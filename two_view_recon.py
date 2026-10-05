"""
Two-view 3D reconstruction: two photos of one object -> sparse point cloud.

Pipeline (each stage is its own function, so you can run/debug one at a time):
  1. load_images        read both photos, check they are the same size
  2. compute_sift       find keypoints + descriptors in each photo
  3. match_features     BFMatcher + Lowe ratio test
  4. save_matches_png   draw the surviving matches so you can eyeball them
  5. estimate_intrinsics  camera matrix K (real calibration, or a guess)
  6. estimate_pose      essential matrix -> R, t  (RANSAC)
  7. triangulate        2D pairs + R, t -> 3D points
  8. filter_points      throw away the bad ones
  9. colorize_and_save  attach colours, show in Open3D, write point_cloud.ply

The result is correct only UP TO SCALE: two photos can never tell you whether the
book was 20 cm or 2 m from the camera. The shape is right; the units are not.

Run with:  ./cv-venv/bin/python two_view_recon.py
"""

import os  # for checking whether a file exists

import cv2  # image reading + all the computer vision maths
import numpy as np  # arrays; the small bits of linear algebra OpenCV does not do
import open3d as o3d  # 3D point cloud handling and the 3D window

# --------------------------------------------------------------------------------------
# Settings you might want to change. Everything else is automatic.
# --------------------------------------------------------------------------------------

# Where the two photos are. We look inside ./photo first, then the current folder.
IMAGE_DIRS = ["photo", "."]
IMAGE_NAMES = ["view1.jpg", "view2.jpg"]

# Lowe ratio test. For every feature in image 1 we compare its best match against its
# second-best match. If the best one is not clearly better, the match is probably a
# coincidence and we throw it away. 0.75 is the classic value from SIFT's paper.
# Lower = stricter (fewer, better matches). Raise it to 0.8 if you have too few matches.
LOWE_RATIO = 0.75

# RANSAC reprojection threshold for findEssentialMat, in PIXELS. A point is called an
# inlier if it is within this many pixels of its epipolar line. 1.0-2.0 is normal.
# Raise it if the object moves slightly between the two shots.
RANSAC_THRESHOLD_PX = 1.5

# Reject a 3D point if re-projecting it back into the photos is off by more than this.
MAX_REPROJECTION_ERROR_PX = 2.0

# Reject a 3D point further away than this, in multiples of the median distance.
# A few matches are wrong in a way that puts them kilometres away; this kills them.
MAX_DISTANCE_FACTOR = 10.0

# Open3D's statistical outlier removal. A point needs at least this many neighbours
# within the local neighbourhood, and they must be within std_ratio standard
# deviations of it, or it is deleted as a speck of dirt.
# This is the MAXIMUM we allow; on a small cloud the script scales it down, because
# "have 20 neighbours" is impossible when there are only 40 points in total.
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

OUTPUT_PLY = "point_cloud.ply"
MATCHES_PNG = "matches.png"
CALIB_FILE = "calib.npz"  # written by calibrate_camera.py


# ======================================================================================
# STAGE 1 - load the images
# ======================================================================================

def load_image(path):
    """Read one image file and return it as a NumPy array of pixels.

    TRAP: cv2.imread does not raise an error when it cannot open a file. It quietly
    returns None, and you only find out much later when some unrelated line complains
    about "unsupported operand" on None. So we check immediately.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(
            f"Could not open '{path}'. Check the file exists, and that the "
            f"spelling matches exactly (Linux is case-sensitive: View1.jpg != view1.jpg)."
        )
    return img


def load_images():
    """Find and load both photos. Returns (img1, img2)."""
    img1 = img2 = None
    for folder in IMAGE_DIRS:
        if not os.path.isdir(folder):
            continue
        candidate = [os.path.join(folder, name) for name in IMAGE_NAMES]
        if all(os.path.exists(p) for p in candidate):
            # cvtColor(BGR -> RGB) is NOT done here. OpenCV works in BGR everywhere and
            # converting now would only tempt us to mix the two up later. We convert
            # only at the very last moment, for display.
            img1, img2 = (load_image(p) for p in candidate)
            print(f"[1] Loaded {candidate[0]} and {candidate[1]}")
            break

    if img1 is None:
        raise FileNotFoundError(
            f"Could not find {' and '.join(IMAGE_NAMES)} in any of {IMAGE_DIRS}.\n"
            f"Put your two photos in ./photo/ and re-run."
        )

    # Both photos must be the same size, otherwise a pixel in one has no partner in
    # the other and nothing can be matched.
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        raise ValueError(
            f"Photo sizes differ: view1 is {w1}x{h1}, view2 is {w2}x{h2}. "
            f"Resize them to the same size, or re-shoot with identical camera settings."
        )

    print(f"    both are {w1} x {h1} pixels, dtype {img1.dtype}")
    return img1, img2


# ======================================================================================
# STAGE 2 - SIFT keypoints and descriptors
# ======================================================================================

def compute_sift(img, label):
    """Find interesting points in one image and describe each one.

    A keypoint is a pixel location that is easy to find again in another photo (a
    corner, a dot, a corner of a letter). Its descriptor is a 128-number fingerprint of
    the local texture around it. If two patches look alike, their fingerprints are close.
    SIFT is rotation- and scale-invariant, which is why it still works if your two photos
    are not perfectly aligned or level.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # SIFT works on single-channel grey
    # nfeatures=0 means "keep as many as you want". contrastThreshold is how strong a
    # feature must stand out from its surroundings to be kept; raising it gives fewer,
    # stronger features.
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # If the photo is blurry, low-contrast or textureless, this comes back as 0. That
    # is not a bug in your code, it is the photo telling you it has nothing to offer.
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError(
            f"SIFT found 0 features in {label}. Usual causes: the photo is too blurry, "
            f"too dark, too flat/low-contrast, or the object fills too little of the frame."
        )

    # keypoints[i].pt is a tuple (x, y) -- note the order: x first, y second.
    print(f"[2] SIFT on {label}: {len(keypoints)} keypoints, "
          f"descriptors shape {descriptors.shape} ({descriptors.dtype})")
    return keypoints, descriptors


# ======================================================================================
# STAGE 3 - match descriptors, then apply Lowe's ratio test
# ======================================================================================

def match_features(descriptors1, descriptors2):
    """Find which feature in image 2 corresponds to each feature in image 1.

    Analogy: imagine sorting a deck of fingerprints from each photo and looking for
    pairs that look alike. The subtlety is that a lot of them look *slightly* alike by
    chance. Lowe's test says: only trust a match if the best candidate is clearly
    better than the runner-up. If the 2nd choice is nearly as good, we have no idea
    which is right, so we drop the pair.
    """
    # NORM_L2 because SIFT descriptors are vectors of floats and we compare them
    # by Euclidean distance. (SIFT's internal matching uses squared L2, but the
    # ratio test works identically either way, and this is the documented pairing.)
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)

    # knnMatch with k=2 returns, for every descriptor in image 1, the two best
    # candidates from image 2.
    knn = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for pair in knn:
        # A descriptor in image 2 can be the nearest neighbour of several descriptors in
        # image 1. Those pairs are only kept if they are mutual: d1->d2 must also be the
        # best match d2->d1. This removes most one-to-many mistakes.
        if len(pair) == 2:
            best, second_best = pair
            if best.distance < LOWE_RATIO * second_best.distance:
                good_matches.append(best)

    print(f"[3] Lowe ratio test at {LOWE_RATIO}: {len(good_matches)} good matches kept")
    if len(good_matches) < 15:
        raise RuntimeError(
            f"Only {len(good_matches)} good matches -- that is too few to recover a pose "
            f"(we need at least 5, and realistically 30+). Causes, in order of likelihood:\n"
            f"  1. Not enough overlap between the two photos -- re-shoot with 60-70% shared.\n"
            f"  2. The object is textureless or repetitive -- SIFT has nothing distinctive.\n"
            f"  3. The object is too small or too blurry in the frame.\n"
            f"  4. You are about to raise LOWE_RATIO to 0.85 just to see what happens."
        )
    return good_matches


def matches_to_points(matches, keypoints1, keypoints2):
    """Turn OpenCV match objects into two matching lists of (x, y) pixel coordinates.

    IMPORTANT SHAPES. OpenCV wants:
      - findEssentialMat / recoverPose / triangulatePoints: float32, shape (N, 1, 2)
        (N points, each a single 1x2 pixel point). Being float32 and C-contiguous matters.
      - the ratio test's .queryIdx / .trainIdx: plain Python ints.
    And a (N, 2) array is NOT interchangeable with an (N, 1, 2) array here.
    """
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return np.ascontiguousarray(pts1), np.ascontiguousarray(pts2)


# ======================================================================================
# STAGE 4 - draw the matches so you can check them by eye
# ======================================================================================

def save_matches_png(img1, img2, keypoints1, keypoints2, matches):
    """Write a side-by-side picture with lines joining matched points.

    Do not skip looking at this file. It is the single fastest way to tell whether the
    rest of the pipeline deserves to be trusted. The lines should look mostly parallel,
    all heading roughly the same way, and never criss-cross. Lots of crossing lines or
    lines going off in random directions means the matches are mostly nonsense.
    """
    # matchColor=(0, 255, 0) green lines, matchesMask=None means draw ALL of them,
    # singlePointColor=(0, 0, 255) red dots at each keypoint.
    canvas = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                             matchColor=(0, 255, 0),
                             singlePointColor=(0, 0, 255))
    cv2.imwrite(MATCHES_PNG, canvas)
    print(f"[4] Wrote {MATCHES_PNG} ({len(matches)} lines). "
          f"Open it and check the lines run roughly parallel.")


# ======================================================================================
# STAGE 5 - camera intrinsics K
# ======================================================================================

def estimate_intrinsics(image_shape):
    """Build the camera matrix K, preferring a real calibration if one exists.

    K answers: "where does the camera sit inside the photo?" It says where the optical
    axis points (cx, cy) and how many pixels wide one radian of view is (fx, fy).

    Two ways to get it:
      (a) Calibrate with a checkerboard using calibrate_camera.py. Accurate.
      (b) Guess: fx = fy = image width, cx = w/2, cy = h/2. That assumes a ~90 degree
          horizontal field of view. Real phone cameras are narrower (~65-70 degrees), so
          the true focal length is more like 1.3x the width -- but a wrong fx mostly
          just rescales the result, and since we are correct only up to scale anyway,
          the guess is fine to get started.
    """
    h, w = image_shape[:2]
    if os.path.exists(CALIB_FILE):
        data = np.load(CALIB_FILE)
        K = data["K"].astype(np.float64)
        dist = data["dist_coeffs"]
        print(f"[5] Using the calibrated camera from {CALIB_FILE}:\n{K}")
        if dist is not None and np.any(np.abs(dist) > 1e-6):
            print(f"    distortion coefficients: {dist.ravel()}")
            print("    NOTE: these points are undistorted only for the projection math "
                  "below; colours are sampled from the ORIGINAL (distorted) photos.")
        return K

    fx = fy = float(w)
    K = np.array([[fx, 0.0, w / 2.0],
                  [0.0, fy, h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    print(f"[5] No {CALIB_FILE} found, so guessing K (fx = fy = width = {w:.0f}):\n{K}")
    print("    Run calibrate_camera.py for a real K, then re-run this script.")
    return K


# ======================================================================================
# STAGE 6 - essential matrix, then relative pose (R, t)
# ======================================================================================

def estimate_pose(pts1, pts2, K):
    """Recover how the second camera sits relative to the first. Returns (R, t, mask).

    The idea in one breath: each matched pair of points defines one line in space -- the
    ray through the pixel in view 1, and the ray through its partner in view 2. If the
    cameras were at the same place, the rays would pass through the same point. Because
    they are not, each pair of rays defines a plane, and the true scene point must lie
    somewhere along both rays. The essential matrix is the algebraic statement of that
    constraint for all pairs at once. RANSAC throws out the lying pairs (mismatches) and
    fits the rest; recoverPose then splits the essential matrix into a rotation R and a
    translation direction t.

    Scale warning: t comes back with length 1, not a real distance. Two photos cannot
    measure absolute size -- if you scale the whole scene up, the photos look identical.
    """
    # threshold is in pixels. prob is how confident RANSAC should be that it found the
    # right answer; 0.999 with maxIters=0 lets OpenCV pick the iteration count itself.
    E, essential_mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=RANSAC_THRESHOLD_PX,
    )
    if E is None or E.shape != (3, 3):
        raise RuntimeError("findEssentialMat could not fit an essential matrix.")

    # TRAP: this mask contains 1 for inlier, 0 for outlier -- but recoverPose's mask
    # contains 255. So never write mask.sum() to count points; use (mask > 0).sum().
    n_essential_inliers = int((essential_mask.ravel() > 0).sum())

    # Signatures differ between OpenCV versions. I tested the classic 4-argument form
    # below on your build (OpenCV 5.0.0) and it works, returning (n, R, t, mask).
    # OpenCV 5 also offers recoverPose(pts1, pts2, K1, None, K2, None[, E]) which
    # re-estimates E internally; both agree to ~1e-7 on synthetic data I tested.
    n_good, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)

    mask = (pose_mask.ravel() > 0)  # boolean array: True = this match is trustworthy
    baseline_deg = np.degrees(np.arccos(np.clip(abs(t[2, 0]), -1.0, 1.0)))

    print(f"[6] Essential matrix: {n_essential_inliers} inliers (RANSAC, {RANSAC_THRESHOLD_PX} px)")
    print(f"    recoverPose: {n_good} points pass the cheirality check "
          f"(i.e. the point is in front of BOTH cameras)")
    print(f"    rotation matrix R:\n{R}")
    print(f"    translation direction t (unit length, scale unknown):\n{t.ravel().round(4)}")
    print(f"    'Baseline' angle of t away from the optical axis: {baseline_deg:.1f} deg")

    if n_good < 8:
        raise RuntimeError(
            f"Only {n_good} points survived pose recovery. The two photos are probably "
            f"too similar (not enough baseline) or the matches are bad. See the notes "
            f"printed by match_features()."
        )
    if baseline_deg < 5.0:
        print("    WARNING: the two cameras barely moved relative to each other. "
              "Parallax is too small to triangulate accurately -- move further apart.")
    return R, t, mask


# ======================================================================================
# STAGE 7 - triangulate
# ======================================================================================

def triangulate(pts1, pts2, K, R, t):
    """Turn each matched pixel pair into one 3D point. Returns points3D, shape (N, 3).

    Both camera poses are written as 3x4 projection matrices:
        P1 = K [ I | 0 ]   the first camera is at the origin, looking down +Z
        P2 = K [ R | t ]   the second camera is rotated by R and moved by t
    triangulatePoints returns each point in homogeneous form, 4 numbers (X, Y, Z, W)
    meaning (X/W, Y/W, Z/W). Dividing through by W is the "convert from homogeneous"
    step -- skip it and your points will have a weird extra scale factor.
    """
    # In world coordinates we choose the first camera to be at the origin, so its
    # projection matrix is K with a zero translation column.
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = np.hstack([K @ R, (K @ t).reshape(3, 1)])

    # TRAP: both point arrays must be float32 with shape (N, 1, 2). int, (N, 2) or
    # (N, 2, 1) all get rejected with a shape error, or worse, quietly misread.
    points4d = cv2.triangulatePoints(P1, P2, pts1, pts2)  # shape (4, N)
    points3d = (points4d[:3] / points4d[3]).T             # -> shape (N, 3)

    depths = points3d[:, 2]
    print(f"[7] Triangulated {len(points3d)} points")
    print(f"    depth (Z) range: {depths.min():.3f} .. {depths.max():.3f}")
    print(f"    finite points  : {int(np.isfinite(points3d).all(axis=1).sum())} "
          f"(the rest are at infinity -- a sign of exactly-parallel rays)")
    return points3d


# ======================================================================================
# STAGE 8 - throw away the bad points
# ======================================================================================

def filter_points(points3d, pts1, pts2, K, R, t, mask):
    """Remove points that cannot possibly be real. Returns points3d, colors, keep_mask.

    Four filters, cheapest first:
      a. only points whose match passed the pose check
      b. behind the camera (negative depth in either view -- physically impossible)
      c. reprojection error: does the point actually land back on the right pixels?
      d. statistical outliers: are there enough neighbours nearby for this to be
         surface detail rather than noise?
    """
    n_start = len(points3d)
    keep = mask.copy()
    print(f"[8] Filtering {n_start} points:")

    # (a) matches that failed RANSAC / cheirality
    keep &= np.isfinite(points3d).all(axis=1)
    print(f"    (a) after finite + pose-inlier test : {int(keep.sum())}")

    # (b) cheirality: depth must be positive in both cameras. A point behind the lens
    # cannot have been photographed.
    z_cam1 = points3d[:, 2]
    z_cam2 = (points3d @ R.T + t.ravel())[:, 2]
    keep &= (z_cam1 > 0) & (z_cam2 > 0)
    print(f"    (b) after behind-the-camera test    : {int(keep.sum())}")

    # (c) reprojection error. Project the 3D point back into both images and compare
    # with where the keypoint actually was. A point built from a bad match will land in
    # the wrong place.
    keep_idx = np.where(keep)[0]
    X = points3d[keep_idx]
    uv1 = X @ K.T
    uv1 = uv1[:, :2] / uv1[:, 2:3]
    err1 = np.linalg.norm(uv1 - pts1[keep_idx, 0, :], axis=1)

    X2 = X @ R.T + t.T
    uv2 = (X2 @ K.T)[:, :2] / (X2 @ K.T)[:, 2:3]
    err2 = np.linalg.norm(uv2 - pts2[keep_idx, 0, :], axis=1)

    reproj_err = np.maximum(err1, err2)
    good_reproj = reproj_err < MAX_REPROJECTION_ERROR_PX
    print(f"    (c) after reprojection test <{MAX_REPROJECTION_ERROR_PX} px  : "
          f"{int(keep_idx.size * good_reproj.mean())} "
          f"(mean error {reproj_err[good_reproj].mean():.2f} px)")
    keep[keep_idx[~good_reproj]] = False

    # (d) statistical outliers, done by Open3D because it is a nearest-neighbour search
    # over 3D positions and it is far too slow to write by hand in numpy.
    #
    # The method compares each point's distance to its k nearest neighbours against the
    # average such distance. So k MUST be smaller than the number of points available --
    # asking for 20 neighbours when only 40 points survive deletes almost everything.
    idx = np.where(keep)[0]
    nb_neighbors = max(4, min(OUTLIER_NB_NEIGHBORS, len(idx) // 6))
    if len(idx) > nb_neighbors + 1:
        scratch = o3d.geometry.PointCloud()
        scratch.points = o3d.utility.Vector3dVector(points3d[idx])
        scratch, kept_local = scratch.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=OUTLIER_STD_RATIO)
        # TRAP: Open3D returns the indices it KEPT, not the ones it deleted -- and that
        # list is SHORTER than the input, so it cannot be used to mask directly.
        # (Writing keep[idx[kept_local]] = False throws away exactly the good points.)
        # Build a "was this point deleted?" boolean instead.
        kept_local = np.asarray(kept_local).ravel()
        deleted = np.ones(len(idx), dtype=bool)
        deleted[kept_local] = False
        keep[idx[deleted]] = False
        print(f"    (d) after statistical outlier test : {int(keep.sum())} "
              f"(k={nb_neighbors} nearest neighbours, {int(deleted.sum())} deleted)")
    else:
        print(f"    (d) statistical outlier test skipped: only {len(idx)} points left")

    # One more guard: anything absurdly far away. With a mostly-good cloud, the median
    # distance is a sensible yardstick; anything 10x beyond it is a broken match.
    idx = np.where(keep)[0]
    if len(idx) > 0:
        dist = np.linalg.norm(points3d[idx], axis=1)
        median = np.median(dist)
        too_far = dist > median * MAX_DISTANCE_FACTOR
        keep[idx[too_far]] = False
        print(f"    (e) after distance < {MAX_DISTANCE_FACTOR} x median "
              f"({median:.2f})     : {int(keep.sum())}")

    print(f"    kept {int(keep.sum())} of {n_start} points "
          f"({100.0 * keep.sum() / max(n_start, 1):.0f}%)")
    if keep.sum() < 20:
        raise RuntimeError(
            f"Only {int(keep.sum())} points survived filtering -- too few to see a shape. "
            f"Loosen MAX_REPROJECTION_ERROR_PX / OUTLIER_STD_RATIO, and check your matches.png."
        )
    return keep


# ======================================================================================
# STAGE 9 - colours, Open3D, save
# ======================================================================================

def colorize_and_save(points3d, pts1, img1, keep):
    """Sample each surviving point's colour from photo 1, build a point cloud, save it."""
    final_points = points3d[keep]
    final_pixels = pts1[keep, 0, :]  # the pixel each point came from

    # Pixel coordinates are floats; image indices must be whole numbers.
    # astype(int) TRUNCATES rather than rounds, and an out-of-range index crashes, so
    # round first then clip into the valid range.
    cols = np.round(final_pixels).astype(int)
    cols[:, 0] = np.clip(cols[:, 0], 0, img1.shape[1] - 1)
    cols[:, 1] = np.clip(cols[:, 1], 0, img1.shape[0] - 1)

    # cv2 gives BGR, Open3D wants RGB -- and Open3D wants 0..1 floats, not 0..255.
    bgr = img1[cols[:, 1], cols[:, 0]].astype(np.float64) / 255.0
    rgb = bgr[:, [2, 1, 0]]

    cloud = o3d.geometry.PointCloud()
    # Vector3dVector is just a typed wrapper Open3D needs; it copies the numpy data.
    cloud.points = o3d.utility.Vector3dVector(final_points)
    cloud.colors = o3d.utility.Vector3dVector(rgb)

    # write_point_cloud infers the format from the .ply extension. 'binary_compressed'
    # is a valid flag but plain 'ascii' is readable in any 3D viewer and in a text editor.
    o3d.io.write_point_cloud(OUTPUT_PLY, cloud, write_ascii=True)
    print(f"[9] Wrote {OUTPUT_PLY}: {len(cloud.points)} coloured points")

    # Open it in the interactive Open3D window. This needs a graphical display; on a
    # headless machine or over a plain SSH session it raises instead of opening.
    try:
        o3d.visualization.draw_geometries([cloud])
        print("    (close the Open3D window to continue)")
    except Exception as exc:
        print(f"    Could not open the Open3D window ({type(exc).__name__}: {exc}).")
        print(f"    Open {OUTPUT_PLY} in MeshLab, CloudCompare, or online at "
              f"https://3dviewer.net instead.")
    return cloud


# ======================================================================================
# put it all together
# ======================================================================================

def main():
    print("=" * 70)
    print("TWO-VIEW 3D RECONSTRUCTION")
    print("=" * 70)

    # 1. load
    img1, img2 = load_images()

    # 2. SIFT
    keypoints1, descriptors1 = compute_sift(img1, "view1")
    keypoints2, descriptors2 = compute_sift(img2, "view2")

    # 3. match + ratio test
    matches = match_features(descriptors1, descriptors2)

    # 4. draw them for inspection
    save_matches_png(img1, img2, keypoints1, keypoints2, matches)

    # 5. intrinsics
    K = estimate_intrinsics(img1.shape)

    # Convert matches to pixel arrays AFTER matching, and only once.
    pts1, pts2 = matches_to_points(matches, keypoints1, keypoints2)
    print(f"    matched pixel arrays: pts1 {pts1.shape} {pts1.dtype}, "
          f"pts2 {pts2.shape} {pts2.dtype}")

    # 6. pose
    R, t, pose_mask = estimate_pose(pts1, pts2, K)

    # 7. triangulate
    points3d = triangulate(pts1, pts2, K, R, t)

    # 8. filter
    keep = filter_points(points3d, pts1, pts2, K, R, t, pose_mask)

    # 9. colour + save
    colorize_and_save(points3d, pts1, img1, keep)

    print("=" * 70)
    print("Done. Sanity checks on your result:")
    print("  * Open matches.png first. Lines should be nearly parallel, not tangled.")
    print("  * Open point_cloud.ply. You should see ONE blob shaped like your object,")
    print("    not a flat sheet (that means it failed and defaulted to something).")
    print("  * If the cloud looks like two overlapping copies of the object, the pose")
    print("    is wrong. If it looks like a stretched blur, K is wrong.")
    print("=" * 70)


if __name__ == "__main__":
    main()