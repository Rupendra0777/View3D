"""
OPTIONAL STAGE 5b - find the real camera intrinsics K with a checkerboard.

Why bother? K says where the camera is inside the photo (focal length in pixels, and
where the optical axis points). two_view_recon.py will guess it if this file does not
exist, and the guess is usually good enough to get a recognisable shape. But a real K
gives a measurably better point cloud, so this is worth doing once per phone/camera.

How it works. Print a chessboard (9x6 squares, say) and photograph it about 15-20 times,
tilting the board each time. Each photo gives us many points where squares meet, and
because we know the board's real geometry, OpenCV can solve for the camera that projects
those known 3D points onto those known 2D points. With enough photos covering enough
different tilts, the answer is well constrained.

Instructions:
  1. Print a chessboard on A4 at 100% size, do NOT let the page scale it. Measure one
     square with a ruler in millimetres -- you need that number below.
  2. Put it on the floor and photograph it from ~20 different angles, filling the frame.
     Hold the phone in ONE hand and move only the board. Vary the tilt a lot; if all
     your photos look the same, the calibration will be bad.
  3. Put the photos in ./calib/ named calib01.jpg, calib02.jpg, ...
  4. Edit PATTERN_COLS / PATTERN_ROWS / SQUARE_SIZE_MM below to match your board.
     These are the number of squares, and the number of INNER corners is one less in
     each direction -- PATTERN_SIZE must be the inner-corner count, not the square count.
  5. Run:  ./cv-venv/bin/python calibrate_camera.py

Output: calib.npz, which two_view_recon.py picks up automatically on its next run.
"""

import os

import cv2
import numpy as np

CALIB_DIRS = ["calib", "photo", "."]
CALIB_NAME_PREFIX = "calib"  # calib01.jpg, calib02.jpg, ...

# --- EDIT THESE THREE TO MATCH YOUR PRINTED BOARD ---------------------------------
PATTERN_COLS = 9  # INNER corners horizontally (a 9-square-wide board has 8)
PATTERN_ROWS = 6  # INNER corners vertically
SQUARE_SIZE_MM = 25.0  # measure one square with a ruler
# ----------------------------------------------------------------------------------

# cornerSubPix works on grey uint8 images and wants (x, y) pixel coords, float32.
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

OUTPUT_FILE = "calib.npz"


def find_calib_images():
    """Return every checkerboard photo we can find, sorted by filename."""
    for folder in CALIB_DIRS:
        if not os.path.isdir(folder):
            continue
        names = sorted(n for n in os.listdir(folder)
                       if n.lower().startswith(CALIB_NAME_PREFIX)
                       and n.lower().endswith((".jpg", ".jpeg", ".png")))
        if names:
            paths = [os.path.join(folder, n) for n in names]
            print(f"Found {len(paths)} calibration photo(s) in ./{folder}")
            return paths
    raise FileNotFoundError(
        f"No photos starting with '{CALIB_NAME_PREFIX}' found in {CALIB_DIRS}.\n"
        f"Take ~20 photos of a printed chessboard and put them in ./calib/."
    )


def object_points_for_one_board():
    """The known 3D positions of every inner corner of the board.

    We can pick Z = 0 for the whole board because the board is flat -- that just fixes
    the world coordinate system, and the answer K does not depend on it.
    """
    pattern = np.zeros((PATTERN_ROWS, PATTERN_COLS, 3), np.float32)
    pattern[:, :, 0] = np.arange(PATTERN_COLS)      # x along the board, in mm
    pattern[:, :, 1] = np.arange(PATTERN_ROWS)      # y along the board, in mm
    return pattern.reshape(-1, 3)                    # -> shape (N, 3)


def main():
    print("=" * 70)
    print("CHECKERBOARD CAMERA CALIBRATION")
    print(f"board: {PATTERN_COLS} x {PATTERN_ROWS} inner corners, "
          f"{SQUARE_SIZE_MM} mm per square")
    print("=" * 70)

    pattern_size = (PATTERN_COLS, PATTERN_ROWS)
    # This 3D point list gets reused for every photo, which is the whole point.
    object_template = object_points_for_one_board()

    obj_points = []  # list of (N, 3) arrays: known board geometry
    img_points = []  # list of (N, 1, 2) arrays: where those corners appeared
    image_size = None
    failed = []

    for path in find_calib_images():
        img = cv2.imread(path)
        if img is None:
            print(f"  SKIP {path}: could not read it")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if not found:
            # The usual cause is not a bad photo: it is that the pattern size is wrong,
            # or the photo is too blurry/dark for the inner corners to be located.
            print(f"  SKIP {os.path.basename(path)}: board not found. "
                  f"(blur? glare? wrong PATTERN_COLS/ROWS? board cropped?)")
            failed.append(path)
            continue

        # Refine the corner positions to sub-pixel accuracy.
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

        obj_points.append(object_template.copy())
        # .reshape(-1, 1, 2) because calibrateCamera wants each point as a 1x2 array.
        img_points.append(corners.reshape(-1, 1, 2))
        image_size = gray.shape[::-1]  # shape[::-1] reverses to (width, height)
        print(f"  OK   {os.path.basename(path)}: {len(corners)} corners")

    n_good = len(obj_points)
    print(f"\n{n_good} usable photos, {len(failed)} skipped.")
    print(f"Calibrating with image size {image_size}")

    if n_good < 8:
        raise RuntimeError(
            f"Only {n_good} usable photos; you need at least 8, and 15-20 is better. "
            f"Retake: bigger board in frame, more varied tilt, more even light."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points=obj_points,
        image_points=img_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    # rms is the average reprojection error, in pixels. Under 1.0 px is good;
    # under 0.5 px is excellent; over 2.0 px means the photos are sloppy.
    print(f"\nRMS reprojection error: {rms:.4f} px")
    print(f"\nCamera matrix K:\n{camera_matrix}")
    print(f"\nDistortion coefficients (k1, k2, p1, p2, k3):\n{dist_coeffs.ravel()}")
    print(f"\nfx = {camera_matrix[0, 0]:.1f} px, fy = {camera_matrix[1, 1]:.1f} px")
    print(f"cx = {camera_matrix[0, 2]:.1f} px, cy = {camera_matrix[1, 2]:.1f} px")

    # The most useful sanity check: fx tells us the real horizontal field of view.
    # fx = width  =>  ~90 degrees.  fx = 1.4 * width  =>  ~70 degrees (typical phone).
    width = image_size[0]
    hfov = np.degrees(2.0 * np.arctan(width / (2.0 * camera_matrix[0, 0])))
    print(f"\nThat implies a horizontal field of view of about {hfov:.1f} degrees.")
    print("(Most phone cameras are 65-75 degrees. A number far from that means the")
    print(" calibration is probably wrong -- check PATTERN_COLS/ROWS and SQUARE_SIZE.)")

    if rms > 2.0:
        print("\nWARNING: RMS is above 2 px. Results will be usable but not great.")

    np.savez(OUTPUT_FILE, K=camera_matrix, dist_coeffs=dist_coeffs, rms=rms)
    print(f"\nSaved {OUTPUT_FILE}. two_view_recon.py will now use this K automatically,")
    print("as long as calib.npz sits in the same folder as two_view_recon.py.")


if __name__ == "__main__":
    main()