from pathlib import Path

import cv2
import numpy as np


project_folder = Path(__file__).parent.parent
photo_folder = project_folder / "photo"
image1 = cv2.imread(str(photo_folder / "view1.jpg"), cv2.IMREAD_COLOR)
image2 = cv2.imread(str(photo_folder / "view2.jpg"), cv2.IMREAD_COLOR)

if image1 is None:
    raise FileNotFoundError(f"Could not read {photo_folder / 'view1.jpg'}")
if image2 is None:
    raise FileNotFoundError(f"Could not read {photo_folder / 'view2.jpg'}")
if image1.shape[:2] != image2.shape[:2]:
    raise ValueError("The two images must have the same height and width.")

# Find distinctive points and describe their surrounding image details.
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

if descriptors1 is None or descriptors2 is None:
    raise RuntimeError("Could not find enough visual details in both images.")

# Match similar descriptors, then keep matches that pass Lowe's ratio test.
matcher = cv2.BFMatcher(cv2.NORM_L2)
nearest_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
good_matches = []
for two_matches in nearest_matches:
    if len(two_matches) == 2:
        best_match, second_best_match = two_matches
        if best_match.distance < 0.75 * second_best_match.distance:
            good_matches.append(best_match)

if len(good_matches) < 5:
    raise RuntimeError(
        f"Only {len(good_matches)} candidate matches found; at least 5 are "
        "needed to estimate camera motion."
    )

# Convert each match into a pair of (x, y) image pixel coordinates.
points1 = np.float32(
    [keypoints1[match.queryIdx].pt for match in good_matches]
)
points2 = np.float32(
    [keypoints2[match.trainIdx].pt for match in good_matches]
)

# Approximate camera intrinsics: focal length is guessed as the image width,
# and the optical center is assumed to be the center of the image.
height, width = image1.shape[:2]
focal_length_guess = float(width)
camera_matrix = np.array(
    [
        [focal_length_guess, 0, width / 2],
        [0, focal_length_guess, height / 2],
        [0, 0, 1],
    ],
    dtype=np.float64,
)

# Estimate the essential matrix and reject matches that disagree with the
# shared camera-motion geometry.
essential_matrix, ransac_mask = cv2.findEssentialMat(
    points1,
    points2,
    camera_matrix,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.5,
)
if essential_matrix is None or ransac_mask is None:
    raise RuntimeError("Could not estimate camera motion from these matches.")

# Recover rotation and translation direction. The translation has unit length,
# because the true distance between camera positions is unknown.
pose_inlier_count, essential_matrix, rotation, translation, pose_mask = (
    cv2.recoverPose(
        points1,
        points2,
        camera_matrix,
        None,
        camera_matrix,
        None,
        E=essential_matrix,
        mask=ransac_mask,
    )
)

# Only triangulate matches that passed RANSAC and the front-of-camera test.
inliers = pose_mask.ravel() > 0
inlier_points1 = points1[inliers]
inlier_points2 = points2[inliers]
inlier_matches = [match for match, keep in zip(good_matches, inliers) if keep]

# Projection matrices map a 3D point to a pixel. Camera 1 is the origin;
# rotation and translation describe camera 2 relative to camera 1.
projection1 = camera_matrix @ np.hstack(
    (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
)
projection2 = camera_matrix @ np.hstack((rotation, translation))

# OpenCV returns homogeneous coordinates (X, Y, Z, W). Divide by W to get
# ordinary 3D coordinates (X/W, Y/W, Z/W).
points4d = cv2.triangulatePoints(
    projection1,
    projection2,
    inlier_points1.T,
    inlier_points2.T,
).T

valid_w = np.abs(points4d[:, 3]) > 1e-8
points3d = np.full((len(points4d), 3), np.nan, dtype=np.float64)
points3d[valid_w] = points4d[valid_w, :3] / points4d[valid_w, 3, np.newaxis]

# A valid reconstructed point must be in front of both cameras.
depth_in_camera1 = points3d[:, 2]
points_in_camera2 = (rotation @ points3d.T + translation).T
depth_in_camera2 = points_in_camera2[:, 2]
in_front = (
    valid_w
    & np.isfinite(points3d).all(axis=1)
    & (depth_in_camera1 > 0)
    & (depth_in_camera2 > 0)
)

# Measure how far each 3D point projects from its observed matched pixels.
rotation_vector, _ = cv2.Rodrigues(rotation)
projected1, _ = cv2.projectPoints(
    points3d, np.zeros(3), np.zeros(3), camera_matrix, None
)
projected2, _ = cv2.projectPoints(
    points3d, rotation_vector, translation, camera_matrix, None
)
projected1 = projected1.reshape(-1, 2)
projected2 = projected2.reshape(-1, 2)
error1 = np.linalg.norm(projected1 - inlier_points1, axis=1)
error2 = np.linalg.norm(projected2 - inlier_points2, axis=1)
mean_reprojection_error = (error1 + error2) / 2

# Exclude invalid points and large reprojection errors from the saved point set.
max_reprojection_error = 4.0
keep = in_front & (mean_reprojection_error <= max_reprojection_error)
final_points = points3d[keep]
final_errors = mean_reprojection_error[keep]
final_matches = [
    match for match, is_valid in zip(inlier_matches, keep) if is_valid
]

print(f"Candidate feature matches: {len(good_matches)}")
print(f"Matches accepted for triangulation: {pose_inlier_count}")
print(f"Points in front of both cameras: {int(in_front.sum())}")
print(
    f"Points kept with mean reprojection error <= {max_reprojection_error:.1f} px: "
    f"{len(final_points)}"
)
if len(final_points) == 0:
    raise RuntimeError(
        "No 3D points survived filtering. Check the image matches and estimated pose."
    )
print(f"Median reprojection error: {float(np.median(final_errors)):.2f} px")
print("First few 3D points (camera 1 coordinate system; arbitrary scale):")
for point in final_points[:5]:
    print(f"  x={point[0]: .4f}, y={point[1]: .4f}, z={point[2]: .4f}")

# Assign each reconstructed point the color of its matching pixel in view 1.
valid_matches = [match for match, is_valid in zip(inlier_matches, keep) if is_valid]
pixel_locations = np.round(
    [keypoints1[match.queryIdx].pt for match in valid_matches]
).astype(int)
pixel_locations[:, 0] = np.clip(pixel_locations[:, 0], 0, width - 1)
pixel_locations[:, 1] = np.clip(pixel_locations[:, 1], 0, height - 1)
colors_bgr = image1[pixel_locations[:, 1], pixel_locations[:, 0]]
colors_rgb = colors_bgr[:, ::-1]

# Write a simple ASCII PLY file so no extra 3D library is needed.
ply_path = project_folder / "point_cloud.ply"
with ply_path.open("w", encoding="ascii") as ply_file:
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write(f"element vertex {len(final_points)}\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")
    for point, color in zip(final_points, colors_rgb):
        ply_file.write(
            f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
            f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
        )

print(f"Saved the colored point cloud: {ply_path}")
print(
    "The point coordinates have arbitrary scale: the two views do not reveal "
    "the real-world camera distance."
)
