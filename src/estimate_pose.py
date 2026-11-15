import argparse
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser(
    description="Estimate the relative viewpoint change between two images."
)
parser.add_argument(
    "--no-display",
    action="store_true",
    help="Save the inlier preview without opening a window.",
)
args = parser.parse_args()

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

# Step 1: find points and descriptors in each view.
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

if descriptors1 is None or descriptors2 is None:
    raise RuntimeError("Could not find enough visual details in both images.")

# Step 2: match similar descriptors and keep confident matches.
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
        f"Only {len(good_matches)} matches found; at least 5 are needed "
        "to estimate the essential matrix."
    )

# Each match links a pixel in view 1 to its corresponding pixel in view 2.
points1 = np.float32(
    [keypoints1[match.queryIdx].pt for match in good_matches]
).reshape(-1, 1, 2)
points2 = np.float32(
    [keypoints2[match.trainIdx].pt for match in good_matches]
).reshape(-1, 1, 2)

# The camera's exact lens settings are unknown, so start with an approximate
# focal length equal to the image width and the optical center at image center.
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

# RANSAC tries to find a camera-motion model supported by the matches, while
# rejecting matches that do not fit the same two-view geometry.
essential_matrix, ransac_mask = cv2.findEssentialMat(
    points1,
    points2,
    camera_matrix,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.5,
)
if essential_matrix is None or ransac_mask is None:
    raise RuntimeError("Could not estimate the essential matrix from these matches.")

# Recover rotation and translation direction. The translation has unit length:
# two views alone cannot tell us the real distance or scale.
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

rotation_cosine = (np.trace(rotation) - 1) / 2
rotation_cosine = np.clip(rotation_cosine, -1.0, 1.0)
rotation_angle_degrees = np.degrees(np.arccos(rotation_cosine))

print(f"View 1 keypoints: {len(keypoints1)}")
print(f"View 2 keypoints: {len(keypoints2)}")
print(f"Feature matches before geometry check: {len(good_matches)}")
print(f"Matches accepted by RANSAC and pose check: {pose_inlier_count}")
print("Estimated relative rotation matrix:")
print(rotation)
print(f"Overall rotation angle: {rotation_angle_degrees:.2f} degrees")
print("Estimated translation direction (unit length; no real-world scale):")
print(translation.ravel())
print(
    "Camera intrinsics are approximate guesses. Treat this pose as an estimate, "
    "especially because the views are AI-generated."
)

# Show only matches that survived both the RANSAC geometry test and pose check.
pose_mask_flat = pose_mask.ravel() > 0
inlier_matches = [
    match for match, is_inlier in zip(good_matches, pose_mask_flat) if is_inlier
]
inlier_preview = sorted(inlier_matches, key=lambda match: match.distance)[:100]
inlier_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_preview,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

output_path = project_folder / "pose_inliers.png"
if not cv2.imwrite(str(output_path), inlier_image):
    raise OSError(f"Could not save the inlier preview to {output_path}")
print(f"Saved geometry-checked matches to: {output_path}")

max_display_width = 1400
if inlier_image.shape[1] > max_display_width:
    scale = max_display_width / inlier_image.shape[1]
    display_size = (
        int(inlier_image.shape[1] * scale),
        int(inlier_image.shape[0] * scale),
    )
    inlier_image = cv2.resize(
        inlier_image, display_size, interpolation=cv2.INTER_AREA
    )

if not args.no_display:
    cv2.imshow("Geometry-checked matches - press any key to close", inlier_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
