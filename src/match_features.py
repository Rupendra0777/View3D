import argparse
from pathlib import Path

import cv2


parser = argparse.ArgumentParser(description="Find and visualize feature matches.")
parser.add_argument(
    "--no-display",
    action="store_true",
    help="Save the match preview without opening a window.",
)
args = parser.parse_args()

project_folder = Path(__file__).parent.parent
photo_folder = project_folder / "photo"
image1_path = photo_folder / "view1.jpg"
image2_path = photo_folder / "view2.jpg"

image1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
image2 = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)

if image1 is None:
    raise FileNotFoundError(f"Could not read {image1_path}")
if image2 is None:
    raise FileNotFoundError(f"Could not read {image2_path}")
if image1.shape[:2] != image2.shape[:2]:
    raise ValueError("The two images must have the same height and width.")

# Find keypoints and descriptors in each image. A descriptor is a numeric
# description of the image area around one keypoint.
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

if descriptors1 is None or descriptors2 is None:
    raise RuntimeError("Could not find enough visual details to match.")
if len(descriptors1) < 2 or len(descriptors2) < 2:
    raise RuntimeError("Each image needs at least two visual details to match.")

# SIFT descriptors are numeric vectors, so BFMatcher uses L2 distance to
# measure how different two descriptors are. Smaller distance means more alike.
matcher = cv2.BFMatcher(cv2.NORM_L2)
nearest_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Keep a match only when its best candidate is clearly better than its second
# best candidate. This is called Lowe's ratio test.
ratio_threshold = 0.75
good_matches = []
for two_matches in nearest_matches:
    if len(two_matches) == 2:
        best_match, second_best_match = two_matches
        if best_match.distance < ratio_threshold * second_best_match.distance:
            good_matches.append(best_match)

print(f"View 1 keypoints: {len(keypoints1)}")
print(f"View 2 keypoints: {len(keypoints2)}")
print(f"Matches kept by the ratio test: {len(good_matches)}")

# Draw only the 100 lowest-distance matches in the preview to keep it readable.
# All good_matches remain available for the next reconstruction step.
preview_matches = sorted(good_matches, key=lambda match: match.distance)[:100]
match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    preview_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

output_path = project_folder / "matches.png"
if not cv2.imwrite(str(output_path), match_image):
    raise OSError(f"Could not save the match preview to {output_path}")
print(f"Saved a preview of up to {len(preview_matches)} matches: {output_path}")

# Show a smaller copy on screen; the saved preview retains its full resolution.
max_display_width = 1400
if match_image.shape[1] > max_display_width:
    scale = max_display_width / match_image.shape[1]
    display_size = (
        int(match_image.shape[1] * scale),
        int(match_image.shape[0] * scale),
    )
    match_image = cv2.resize(
        match_image, display_size, interpolation=cv2.INTER_AREA
    )

if not args.no_display:
    cv2.imshow("Feature matches - press any key to close", match_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
