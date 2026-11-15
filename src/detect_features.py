from pathlib import Path

import cv2


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

# SIFT looks for distinctive areas, such as corners and textured details.
sift = cv2.SIFT_create()

# Keypoints are the locations SIFT found. Descriptors are number-arrays that
# describe the appearance around each keypoint; we will use them for matching.
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

print(f"View 1: found {len(keypoints1)} distinctive points.")
print(f"View 2: found {len(keypoints2)} distinctive points.")
print("The points are detected separately; we have not matched them yet.")

# Draw each keypoint on its original image. The circle indicates its location
# and approximate size.
image1_with_points = cv2.drawKeypoints(
    image1,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
image2_with_points = cv2.drawKeypoints(
    image2,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

both_images = cv2.hconcat([image1_with_points, image2_with_points])

# Resize only for display so both views fit on a typical screen.
max_display_width = 1400
if both_images.shape[1] > max_display_width:
    scale = max_display_width / both_images.shape[1]
    display_size = (
        int(both_images.shape[1] * scale),
        int(both_images.shape[0] * scale),
    )
    both_images = cv2.resize(
        both_images, display_size, interpolation=cv2.INTER_AREA
    )

cv2.imshow(
    "SIFT keypoints: View 1 (left), View 2 (right) - press any key to close",
    both_images,
)
cv2.waitKey(0)
cv2.destroyAllWindows()
