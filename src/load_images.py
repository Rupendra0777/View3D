from pathlib import Path

import cv2


# Find the photo folder next to this script, even if the script is run from
# another working directory.
project_folder = Path(__file__).parent.parent
photo_folder = project_folder / "photo"
image1_path = photo_folder / "view1.jpg"
image2_path = photo_folder / "view2.jpg"

# OpenCV loads color images in BGR order (blue, green, red).
image1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
image2 = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)

# imread returns None when it cannot read a file, so check both images.
if image1 is None:
    raise FileNotFoundError(f"Could not read the first image: {image1_path}")
if image2 is None:
    raise FileNotFoundError(f"Could not read the second image: {image2_path}")

# Both views must have the same height and width for later steps.
if image1.shape[:2] != image2.shape[:2]:
    raise ValueError(
        "The images must have the same dimensions. "
        f"First image: {image1.shape[1]}x{image1.shape[0]}, "
        f"second image: {image2.shape[1]}x{image2.shape[0]}."
    )

height, width = image1.shape[:2]
print(f"Both images loaded successfully: {width} x {height} pixels.")

# Put the views next to each other so they are easy to compare.
both_images = cv2.hconcat([image1, image2])

# Shrink only the display copy if it is too wide for the screen.
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

cv2.imshow("View 1 (left) and View 2 (right) - press any key to close", both_images)
cv2.waitKey(0)
cv2.destroyAllWindows()
