"""Load an image, make grayscale and edge versions, and display the results."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "images" / "sample.jpg"
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    """Run the image-processing example for one image file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"input image (default: {DEFAULT_IMAGE})",
    )
    args = parser.parse_args()

    image_path = args.image.expanduser()
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path

    if not image_path.is_file():
        parser.error(
            f"image not found: {image_path}\n"
            "Add a JPG, PNG, or other supported image at "
            f"{DEFAULT_IMAGE}, or pass an image path as an argument."
        )

    # OpenCV loads color images in BGR order. Matplotlib expects RGB order.
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        parser.error(
            f"could not decode image: {image_path}. "
            "Check that the file is a valid image supported by OpenCV."
        )

    height, width, channels = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for output_path, output_image in (
        (RESULTS_DIR / "grayscale.png", grayscale),
        (RESULTS_DIR / "edges.png", edges),
    ):
        if not cv2.imwrite(str(output_path), output_image):
            parser.error(f"could not save output image: {output_path}")

    print(f"Loaded: {image_path}")
    print(f"Color image: shape (height, width, channels) = {image_bgr.shape}")
    print(f"Pixel type: {image_bgr.dtype}")
    print(f"Grayscale image shape: {grayscale.shape}")
    print(
        "Pixel indexing: image[row, column] means image[y, x]; "
        f"valid rows are 0..{height - 1} and columns are 0..{width - 1}."
    )
    print(f"Saved grayscale image: {RESULTS_DIR / 'grayscale.png'}")
    print(f"Saved edge image: {RESULTS_DIR / 'edges.png'}")

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original RGB ({width} x {height}, {channels} channels)")
    axes[1].imshow(grayscale, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale")
    axes[2].imshow(edges, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Gaussian blur + Canny edges")
    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
