"""Detect SIFT features in two views and save keypoint visualizations."""

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_1 = PROJECT_ROOT / "data" / "images" / "object1.jpg"
DEFAULT_IMAGE_2 = PROJECT_ROOT / "data" / "images" / "object2.jpg"
RESULTS_DIR = PROJECT_ROOT / "results" / "week2"


def load_grayscale(path: Path) -> np.ndarray:
    """Load one color image from disk and return its grayscale pixel array."""
    if not path.is_file():
        raise FileNotFoundError(
            f"image file not found: {path}. Add two photographs of the same "
            "stationary object to data/images/object1.jpg and "
            "data/images/object2.jpg, or pass paths with --image1 and --image2."
        )

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"OpenCV could not decode {path}; check that it is a valid image file."
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_sift_features(
    grayscale: np.ndarray,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Find local SIFT keypoints and describe the image around each one."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(grayscale, None)
    return keypoints, descriptors


def save_keypoint_visualization(
    grayscale: np.ndarray, keypoints: list[cv2.KeyPoint], output_path: Path
) -> None:
    """Draw detected keypoints on a grayscale image and save the visualization."""
    visualization = cv2.drawKeypoints(
        grayscale,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"could not save keypoint visualization: {output_path}")


def load_and_detect(
    image1_path: Path, image2_path: Path
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[cv2.KeyPoint],
    list[cv2.KeyPoint],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Load both views, detect features, and save one keypoint image per view."""
    grayscale1 = load_grayscale(image1_path)
    grayscale2 = load_grayscale(image2_path)

    keypoints1, descriptors1 = detect_sift_features(grayscale1)
    keypoints2, descriptors2 = detect_sift_features(grayscale2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_keypoint_visualization(
        grayscale1, keypoints1, RESULTS_DIR / "keypoints_object1.png"
    )
    save_keypoint_visualization(
        grayscale2, keypoints2, RESULTS_DIR / "keypoints_object2.png"
    )

    print(f"Image 1: {image1_path} — grayscale shape {grayscale1.shape}")
    print(
        f"  keypoints: {len(keypoints1)}; "
        f"descriptors: {describe_descriptor_array(descriptors1)}"
    )
    print(f"Image 2: {image2_path} — grayscale shape {grayscale2.shape}")
    print(
        f"  keypoints: {len(keypoints2)}; "
        f"descriptors: {describe_descriptor_array(descriptors2)}"
    )
    print(f"Saved keypoint visualizations to: {RESULTS_DIR}")

    return (
        grayscale1,
        grayscale2,
        keypoints1,
        keypoints2,
        descriptors1,
        descriptors2,
    )


def describe_descriptor_array(descriptors: np.ndarray | None) -> str:
    """Format descriptor dimensions, including the no-features case."""
    if descriptors is None:
        return "none (no descriptors were found)"
    return f"{descriptors.shape[0]} rows x {descriptors.shape[1]} values"


def image_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared optional image-path arguments to a command-line parser."""
    parser.add_argument(
        "--image1",
        type=Path,
        default=DEFAULT_IMAGE_1,
        help=f"first view (default: {DEFAULT_IMAGE_1})",
    )
    parser.add_argument(
        "--image2",
        type=Path,
        default=DEFAULT_IMAGE_2,
        help=f"second view (default: {DEFAULT_IMAGE_2})",
    )


def main() -> None:
    """Run SIFT detection on both default images or user-specified paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    image_arguments(parser)
    args = parser.parse_args()

    try:
        load_and_detect(args.image1.expanduser(), args.image2.expanduser())
    except (FileNotFoundError, ValueError, cv2.error, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
