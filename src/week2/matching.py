"""Match SIFT descriptors between two views and save the accepted matches."""

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from .sift import (
        RESULTS_DIR,
        describe_descriptor_array,
        image_arguments,
        load_and_detect,
    )
except ImportError:
    from sift import (
        RESULTS_DIR,
        describe_descriptor_array,
        image_arguments,
        load_and_detect,
    )


LOWE_RATIO = 0.75
# Eight correspondences are a useful minimum for the common Week 3
# fundamental-matrix approach. This is a warning threshold, not an expected
# match count or a claim that the matches are geometrically correct.
MIN_MATCHES_FOR_GEOMETRY = 8


def ratio_test_matches(
    descriptors1: np.ndarray | None,
    descriptors2: np.ndarray | None,
    ratio: float = LOWE_RATIO,
) -> tuple[list[cv2.DMatch], int]:
    """Return descriptor pairs passing Lowe's ratio test and the raw pair count."""
    if descriptors1 is None or descriptors2 is None:
        return [], 0
    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return [], 0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    neighbors = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Some query descriptors can have fewer than two neighbors; those rows
    # cannot be evaluated by the ratio test and are skipped.
    pairs = [pair for pair in neighbors if len(pair) == 2]
    accepted = [
        best
        for best, second in pairs
        if best.distance < ratio * second.distance
    ]
    return accepted, len(pairs)


def save_match_visualization(
    grayscale1: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    grayscale2: np.ndarray,
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    output_path: Path,
) -> None:
    """Draw accepted matches between the two grayscale images and save them."""
    visualization = cv2.drawMatches(
        grayscale1,
        keypoints1,
        grayscale2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"could not save match visualization: {output_path}")


def main() -> None:
    """Run feature detection, matching, filtering, and diagnostic output."""
    parser = argparse.ArgumentParser(description=__doc__)
    image_arguments(parser)
    parser.add_argument(
        "--ratio",
        type=float,
        default=LOWE_RATIO,
        help=f"Lowe ratio threshold between 0 and 1 (default: {LOWE_RATIO})",
    )
    args = parser.parse_args()

    if not 0.0 < args.ratio < 1.0:
        parser.error("--ratio must be greater than 0 and less than 1")

    try:
        (
            grayscale1,
            grayscale2,
            keypoints1,
            keypoints2,
            descriptors1,
            descriptors2,
        ) = load_and_detect(args.image1.expanduser(), args.image2.expanduser())

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        matches, comparable_pairs = ratio_test_matches(
            descriptors1, descriptors2, args.ratio
        )
        match_image_path = RESULTS_DIR / "matches_lowe_075.png"
        if args.ratio != LOWE_RATIO:
            match_image_path = RESULTS_DIR / f"matches_lowe_{args.ratio:.2f}.png"
        save_match_visualization(
            grayscale1,
            keypoints1,
            grayscale2,
            keypoints2,
            matches,
            match_image_path,
        )
    except (FileNotFoundError, ValueError, cv2.error, OSError) as error:
        parser.error(str(error))

    print(f"SIFT descriptor matching (Lowe ratio: {args.ratio:.2f})")
    print(f"Descriptor array 1: {describe_descriptor_array(descriptors1)}")
    print(f"Descriptor array 2: {describe_descriptor_array(descriptors2)}")
    print(f"Neighbor pairs checked by the ratio test: {comparable_pairs}")
    print(f"Accepted descriptor matches: {len(matches)}")
    print(f"Saved match visualization: {match_image_path}")
    print(
        "Descriptor matches are candidates, not proof of geometric correctness. "
        "Inspect the image pair and match visualization."
    )

    if descriptors1 is None or descriptors2 is None:
        print(
            "No descriptors were found in at least one image. Try clear, "
            "well-lit photos with visible texture and sharp focus."
        )
    elif len(matches) < MIN_MATCHES_FOR_GEOMETRY:
        print(
            f"Warning: fewer than {MIN_MATCHES_FOR_GEOMETRY} accepted matches; "
            "this is usually too few for reliable Week 3 camera geometry. "
            "Check that these are overlapping views of the same stationary "
            "object and that the object has distinctive, in-focus texture."
        )


if __name__ == "__main__":
    main()
