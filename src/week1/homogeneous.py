"""Demonstrate a 4x4 rigid transformation with homogeneous coordinates."""

import numpy as np


def main() -> None:
    """Rotate a 3D point and then translate it using one 4x4 matrix."""
    angle_radians = np.deg2rad(90.0)
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    rotation_z = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([2.0, -1.0, 3.0])

    # The last row keeps the homogeneous coordinate w unchanged.
    transform = np.eye(4)
    transform[:3, :3] = rotation_z
    transform[:3, 3] = translation

    point = np.array([1.0, 0.0, 0.0])
    point_homogeneous = np.append(point, 1.0)
    transformed_homogeneous = transform @ point_homogeneous
    transformed_point = transformed_homogeneous[:3] / transformed_homogeneous[3]

    print("4x4 transformation (rotate 90 degrees around z, then translate):")
    print(f"T =\n{transform}")
    print(f"3D point: {point}")
    print(f"Homogeneous point [x, y, z, 1]: {point_homogeneous}")
    print(f"T @ point_homogeneous: {transformed_homogeneous}")
    print(f"Transformed 3D point: {transformed_point}")
    print(
        "A point uses w=1 so translation affects it. A direction uses w=0 "
        "so translation does not affect it."
    )


if __name__ == "__main__":
    main()
