"""Create a 3D rotation matrix and apply it to a vector."""

import numpy as np


def rotation_matrix_z(angle_degrees: float) -> np.ndarray:
    """Return a 3x3 matrix that rotates vectors around the z-axis."""
    angle_radians = np.deg2rad(angle_degrees)
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def main() -> None:
    """Rotate a vector by 90 degrees around the z-axis."""
    angle_degrees = 90.0
    rotation = rotation_matrix_z(angle_degrees)
    vector = np.array([1.0, 0.0, 0.0])
    rotated_vector = rotation @ vector

    print(f"Rotation around z by {angle_degrees:.0f} degrees:")
    print(f"R =\n{rotation}")
    print(f"Original vector: {vector}")
    print(f"R @ vector:      {rotated_vector}")
    print(
        "Vectors are columns conceptually: the matrix changes direction "
        "without changing the vector's length."
    )
    print(f"det(R) = {np.linalg.det(rotation):.1f}")
    print(f"R.T @ R =\n{rotation.T @ rotation}")


if __name__ == "__main__":
    main()
