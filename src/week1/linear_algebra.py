"""A small NumPy demonstration of vectors and matrix multiplication."""

import numpy as np


def main() -> None:
    """Print example vector operations and a matrix product."""
    vector_a = np.array([1, 2, 3], dtype=float)
    vector_b = np.array([4, 5, 6], dtype=float)

    print("Vectors")
    print(f"a = {vector_a}")
    print(f"b = {vector_b}")
    print(f"a + b = {vector_a + vector_b}")
    print(f"2 * a = {2 * vector_a}")
    print(f"a dot b = {np.dot(vector_a, vector_b)} (one number)")
    print(f"a cross b = {np.cross(vector_a, vector_b)} (a perpendicular vector)")
    print(f"length of a = {np.linalg.norm(vector_a):.3f}")

    matrix_a = np.array([[1, 2], [3, 4]], dtype=float)
    matrix_b = np.array([[5, 6], [7, 8]], dtype=float)
    product = matrix_a @ matrix_b

    print("\nMatrices")
    print(f"A =\n{matrix_a}")
    print(f"B =\n{matrix_b}")
    print(f"A @ B =\n{product}")
    print(
        "The @ operator performs matrix multiplication: each output entry "
        "is a row of A dotted with a column of B."
    )


if __name__ == "__main__":
    main()
