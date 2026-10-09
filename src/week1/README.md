# Week 1: Images, vectors, and transformations

These four small scripts introduce the image and linear-algebra operations used
later in the 3D reconstruction project. They run independently from the project
root and use the existing `cv-venv`.

## Environment check

The current `cv-venv` has all Week 1 dependencies:

- OpenCV (`cv2`), from `opencv-contrib-python` 5.0.0.93
- NumPy 2.5.3
- Matplotlib 3.11.2
- Open3D 0.19.0 (not needed by these Week 1 scripts)

No packages are missing, so no installation is needed. If you later find a
dependency missing, install it into this project environment, not system Python:

```bash
./cv-venv/bin/python -m pip install numpy
./cv-venv/bin/python -m pip install matplotlib
./cv-venv/bin/python -m pip install opencv-contrib-python
```

For OpenCV, install `opencv-contrib-python` only if importing `cv2` fails; it
provides the OpenCV module and contrib features. Do not install a second OpenCV
wheel if `cv2` already imports.

## 1. Load, display, and process an image

Put an image named `sample.jpg` in `data/images/`. The folder is intentionally
empty right now; no input image was found when Week 1 was set up. To add one,
copy a JPG, PNG, or another OpenCV-readable image into that folder and name it
`sample.jpg`. You can also pass a different image path to the script.

```bash
./cv-venv/bin/python src/week1/image_basics.py
```

Or:

```bash
./cv-venv/bin/python src/week1/image_basics.py data/images/my_photo.png
```

The script loads a color image with OpenCV. OpenCV stores color channels in
BGR order; the script converts BGR to RGB before showing the color image with
Matplotlib. It converts the image to grayscale, smooths it with a 5x5 Gaussian
kernel, then finds edges with Canny thresholds 50 and 150. A grayscale pixel
contains one intensity (0 is black, 255 is white); a color pixel has three
channel values.

An image array has shape `(height, width, channels)` for color or `(height,
width)` for grayscale. Its top-left pixel is at `(row=0, column=0)`. In NumPy,
`image[row, column]` means `image[y, x]`: rows increase downwards and columns
increase to the right. Geometric points are often written `(x, y)`, so remember
that indexing order and coordinate order differ.

Expected output: a window containing the original, grayscale, and edge images,
plus `results/grayscale.png` and `results/edges.png`. The terminal prints the
image dimensions, array type, and indexing reminder. On a headless session or
without a graphical display, inspect the saved output PNGs instead of expecting
a display window.

To verify it worked, confirm the output window has three panels (or open the
two saved PNGs), the grayscale image has no color, and the edge image shows
boundaries rather than the original full-color scene. If the input is absent
or unreadable, the script exits with an actionable error instead of skipping
the example.

## 2. NumPy vector and matrix operations

```bash
./cv-venv/bin/python src/week1/linear_algebra.py
```

Expected values include `a + b = [5. 7. 9.]`, `a dot b = 32.0`, and
`A @ B = [[19. 22.], [43. 50.]]`. Addition and scalar multiplication are
elementwise. A dot product combines two vectors into one number; a cross
product of 3D vectors gives a perpendicular vector. The `@` operator performs
matrix multiplication: multiply each row of the left matrix by each column of
the right matrix and sum the products.

Verify by comparing the printed dot product with `1*4 + 2*5 + 3*6`, and the
first matrix-product entry with `1*5 + 2*7`.

## 3. 3D rotation

```bash
./cv-venv/bin/python src/week1/rotation.py
```

The rotation matrix is built for an angle `theta` around the z-axis:

```text
[[cos(theta), -sin(theta), 0],
 [sin(theta),  cos(theta), 0],
 [0,           0,          1]]
```

The script applies it using `R @ vector`. With a 90-degree rotation,
`[1, 0, 0]` becomes approximately `[0, 1, 0]`. The z coordinate stays fixed;
the x-y direction turns counterclockwise when viewed from positive z. A valid
rotation preserves lengths (`det(R) = 1` and `R.T @ R` is the identity matrix).

Verify the transformed vector and the printed determinant/identity matrix.

## 4. Homogeneous coordinates

```bash
./cv-venv/bin/python src/week1/homogeneous.py
```

A 3D rotation and translation can be combined into a 4x4 matrix:

```text
[[R00, R01, R02, tx],
 [R10, R11, R12, ty],
 [R20, R21, R22, tz],
 [0,   0,   0,   1 ]]
```

Append `w=1` to a point `(x, y, z)` and multiply `T @ [x, y, z, 1]`. The
rotation acts on the first three coordinates and the last column adds the
translation. Here `[1, 0, 0]` rotates to `[0, 1, 0]` and then translates to
`[2, 0, 3]`. For a direction rather than a point, use `w=0`; this prevents
translation from changing the direction. If the resulting `w` is not 1,
convert back to ordinary coordinates by dividing x, y, and z by w.

Verify that the script prints the transformed point `[2, 0, 3]` (allowing for
tiny floating-point rounding).

## Week 1 checklist

- [ ] Add your own image to `data/images/` and run the image example.
- [ ] View the grayscale and edge outputs in the display or in `results/`.
- [ ] Explain why OpenCV array indexing is `[row, column]` but points are often
  written `(x, y)`.
- [ ] Run the NumPy example and explain the dot product and `@`.
- [ ] Explain what the rotation matrix does to `[1, 0, 0]`.
- [ ] Explain why a homogeneous point uses `w=1` and a direction uses `w=0`.

## Check your understanding

1. If `image.shape` is `(480, 640, 3)`, what are the image width and height, and
   what does the `3` represent?
2. What does `a dot b` return, and how is that different from `A @ B`?
3. In a 4x4 transform, why does a point use `w=1` while a direction uses `w=0`?
