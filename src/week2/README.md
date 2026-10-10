# Week 2: SIFT feature detection and matching

Week 2 reads `data/images/object1.jpg` and `data/images/object2.jpg`. They must
be photographs of the **same stationary physical object** from different
viewpoints, with substantial overlap. The scripts cannot prove that the images
show the same object or that it stayed still. If the two requested files are
missing, add your own pair there; do not use unrelated images and interpret the
result as valid correspondences.

No extra package is required. The existing project `cv-venv` already provides
OpenCV 5.0.0.93, NumPy, and Matplotlib. Continue running the scripts from the
project root with `./cv-venv/bin/python`.

## Run commands

Detect SIFT keypoints in both images and save per-image visualizations:

```bash
./cv-venv/bin/python src/week2/sift.py
```

Match SIFT descriptors, apply Lowe's ratio test at the initial threshold 0.75,
and save the accepted matches:

```bash
./cv-venv/bin/python src/week2/matching.py
```

Both scripts accept optional image paths:

```bash
./cv-venv/bin/python src/week2/matching.py \
  --image1 data/images/object1.jpg \
  --image2 data/images/object2.jpg
```

The matching script also accepts `--ratio 0.75`, but keep the initial threshold
until you inspect the images and diagnostics. Changing it cannot establish that
the inputs are a valid same-object pair.

If the input files are missing, add two original photo files by copying them
into the expected locations:

```bash
cp /path/to/your/first_view.jpg data/images/object1.jpg
cp /path/to/your/second_view.jpg data/images/object2.jpg
```

Keep the object still, change the camera viewpoint, use even lighting and sharp
focus, and include distinctive texture with substantial overlap in both views.
Avoid blank, glossy, or repetitive surfaces where possible.

## What the output means

- A **keypoint** is a detected image location with a scale and orientation,
  such as a corner or textured detail that SIFT considers distinctive.
- A **descriptor** is a numeric summary of the local image appearance around a
  keypoint. SIFT typically returns one 128-value descriptor per keypoint; the
  script prints actual shapes rather than assuming every keypoint has one.
- A **match** pairs descriptors that look similar across the two images.
  `BFMatcher` compares descriptor vectors with Euclidean distance
  (`NORM_L2`). `knnMatch(k=2)` finds the two nearest candidates for each
  descriptor. Lowe's test accepts the best candidate only if its distance is
  less than `0.75` times the second-best distance.

The ratio test removes many ambiguous matches, but accepted descriptor matches
are **not guaranteed to be geometrically correct**. Some can still be outliers.
Week 3 will use image coordinates from these candidates to estimate camera
geometry (for example, an epipolar/fundamental-matrix relationship), reject
inconsistent points with a robust method, and ultimately estimate relative
camera pose. The Week 2 match drawing is a visual starting point, not a
reconstructed 3D result.

## Saved files and inspection

Outputs are written to `results/week2/`:

- `keypoints_object1.png`: detected SIFT keypoints over the first grayscale
  image. Larger circles indicate detected feature scale; the direction marker
  indicates orientation.
- `keypoints_object2.png`: the same visualization for the second image.
- `matches_lowe_075.png`: accepted descriptor matches, with the first image on
  the left and the second image on the right. For a custom threshold, the
  filename includes that value.

Open the results in your desktop image viewer, for example:

```bash
xdg-open results/week2/matches_lowe_075.png
xdg-open results/week2/keypoints_object1.png
xdg-open results/week2/keypoints_object2.png
```

In the match image, inspect whether lines connect visually corresponding
details on the object. Plausible lines should generally not be random or
concentrated only on unrelated background regions. A line is still only a
descriptor match; geometric verification comes later.

The terminal reports keypoints, descriptor-array sizes, neighbor-pair count,
accepted matches, and the output path. If descriptors are absent, SIFT found no
usable local texture in that image. If there are few accepted matches, first
check that both files are readable photographs of the same stationary object,
with enough shared view, sharp detail, and non-repeating texture. If matches
cluster on the background or connect unrelated-looking details, improve or
retake the pair before tuning the ratio threshold.
