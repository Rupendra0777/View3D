import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"No points found in '{path}'")

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.2, 0.6, 0.9])

    return pcd


def render_to_image(pcd: o3d.geometry.PointCloud, width: int = 1200, height: int = 900) -> o3d.geometry.Image:
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 8.0

    renderer.scene.add_geometry(
        "point_cloud",
        pcd,
        material,
        add_downsampled_copy_for_fast_rendering=False,
    )

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    distance = max(extent) * 1.8

    eye = center + np.array([0.0, -distance, distance * 0.6], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    renderer.scene.camera.look_at(center, eye, up)

    return renderer.render_to_image()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a 3D point cloud from a PLY file.")
    parser.add_argument(
        "--input",
        type=str,
        default="point_cloud.ply",
        help="Path to the input .ply point cloud file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="images/point_cloud_visualized.png",
        help="Where to save the rendered visualization image",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive Open3D window as well as saving the image",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = load_point_cloud(input_path)
    image = render_to_image(pcd)
    o3d.io.write_image(str(output_path), image)
    print(f"Saved rendered point cloud to: {output_path}")

    if args.show:
        o3d.visualization.draw_geometries([pcd], window_name="3D point cloud")


if __name__ == "__main__":
    main()
