"""Step 2 (run under NPBG's torch_env, needs OpenEXR + torch): project the directly
mesh-sampled points (from mesh_sample_geometry.py) into every real captured training-split
view, assign each point the genuine ray-traced colour from whichever view(s) actually see it
(depth-agreement visibility check, not UV texture lookup), run the same greedy
coverage/colour-complexity selection as sample.py, and write the result as NPBG's native
point_cloud.npz + manifest.json - replacing only the point_cloud block build.py's own default
voxel_downsample run produced, keeping its captured records/ untouched.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data0/home/anthony6/NPBG")
from capture import intrinsics, load_camera_path, read_frame, view_matrix  # noqa: E402

POINT_FEATURES = ("r", "g", "b", "n_x", "n_y", "n_z", "albedo_r", "albedo_g", "albedo_b", "roughness")
EPSILON_FRACTION = 0.10  # of the object's own world-space bounding-box diagonal, per calibration


def shade_lambertian(albedo: np.ndarray, normals: np.ndarray, light_dir=(0.4, 0.6, 0.7), ambient=0.25) -> np.ndarray:
    light = np.asarray(light_dir, dtype=np.float64)
    light = light / np.linalg.norm(light)
    ndotl = np.clip(normals @ light, 0.0, None)
    shade = ambient + (1.0 - ambient) * ndotl
    return np.clip(albedo * shade[:, None], 0.0, 1.0)


def assign_colors_by_projection(
    positions: np.ndarray, scene_root: Path, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average real captured rgb across every train-split view where each point is
    genuinely visible (depth-agreement within epsilon, ground-truth coverage says a real
    surface is there). Returns (colors, visible_view_count) - zero count means no train
    view ever saw that point, a real gap this one-directional pipeline cannot fill
    """
    manifest = json.loads((scene_root / "manifest.json").read_text())
    camera_path = manifest["point_cloud"]["camera_path"]
    width, height = manifest["point_cloud"]["width"], manifest["point_cloud"]["height"]
    fovy = manifest["point_cloud"]["fovy"]
    train_indices = [r["frame"] for r in manifest["records"] if r["split"] == "train"]

    positions_t = positions  # numpy is fine here, small point counts
    color_sum = np.zeros((positions.shape[0], 3), dtype=np.float64)
    color_count = np.zeros(positions.shape[0], dtype=np.int64)

    full_path = Path("/data0/home/anthony6/VulkanRenderer") / camera_path
    cam_positions, cam_eulers = load_camera_path(str(full_path))
    K = intrinsics(width, height, fovy)

    capture_dir = scene_root / "capture"
    for index in train_indices:
        exr_path = capture_dir / f"frame_{index:06d}.exr"
        if not exr_path.is_file():
            continue
        frame = read_frame(exr_path, "rt_viewer")
        depth_map = frame["depth"]
        coverage_map = frame["coverage"][0]
        rgb_map = frame["rgb"]  # [3,H,W], linear - matches positions' own linear space

        view = view_matrix(cam_positions[index], cam_eulers[index])
        from rasterize import Intrinsics, project
        import torch as _torch

        view_t = _torch.from_numpy(view).float().unsqueeze(0)
        intr = Intrinsics(
            _torch.tensor([K[0, 0]]), _torch.tensor([K[1, 1]]),
            _torch.tensor([K[0, 2]]), _torch.tensor([K[1, 2]]),
        )
        points_t = _torch.from_numpy(positions_t).float()
        u, v, depth = project(points_t, view_t, intr)
        u, v, depth = u[0].numpy(), v[0].numpy(), depth[0].numpy()

        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (depth > 1e-4)
        ui, vi = u.astype(int), v.astype(int)
        hit = np.zeros(positions.shape[0], dtype=bool)
        hit[in_bounds] = coverage_map[vi[in_bounds], ui[in_bounds]] > 0.5
        captured_depth = np.full(positions.shape[0], np.inf)
        captured_depth[hit] = depth_map[vi[hit], ui[hit]]

        visible = hit & (np.abs(depth - captured_depth) < epsilon)
        if not visible.any():
            continue
        rgb_at_pixel = rgb_map[:, vi[visible], ui[visible]].T  # [n_visible, 3]
        color_sum[visible] += rgb_at_pixel
        color_count[visible] += 1

    colors = np.zeros((positions.shape[0], 3), dtype=np.float64)
    seen = color_count > 0
    colors[seen] = color_sum[seen] / color_count[seen, None]
    return colors.astype(np.float32), color_count


def knn_chunked(positions_t: torch.Tensor, k: int, chunk_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Same result as torch.cdist(positions, positions).topk(k), but never materializes the
    full N x N matrix - a single cdist call needs N^2 x 4 bytes (40GB+ at N=100k), which is
    exactly what made heavy oversampling impractical before. Chunking the query side caps
    memory at chunk_size x N regardless of how large N gets.
    """
    n = positions_t.shape[0]
    dist_chunks, idx_chunks = [], []
    for start in range(0, n, chunk_size):
        chunk_dist = torch.cdist(positions_t[start : start + chunk_size], positions_t)
        chunk_d, chunk_i = chunk_dist.topk(k, dim=-1, largest=False)
        dist_chunks.append(chunk_d)
        idx_chunks.append(chunk_i)
    return torch.cat(dist_chunks, dim=0), torch.cat(idx_chunks, dim=0)


def greedy_select(positions, normals, colors, radius_fraction=0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    positions_t = torch.tensor(positions, dtype=torch.float32, device=device)
    normals_t = torch.tensor(normals, dtype=torch.float32, device=device)
    colors_t = torch.tensor(colors, dtype=torch.float32, device=device)

    diag = float(np.linalg.norm(positions.max(0) - positions.min(0)))
    radius = radius_fraction * diag
    k = min(64, positions.shape[0])
    dist, idx = knn_chunked(positions_t, k)
    idx[dist > radius] = -1

    neighbor_normals = normals_t[idx.clamp(min=0)]
    normal_sim = (neighbor_normals * normals_t.unsqueeze(1)).sum(dim=-1)
    valid = (idx >= 0) & (normal_sim > 0.9)

    neighbor_colors = colors_t[idx.clamp(min=0)]
    w_valid = valid.unsqueeze(-1).float()
    n_valid = w_valid.sum(dim=1).clamp(min=1)
    mean_color = (neighbor_colors * w_valid).sum(dim=1) / n_valid
    color_complexity = (((neighbor_colors - mean_color.unsqueeze(1)) ** 2) * w_valid).sum(dim=1).sum(dim=-1)

    n = positions_t.shape[0]
    covered = torch.zeros(n, dtype=torch.bool, device=device)
    selected_mask = torch.zeros(n, dtype=torch.bool, device=device)
    selected = []
    while True:
        neighbor_covered = covered[idx.clamp(min=0)]
        new_coverage = (valid & ~neighbor_covered).sum(dim=-1).float()
        weighted = new_coverage * (1 + color_complexity)
        weighted[selected_mask] = -1
        if new_coverage.max().item() / n < 0.0001:
            break
        best = weighted.argmax().item()
        selected.append(best)
        selected_mask[best] = True
        covered[idx[best][valid[best]]] = True
    return np.array(selected)


def main(scene_name: str) -> None:
    scene_root = Path("/data0/home/anthony6/NPBG/data") / scene_name
    geometry = np.load(f"/tmp/{scene_name}_geometry.npz")
    positions, normals = geometry["positions"], geometry["normals"]
    albedo, roughness, diagonal = geometry["albedo"], geometry["roughness"], float(geometry["diagonal"])

    epsilon = EPSILON_FRACTION * diagonal
    projected_colors, visible_counts = assign_colors_by_projection(positions, scene_root, epsilon)

    unseen = visible_counts == 0
    print(f"{scene_name}: {unseen.sum()}/{len(unseen)} points never visible in any train view "
          f"({100*unseen.mean():.1f}%) - falling back to Lambertian shade for those only")
    if unseen.any():
        fallback = shade_lambertian(albedo[unseen], normals[unseen])
        projected_colors[unseen] = fallback

    selected = greedy_select(positions, normals, projected_colors)
    print(f"{scene_name}: {positions.shape[0]} sampled -> {len(selected)} selected "
          f"({positions.shape[0]/len(selected):.1f}x reduction)")

    out_positions = positions[selected].astype(np.float32)
    features = np.concatenate(
        [projected_colors[selected], normals[selected], albedo[selected], roughness[selected]], axis=1
    ).astype(np.float16)
    assert features.shape[1] == len(POINT_FEATURES)

    npz_path = scene_root / "point_cloud.npz"
    np.savez(npz_path, positions=out_positions, features=features)
    digest = hashlib.sha256()
    with npz_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    manifest_path = scene_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["point_cloud"]["num_points"] = int(out_positions.shape[0])
    manifest["point_cloud"]["sha256"] = digest.hexdigest()
    manifest["point_cloud"]["renderer"] = (
        "mesh_sample_projected (direct trimesh surface sampling for position/normal, real "
        "ray-traced rgb assigned by projecting into every train-split captured view with "
        f"depth-agreement visibility check, epsilon={epsilon:.5f} = {EPSILON_FRACTION} x "
        f"diagonal; {int(unseen.sum())} points with zero visible views fall back to a "
        "single-light Lambertian shade of albedo)"
    )
    manifest["point_cloud"]["voxel_size"] = None
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {npz_path} and updated {manifest_path}")


if __name__ == "__main__":
    main(sys.argv[1])
