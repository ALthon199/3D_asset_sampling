"""Step 1 (run under 3d_stable, needs trimesh): direct, uniform surface sampling for
position/normal/roughness - unchanged from the mesh-only approach. Colour is deliberately
NOT assigned here; that needs the real captured renders and OpenEXR, which live in NPBG's
torch_env instead. Writes an intermediate npz that mesh_build_cloud.py (torch_env) consumes.
"""

import math
import sys

import numpy as np
import trimesh

# The point of this stage is to heavily oversample: greedy_select (now with chunked k-NN, see
# mesh_build_cloud.py) prunes the real "optimized" cloud out of whatever gets handed to it, so a
# bigger raw candidate pool only helps - more texture/geometric detail survives into the
# selection step. Faces and texture resolution still set the relative scale across assets
# (damaged_helmet's 15k textured triangles deserve a bigger pool than box's 12 untextured ones),
# but even the floor is itself a heavy oversample, not a bare-minimum count.
POINTS_PER_FACE_TEXEL_SQRT = 800
MIN_SAMPLE_COUNT = 20000
MAX_SAMPLE_COUNT = 200000
FALLBACK_TEXTURE_MEGAPIXELS = 0.25


def sample_texture(texture, uvs: np.ndarray) -> np.ndarray:
    array = np.array(texture.convert("RGB")) / 255.0
    h, w = array.shape[:2]
    px = (uvs[:, 0] * w).astype(int).clip(0, w - 1)
    py = ((1 - uvs[:, 1]) * h).astype(int).clip(0, h - 1)
    return array[py, px]


def adaptive_sample_count(num_faces: int, texture) -> int:
    texture_megapixels = (texture.size[0] * texture.size[1] / 1e6) if texture is not None else FALLBACK_TEXTURE_MEGAPIXELS
    detail_score = num_faces * texture_megapixels
    raw_count = POINTS_PER_FACE_TEXEL_SQRT * math.sqrt(detail_score)
    return int(max(MIN_SAMPLE_COUNT, min(MAX_SAMPLE_COUNT, raw_count)))


def main(mesh_path: str, scale: float, out_path: str, sample_count: int | None = None, seed: int = 0) -> None:
    np.random.seed(seed)
    scene = trimesh.load(mesh_path)
    geometry = list(scene.geometry.values())[0]
    material = geometry.visual.material

    if sample_count is None:
        sample_count = adaptive_sample_count(len(geometry.faces), getattr(material, "baseColorTexture", None))

    points, face_indices = trimesh.sample.sample_surface(geometry, sample_count)
    bary = trimesh.triangles.points_to_barycentric(geometry.triangles[face_indices], points)
    uvs = geometry.visual.uv
    face_uvs = uvs[geometry.faces[face_indices]]
    sampled_uvs = np.einsum("ij,ijk->ik", bary, face_uvs)

    albedo = sample_texture(material.baseColorTexture, sampled_uvs)
    normals = geometry.face_normals[face_indices]
    roughness_texture = getattr(material, "metallicRoughnessTexture", None)
    if roughness_texture is not None:
        roughness = sample_texture(roughness_texture, sampled_uvs)[:, 1:2]
    else:
        roughness = np.full((points.shape[0], 1), 0.5, dtype=np.float64)

    # World-space (scene-relative), matching the scale the renderer actually placed this
    # model at - everything downstream (colour projection, the final saved cloud) works in
    # this scaled space, not the mesh's own native units
    positions_world = (points * scale).astype(np.float32)
    normals = normals.astype(np.float32)  # direction unaffected by uniform scale
    diagonal_world = float(np.linalg.norm(geometry.extents)) * scale

    np.savez(
        out_path,
        positions=positions_world,
        normals=normals,
        albedo=albedo.astype(np.float32),
        roughness=roughness.astype(np.float32),
        diagonal=diagonal_world,
    )
    print(f"wrote {out_path}: {positions_world.shape[0]} sampled points, world diagonal={diagonal_world:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3])
