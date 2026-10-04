import trimesh
import numpy as np
from PIL import Image
import torch
import os

np.random.seed(0)

# load
scene = trimesh.load('assets/easy_meshes/avocado.glb')
geometry = list(scene.geometry.values())[0]

# extract albedo
albedo = np.array(geometry.visual.material.baseColorTexture.convert('RGB')) / 255.0
h, w = albedo.shape[:2]

# sample points with face indices so we can interpolate UVs
points, face_indices = trimesh.sample.sample_surface(geometry, 10000)

# get barycentric coordinates
bary = trimesh.triangles.points_to_barycentric(
    geometry.triangles[face_indices], points
)

# get UVs at triangle vertices
uvs = geometry.visual.uv                      # (N_vertices, 2)
face_uvs = uvs[geometry.faces[face_indices]]  # (N_samples, 3, 2)

# interpolate UVs at sampled points
sampled_uvs = np.einsum('ij,ijk->ik', bary, face_uvs)  # (N_samples, 2)

# sample albedo at UV locations
px = (sampled_uvs[:, 0] * w).astype(int).clip(0, w-1)
py = ((1 - sampled_uvs[:, 1]) * h).astype(int).clip(0, h-1)

colors = albedo[py, px]  # (N_samples, 3)
normals = geometry.face_normals[face_indices]



points  = torch.tensor(points,  dtype=torch.float32).cuda()  # (N, 3) on GPU
normals = torch.tensor(normals, dtype=torch.float32).cuda()  # (N, 3) on GPU
colors  = torch.tensor(colors,  dtype=torch.float32).cuda()  # (N, 3) on GPU



# precompute once — expensive part, done before loop
# K nearest neighbors within a radius scaled to the mesh size (5% of the bbox diagonal)
radius = 0.05 * float(np.linalg.norm(geometry.extents))
dist, idx = torch.cdist(points, points).topk(64, dim=-1, largest=False)
idx[dist > radius] = -1

neighbor_normals = normals[idx.clamp(min=0)]
normal_sim = (neighbor_normals * normals.unsqueeze(1)).sum(dim=-1)
valid = (idx >= 0) & (normal_sim > 0.9)

# color variance over valid neighbors only
neighbor_colors = colors[idx.clamp(min=0)]
w_valid = valid.unsqueeze(-1).float()
n_valid = w_valid.sum(dim=1)
mean_color = (neighbor_colors * w_valid).sum(dim=1) / n_valid
color_complexity = (((neighbor_colors - mean_color.unsqueeze(1)) ** 2) * w_valid).sum(dim=1) / n_valid
color_complexity = color_complexity.sum(dim=-1)

# greedy loop — only the selection is sequential
selected = []
number_of_points = points.shape[0]
covered = torch.zeros(number_of_points, dtype=torch.bool, device='cuda')
selected_mask = torch.zeros(number_of_points, dtype=torch.bool, device='cuda')

while True:
    # these three lines replace an entire nested CPU loop
    neighbor_covered = covered[idx.clamp(min=0)]
    new_coverage = (valid & ~neighbor_covered).sum(dim=-1).float()
    weighted = new_coverage * (1 + color_complexity)
    
    weighted[selected_mask] = -1
    
    if new_coverage.max().item() / number_of_points < 0.001:
        break
    
    best = weighted.argmax().item()
    selected.append(best)
    selected_mask[best] = True
    covered[idx[best][valid[best]]] = True

print(f"oversampled: {number_of_points} points")
print(f"selected: {len(selected)} points ({100*len(selected)/number_of_points:.1f}%)")
print(f"reduction ratio: {number_of_points / len(selected):.1f}x")

selected = np.array(selected)

os.makedirs('optimized_npz', exist_ok=True)
np.savez(
    'optimized_npz/avocado_reduced.npz',
    points  = points[selected].cpu().numpy(),   # (K, 3) selected only
    normals = normals[selected].cpu().numpy(),   # (K, 3) selected only
    colors  = colors[selected].cpu().numpy(),    # (K, 3) selected only
)

# headless preview: orthographic projection onto the XY plane, nearest point (largest z) wins
size = 800
p = points[selected].cpu().numpy()
c = (colors[selected].cpu().numpy() * 255).astype(np.uint8)
xy = (p[:, :2] - p[:, :2].min(0)) / (p[:, :2].max(0) - p[:, :2].min(0)).max()
px = (xy[:, 0] * (size - 5)).astype(int)
py = ((1 - xy[:, 1]) * (size - 5)).astype(int)
img = np.full((size, size, 3), 255, dtype=np.uint8)
for j in np.argsort(p[:, 2]):
    img[py[j]:py[j] + 4, px[j]:px[j] + 4] = c[j]
Image.fromarray(img).save('points.png')
