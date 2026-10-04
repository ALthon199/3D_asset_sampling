#!/bin/bash
# Orchestrates both halves of the mesh-sampling pipeline for all 4 working assets, once
# their captures (data/mesh_<name>/) exist. Part 1 needs trimesh (3d_stable), part 2 needs
# OpenEXR + torch (NPBG's torch_env) - see mesh_sample_geometry.py / mesh_build_cloud.py.
set -euo pipefail
cd /home/anthony6/3D_asset_sampling

PY_3D=/home/anthony6/anaconda3/envs/3d_stable/bin/python
PY_TORCH=/home/anthony6/anaconda3/envs/torch_env/bin/python

declare -A SCALES=(
  [mesh_avocado]="1.002360"
  [mesh_damaged_helmet]="0.085126"
  [mesh_boombox]="10.181222"
  [mesh_box]="0.115470"
)
declare -A MESH_PATHS=(
  [mesh_avocado]="assets/easy_meshes/avocado.glb"
  [mesh_damaged_helmet]="assets/easy_meshes/damaged_helmet.glb"
  [mesh_boombox]="assets/easy_meshes/boombox.glb"
  [mesh_box]="assets/easy_meshes/box.glb"
)

for scene in mesh_avocado mesh_damaged_helmet mesh_boombox mesh_box; do
  echo "=== $scene: sampling geometry ==="
  "$PY_3D" mesh_sample_geometry.py "${MESH_PATHS[$scene]}" "${SCALES[$scene]}" "/tmp/${scene}_geometry.npz"
  echo "=== $scene: projecting colors + selecting ==="
  "$PY_TORCH" mesh_build_cloud.py "$scene"
done
echo "all done"
