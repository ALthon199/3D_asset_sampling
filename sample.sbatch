#!/bin/bash
#SBATCH --job-name=asset_sample
#SBATCH --output=logs/asset_sample_%j.log
#SBATCH --error=logs/asset_sample_%j.log
#SBATCH --account=generic
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
PY=${PY:-/home/anthony6/anaconda3/envs/3d_stable/bin/python}

cd "$ROOT"
"$PY" sample.py
