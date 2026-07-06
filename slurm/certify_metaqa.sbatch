#!/bin/bash
#SBATCH --job-name=cert_metaqa
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --array=0-5
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.out
#
# Deletion-certificate sweep over MetaQA {1,2,3}-hop x {dev,test}.
# One array cell per (hops, split). Certificate computation is CPU-bound
# (NetworkX max-flow), so this runs on CPU cores, no GPU requested. The
# large train splits are a separate, later run (needed only for P8).
#
# Submit from the repo root:  sbatch slurm/certify_metaqa.sbatch
set -euo pipefail

export UV_CACHE_DIR="${SCRATCH}/.uv-cache"
export UV_LINK_MODE=copy

HOPS_LIST=(1 1 2 2 3 3)
SPLIT_LIST=(dev test dev test dev test)
HOPS=${HOPS_LIST[$SLURM_ARRAY_TASK_ID]}
SPLIT=${SPLIT_LIST[$SLURM_ARRAY_TASK_ID]}
DATASET="metaqa-${HOPS}hop"
OUT="${SCRATCH}/runs/certificates/${DATASET}_${SPLIT}"

echo "cell ${SLURM_ARRAY_TASK_ID}: ${DATASET} ${SPLIT} -> ${OUT}"
uv run python scripts/compute_certificates.py \
    --dataset "${DATASET}" --split "${SPLIT}" \
    --workers "${SLURM_CPUS_PER_TASK}" --out-dir "${OUT}"
