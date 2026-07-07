#!/bin/bash
#SBATCH --job-name=cert_2wiki
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=logs/slurm/%x_%A.out
#SBATCH --error=logs/slurm/%x_%A.out
#
# Deletion-certificate over 2WikiMultiHopQA (compositional/inference, the
# Wikidata-id-annotated subset). CPU-bound; requires the cached KG from
# scripts/build_2wiki_kg.py. Submit from the repo root:
#   sbatch slurm/certify_2wiki.sbatch
set -euo pipefail

export UV_CACHE_DIR="${SCRATCH}/.uv-cache"
export UV_LINK_MODE=copy

SPLIT="${1:-dev}"
OUT="${SCRATCH}/runs/certificates/2wikimultihopqa_${SPLIT}"

echo "2wikimultihopqa ${SPLIT} -> ${OUT}"
uv run python scripts/compute_certificates.py \
    --dataset 2wikimultihopqa --split "${SPLIT}" \
    --workers "${SLURM_CPUS_PER_TASK}" --out-dir "${OUT}"
