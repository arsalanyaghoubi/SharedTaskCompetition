#!/bin/bash
#SBATCH --job-name=test_70b_base
#SBATCH --output=logs/test_70b_base_%j.out
#SBATCH --error=logs/test_70b_base_%j.err

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

echo "========================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU(s) allocated: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

# Load modules
module purge
module load anaconda cuda/12.4

# Activate environment
conda activate synur

# Set paths
REPO_DIR=/data/projects/mtootooni_01/SharedTaskCompetition
SYNUR_DIR=${REPO_DIR}/synur_pipeline
MODEL_PATH=/data/shared/models/llama/Llama-3.3-70B-Instruct

# Change to working directory
cd ${SYNUR_DIR}
mkdir -p ../valkyrie_scripts/logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=============================================="
echo "Llama-3.3-70B-Instruct (Base) on TEST set"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo ""

# Run inference with base model (no SFT)
python run_pipeline.py \
    --data test \
    --model-path "${MODEL_PATH}" \
    --embedding-model "${REPO_DIR}/models/bge-base-en-v1.5" \
    --hybrid-alpha 0.6 \
    --top-n 60 \
    --temperature 0 \
    --enhanced-prompts \
    --load-in-4bit

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "========================================================================"

exit $EXIT_CODE
