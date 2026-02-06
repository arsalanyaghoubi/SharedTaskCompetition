#!/bin/bash
#SBATCH --job-name=verify_70b_base
#SBATCH --output=logs/verify_70b_base_%j.out
#SBATCH --error=logs/verify_70b_base_%j.err

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00

echo "========================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU(s) allocated: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

# Load modules (correct names for Valkyrie)
module purge
module load anaconda cuda/12.4

# Activate environment
conda activate synur

# Set paths (correct base path for Valkyrie)
REPO_DIR=/data/projects/mtootooni_01/SharedTaskCompetition
SYNUR_DIR=${REPO_DIR}/synur_pipeline
MODEL_PATH=/data/shared/models/llama/Llama-3.3-70B-Instruct

# Change to working directory
cd ${SYNUR_DIR}
mkdir -p ../valkyrie_scripts/logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Input: Same predictions used for GPT-4o-mini verification (temp=0 baseline)
INPUT_PREDICTIONS="${SYNUR_DIR}/outputs/Llama-3.3-70B-Instruct+sft_20260204_011858/dev_BGE_top60_hybrid0.6_enhanced/baseline/run_20260205_024245/predictions.json"

# Output paths
OUTPUT_JSON="${SYNUR_DIR}/outputs/Llama-3.3-70B-Instruct+sft_20260204_011858/dev_BGE_top60_hybrid0.6_enhanced/post_processing/verified_predictions_70b_base.json"
OUTPUT_SUBMISSION="${SYNUR_DIR}/outputs/Llama-3.3-70B-Instruct+sft_20260204_011858/dev_BGE_top60_hybrid0.6_enhanced/post_processing/verified_submission_70b_base.jsonl"

echo "=============================================="
echo "Verification Pass with Llama-3.3-70B-Instruct"
echo "=============================================="
echo "Input: ${INPUT_PREDICTIONS}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_JSON}"
echo ""

# Run verification with local model (8-bit to fit in memory)
python verification_pass.py \
    --predictions "${INPUT_PREDICTIONS}" \
    --data "${REPO_DIR}/Data/dev.jsonl" \
    --output "${OUTPUT_JSON}" \
    --submission "${OUTPUT_SUBMISSION}" \
    --schema "${REPO_DIR}/Data/synur_schema.json" \
    --model-path "${MODEL_PATH}" \
    --load-in-8bit \
    --eval

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "========================================================================"

exit $EXIT_CODE
