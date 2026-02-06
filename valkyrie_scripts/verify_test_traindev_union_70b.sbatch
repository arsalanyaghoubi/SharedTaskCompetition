#!/bin/bash
#SBATCH --job-name=verify_union_70b
#SBATCH --output=logs/verify_union_70b_%j.out
#SBATCH --error=logs/verify_union_70b_%j.err

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

# Input: Union voted predictions from train+dev SFT model on TEST set
INPUT_PREDICTIONS="${SYNUR_DIR}/outputs/Llama-3.3-70B-Instruct+sft_20260204_102723/test_BGE_top60_hybrid0.6_enhanced/union_voted_predictions.json"
OUTPUT_DIR="${SYNUR_DIR}/outputs/Llama-3.3-70B-Instruct+sft_20260204_102723/test_BGE_top60_hybrid0.6_enhanced"
DATA_FILE="${REPO_DIR}/Data/SYNUR_testset_input.jsonl"

# Output paths
OUTPUT_JSON="${OUTPUT_DIR}/union_verified_predictions_70b.json"
OUTPUT_SUBMISSION="${OUTPUT_DIR}/union_verified_submission_70b.jsonl"

echo "=============================================="
echo "Verification Pass with Llama-3.3-70B-Instruct"
echo "Input: Union Voted TEST Predictions (train+dev SFT)"
echo "=============================================="
echo "Input: ${INPUT_PREDICTIONS}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_JSON}"
echo ""

# Run verification with local model (8-bit to fit in memory)
python verification_pass.py \
    --predictions "${INPUT_PREDICTIONS}" \
    --data "${DATA_FILE}" \
    --output "${OUTPUT_JSON}" \
    --submission "${OUTPUT_SUBMISSION}" \
    --schema "${REPO_DIR}/Data/synur_schema.json" \
    --model-path "${MODEL_PATH}" \
    --load-in-8bit

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "========================================================================"

exit $EXIT_CODE
