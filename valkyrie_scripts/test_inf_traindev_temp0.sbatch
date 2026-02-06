#!/bin/bash

#SBATCH --job-name=test_traindev_t0
#SBATCH --output=logs/test_traindev_t0_%j.out
#SBATCH --error=logs/test_traindev_t0_%j.err

# Resource requests
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

echo "========================================================================"
echo "TEST SET INFERENCE - Train+Dev SFT Model - Temperature 0.0"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU(s) allocated: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

module purge
module load anaconda cuda/12.4
conda activate synur

cd /data/projects/mtootooni_01/SharedTaskCompetition/synur_pipeline
mkdir -p ../valkyrie_scripts/logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Train+Dev SFT model (3 epochs, 223 examples)
SFT_MODEL="sft_output_70b_traindev/sft_20260204_102723/final"

echo ""
echo "Using Train+Dev fine-tuned model: $SFT_MODEL"
echo "Temperature: 0.0 (deterministic)"
echo "Data: TEST SET"
echo ""

python run_pipeline.py \
    --data test \
    --model llama-70b-sft-traindev \
    --model-path $SFT_MODEL \
    --base-model-path /data/shared/models/llama/Llama-3.3-70B-Instruct \
    --load-in-4bit \
    --embedding-model ../models/bge-base-en-v1.5 \
    --hybrid --hybrid-alpha 0.6 \
    --enhanced-prompts

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE
