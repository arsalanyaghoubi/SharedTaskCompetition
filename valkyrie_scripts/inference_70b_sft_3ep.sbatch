#!/bin/bash

#SBATCH --job-name=synur_inf_70b_3ep
#SBATCH --output=logs/inf_70b_3ep_%j.out
#SBATCH --error=logs/inf_70b_3ep_%j.err

# Resource requests
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

echo "========================================================================"
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

# Use the specific 3-epoch model
SFT_MODEL="sft_output_70b/sft_20260204_011858/final"

echo ""
echo "Using 3-epoch fine-tuned model: $SFT_MODEL"
echo ""

echo "Running 70B SFT (3-epoch) model inference on dev set"

python run_pipeline.py \
    --data dev \
    --model llama-3.3-70b-sft-3ep \
    --model-path $SFT_MODEL \
    --base-model-path /data/shared/models/llama/Llama-3.3-70B-Instruct \
    --load-in-4bit \
    --embedding-model ../models/bge-base-en-v1.5 \
    --hybrid --hybrid-alpha 0.6 \
    --enhanced-prompts

# Capture exit code
INFERENCE_EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "========================================================================"

exit $INFERENCE_EXIT_CODE
