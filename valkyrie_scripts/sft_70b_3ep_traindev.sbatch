#!/bin/bash

#SBATCH --job-name=synur_sft_70b_traindev
#SBATCH --output=logs/sft_70b_traindev_%j.out
#SBATCH --error=logs/sft_70b_traindev_%j.err

# Resource requests
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00

#===============================================================================
echo "========================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Requested CPUs: $SLURM_CPUS_PER_TASK"
echo "Requested Memory: $SLURM_MEM_PER_NODE"
echo "GPU(s) allocated: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

module purge
module load anaconda cuda/12.4

conda activate synur

cd /data/projects/mtootooni_01/SharedTaskCompetition/synur_pipeline

# Create logs directory if it doesn't exist
mkdir -p ../valkyrie_scripts/logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Print environment info
echo ""
echo "Environment info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""


echo "Starting SFT training on TRAIN+DEV with 3 epochs"
echo ""

python sft_train.py \
    --data train+dev \
    --model-path /data/shared/models/llama/Llama-3.3-70B-Instruct \
    --embedding-model ../models/bge-base-en-v1.5 \
    --output-dir ./sft_output_70b_traindev \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --lr 1e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-seq-length 4096 \
    --use-4bit

# Capture training exit code before any other commands
TRAIN_EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "========================================================================"

# Print resource usage summary (ignore errors if sacct fails)
echo ""
echo "Resource usage summary:"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,CPUTime 2>/dev/null || echo "(sacct unavailable)"

# Exit with training exit code, not sacct exit code
exit $TRAIN_EXIT_CODE
