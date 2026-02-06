#!/bin/bash


echo "=============================================="
echo "Setting up SYNUR environment on Valkyrie"
echo "=============================================="

# Get the directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"

# Load modules
module purge
module load anaconda cuda/12.4

# Create conda environment (skip if exists)
if conda env list | grep -q "^synur "; then
    echo "Conda environment 'synur' already exists, skipping creation"
else
    echo "Creating conda environment 'synur'..."
    conda create -n synur python=3.11 -y
fi

# Activate environment (this only works if script is sourced)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate synur

# Verify activation
echo "Active environment: $CONDA_DEFAULT_ENV"

# Install packages from requirements.txt
echo "Installing Python packages..."
pip install -r "$PROJECT_DIR/synur_pipeline/requirements.txt"

# Verify key packages
echo ""
echo "Verifying installations"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
python -c "import sentence_transformers; print(f'SentenceTransformers: {sentence_transformers.__version__}')"
