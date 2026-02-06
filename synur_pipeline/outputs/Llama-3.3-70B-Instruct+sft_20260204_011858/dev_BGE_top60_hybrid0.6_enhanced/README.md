# Llama-3.3-70B-Instruct-SFT (3-epoch) Outputs

This directory contains outputs from the fine-tuned Llama-3.3-70B model trained for 3 epochs.

## Directory Structure

### baseline/
Best performing runs with temperature=0
- `20260205_024245`: Best run (F1=76.45%)
- `20260204_060622`: Initial run (F1=76.24%)

### self_consistency/
Runs for self-consistency voting (temperature=0.3, different seeds)
- `20260205_024318`: SC1 (seed=42, F1=75.60%)
- `20260205_020844`: SC2 (seed=123, F1=75.32%)
- `20260205_034609`: SC3 (seed=456, F1=76.66%)
- Use `self_consistency_vote.py --runs` with these predictions for majority voting

### llm_segmentation/
Experiments with LLM-based transcript segmentation
- `20260204_134618`: Using same Llama model for segmentation (F1=70.57%)
- `20260204_174545`: Using GPT-4o-mini for segmentation (F1=72.93%)

### post_processing/
Outputs from verification and voting post-processing
- `verified_*`: Outputs from verification_pass.py
- `voted_*`: Outputs from self_consistency_vote.py
