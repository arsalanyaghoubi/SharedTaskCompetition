# Pipeline Outputs

This folder contains extraction results organized by model and configuration.

## Configuration String Format

`{data_split}_{embedding_model}_{top_n}_{options}`

Examples:
- `train_openai-small_top60_enhanced` - Train set, OpenAI embeddings, top-60, enhanced prompts
- `train_BGE_top60_hybrid0.6_enhanced` - Train set, BGE embeddings, hybrid search (α=0.6), enhanced prompts

## File Descriptions

| File | Purpose |
|------|---------|
| `predictions_*.json` | Detailed predictions with gold labels. Use with `error_analysis.py` |
| `submission_*.jsonl` | Competition submission format |
| `summary_*.json` | Run configuration and official P/R/F1 scores |