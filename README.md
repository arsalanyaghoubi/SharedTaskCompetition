# SharedTaskCompetition

## Repository Structure

```
SharedTaskCompetition/
 Data/                    # Competition data files
    train.jsonl          # Training data with transcripts and gold observations
    dev.jsonl            # Development/evaluation data
    synur_schema.json    # Schema defining all possible observations
 synur_pipeline/          # Extraction pipeline implementation
    config.py            # Pipeline configuration
    schema_rag.py        # RAG-based schema filtering
    segmentation.py      # Transcript segmentation (LLM and rule-based)
    extraction.py        # LLM-based observation extraction
    evaluation.py        # F1 scoring
    run_pipeline.py      # Main script
    requirements.txt     # Dependencies
    .env.example         # Template for API keys
 main.py                  # Data loading utilities
 CompetitionBluePrint.docx
```

## Example

### 1. Setup

```bash
cd synur_pipeline
pip install -r requirements.txt

# Copy .env.example to .env and add your OpenAI API key
cp .env.example .env
# Edit .env with your API key
```

### 2. Run the Pipeline

```bash
# Zero-shot with gpt-4o-mini for 5 transcripts
python run_pipeline.py --data dev --model gpt-4o --zero-shot --limit 5
```


## Pipeline Architecture (based on the paper)

1. **Segmentation**: Split transcripts into clinically relevant segments
2. **Schema RAG**: Retrieve top-N relevant schema concepts per segment
3. **Extraction**: LLM extracts observations matching the schema
4. **Evaluation**: F1 score

## Expected Results

Paper's reported results on SYNUR:
- GPT-4o zero-shot: **0.883 F1**
- GPT-4o few-shot: ~**0.900 F1**
