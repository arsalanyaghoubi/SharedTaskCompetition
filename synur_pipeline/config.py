"""
SYNUR Pipeline Configuration

This module contains all configurable parameters for the extraction pipeline.

IMPORTANT: What's from the paper vs. our choices:
- LLM models (gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini): FROM PAPER
- Embedding model (all-MiniLM-L6-v2): OUR CHOICE (paper doesn't specify)
- top_n_schema_rows (20): OUR CHOICE (paper doesn't specify)
- Prompts: Paper provides "simplified" versions in Appendix A.1; we include both.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:

    # File paths
    schema_path: Path = Path("../Data/synur_schema.json")
    train_path: Path = Path("../Data/train.jsonl")
    dev_path: Path = Path("../Data/dev.jsonl")

    # Model settings
    embedding_model: str = "text-embedding-3-small" # OpenAI embedding model. Can also try text-embedding-3-large or sentence-transformers like all-MiniLM-L6-v2
    
    top_n_schema_rows: int = 60 # Paper doesn't specify this either. I tested it out and it seems like 60 is a potential sweet spot between performance and cost, but we can try lower or higher. Max is 193.
    
    llm_model: str = "gpt-4o-mini" # Cheapest model, can also try gpt-4o, gpt-4.1, gpt-4.1-mini (what paper tried).
    temperature: float = 0.0  # Deterministic outputs for reproducibility
    max_tokens: int = 2048

    use_paper_prompts: bool = True #True uses the paper's simplified prompts, False uses some more detailed prompts.
    
    # Few-shot settings
    use_few_shot: bool = False  # Default to zero-shot for baseline reproduction
    num_few_shot_examples: int = 2  # Paper shows 1-2 examples are sufficient
    
    # Evaluation
    output_dir: Path = Path("./outputs")
