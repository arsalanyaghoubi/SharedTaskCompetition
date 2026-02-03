from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:

    # File paths
    schema_path: Path = Path("../Data/synur_schema.json")
    train_path: Path = Path("../Data/train.jsonl")
    dev_path: Path = Path("../Data/dev.jsonl")

    # Model settings
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model. Can also try text-embedding-3-large or sentence-transformers like all-MiniLM-L6-v2
    
    top_n_schema_rows: int = 60  # Paper doesn't specify this either. I tested it out and it seems like 60 is a potential sweet spot between performance and cost, but we can try lower or higher. Max is 193.
    
    # LLM Model settings
    llm_model: str = "gpt-4o-mini"  # Model name for display/logging
    llm_model_path: Optional[str] = None  # Path to local model folder (if None, uses OpenAI)
    lora_path: Optional[str] = None  # Path to LoRA adapter folder (for SFT models)
    
    temperature: float = 0.0  # Deterministic outputs for reproducibility
    max_tokens: int = 4096

    use_paper_prompts: bool = True # True uses the paper's simplified prompts, False uses some more detailed prompts.
    
    # Hybrid search settings
    use_hybrid_search: bool = False
    hybrid_alpha: float = 0.6
    
    # Quantization settings (for large models)
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    # Few-shot settings
    use_few_shot: bool = False  # Default to zero-shot for baseline reproduction
    num_few_shot_examples: int = 2  # Paper shows 1-2 examples are sufficient
    
    # Evaluation
    output_dir: Path = Path("./outputs")
    
    def get_model_display_name(self):
        if self.llm_model_path:
            base_name = Path(self.llm_model_path).name
            if self.lora_path:
                # Extract SFT run name from lora path (e.g., sft_20260203_131139)
                lora_name = Path(self.lora_path).parent.name
                return f"{base_name}+{lora_name}"
            return base_name
        return self.llm_model
