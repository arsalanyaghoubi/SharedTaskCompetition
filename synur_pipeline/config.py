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
    base_model_path: Optional[str] = None  # Override base model path for PEFT models (run works on both plantain and valkyrie)
    
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
            model_path = Path(self.llm_model_path)
            
            # Check if this is an SFT model (path contains sft_output)
            path_str = str(model_path)
            if "sft_output" in path_str:
                # Extract the SFT checkpoint name (e.g., sft_20260204_011858)
                # Path is like: sft_output_70b/sft_20260204_011858/final
                parts = model_path.parts
                for i, part in enumerate(parts):
                    if part.startswith("sft_2"):  # SFT timestamp folder
                        sft_name = part
                        # Determine base model from the sft_output folder name
                        if i > 0 and "70b" in parts[i-1].lower():
                            base_model = "Llama-3.3-70B-Instruct"
                        elif i > 0 and "8b" in parts[i-1].lower():
                            base_model = "Meta-Llama-3-8B-Instruct"
                        else:
                            base_model = "Llama"
                        return f"{base_model}+{sft_name}"
            
            # For non-SFT models, use the path name
            base_name = model_path.name
            if self.lora_path:
                # Extract SFT run name from lora path (e.g., sft_20260203_131139)
                lora_name = Path(self.lora_path).parent.name
                return f"{base_name}+{lora_name}"
            return base_name
        return self.llm_model
