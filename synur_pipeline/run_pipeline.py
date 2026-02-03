"""
Extraction pipeline:
1. Load data and schema
2. Segment transcripts
3. RAG-filter schema
4. Extract observations via LLM
5. Evaluate against annotations

Usage:
    cd synur_pipeline
    
    # OpenAI models (requires OPENAI_API_KEY in .env)
    python run_pipeline.py --data dev --model gpt-4o-mini
    python run_pipeline.py --data dev --model gpt-4o --few-shot
    
    # Local models (point to your model folder)
    python run_pipeline.py --data dev --model llama-3.2-3b --model-path /path/to/Llama-3.2-3B-Instruct
    python run_pipeline.py --data dev --model mistral-7b --model-path /models/Mistral-7B-Instruct-v0.3
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get the directory where this script lives
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add script directory to path for imports
sys.path.insert(0, str(SCRIPT_DIR))

from config import PipelineConfig
from schema_rag import SchemaRAG, HybridSchemaRAG, FewShotRAG
from segmentation import segment_transcript, segment_transcript_simple
from extraction import extract_observations, extract_from_full_transcript


# Global model storage (loaded once, reused)
_openai_client = None
_local_model = None
_local_tokenizer = None
_model_name = None
_model_path = None
_max_tokens = 4096


def init_llm(model_name, model_path=None, max_tokens=4096):

    global _openai_client, _local_model, _local_tokenizer, _model_name, _model_path, _max_tokens
    
    _model_name = model_name
    _model_path = model_path
    _max_tokens = max_tokens
    
    if model_path:
        print(f"Loading model from {model_path}")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token
        
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _local_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
        print(f"Model loaded")
    else:
        from openai import OpenAI
        _openai_client = OpenAI()


def call_llm(messages, temperature=0.0, json_mode=False):

    if _model_path:
        # Local model
        prompt = _local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if json_mode:
            prompt += "\nRespond with valid JSON only:\n"
        
        inputs = _local_tokenizer(prompt, return_tensors="pt").to(_local_model.device)
        
        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=_max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=_local_tokenizer.pad_token_id
        )
        
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return _local_tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        # OpenAI
        kwargs = {"model": _model_name, "messages": messages, "temperature": temperature, "max_tokens": _max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = _openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


def get_model_name():
    return _model_name

def load_jsonl(path):

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_observations(obs_field):

    if isinstance(obs_field, str):
        return json.loads(obs_field)
    return obs_field


def run_pipeline(config, use_llm_segmentation=False, verbose=True, limit=None):

    # Load .env from project root
    load_dotenv(PROJECT_ROOT / ".env")
    
    # Initialize LLM (OpenAI or local model)
    if verbose:
        print(f"Initializing LLM")
    
    init_llm(config.llm_model, config.llm_model_path, config.max_tokens)
    
    if verbose:
        print(f"Using model: {get_model_name()}")
    
    # Load schema and initialize RAG
    if verbose:
        print(f"Loading schema from {config.schema_path}")
    
    # Use hybrid search if configured
    if getattr(config, 'use_hybrid_search', False):
        alpha = getattr(config, 'hybrid_alpha', 0.6)
        schema_rag = HybridSchemaRAG(config.schema_path, config.embedding_model, alpha=alpha)
        if verbose:
            print(f"  Using HYBRID search (alpha={alpha})")
    else:
        schema_rag = SchemaRAG(config.schema_path, config.embedding_model)
    
    if verbose:
        print(f"  Loaded {len(schema_rag.schema_rows)} schema rows.")
    
    # Load evaluation data
    if verbose:
        print(f"Loading evaluation data from {config.dev_path}.")
    eval_data = load_jsonl(config.dev_path)
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        eval_data = eval_data[:limit]
        if verbose:
            print(f"Limited to {len(eval_data)} transcripts (--limit {limit})")
    else:
        if verbose:
            print(f"Loaded {len(eval_data)} transcripts.")
    
    # Load training data for few-shot examples (RAG-based selection)
    few_shot_rag = None
    if config.use_few_shot:
        if verbose:
            print(f"Loading training data for few-shot examples (RAG-based selection).")
        train_data = load_jsonl(config.train_path)
        few_shot_rag = FewShotRAG(train_data, config.embedding_model)
        if verbose:
            print(f"Embedded {len(train_data)} training transcripts for similarity matching.")
    
    # Process each transcript
    all_predictions = []
    all_gold = []
    
    if verbose:
        print(f"\nProcessing transcripts with {config.llm_model}.")
        print(f"Prompt mode: {'Paper (Appendix A.1)' if config.use_paper_prompts else 'Enhanced'}.")
    
    for item in tqdm(eval_data, disable=not verbose):
        transcript_id = item.get("id", "unknown")
        transcript = item["transcript"]
        gold_obs = parse_observations(item["observations"])
        
        # Get few-shot examples via RAG (most similar training transcripts)
        few_shot_examples = None
        if few_shot_rag:
            few_shot_examples = few_shot_rag.retrieve(transcript, top_n=config.num_few_shot_examples)
        
        # Step 1: Segment the transcript
        if use_llm_segmentation:
            segments = segment_transcript(
                transcript,
                call_llm,
                use_paper_prompt=config.use_paper_prompts
            )
        else:
            segments = segment_transcript_simple(transcript)
        
        # Steps 2-3: RAG filter + Extract observations
        predictions = extract_from_full_transcript(
            transcript=transcript,
            segments=segments,
            schema_rag=schema_rag,
            call_llm=call_llm,
            top_n_schema=config.top_n_schema_rows,
            few_shot_examples=few_shot_examples,
            temperature=config.temperature,
            use_paper_prompt=config.use_paper_prompts
        )
        
        all_predictions.append(predictions)
        all_gold.append(gold_obs)
    
    
    return all_predictions, all_gold


def run_official_eval(pred_jsonl_path, ref_jsonl_path):
    import subprocess
    
    eval_script = PROJECT_ROOT / "mediqa_synur_eval_script.py"

    
    result = subprocess.run(
        ["python", str(eval_script), "-r", str(ref_jsonl_path), "-p", str(pred_jsonl_path)],
        capture_output=True,
        text=True
    )
    
    lines = result.stdout.strip().split("\n")
    try:
        idx = lines.index("Final Results:")
        precision = float(lines[idx + 1])
        recall = float(lines[idx + 2])
        f1 = float(lines[idx + 3])
        return {"precision": precision, "recall": recall, "f1": f1}
    except (ValueError, IndexError) as e:
        print(f"Error parsing official eval output: {e}")
        print(f"Output was: {result.stdout}")
        return None


def validate_observations(observations):

    valid = []
    for obs in observations:
        # Must be a dict
        if not isinstance(obs, dict):
            continue
        # Must have required fields
        if 'id' not in obs:
            continue
        if 'value' not in obs or obs['value'] is None:
            continue
        # value_type is optional but good to have
        valid.append(obs)
    return valid


def save_results(config, all_predictions, all_gold, eval_data, use_llm_segmentation=False, limit=None, data_split="train"):
    
    model_name = config.get_model_display_name()
    
    emb_short = config.embedding_model.split("/")[-1]
    if emb_short == "text-embedding-3-small":
        emb_short = "openai-small"
    elif "MiniLM" in emb_short:
        emb_short = "MiniLM"
    elif "bge" in emb_short.lower():
        emb_short = "BGE"
    
    # Build config string
    config_parts = [
        data_split,
        emb_short,
        f"top{config.top_n_schema_rows}"
    ]
    if getattr(config, 'use_hybrid_search', False):
        config_parts.append(f"hybrid{config.hybrid_alpha}")
    if not config.use_paper_prompts:
        config_parts.append("enhanced")
    if config.use_few_shot:
        config_parts.append(f"{config.num_few_shot_examples}shot")
    
    config_folder = "_".join(config_parts)
    
    # Create organized output directory
    run_dir = config.output_dir / model_name / config_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions (for error analysis)
    predictions_path = run_dir / f"predictions_{timestamp}.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        output = []
        for i, (preds, gold) in enumerate(zip(all_predictions, all_gold)):
            output.append({
                "id": eval_data[i].get("id", str(i)),
                "predictions": preds,
                "gold": gold
            })
        json.dump(output, f, indent=2)
    
    # Save predictions in official JSONL format
    official_pred_path = run_dir / f"submission_{timestamp}.jsonl"
    total_obs = 0
    valid_obs = 0
    with open(official_pred_path, "w", encoding="utf-8") as f:
        for i, preds in enumerate(all_predictions):
            # Validate observations before writing
            validated_preds = validate_observations(preds)
            total_obs += len(preds)
            valid_obs += len(validated_preds)
            record = {
                "id": eval_data[i].get("id", str(i)),
                "observations": validated_preds
            }
            f.write(json.dumps(record) + "\n")
    
    # Report filtered observations
    filtered_count = total_obs - valid_obs
    if filtered_count > 0:
        print(f"\n[INFO] Filtered {filtered_count} malformed observations (missing id/value)")
        print(f"[INFO] {valid_obs}/{total_obs} observations retained for evaluation")
    
    # Create temporary reference file for evaluation
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as ref_file:
        for i, gold in enumerate(all_gold):
            record = {
                "id": eval_data[i].get("id", str(i)),
                "observations": gold
            }
            ref_file.write(json.dumps(record) + "\n")
        ref_path = ref_file.name
    
    # Run official evaluation
    official_result = run_official_eval(official_pred_path, ref_path)
    
    # Clean up temp file
    os.remove(ref_path)
    
    # Save summary with all config details
    summary_path = run_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            # Run metadata
            "timestamp": timestamp,
            "run_folder": str(run_dir.relative_to(config.output_dir)),
            
            # Model settings
            "llm_model": config.llm_model,
            "llm_model_path": config.llm_model_path,
            "model_display_name": model_name,
            "embedding_model": config.embedding_model,
            "temperature": config.temperature,
            
            # RAG settings
            "top_n_schema_rows": config.top_n_schema_rows,
            "hybrid_search": getattr(config, 'use_hybrid_search', False),
            "hybrid_alpha": getattr(config, 'hybrid_alpha', None) if getattr(config, 'use_hybrid_search', False) else None,
            
            # Prompt settings
            "use_paper_prompts": config.use_paper_prompts,
            "use_enhanced_prompts": not config.use_paper_prompts,
            
            # Few-shot settings
            "few_shot": config.use_few_shot,
            "num_few_shot_examples": config.num_few_shot_examples if config.use_few_shot else 0,
            
            # Segmentation
            "segmentation": "llm" if use_llm_segmentation else "simple",
            
            # Data
            "data_split": data_split,
            "num_transcripts": len(all_predictions),
            "limit": limit,
            
            # Observation counts
            "total_observations": total_obs,
            "valid_observations": valid_obs,
            "filtered_malformed": filtered_count,
            
            # Official evaluation results
            "precision": official_result["precision"] if official_result else None,
            "recall": official_result["recall"] if official_result else None,
            "f1": official_result["f1"] if official_result else None
        }, f, indent=2)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    if official_result:
        print(f"  Precision: {official_result['precision']:.4f}")
        print(f"  Recall:    {official_result['recall']:.4f}")
        print(f"  F1:        {official_result['f1']:.4f}  ({official_result['f1']*100:.1f}%)")
    else:
        print("Official evaluation failed - check eval script output above")
    
    print(f"\nFiles saved to: {run_dir.relative_to(config.output_dir)}/")
    print(f"  predictions_{timestamp}.json  (for error analysis)")
    print(f"  submission_{timestamp}.jsonl  (official format)")
    print(f"  summary_{timestamp}.json      (config + metrics)")


def main():
    parser = argparse.ArgumentParser(description="SYNUR Extraction Pipeline")
    parser.add_argument("--data", choices=["train", "dev"], default="train",
                        help="Dataset to evaluate on")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Model name for display (e.g., gpt-4o-mini, llama-3.2-3b)")
    parser.add_argument("--model-path", default=None,
                        help="Path to local model folder. If provided, uses local model instead of OpenAI.")
    parser.add_argument("--few-shot", action="store_true",
                        help="Use few-shot examples")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Use zero-shot")
    parser.add_argument("--num-examples", type=int, default=2,
                        help="Number of few-shot examples (RAG-selected)")
    parser.add_argument("--top-n", type=int, default=60,
                        help="Number of schema rows to retrieve per segment")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="Embedding model (text-embedding-3-small, all-MiniLM-L6-v2, etc.)")
    parser.add_argument("--llm-segmentation", action="store_true",
                        help="Use LLM for segmentation")
    parser.add_argument("--enhanced-prompts", action="store_true",
                        help="Use enhanced prompts instead of paper's exact prompts")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output files (default: synur_pipeline/outputs)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of transcripts to process")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid search (BM25 + dense embeddings)")
    parser.add_argument("--hybrid-alpha", type=float, default=0.6,
                        help="Alpha for hybrid search (1.0=dense only, 0.0=BM25 only, default=0.6)")
    
    args = parser.parse_args()
    
    # Determine few-shot setting (default to zero-shot)
    use_few_shot = args.few_shot
    
    # Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "outputs"
    
    # Create config with absolute paths (works from any directory)
    config = PipelineConfig(
        schema_path=PROJECT_ROOT / "Data/synur_schema.json",
        train_path=PROJECT_ROOT / "Data/train.jsonl",
        dev_path=PROJECT_ROOT / "Data" / f"{args.data}.jsonl",
        llm_model=args.model,
        llm_model_path=args.model_path,
        embedding_model=args.embedding_model,
        use_few_shot=use_few_shot,
        num_few_shot_examples=args.num_examples,
        top_n_schema_rows=args.top_n,
        use_paper_prompts=not args.enhanced_prompts,  # Default to paper prompts
        output_dir=output_dir
    )
    
    # Add hybrid search settings (not in dataclass, added dynamically)
    config.use_hybrid_search = args.hybrid
    config.hybrid_alpha = args.hybrid_alpha
    
    # Determine display name for model
    model_display = config.get_model_display_name()
    
    print("=" * 60)
    print("SYNUR Extraction Pipeline")
    print(f"Model: {model_display}")
    if config.llm_model_path:
        print(f"Model path: {config.llm_model_path}")
    print(f"Embedding: {config.embedding_model}")
    print(f"Mode: {'Few-shot' if config.use_few_shot else 'Zero-shot'}")
    if config.use_few_shot:
        print(f"Examples: {config.num_few_shot_examples}")
    print(f"Prompts: {'Paper (Appendix A.1)' if config.use_paper_prompts else 'Enhanced'}")
    print(f"Top-N schema rows: {config.top_n_schema_rows}")
    print(f"Retrieval: {'Hybrid (BM25 + dense, alpha=' + str(config.hybrid_alpha) + ')' if config.use_hybrid_search else 'Dense only'}")
    print(f"Data: {args.data}")
    print(f"Segmentation: {'LLM' if args.llm_segmentation else 'Simple rules'}")
    if args.limit:
        print(f"Limit: {args.limit} transcripts")
    print("=" * 60)
    
    # Run pipeline
    all_predictions, all_gold = run_pipeline(
        config,
        use_llm_segmentation=args.llm_segmentation,
        verbose=True,
        limit=args.limit
    )
    
    # Save results and run official evaluation
    eval_data = load_jsonl(config.dev_path)
    save_results(
        config, all_predictions, all_gold, eval_data,
        use_llm_segmentation=args.llm_segmentation,
        limit=args.limit,
        data_split=args.data
    )


if __name__ == "__main__":
    main()

