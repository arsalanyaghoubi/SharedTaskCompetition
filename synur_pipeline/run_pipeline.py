"""
Extraction pipeline:
1. Load data and schema
2. Segment transcripts
3. RAG-filter schema
4. Extract observations via LLM
5. Evaluate against annotations

Usage:
    cd synur_pipeline
    python run_pipeline.py --data dev --model gpt-4o --few-shot
    python run_pipeline.py --data dev --model gpt-4o-mini --zero-shot
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Get the directory where this script lives
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add script directory to path for imports
sys.path.insert(0, str(SCRIPT_DIR))

from config import PipelineConfig
from schema_rag import SchemaRAG, FewShotRAG
from segmentation import segment_transcript, segment_transcript_simple
from extraction import extract_observations, extract_from_full_transcript
from evaluation import evaluate_single, evaluate_dataset, compute_error_analysis, EvaluationResult


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


def run_pipeline(config, use_llm_segmentation=False, verbose=True, limit=None, partial_credit=False):

    # Load .env from project root
    load_dotenv(PROJECT_ROOT / ".env")
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load schema and initialize RAG
    if verbose:
        print(f"Loading schema from {config.schema_path}.")
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
                transcript, client, config.llm_model,
                use_paper_prompt=config.use_paper_prompts
            )
        else:
            segments = segment_transcript_simple(transcript)
        
        # Steps 2-3: RAG filter + Extract observations
        predictions = extract_from_full_transcript(
            transcript=transcript,
            segments=segments,
            schema_rag=schema_rag,
            client=client,
            model=config.llm_model,
            top_n_schema=config.top_n_schema_rows,
            few_shot_examples=few_shot_examples,
            temperature=config.temperature,
            use_paper_prompt=config.use_paper_prompts
        )
        
        all_predictions.append(predictions)
        all_gold.append(gold_obs)
    
    # Step 4: Evaluate
    if verbose:
        print("\nEvaluating predictions.")
        if partial_credit:
            print("Using partial credit for MULTI_SELECT.")
    result = evaluate_dataset(all_predictions, all_gold, partial_credit=partial_credit)
    
    return all_predictions, all_gold, result


def save_results(config, all_predictions, all_gold, result, eval_data, use_llm_segmentation=False, partial_credit=False, limit=None):

    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    predictions_path = config.output_dir / f"predictions_{config.llm_model}_{timestamp}.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        output = []
        for i, (preds, gold) in enumerate(zip(all_predictions, all_gold)):
            output.append({
                "id": eval_data[i].get("id", str(i)),
                "predictions": preds,
                "gold": gold,
                "single_result": {
                    "precision": evaluate_single(preds, gold).precision,
                    "recall": evaluate_single(preds, gold).recall,
                    "f1": evaluate_single(preds, gold).f1
                }
            })
        json.dump(output, f, indent=2)
    
    # Save summary
    summary_path = config.output_dir / f"summary_{config.llm_model}_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            # Run metadata
            "timestamp": timestamp,
            
            # Model settings
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "temperature": config.temperature,
            
            # RAG settings
            "top_n_schema_rows": config.top_n_schema_rows,
            
            # Prompt settings
            "use_paper_prompts": config.use_paper_prompts,
            "use_enhanced_prompts": not config.use_paper_prompts,
            
            # Few-shot settings
            "few_shot": config.use_few_shot,
            "num_few_shot_examples": config.num_few_shot_examples if config.use_few_shot else 0,
            
            # Segmentation
            "segmentation": "llm" if use_llm_segmentation else "simple",
            
            # Evaluation settings
            "partial_credit": partial_credit,
            "num_transcripts": len(all_predictions),
            "limit": limit,
            
            # Results
            "true_positives": result.true_positives,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1
        }, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="SYNUR Extraction Pipeline")
    parser.add_argument("--data", choices=["train", "dev"], default="train",
                        help="Dataset to evaluate on")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model to use (gpt-4o, gpt-4o-mini, gpt-4.1, etc.)")
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
    parser.add_argument("--partial-credit", action="store_true",
                        help="Use partial credit for MULTI_SELECT")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output files (default: synur_pipeline/outputs)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of transcripts to process")
    
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
        embedding_model=args.embedding_model,
        use_few_shot=use_few_shot,
        num_few_shot_examples=args.num_examples,
        top_n_schema_rows=args.top_n,
        use_paper_prompts=not args.enhanced_prompts,  # Default to paper prompts
        output_dir=output_dir
    )
    print("=" * 60)
    print("SYNUR Extraction Pipeline")
    print(f"Model: {config.llm_model}")
    print(f"Embedding: {config.embedding_model}")
    print(f"Mode: {'Few-shot' if config.use_few_shot else 'Zero-shot'}")
    if config.use_few_shot:
        print(f"Examples: {config.num_few_shot_examples}")
    print(f"Prompts: {'Paper (Appendix A.1)' if config.use_paper_prompts else 'Enhanced'}")
    print(f"Top-N schema rows: {config.top_n_schema_rows}")
    print(f"Data: {args.data}")
    print(f"Segmentation: {'LLM' if args.llm_segmentation else 'Simple rules'}")
    if args.limit:
        print(f"Limit: {args.limit} transcripts")
    if args.partial_credit:
        print(f"Evaluation: Partial credit for MULTI_SELECT")
    print("=" * 60)
    
    # Run pipeline
    all_predictions, all_gold, result = run_pipeline(
        config,
        use_llm_segmentation=args.llm_segmentation,
        verbose=True,
        limit=args.limit,
        partial_credit=args.partial_credit
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS" + (" (with partial credit)" if args.partial_credit else ""))
    print("=" * 60)
    print(result)
    print(f"\nF1 Score: {result.f1 * 100:.1f}%")
    
    
    # Save results
    eval_data = load_jsonl(config.dev_path)
    save_results(
        config, all_predictions, all_gold, result, eval_data,
        use_llm_segmentation=args.llm_segmentation,
        partial_credit=args.partial_credit,
        limit=args.limit
    )


if __name__ == "__main__":
    main()

