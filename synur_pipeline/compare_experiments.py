import argparse
import json
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMA_PATH = PROJECT_ROOT / "Data" / "synur_schema.json"
TRAIN_PATH = PROJECT_ROOT / "Data" / "train.jsonl"
DEV_PATH = PROJECT_ROOT / "Data" / "dev.jsonl"


def load_schema():

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    return {str(s['id']): s for s in schema}


def load_reference_data(data_split):

    data_path = DEV_PATH if data_split == "dev" else TRAIN_PATH
    if not data_path.exists():
        return {}
    
    gold_by_transcript = {}
    with open(data_path) as f:
        for line in f:
            record = json.loads(line)
            tid = str(record['id'])
            obs = record.get('observations', [])
            if isinstance(obs, str):
                obs = json.loads(obs)
            gold_by_transcript[tid] = obs
    return gold_by_transcript


def discover_experiments():

    experiments = []
    
    for summary_file in OUTPUTS_DIR.glob("**/summary_*.json"):
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            
            # Skip test runs with limit
            if summary.get("limit") is not None:
                continue
            
            timestamp = summary.get("timestamp", "")
            
            # Find corresponding files
            pred_file = summary_file.parent / f"predictions_{timestamp}.json"
            submission_file = summary_file.parent / f"submission_{timestamp}.jsonl"
            
            exp = {
                "summary_file": summary_file,
                "predictions_file": pred_file if pred_file.exists() else None,
                "submission_file": submission_file if submission_file.exists() else None,
                "timestamp": timestamp,
                "model": summary.get("model_display_name", summary.get("llm_model", "unknown")),
                "data_split": summary.get("data_split", "unknown"),
                "embedding": summary.get("embedding_model", "").split("/")[-1],
                "hybrid": summary.get("hybrid_search", False),
                "hybrid_alpha": summary.get("hybrid_alpha"),
                "few_shot": summary.get("few_shot", False),
                "num_examples": summary.get("num_few_shot_examples", 0),
                "lora_path": summary.get("lora_path"),
                "quantization": summary.get("quantization"),
                "segmentation": summary.get("segmentation", "simple"),
                "num_transcripts": summary.get("num_transcripts", 0),
                "precision": summary.get("precision"),
                "recall": summary.get("recall"),
                "f1": summary.get("f1"),
                "total_obs": summary.get("total_observations", 0),
                "valid_obs": summary.get("valid_observations", 0),
            }
            
            exp["short_name"] = build_short_name(exp)
            exp["full_name"] = build_full_name(exp)
            
            experiments.append(exp)
            
        except Exception as e:
            print(f"Warning: Could not parse {summary_file}: {e}")
    
    return experiments


def build_short_name(exp):

    parts = []
    
    # Model
    model = exp["model"]
    if "gpt-4o-mini" in model.lower():
        parts.append("GPT")
    elif exp["lora_path"]:
        parts.append("Llama-SFT")
    elif "70b" in model.lower():
        parts.append("Llama-70B")
    elif "8b" in model.lower():
        parts.append("Llama-8B")
    else:
        parts.append(model[:10])
    
    # Embedding
    emb = exp["embedding"]
    if "bge" in emb.lower():
        parts.append("BGE")
    elif "minilm" in emb.lower():
        parts.append("MiniLM")
    else:
        parts.append("OpenAI")
    
    # Retrieval
    parts.append("Hyb" if exp["hybrid"] else "Den")
    
    # Few-shot
    if exp["few_shot"] and exp["num_examples"] > 0:
        parts.append(f"{exp['num_examples']}sh")
    
    # Segmentation
    if exp["segmentation"] == "llm":
        parts.append("LLMseg")
    
    # Quantization
    if exp["quantization"]:
        parts.append(exp["quantization"].replace("-bit", "b"))
    
    return "+".join(parts)


def build_full_name(exp):

    parts = []
    
    model = exp["model"]
    if "gpt-4o-mini" in model.lower():
        parts.append("GPT-4o-mini")
    elif exp["lora_path"]:
        parts.append("Llama-8B-SFT")
    elif "70b" in model.lower():
        parts.append("Llama-3.3-70B")
    elif "8b" in model.lower():
        parts.append("Llama-3-8B")
    else:
        parts.append(model)
    
    emb = exp["embedding"]
    if "bge" in emb.lower():
        parts.append("BGE")
    elif "minilm" in emb.lower():
        parts.append("MiniLM")
    else:
        parts.append("OpenAI")
    
    parts.append("Hybrid" if exp["hybrid"] else "Dense")
    
    if exp["segmentation"] == "llm":
        parts.append("LLM-seg")
    
    if exp["few_shot"]:
        parts.append(f"{exp['num_examples']}-shot")
    else:
        parts.append("0-shot")
    
    return " + ".join(parts)


def normalize_value(value):

    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return sorted([normalize_value(v) for v in value])
    if isinstance(value, (int, float)):
        return float(value)
    return str(value).strip().lower()


def compute_error_analysis(exp, gold_by_transcript, schema):

    submission_file = exp.get("submission_file")
    if not submission_file or not submission_file.exists():
        return None
    
    if not gold_by_transcript:
        return None
    
    pred_by_transcript = {}
    with open(submission_file) as f:
        for line in f:
            record = json.loads(line)
            tid = str(record['id'])
            pred_by_transcript[tid] = record.get('observations', [])
    
    stats = {"correct": 0, "wrong_value": 0, "missed": 0, "hallucinated": 0}
    
    for tid, gold_obs in gold_by_transcript.items():
        preds = pred_by_transcript.get(tid, [])
        valid_preds = [p for p in preds if isinstance(p, dict) and 'id' in p and 'value' in p]
        
        gold_by_concept = {str(g['id']): g for g in gold_obs}
        pred_by_concept = {str(p['id']): p for p in valid_preds}
        
        for gold in gold_obs:
            gid = str(gold['id'])
            if gid not in pred_by_concept:
                stats["missed"] += 1
            elif normalize_value(pred_by_concept[gid].get('value')) == normalize_value(gold.get('value')):
                stats["correct"] += 1
            else:
                stats["wrong_value"] += 1
        
        for pred in valid_preds:
            if str(pred['id']) not in gold_by_concept:
                stats["hallucinated"] += 1
    
    return stats


def filter_experiments(experiments, data_filter=None, model_filter=None):

    filtered = experiments
    
    if data_filter:
        filtered = [e for e in filtered if e["data_split"] == data_filter]
    
    if model_filter:
        model_lower = model_filter.lower()
        filtered = [e for e in filtered 
                   if model_lower in e["model"].lower() 
                   or model_lower in e["short_name"].lower()]
    
    return filtered


def export_csv(experiments, output_path):

    experiments = sorted(experiments, key=lambda x: (x.get("data_split") != "dev", -(x.get("f1") or 0)))
    
    schema = load_schema()
    gold_data_cache = {}
    
    for exp in experiments:
        split = exp.get("data_split", "unknown")
        if split not in gold_data_cache:
            gold_data_cache[split] = load_reference_data(split)
        exp["error_stats"] = compute_error_analysis(exp, gold_data_cache.get(split, {}), schema)
    
    fieldnames = [
        "short_name", "full_name", "data_split", "model", "embedding", 
        "hybrid", "segmentation", "few_shot", "num_examples", "lora_path", 
        "quantization", "precision", "recall", "f1", "num_transcripts", 
        "correct", "wrong_value", "missed", "hallucinated", "timestamp"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for exp in experiments:
            row = {k: exp.get(k) for k in fieldnames}
            stats = exp.get("error_stats") or {}
            row["correct"] = stats.get("correct", "")
            row["wrong_value"] = stats.get("wrong_value", "")
            row["missed"] = stats.get("missed", "")
            row["hallucinated"] = stats.get("hallucinated", "")
            writer.writerow(row)
    
    print(f"Saved {len(experiments)} experiments to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiments")
    parser.add_argument("--data", choices=["train", "dev"], help="Filter by data split")
    parser.add_argument("--model", type=str, help="Filter by model name")
    parser.add_argument("--csv", type=str, help="Output CSV path (default: outputs/comparison_results.csv)")
    
    args = parser.parse_args()
    
    experiments = discover_experiments()
    print(f"Found {len(experiments)} experiments")
    
    experiments = filter_experiments(experiments, args.data, args.model)
    if args.data or args.model:
        print(f"After filtering: {len(experiments)} experiments")
    
    csv_path = args.csv or (OUTPUTS_DIR / "comparison_results.csv")
    export_csv(experiments, csv_path)


if __name__ == "__main__":
    main()
