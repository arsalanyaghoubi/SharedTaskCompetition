"""
Evaluate Schema RAG Retrieval Quality

This script analyzes whether the RAG retrieval step is a bottleneck.
For each gold observation, we check if its schema item was retrieved
for the corresponding transcript segment.

Key metrics:
- Recall@K: What % of gold schema items are in the top-K retrieved?
- Per-segment coverage: How many gold items are covered per segment?
- Most frequently missed: Which schema items are consistently NOT retrieved?

Usage:
    python evaluate_rag.py --data train --top-n 60
    python evaluate_rag.py --data train --top-n 60 --embedding-model all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import sys
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from schema_rag import SchemaRAG
from segmentation import segment_transcript_simple


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


def evaluate_retrieval(data, schema_rag, top_n=60, verbose=True):
    """
    For each transcript, segment it and retrieve schema items.
    Then check if the gold observation schema IDs are in the retrieved set.
    """
    
    results = {
        "total_gold_observations": 0,
        "retrieved_gold_observations": 0,
        "missed_gold_observations": 0,
        "transcripts_analyzed": 0,
        "segments_analyzed": 0,
        "missed_by_schema_id": defaultdict(int),
        "retrieved_by_schema_id": defaultdict(int),
        "per_transcript_recall": [],
    }
    
    # Build schema ID to name mapping
    id_to_name = {row.id: row.name for row in schema_rag.schema_rows}
    
    for item in data:
        transcript_id = item.get("id", "unknown")
        transcript = item["transcript"]
        gold_obs = parse_observations(item["observations"])
        
        # Get all gold schema IDs for this transcript
        gold_ids = set(str(obs.get("id")) for obs in gold_obs)
        
        # Segment the transcript (using simple segmentation)
        segments = segment_transcript_simple(transcript)
        
        # Retrieve schema items for each segment and track which are covered
        all_retrieved_ids = set()
        for segment in segments:
            retrieved_rows = schema_rag.retrieve(segment, top_n=top_n)
            retrieved_ids = set(row.id for row in retrieved_rows)
            all_retrieved_ids.update(retrieved_ids)
            results["segments_analyzed"] += 1
        
        # Check coverage
        retrieved_gold = gold_ids & all_retrieved_ids
        missed_gold = gold_ids - all_retrieved_ids
        
        results["total_gold_observations"] += len(gold_ids)
        results["retrieved_gold_observations"] += len(retrieved_gold)
        results["missed_gold_observations"] += len(missed_gold)
        results["transcripts_analyzed"] += 1
        
        # Track per-transcript recall
        transcript_recall = len(retrieved_gold) / len(gold_ids) if gold_ids else 1.0
        results["per_transcript_recall"].append({
            "id": transcript_id,
            "gold_count": len(gold_ids),
            "retrieved_count": len(retrieved_gold),
            "missed_count": len(missed_gold),
            "recall": transcript_recall,
            "missed_ids": list(missed_gold)
        })
        
        # Track which schema items are missed/retrieved
        for schema_id in retrieved_gold:
            results["retrieved_by_schema_id"][schema_id] += 1
        for schema_id in missed_gold:
            results["missed_by_schema_id"][schema_id] += 1
    
    # Calculate aggregate metrics
    recall = results["retrieved_gold_observations"] / results["total_gold_observations"]
    
    # Sort missed by frequency
    missed_sorted = sorted(
        results["missed_by_schema_id"].items(),
        key=lambda x: -x[1]
    )
    
    # Print report
    print("\n" + "=" * 70)
    print("SCHEMA RAG RETRIEVAL ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Embedding model: {schema_rag.embedding_model_name}")
    print(f"  Top-N retrieved: {top_n}")
    print(f"  Transcripts: {results['transcripts_analyzed']}")
    print(f"  Total segments: {results['segments_analyzed']}")
    
    print(f"\n" + "-" * 70)
    print("RETRIEVAL COVERAGE (Schema ID Recall)")
    print("-" * 70)
    print(f"  Total gold observations: {results['total_gold_observations']}")
    print(f"  Retrieved (in top-{top_n}): {results['retrieved_gold_observations']}")
    print(f"  Missed (not retrieved): {results['missed_gold_observations']}")
    print(f"\n  >>> RECALL@{top_n}: {recall:.1%} <<<")
    
    if recall < 1.0:
        print(f"\n  This means {results['missed_gold_observations']} gold observations")
        print(f"  CANNOT be extracted because their schema items weren't retrieved.")
        print(f"  This is the ceiling on recall - model can't do better than {recall:.1%}")
    
    print(f"\n" + "-" * 70)
    print("MOST FREQUENTLY MISSED SCHEMA ITEMS")
    print("-" * 70)
    if missed_sorted:
        print(f"  {'ID':<6} {'Count':<8} Name")
        print(f"  {'-'*6} {'-'*8} {'-'*40}")
        for schema_id, count in missed_sorted[:20]:
            name = id_to_name.get(schema_id, "Unknown")
            print(f"  {schema_id:<6} {count:<8} {name}")
    else:
        print("  No schema items were missed! RAG retrieval is not a bottleneck.")
    
    # Analyze transcripts with worst recall
    worst_transcripts = sorted(
        results["per_transcript_recall"],
        key=lambda x: x["recall"]
    )[:10]
    
    print(f"\n" + "-" * 70)
    print("TRANSCRIPTS WITH WORST RETRIEVAL RECALL")
    print("-" * 70)
    print(f"  {'ID':<12} {'Gold':<6} {'Retrieved':<10} {'Missed':<8} {'Recall':<8}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
    for t in worst_transcripts:
        if t["recall"] < 1.0:
            print(f"  {t['id']:<12} {t['gold_count']:<6} {t['retrieved_count']:<10} {t['missed_count']:<8} {t['recall']:.1%}")
    
    print("\n" + "=" * 70)
    
    return results


def compare_top_n(data, schema_rag, top_n_values=[20, 40, 60, 80, 100, 120]):
    """Compare recall at different top-N values."""
    
    print("\n" + "=" * 70)
    print("RECALL vs TOP-N COMPARISON")
    print("=" * 70)
    print(f"\n  {'Top-N':<10} {'Recall':<10} {'Missed':<10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    
    for top_n in top_n_values:
        results = evaluate_retrieval_silent(data, schema_rag, top_n)
        recall = results["retrieved_gold_observations"] / results["total_gold_observations"]
        missed = results["missed_gold_observations"]
        print(f"  {top_n:<10} {recall:.1%}      {missed}")
    
    print("\n" + "=" * 70)


def evaluate_retrieval_silent(data, schema_rag, top_n):
    """Silent version for comparison."""
    results = {
        "total_gold_observations": 0,
        "retrieved_gold_observations": 0,
        "missed_gold_observations": 0,
    }
    
    for item in data:
        transcript = item["transcript"]
        gold_obs = parse_observations(item["observations"])
        gold_ids = set(str(obs.get("id")) for obs in gold_obs)
        
        segments = segment_transcript_simple(transcript)
        all_retrieved_ids = set()
        for segment in segments:
            retrieved_rows = schema_rag.retrieve(segment, top_n=top_n)
            all_retrieved_ids.update(row.id for row in retrieved_rows)
        
        retrieved_gold = gold_ids & all_retrieved_ids
        missed_gold = gold_ids - all_retrieved_ids
        
        results["total_gold_observations"] += len(gold_ids)
        results["retrieved_gold_observations"] += len(retrieved_gold)
        results["missed_gold_observations"] += len(missed_gold)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Schema RAG Retrieval")
    parser.add_argument("--data", choices=["train", "dev"], default="train",
                        help="Dataset to evaluate on")
    parser.add_argument("--top-n", type=int, default=60,
                        help="Number of schema rows to retrieve")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="Embedding model for RAG")
    parser.add_argument("--compare-top-n", action="store_true",
                        help="Compare recall at different top-N values")
    
    args = parser.parse_args()
    
    # Paths
    project_root = SCRIPT_DIR.parent
    schema_path = project_root / "Data" / "synur_schema.json"
    data_path = project_root / "Data" / f"{args.data}.jsonl"
    
    print(f"Loading schema from {schema_path}")
    schema_rag = SchemaRAG(schema_path, args.embedding_model)
    print(f"Loaded {len(schema_rag.schema_rows)} schema rows")
    
    print(f"Loading data from {data_path}")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} transcripts")
    
    if args.compare_top_n:
        compare_top_n(data, schema_rag)
    else:
        evaluate_retrieval(data, schema_rag, top_n=args.top_n)


if __name__ == "__main__":
    main()
