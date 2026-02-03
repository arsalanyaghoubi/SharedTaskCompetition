import argparse
import json
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import numpy as np

import sys
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Load .env for OpenAI API key
load_dotenv(PROJECT_ROOT / ".env")

from schema_rag import SchemaRAG
from segmentation import segment_transcript_simple

# Try to import BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank_bm25 not installed. Hybrid search disabled.")


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


def evaluate_retrieval(data, schema_rag, top_n=60, use_hybrid=False, alpha=0.6, bm25=None):

    total_gold = 0
    retrieved_gold = 0
    
    for item in data:
        transcript = item["transcript"]
        gold_obs = parse_observations(item["observations"])
        gold_ids = set(str(obs.get("id")) for obs in gold_obs)
        
        segments = segment_transcript_simple(transcript)
        all_retrieved_ids = set()
        
        for segment in segments:
            if use_hybrid and bm25 is not None:
                # Hybrid retrieval
                seg_emb = schema_rag._embed_single(segment)
                dense_scores = np.dot(schema_rag.schema_embeddings, seg_emb)
                sparse_scores = np.array(bm25.get_scores(segment.lower().split()))
                
                # Normalize
                def norm(s):
                    s = np.array(s)
                    if s.max() - s.min() == 0:
                        return np.zeros_like(s)
                    return (s - s.min()) / (s.max() - s.min())
                
                hybrid_scores = alpha * norm(dense_scores) + (1 - alpha) * norm(sparse_scores)
                top_indices = np.argsort(hybrid_scores)[-top_n:][::-1]
                all_retrieved_ids.update(schema_rag.schema_rows[i].id for i in top_indices)
            else:
                # Dense-only retrieval
                retrieved_rows = schema_rag.retrieve(segment, top_n=top_n)
                all_retrieved_ids.update(row.id for row in retrieved_rows)
        
        retrieved_gold += len(gold_ids & all_retrieved_ids)
        total_gold += len(gold_ids)
    
    recall = retrieved_gold / total_gold if total_gold > 0 else 0
    missed = total_gold - retrieved_gold
    return recall, missed, total_gold


def run_full_analysis(data, schema_path):

    # Embedding models to test
    embedding_models = [
        ("all-MiniLM-L6-v2", "MiniLM-L6 (local)"),
        ("all-MiniLM-L12-v2", "MiniLM-L12 (local)"),
        ("all-mpnet-base-v2", "MPNet (local)"),
        ("BAAI/bge-base-en-v1.5", "BGE-base (local)"),
        ("text-embedding-3-small", "OpenAI small"),
    ]
    
    # Top-N values to test
    top_n_values = [10, 20, 40, 60, 80, 120]
    
    total_gold = None
    
    print("\n" + "=" * 90)
    print("COMPREHENSIVE SCHEMA RETRIEVAL ANALYSIS")
    print("=" * 90)
    print(f"\nDataset: {len(data)} transcripts")
    print(f"Testing {len(embedding_models)} embedding models × {len(top_n_values)} top-N values")
    print("\n" + "-" * 90)
    
    # Results table
    results = {}
    
    for model_id, model_name in embedding_models:
        print(f"\nLoading {model_name}...", end=" ", flush=True)
        try:
            rag = SchemaRAG(schema_path, model_id)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        
        results[model_name] = {}
        for top_n in top_n_values:
            recall, missed, total = evaluate_retrieval(data, rag, top_n=top_n)
            results[model_name][top_n] = {"recall": recall, "missed": missed}
            if total_gold is None:
                total_gold = total
        
        del rag
    
    # Print results table
    print("\n" + "=" * 90)
    print("RECALL BY EMBEDDING MODEL AND TOP-N")
    print("=" * 90)
    print(f"\nTotal gold observations: {total_gold}")
    print("\nRecall (% of gold schema items retrieved):")
    print()
    
    # Header
    header = f"{'Model':<22}" + "".join(f"{'N='+str(n):>10}" for n in top_n_values)
    print(header)
    print("-" * len(header))
    
    # Data rows
    best_model = None
    best_recall = 0
    best_top_n = 60
    
    for model_name, top_n_results in results.items():
        row = f"{model_name:<22}"
        for top_n in top_n_values:
            if top_n in top_n_results:
                recall = top_n_results[top_n]["recall"]
                row += f"{recall*100:>9.1f}%"
                if recall > best_recall:
                    best_recall = recall
                    best_model = model_name
                    best_top_n = top_n
            else:
                row += f"{'—':>10}"
        print(row)
    
    print("-" * len(header))
    print(f"\nBest: {best_model} at N={best_top_n} ({best_recall:.1%} recall)")
    
    # Print missed observations table
    print("\n" + "=" * 90)
    print("MISSED OBSERVATIONS (ceiling on extraction recall)")
    print("=" * 90)
    print()
    
    header = f"{'Model':<22}" + "".join(f"{'N='+str(n):>10}" for n in top_n_values)
    print(header)
    print("-" * len(header))
    
    for model_name, top_n_results in results.items():
        row = f"{model_name:<22}"
        for top_n in top_n_values:
            if top_n in top_n_results:
                missed = top_n_results[top_n]["missed"]
                row += f"{missed:>10}"
            else:
                row += f"{'—':>10}"
        print(row)
    
    print("-" * len(header))
    
    return results, best_model, best_top_n


def run_hybrid_analysis(data, schema_path, embedding_model, top_n_values=[40, 60, 80]):

    if not HAS_BM25:
        print("Error: rank_bm25 not installed. Run: pip install rank_bm25")
        return
    
    print(f"\nLoading {embedding_model}...")
    rag = SchemaRAG(schema_path, embedding_model)
    
    # Build BM25 index
    print("Building BM25 index...")
    tokenized_corpus = [row.to_embedding_text().lower().split() for row in rag.schema_rows]
    bm25 = BM25Okapi(tokenized_corpus)
    
    alpha_values = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]
    
    print("\n" + "=" * 90)
    print("HYBRID SEARCH ANALYSIS (Dense + BM25)")
    print("=" * 90)
    print(f"\nEmbedding model: {embedding_model}")
    print(f"Alpha: 1.0 = dense only, 0.0 = BM25 only")
    print()
    
    for top_n in top_n_values:
        print(f"\n--- Top-N = {top_n} ---")
        print(f"{'Alpha':<10} {'Method':<25} {'Recall':>10} {'Missed':>10}")
        print("-" * 55)
        
        for alpha in alpha_values:
            if alpha == 1.0:
                method = "Dense only"
            elif alpha == 0.0:
                method = "BM25 only"
            else:
                method = f"Hybrid ({alpha:.1f}×dense + {1-alpha:.1f}×BM25)"
            
            recall, missed, _ = evaluate_retrieval(
                data, rag, top_n=top_n, 
                use_hybrid=True, alpha=alpha, bm25=bm25
            )
            print(f"{alpha:<10.1f} {method:<25} {recall:>9.1%} {missed:>10}")
    
    print("\n" + "=" * 90)


def run_single_evaluation(data, schema_rag, top_n, verbose=True):

    results = {
        "total_gold_observations": 0,
        "retrieved_gold_observations": 0,
        "missed_gold_observations": 0,
        "transcripts_analyzed": 0,
        "segments_analyzed": 0,
        "missed_by_schema_id": defaultdict(int),
        "per_transcript_recall": [],
    }
    
    id_to_name = {row.id: row.name for row in schema_rag.schema_rows}
    
    for item in data:
        transcript_id = item.get("id", "unknown")
        transcript = item["transcript"]
        gold_obs = parse_observations(item["observations"])
        gold_ids = set(str(obs.get("id")) for obs in gold_obs)
        
        segments = segment_transcript_simple(transcript)
        all_retrieved_ids = set()
        
        for segment in segments:
            retrieved_rows = schema_rag.retrieve(segment, top_n=top_n)
            all_retrieved_ids.update(row.id for row in retrieved_rows)
            results["segments_analyzed"] += 1
        
        retrieved_gold = gold_ids & all_retrieved_ids
        missed_gold = gold_ids - all_retrieved_ids
        
        results["total_gold_observations"] += len(gold_ids)
        results["retrieved_gold_observations"] += len(retrieved_gold)
        results["missed_gold_observations"] += len(missed_gold)
        results["transcripts_analyzed"] += 1
        
        transcript_recall = len(retrieved_gold) / len(gold_ids) if gold_ids else 1.0
        results["per_transcript_recall"].append({
            "id": transcript_id,
            "recall": transcript_recall,
            "missed_ids": list(missed_gold)
        })
        
        for schema_id in missed_gold:
            results["missed_by_schema_id"][schema_id] += 1
    
    recall = results["retrieved_gold_observations"] / results["total_gold_observations"]
    
    missed_sorted = sorted(results["missed_by_schema_id"].items(), key=lambda x: -x[1])
    
    if verbose:
        print("\n" + "=" * 70)
        print("SCHEMA RAG RETRIEVAL ANALYSIS")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Embedding model: {schema_rag.embedding_model_name}")
        print(f"  Top-N retrieved: {top_n}")
        print(f"  Transcripts: {results['transcripts_analyzed']}")
        
        print(f"\n" + "-" * 70)
        print("RETRIEVAL COVERAGE")
        print("-" * 70)
        print(f"  Total gold observations: {results['total_gold_observations']}")
        print(f"  Retrieved: {results['retrieved_gold_observations']}")
        print(f"  Missed: {results['missed_gold_observations']}")
        print(f"\n  >>> RECALL@{top_n}: {recall:.1%} <<<")
        
        if recall < 1.0:
            print(f"\n  {results['missed_gold_observations']} observations can NEVER be extracted")
            print(f"  because their schema items weren't in top-{top_n}.")
        
        if missed_sorted:
            print(f"\n" + "-" * 70)
            print("MOST FREQUENTLY MISSED SCHEMA ITEMS")
            print("-" * 70)
            print(f"  {'ID':<6} {'Count':<8} Name")
            print(f"  {'-'*6} {'-'*8} {'-'*40}")
            for schema_id, count in missed_sorted[:15]:
                name = id_to_name.get(schema_id, "Unknown")
                print(f"  {schema_id:<6} {count:<8} {name}")
        
        print("\n" + "=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Schema RAG Retrieval Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", choices=["train", "dev"], default="train",
                        help="Dataset to evaluate on")
    parser.add_argument("--top-n", type=int, default=60,
                        help="Number of schema rows to retrieve")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="Embedding model for RAG")
    parser.add_argument("--full-analysis", action="store_true",
                        help="Compare all embedding models across all top-N values")
    parser.add_argument("--hybrid", action="store_true",
                        help="Evaluate hybrid search (BM25 + dense)")
    parser.add_argument("--compare-top-n", action="store_true",
                        help="Compare recall at different top-N values for one model")
    
    args = parser.parse_args()
    
    # Paths
    schema_path = PROJECT_ROOT / "Data" / "synur_schema.json"
    data_path = PROJECT_ROOT / "Data" / f"{args.data}.jsonl"
    
    print(f"Loading data from {data_path}")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} transcripts")
    
    if args.full_analysis:
        results, best_model, best_top_n = run_full_analysis(data, schema_path)
        
        # Automatically run hybrid analysis on top models
        top_models_for_hybrid = [
            ("BAAI/bge-base-en-v1.5", "BGE-base (local)"),
            ("text-embedding-3-small", "OpenAI small"),
        ]
        print("\n" + "#" * 90)
        print("AUTOMATIC HYBRID ANALYSIS FOR TOP MODELS")
        print("#" * 90)
        
        for model_id, model_name in top_models_for_hybrid:
            print(f"\n>>> {model_name} <<<")
            try:
                run_hybrid_analysis(data, schema_path, model_id, top_n_values=[40, 60, 80])
            except Exception as e:
                print(f"Failed: {e}")
    elif args.hybrid:
        run_hybrid_analysis(data, schema_path, args.embedding_model, top_n_values=[args.top_n])
    elif args.compare_top_n:
        print(f"\nLoading {args.embedding_model}...")
        rag = SchemaRAG(schema_path, args.embedding_model)
        
        print("\n" + "=" * 50)
        print("RECALL vs TOP-N")
        print("=" * 50)
        print(f"\n{'Top-N':<10} {'Recall':>10} {'Missed':>10}")
        print("-" * 30)
        
        for top_n in [10, 20, 40, 60, 80, 120]:
            recall, missed, _ = evaluate_retrieval(data, rag, top_n=top_n)
            print(f"{top_n:<10} {recall:>9.1%} {missed:>10}")
        
        print("=" * 50)
    else:
        print(f"\nLoading {args.embedding_model}...")
        rag = SchemaRAG(schema_path, args.embedding_model)
        run_single_evaluation(data, rag, args.top_n)


if __name__ == "__main__":
    main()
