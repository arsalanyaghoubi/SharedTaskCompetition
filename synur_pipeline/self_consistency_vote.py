import argparse
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_predictions(path):

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_value(value):

    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return tuple(sorted([v.lower() if isinstance(v, str) else v for v in value]))
    return value


def vote_on_transcript(transcript_predictions, threshold, num_runs):

    # Group predictions by ID across all runs
    id_to_values = defaultdict(list)
    id_to_full_obs = {}
    
    for run_idx, preds in enumerate(transcript_predictions):
        for obs in preds:
            obs_id = obs.get('id')
            if obs_id is None:
                continue
            
            value = obs.get('value')
            id_to_values[obs_id].append((value, run_idx))
            
            # Store full observation
            if obs_id not in id_to_full_obs:
                id_to_full_obs[obs_id] = obs
    
    # Vote on which IDs to include
    voted_predictions = []
    
    for obs_id, value_list in id_to_values.items():
        num_votes = len(value_list)
        
        if num_votes >= threshold:
            values = [v for v, _ in value_list]
            
            # Count normalized values
            value_counts = Counter([normalize_value(v) for v in values])
            most_common_normalized = value_counts.most_common(1)[0][0]
            
            # Find original (non-normalized) value that matches
            for v in values:
                if normalize_value(v) == most_common_normalized:
                    voted_value = v
                    break
            
            # Create voted observation
            base_obs = id_to_full_obs[obs_id].copy()
            base_obs['value'] = voted_value
            base_obs['_vote_count'] = num_votes
            base_obs['_vote_threshold'] = threshold
            
            voted_predictions.append(base_obs)
    
    return voted_predictions


def run_voting(run_paths, threshold=None, output_path=None, verbose=True):

    num_runs = len(run_paths)
    
    if threshold is None:
        threshold = (num_runs // 2) + 1  # Majority
    
    if verbose:
        print(f"\n{'='*60}")
        print("SELF-CONSISTENCY VOTING")
        print(f"{'='*60}")
        print(f"Number of runs: {num_runs}")
        print(f"Vote threshold: {threshold}/{num_runs}")
        print(f"Run files:")
        for p in run_paths:
            print(f"  - {p}")
        print()
    
    # Load all runs
    all_runs = []
    for path in run_paths:
        preds = load_predictions(path)
        all_runs.append(preds)
        if verbose:
            print(f"Loaded {len(preds)} transcripts from {Path(path).name}")
    
    # Verify all runs have same number of transcripts
    num_transcripts = len(all_runs[0])
    for i, run in enumerate(all_runs):
        if len(run) != num_transcripts:
            raise ValueError(f"Run {i} has {len(run)} transcripts, expected {num_transcripts}")
    
    # Vote on each transcript
    voted_output = []
    stats = {
        'total_ids_seen': 0,
        'ids_passed_threshold': 0,
        'ids_filtered_out': 0,
    }
    
    for t_idx in range(num_transcripts):
        transcript_id = all_runs[0][t_idx].get('id', str(t_idx))
        
        # Gather predictions from all runs for this transcript
        transcript_preds = [run[t_idx].get('predictions', []) for run in all_runs]
        
        # Get gold labels (should be same across runs)
        gold = all_runs[0][t_idx].get('gold', [])
        
        # Count unique IDs seen across runs (before voting)
        all_ids = set()
        for preds in transcript_preds:
            for obs in preds:
                if obs.get('id'):
                    all_ids.add(obs.get('id'))
        stats['total_ids_seen'] += len(all_ids)
        
        # Vote
        voted_preds = vote_on_transcript(transcript_preds, threshold, num_runs)
        stats['ids_passed_threshold'] += len(voted_preds)
        stats['ids_filtered_out'] += len(all_ids) - len(voted_preds)
        
        voted_output.append({
            'id': transcript_id,
            'predictions': voted_preds,
            'gold': gold
        })
    
    if verbose:
        print(f"\n{'='*60}")
        print("VOTING STATISTICS")
        print(f"{'='*60}")
        print(f"Total unique IDs seen across all runs: {stats['total_ids_seen']}")
        print(f"IDs that passed threshold ({threshold}/{num_runs}): {stats['ids_passed_threshold']}")
        print(f"IDs filtered out: {stats['ids_filtered_out']}")
        
        # Per-run comparison
        print(f"\n{'='*60}")
        print("PER-RUN OBSERVATION COUNTS")
        print(f"{'='*60}")
        for i, run in enumerate(all_runs):
            total_obs = sum(len(t.get('predictions', [])) for t in run)
            print(f"Run {i+1}: {total_obs} observations")
        
        voted_total = sum(len(t['predictions']) for t in voted_output)
        print(f"Voted:  {voted_total} observations")
    
    # Save output
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(voted_output, f, indent=2)
        if verbose:
            print(f"\nSaved voted predictions to: {output_path}")
    
    return voted_output, stats


def create_submission_jsonl(voted_output, output_path, schema_path=None):

    if schema_path:
        from run_pipeline import load_schema_for_validation, validate_observations_with_schema
        schema_by_id = load_schema_for_validation(schema_path)
    else:
        schema_by_id = None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in voted_output:
            preds = item['predictions']
            
            clean_preds = []
            for obs in preds:
                clean_obs = {k: v for k, v in obs.items() if not k.startswith('_')}
                clean_preds.append(clean_obs)
            
            if schema_by_id:
                clean_preds, _ = validate_observations_with_schema(clean_preds, schema_by_id)
            
            record = {
                'id': item['id'],
                'observations': clean_preds
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"Created submission file: {output_path}")


def run_official_eval(pred_jsonl_path, ref_jsonl_path):

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
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return None


def create_reference_jsonl(voted_output, output_path):

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in voted_output:
            gold = item.get('gold', [])
            record = {
                'id': item['id'],
                'observations': gold
            }
            f.write(json.dumps(record) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Self-consistency voting for predictions")
    parser.add_argument('--runs', nargs='+', required=True,
                        help='Paths to prediction JSON files from different runs')
    parser.add_argument('--output', type=str, default='voted_predictions.json',
                        help='Output path for voted predictions')
    parser.add_argument('--submission', type=str, default=None,
                        help='Also create submission JSONL file at this path')
    parser.add_argument('--schema', type=str, default='../Data/synur_schema.json',
                        help='Schema path for validation in submission file')
    parser.add_argument('--threshold', type=int, default=None,
                        help='Minimum votes to include observation (default: majority)')
    parser.add_argument('--eval', action='store_true',
                        help='Run official evaluation after voting')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Run voting
    voted_output, stats = run_voting(
        run_paths=args.runs,
        threshold=args.threshold,
        output_path=args.output,
        verbose=not args.quiet
    )
    
    # Resolve schema path
    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = SCRIPT_DIR / args.schema
    
    # Create submission file
    if args.submission:
        create_submission_jsonl(
            voted_output,
            args.submission,
            str(schema_path) if schema_path.exists() else None
        )
    
    # Run evaluation
    if args.eval:
        has_gold = any(len(item.get('gold', [])) > 0 for item in voted_output)
        
        if not has_gold:
            print("\n[WARNING] No gold labels found in predictions - cannot evaluate")
        else:
            # Create temp files for evaluation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as pred_file, \
                 tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as ref_file:
                
                # Write predictions
                for item in voted_output:
                    preds = item['predictions']
                    clean_preds = [{k: v for k, v in obs.items() if not k.startswith('_')} for obs in preds]
                    
                    # Apply schema validation
                    if schema_path.exists():
                        from run_pipeline import load_schema_for_validation, validate_observations_with_schema
                        schema_by_id = load_schema_for_validation(str(schema_path))
                        clean_preds, _ = validate_observations_with_schema(clean_preds, schema_by_id)
                    
                    pred_file.write(json.dumps({'id': item['id'], 'observations': clean_preds}) + '\n')
                
                # Write reference
                for item in voted_output:
                    ref_file.write(json.dumps({'id': item['id'], 'observations': item.get('gold', [])}) + '\n')
                
                pred_path = pred_file.name
                ref_path = ref_file.name
            
            # Run evaluation
            print(f"\n{'='*60}")
            print("EVALUATION RESULTS")
            print(f"{'='*60}")
            
            results = run_official_eval(pred_path, ref_path)
            
            if results:
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall:    {results['recall']:.4f}")
                print(f"F1 Score:  {results['f1']:.4f}")
            
            Path(pred_path).unlink()
            Path(ref_path).unlink()


if __name__ == '__main__':
    main()
