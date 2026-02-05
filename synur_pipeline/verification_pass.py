import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(PROJECT_ROOT / ".env")


VERIFICATION_SYSTEM_PROMPT = """You are a clinical documentation verification expert.

Your task is to verify whether extracted clinical observations are actually supported by the source nursing transcript.

For each observation, determine:
- KEEP: The observation is directly stated or clearly implied in the transcript
- REMOVE: The observation is NOT supported, was incorrectly inferred, or fabricated

Be conservative - only mark REMOVE if you are confident the observation is not supported.
Small variations in wording are acceptable if the meaning matches."""


VERIFICATION_USER_PROMPT = """TRANSCRIPT:
{transcript}

EXTRACTED OBSERVATIONS TO VERIFY:
{observations}

For each observation ID, respond with either "KEEP" or "REMOVE".
Return a JSON object mapping observation IDs to decisions.

Example response format:
{{"40": "KEEP", "56": "REMOVE", "89": "KEEP"}}

Your verification:"""


_local_model = None
_local_tokenizer = None
_openai_client = None
_model_type = None  # 'local' or 'openai'


def init_local_model(model_path, load_in_4bit=False, load_in_8bit=False):

    global _local_model, _local_tokenizer, _model_type
    
    _model_type = 'local'
    print(f"Loading verification model from {model_path}")
    
    _local_tokenizer = AutoTokenizer.from_pretrained(model_path)
    if _local_tokenizer.pad_token is None:
        _local_tokenizer.pad_token = _local_tokenizer.eos_token
    
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    dtype = torch.bfloat16 if (load_in_4bit or load_in_8bit) else torch.float16
    
    _local_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    
    print(f"Verification model loaded (device_map=auto)")


def init_openai_model():

    global _openai_client, _model_type
    
    _model_type = 'openai'
    from openai import OpenAI
    _openai_client = OpenAI()


def load_predictions(path):

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(path):

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def call_verifier_local(transcript, observations):

    global _local_model, _local_tokenizer
    
    # Format observations
    obs_formatted = []
    for obs in observations:
        obs_id = obs.get('id', 'unknown')
        name = obs.get('name', '')
        value = obs.get('value', '')
        obs_formatted.append(f"- ID {obs_id}: {name} = {value}")
    
    obs_text = '\n'.join(obs_formatted)
    
    user_prompt = VERIFICATION_USER_PROMPT.format(
        transcript=transcript,
        observations=obs_text
    )
    
    messages = [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = _local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _local_tokenizer(prompt, return_tensors="pt").to(_local_model.device)
    
    with torch.no_grad():
        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=_local_tokenizer.pad_token_id
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = _local_tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Parse JSON from response
    try:
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            decisions = json.loads(json_match.group())
        else:
            decisions = json.loads(response)
        return decisions
    except json.JSONDecodeError:
        decisions = {}
        for obs in observations:
            obs_id = str(obs.get('id', ''))
            if re.search(rf'(?:ID\s*)?{obs_id}["\s:]*REMOVE', response, re.IGNORECASE):
                decisions[obs_id] = 'REMOVE'
            else:
                decisions[obs_id] = 'KEEP'
        return decisions


def call_verifier_openai(client, model, transcript, observations, max_retries=3):

    # Format observations
    obs_formatted = []
    for obs in observations:
        obs_id = obs.get('id', 'unknown')
        name = obs.get('name', '')
        value = obs.get('value', '')
        obs_formatted.append(f"- ID {obs_id}: {name} = {value}")
    
    obs_text = '\n'.join(obs_formatted)
    
    user_prompt = VERIFICATION_USER_PROMPT.format(
        transcript=transcript,
        observations=obs_text
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            decisions = json.loads(result)
            return decisions
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Verifier error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"Verifier failed after {max_retries} attempts: {e}")
                return {obs.get('id', ''): 'KEEP' for obs in observations}


def call_verifier(client, model, transcript, observations, max_retries=3):

    if _model_type == 'local':
        return call_verifier_local(transcript, observations)
    else:
        return call_verifier_openai(client, model, transcript, observations, max_retries)


def verify_transcript(client, model, transcript, predictions, verbose=False):

    if not predictions:
        return [], {'total': 0, 'kept': 0, 'removed': 0}
    
    decisions = call_verifier(client, model, transcript, predictions)
    
    verified = []
    stats = {'total': len(predictions), 'kept': 0, 'removed': 0}
    
    for obs in predictions:
        obs_id = str(obs.get('id', ''))
        decision = decisions.get(obs_id, 'KEEP')
        
        if decision.upper() == 'KEEP':
            verified.append(obs)
            stats['kept'] += 1
        else:
            stats['removed'] += 1
            if verbose:
                print(f"REMOVED: ID {obs_id} - {obs.get('name', '')} = {obs.get('value', '')}")
    
    return verified, stats


def create_submission_jsonl(verified_output, output_path, schema_path=None):

    if schema_path:
        from run_pipeline import load_schema_for_validation, validate_observations_with_schema
        schema_by_id = load_schema_for_validation(schema_path)
    else:
        schema_by_id = None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in verified_output:
            preds = item['predictions']
            
            clean_preds = [{k: v for k, v in obs.items() if not k.startswith('_')} for obs in preds]
            
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


def verify_segment_level(client, model, segments, predictions, verbose=False):

    if not predictions:
        return [], {'total': 0, 'kept': 0, 'removed': 0}
    
    # Group predictions by segment
    preds_by_segment = defaultdict(list)
    for obs in predictions:
        seg_idx = obs.get('_segment_idx', 0) 
        preds_by_segment[seg_idx].append(obs)
    
    verified = []
    stats = {'total': len(predictions), 'kept': 0, 'removed': 0}
    
    for seg_idx, seg_preds in preds_by_segment.items():
        if seg_idx < len(segments):
            segment_text = segments[seg_idx]
        else:
            segment_text = ' '.join(segments)
        
        decisions = call_verifier(client, model, segment_text, seg_preds)
        
        for obs in seg_preds:
            obs_id = str(obs.get('id', ''))
            decision = decisions.get(obs_id, 'KEEP')
            
            if decision.upper() == 'KEEP':
                verified.append(obs)
                stats['kept'] += 1
            else:
                stats['removed'] += 1
                if verbose:
                    print(f"REMOVED (seg {seg_idx}): ID {obs_id} - {obs.get('name', '')} = {obs.get('value', '')}")
    
    return verified, stats


def run_verification(predictions_path, data_path, output_path, verifier_model='gpt-4o-mini', segment_level=False, verbose=True):

    if verbose:
        print(f"\n{'='*60}")
        print("VERIFICATION PASS")
        print(f"{'='*60}")
        print(f"Predictions: {predictions_path}")
        print(f"Data: {data_path}")
        print(f"Verifier model: {verifier_model}")
        print(f"Mode: {'Segment-level' if segment_level else 'Transcript-level'}")
        print(f"Backend: {_model_type}")
        print()
    
    predictions = load_predictions(predictions_path)
    data = load_jsonl(data_path)
    
    transcript_lookup = {item.get('id', str(i)): item.get('transcript', '') for i, item in enumerate(data)}
    
    client = _openai_client
    
    verified_output = []
    total_stats = {'total': 0, 'kept': 0, 'removed': 0}
    
    for item in tqdm(predictions, desc="Verifying", disable=not verbose):
        transcript_id = item.get('id', 'unknown')
        preds = item.get('predictions', [])
        gold = item.get('gold', [])
        segments = item.get('segments', []) 
        
        transcript = transcript_lookup.get(transcript_id, '')
        if not transcript and not segments:
            print(f"  [WARNING] No transcript found for ID {transcript_id}")
            verified_output.append(item)
            continue
        
        if segment_level and segments:
            verified_preds, stats = verify_segment_level(
                client, verifier_model, segments, preds,
                verbose=(verbose and len(preds) > 0)
            )
        else:
            verified_preds, stats = verify_transcript(
                client, verifier_model, transcript, preds, 
                verbose=(verbose and len(preds) > 0)
            )
        
        for key in ['total', 'kept', 'removed']:
            total_stats[key] += stats[key]
        
        verified_output.append({
            'id': transcript_id,
            'predictions': verified_preds,
            'gold': gold,
            'segments': segments
        })
    
    if verbose:
        print(f"\n{'='*60}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total observations: {total_stats['total']}")
        print(f"Kept: {total_stats['kept']} ({100*total_stats['kept']/max(1,total_stats['total']):.1f}%)")
        print(f"Removed: {total_stats['removed']} ({100*total_stats['removed']/max(1,total_stats['total']):.1f}%)")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(verified_output, f, indent=2)
    
    if verbose:
        print(f"\nSaved verified predictions to: {output_path}")
    
    return verified_output, total_stats


def main():
    parser = argparse.ArgumentParser(description="Verification pass for predictions")
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--data', type=str, default='../Data/dev.jsonl',
                        help='Path to data JSONL file (for transcripts)')
    parser.add_argument('--output', type=str, default='verified_predictions.json',
                        help='Output path for verified predictions')
    parser.add_argument('--submission', type=str, default=None,
                        help='Also create submission JSONL file at this path')
    parser.add_argument('--schema', type=str, default='../Data/synur_schema.json',
                        help='Schema path for validation in submission file')
    parser.add_argument('--verifier', type=str, default='gpt-4o-mini',
                        help='Model to use for verification (default: gpt-4o-mini). For local models, this is ignored.')
    parser.add_argument('--segment-level', action='store_true',
                        help='Verify at segment level instead of transcript level')
    parser.add_argument('--eval', action='store_true',
                        help='Run official evaluation after verification')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to local model for verification (uses local model instead of OpenAI)')
    parser.add_argument('--load-in-4bit', action='store_true',
                        help='Load local model in 4-bit quantization')
    parser.add_argument('--load-in-8bit', action='store_true',
                        help='Load local model in 8-bit quantization')
    
    args = parser.parse_args()
    
    if args.model_path:
        print(f"Initializing local model: {args.model_path}")
        init_local_model(args.model_path, load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit)
        verifier_name = args.model_path.split('/')[-1] 
    else:
        print(f"Initializing OpenAI client for model: {args.verifier}")
        init_openai_model()
        verifier_name = args.verifier
    
    predictions_path = Path(args.predictions)
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = SCRIPT_DIR / args.data
    
    # Run verification
    verified_output, stats = run_verification(
        predictions_path=str(predictions_path),
        data_path=str(data_path),
        output_path=args.output,
        verifier_model=verifier_name,
        segment_level=args.segment_level,
        verbose=not args.quiet
    )
    
    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = SCRIPT_DIR / args.schema
    
    if args.submission:
        create_submission_jsonl(
            verified_output,
            args.submission,
            str(schema_path) if schema_path.exists() else None
        )
    
    # Run evaluation
    if args.eval:
        has_gold = any(len(item.get('gold', [])) > 0 for item in verified_output)
        
        if not has_gold:
            print("\nNo gold labels found in predictions. Cannot evaluate")
        else:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as pred_file, \
                 tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as ref_file:
                for item in verified_output:
                    preds = item['predictions']
                    clean_preds = [{k: v for k, v in obs.items() if not k.startswith('_')} for obs in preds]
                    
                    if schema_path.exists():
                        from run_pipeline import load_schema_for_validation, validate_observations_with_schema
                        schema_by_id = load_schema_for_validation(str(schema_path))
                        clean_preds, _ = validate_observations_with_schema(clean_preds, schema_by_id)
                    
                    pred_file.write(json.dumps({'id': item['id'], 'observations': clean_preds}) + '\n')
                
                for item in verified_output:
                    ref_file.write(json.dumps({'id': item['id'], 'observations': item.get('gold', [])}) + '\n')
                
                pred_path = pred_file.name
                ref_path = ref_file.name
            
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
