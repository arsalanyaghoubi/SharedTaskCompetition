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
    
    # Parallel execution on multiple GPUs (for local models)
    python run_pipeline.py --data dev --model llama-3-8b --model-path /path/to/model --num-gpus 6 --gpus 2,3,4,5,6,7
"""

import argparse
import json
import os
import sys
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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
_max_tokens = 8192
_is_peft_model = False

_seg_model = None
_seg_tokenizer = None
_seg_model_path = None
_seg_openai_model = None
_seg_openai_client = None


def init_llm(model_name, model_path=None, max_tokens=4096, lora_path=None, load_in_4bit=False, load_in_8bit=False, force_single_gpu=False, base_model_path=None):


    global _openai_client, _local_model, _local_tokenizer, _model_name, _model_path, _max_tokens, _is_peft_model
    
    _model_name = model_name
    _model_path = model_path
    _max_tokens = max_tokens
    _is_peft_model = False
    
    if model_path:
        print(f"Loading model from {model_path}")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token
        
        # Determine device mapping:
        # - force_single_gpu=True: Use cuda:0 (for data parallelism workers)
        # - Otherwise: Use "auto" (allows model parallelism for large models)
        device_map = "cuda:0" if force_single_gpu else "auto"
        
        # Setup quantization config if requested
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print(f"  Using 4-bit quantization (NF4)")
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print(f"  Using 8-bit quantization")
        
        # Determine dtype (bfloat16 for quantized, float16 otherwise)
        dtype = torch.bfloat16 if (load_in_4bit or load_in_8bit) else (torch.float16 if torch.cuda.is_available() else torch.float32)
        
        # Check if this is a PEFT/LoRA model
        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.exists() or lora_path:
            # Load as PEFT model
            from peft import PeftModel, PeftConfig
            
            if lora_path:
                # Separate base model and LoRA adapter
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=dtype, 
                    device_map=device_map,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                )
                _local_model = PeftModel.from_pretrained(base_model, lora_path)
                print(f"Loaded LoRA adapter from {lora_path}")
            else:
                # LoRA adapter is in model_path, need base model path from config
                peft_config = PeftConfig.from_pretrained(model_path)
                # Use override if provided, otherwise use config's base model path
                actual_base_path = base_model_path or peft_config.base_model_name_or_path
                base_model = AutoModelForCausalLM.from_pretrained(
                    actual_base_path, 
                    torch_dtype=dtype, 
                    device_map=device_map,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                )
                _local_model = PeftModel.from_pretrained(base_model, model_path)
                print(f"Loaded PEFT model with base: {actual_base_path}")
            
            _is_peft_model = True
        else:
            _local_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=dtype, 
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
        
        print(f"Model loaded (device_map={device_map})")
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
        # OpenAI with retry logic for rate limits
        import time
        kwargs = {"model": _model_name, "messages": messages, "temperature": temperature, "max_tokens": _max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = _openai_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    print(f"\n  Rate limit hit, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        
        # Final attempt
        response = _openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


def init_llm_segmentation(model_path, load_in_4bit=False, load_in_8bit=False):

    global _seg_model, _seg_tokenizer, _seg_model_path
    
    _seg_model_path = model_path
    print(f"Loading segmentation model from {model_path}")
    
    _seg_tokenizer = AutoTokenizer.from_pretrained(model_path)
    if _seg_tokenizer.pad_token is None:
        _seg_tokenizer.pad_token = _seg_tokenizer.eos_token
    
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    dtype = torch.bfloat16 if (load_in_4bit or load_in_8bit) else torch.float16
    
    _seg_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )


def init_llm_segmentation_openai(model_name):

    global _seg_openai_model, _seg_openai_client
    
    load_dotenv(PROJECT_ROOT / ".env")
    
    from openai import OpenAI
    _seg_openai_model = model_name
    _seg_openai_client = OpenAI()
    print(f"Segmentation using OpenAI: {model_name}")


def call_llm_segmentation(messages, temperature=0.0, json_mode=False):

    if _seg_openai_client is not None:
        import time
        kwargs = {
            "model": _seg_openai_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = _seg_openai_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = 2 ** attempt
                    print(f"\nRate limit hit, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise e
        response = _seg_openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    if _seg_model is None:
        return call_llm(messages, temperature, json_mode)
    
    prompt = _seg_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if json_mode:
        prompt += "\nRespond with valid JSON only:\n"
    
    inputs = _seg_tokenizer(prompt, return_tensors="pt").to(_seg_model.device)
    
    outputs = _seg_model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=_seg_tokenizer.pad_token_id
    )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return _seg_tokenizer.decode(new_tokens, skip_special_tokens=True)


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


def run_pipeline(config, use_llm_segmentation=False, verbose=True, limit=None, shard=None):

    # Load .env from project root
    load_dotenv(PROJECT_ROOT / ".env")
    
    # Initialize LLM (OpenAI or local model)
    if verbose:
        print(f"Initializing LLM")
    
    init_llm(
        config.llm_model, 
        config.llm_model_path, 
        config.max_tokens, 
        config.lora_path,
        load_in_4bit=getattr(config, 'load_in_4bit', False),
        load_in_8bit=getattr(config, 'load_in_8bit', False),
        force_single_gpu=False,  # Use device_map="auto" for model parallelism
        base_model_path=getattr(config, 'base_model_path', None)
    )
    
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
    
    # Apply sharding if specified
    if shard:
        shard_num, total_shards = map(int, shard.split('/'))
        shard_size = len(eval_data) // total_shards
        start_idx = (shard_num - 1) * shard_size
        end_idx = start_idx + shard_size if shard_num < total_shards else len(eval_data)
        eval_data = eval_data[start_idx:end_idx]
        if verbose:
            print(f"Shard {shard_num}/{total_shards}: processing transcripts {start_idx}-{end_idx-1} ({len(eval_data)} transcripts)")
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
    all_segments = []  # Track segments for each transcript
    
    if verbose:
        print(f"\nProcessing transcripts with {config.llm_model}.")
        print(f"Prompt mode: {'Paper (Appendix A.1)' if config.use_paper_prompts else 'Enhanced'}.")
    
    for item in tqdm(eval_data, disable=not verbose):
        transcript_id = item.get("id", "unknown")
        transcript = item["transcript"]
        gold_obs = parse_observations(item.get("observations", [])) if "observations" in item else []
        
        # Get few-shot examples via RAG (most similar training transcripts)
        few_shot_examples = None
        if few_shot_rag:
            few_shot_examples = few_shot_rag.retrieve(transcript, top_n=config.num_few_shot_examples)
        
        # Step 1: Segment the transcript
        if use_llm_segmentation:
            segments = segment_transcript(
                transcript,
                call_llm_segmentation,
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
        all_segments.append(segments)
    
    
    return all_predictions, all_gold, all_segments


def _run_shard_worker(args_tuple):

    (shard_num, total_shards, gpu_id, config_dict, use_llm_segmentation, 
     eval_data_slice, eval_data_indices, temp_dir) = args_tuple
    
    # Set GPU visibility for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Reconstruct config
    config = PipelineConfig(**config_dict)
    
    # Load .env from project root
    load_dotenv(PROJECT_ROOT / ".env")
    
    # Initialize LLM for this worker
    init_llm(
        config.llm_model, 
        config.llm_model_path, 
        config.max_tokens, 
        config.lora_path,
        load_in_4bit=getattr(config, 'load_in_4bit', False),
        load_in_8bit=getattr(config, 'load_in_8bit', False),
        force_single_gpu=True,
        base_model_path=getattr(config, 'base_model_path', None)
    )
    
    # Initialize schema RAG
    if getattr(config, 'use_hybrid_search', False):
        alpha = getattr(config, 'hybrid_alpha', 0.6)
        schema_rag = HybridSchemaRAG(config.schema_path, config.embedding_model, alpha=alpha)
    else:
        schema_rag = SchemaRAG(config.schema_path, config.embedding_model)
    
    # Load few-shot RAG if needed
    few_shot_rag = None
    if config.use_few_shot:
        train_data = load_jsonl(config.train_path)
        few_shot_rag = FewShotRAG(train_data, config.embedding_model)
    
    # Process this shard's data
    shard_predictions = []
    shard_gold = []
    shard_ids = []
    shard_segments = []  # Track segments
    
    for item in tqdm(eval_data_slice, desc=f"Shard {shard_num}/{total_shards}", position=shard_num-1):
        transcript_id = item.get("id", "unknown")
        transcript = item["transcript"]
        gold_obs = parse_observations(item.get("observations", [])) if "observations" in item else []
        
        # Get few-shot examples if configured
        few_shot_examples = None
        if few_shot_rag:
            few_shot_examples = few_shot_rag.retrieve(transcript, top_n=config.num_few_shot_examples)
        
        # Segment
        if use_llm_segmentation:
            segments = segment_transcript(transcript, call_llm_segmentation, use_paper_prompt=config.use_paper_prompts)
        else:
            segments = segment_transcript_simple(transcript)
        
        # Extract
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
        
        shard_predictions.append(predictions)
        shard_gold.append(gold_obs)
        shard_ids.append(transcript_id)
        shard_segments.append(segments)
    
    # Save shard results to temp file
    shard_file = Path(temp_dir) / f"shard_{shard_num}.json"
    with open(shard_file, "w") as f:
        json.dump({
            "shard_num": shard_num,
            "indices": eval_data_indices,
            "ids": shard_ids,
            "predictions": shard_predictions,
            "gold": shard_gold,
            "segments": shard_segments
        }, f)
    
    return shard_num, len(shard_predictions)


def run_pipeline_parallel(config, gpus, use_llm_segmentation=False, verbose=True, limit=None):
    """Run pipeline in parallel across multiple GPUs."""
    
    num_gpus = len(gpus)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PARALLEL EXECUTION: {num_gpus} GPUs")
        print(f"GPUs: {gpus}")
        print(f"{'='*60}\n")
    
    # Load evaluation data
    eval_data = load_jsonl(config.dev_path)
    if limit is not None and limit > 0:
        eval_data = eval_data[:limit]
    
    if verbose:
        print(f"Total transcripts: {len(eval_data)}")
        print(f"Transcripts per GPU: ~{len(eval_data) // num_gpus}")
    
    # Create temp directory for shard results
    temp_dir = tempfile.mkdtemp(prefix="synur_shards_")
    
    # Prepare config dict for serialization (can't pickle PipelineConfig directly across processes)
    config_dict = {
        "schema_path": str(config.schema_path),
        "train_path": str(config.train_path),
        "dev_path": str(config.dev_path),
        "llm_model": config.llm_model,
        "llm_model_path": config.llm_model_path,
        "lora_path": config.lora_path,
        "embedding_model": config.embedding_model,
        "use_few_shot": config.use_few_shot,
        "num_few_shot_examples": config.num_few_shot_examples,
        "top_n_schema_rows": config.top_n_schema_rows,
        "use_paper_prompts": config.use_paper_prompts,
        "output_dir": str(config.output_dir),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    # Add hybrid search settings
    config_dict["use_hybrid_search"] = getattr(config, 'use_hybrid_search', False)
    config_dict["hybrid_alpha"] = getattr(config, 'hybrid_alpha', 0.6)
    
    # Add quantization settings
    config_dict["load_in_4bit"] = getattr(config, 'load_in_4bit', False)
    config_dict["load_in_8bit"] = getattr(config, 'load_in_8bit', False)
    
    # Add base model path override
    config_dict["base_model_path"] = getattr(config, 'base_model_path', None)
    
    # Split data into shards
    shard_size = len(eval_data) // num_gpus
    worker_args = []
    
    for i, gpu_id in enumerate(gpus):
        shard_num = i + 1
        start_idx = i * shard_size
        end_idx = start_idx + shard_size if shard_num < num_gpus else len(eval_data)
        
        eval_data_slice = eval_data[start_idx:end_idx]
        eval_data_indices = list(range(start_idx, end_idx))
        
        worker_args.append((
            shard_num, num_gpus, gpu_id, config_dict, use_llm_segmentation,
            eval_data_slice, eval_data_indices, temp_dir
        ))
    
    if verbose:
        print(f"\nLaunching {num_gpus} workers...")
    
    # Run workers in parallel using spawn to avoid CUDA issues
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        futures = [executor.submit(_run_shard_worker, args) for args in worker_args]
        
        # Wait for all to complete
        for future in as_completed(futures):
            try:
                shard_num, count = future.result()
                if verbose:
                    print(f"  Shard {shard_num} completed: {count} transcripts")
            except Exception as e:
                print(f"  ERROR in shard: {e}")
                raise
    
    if verbose:
        print(f"\nAll shards completed. Merging results...")
    
    # Merge results from all shards
    all_results = []
    for i in range(1, num_gpus + 1):
        shard_file = Path(temp_dir) / f"shard_{i}.json"
        with open(shard_file, "r") as f:
            all_results.append(json.load(f))
    
    # Sort by original indices and combine
    all_results.sort(key=lambda x: x["indices"][0] if x["indices"] else 0)
    
    all_predictions = []
    all_gold = []
    all_segments = []
    for result in all_results:
        all_predictions.extend(result["predictions"])
        all_gold.extend(result["gold"])
        all_segments.extend(result.get("segments", [[] for _ in result["predictions"]]))
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    if verbose:
        print(f"Merged {len(all_predictions)} predictions from {num_gpus} shards")
    
    return all_predictions, all_gold, all_segments


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


def load_schema_for_validation(schema_path):

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    schema_by_id = {}
    for row in schema:
        schema_by_id[row["id"]] = {
            "name": row.get("name", ""),
            "value_type": row.get("value_type", "STRING"),
            "value_enum": row.get("value_enum", []),
            "enum_lower_map": {v.lower(): v for v in row.get("value_enum", [])}
        }
    return schema_by_id


def validate_observations_with_schema(observations, schema_by_id):

    valid = []
    stats = {
        "total": 0,
        "structural_invalid": 0,
        "invalid_id": 0,
        "case_fixed": 0,
        "invalid_enum_removed": 0,
    }
    
    for obs in observations:
        stats["total"] += 1
        
        # Basic structural validation
        if not isinstance(obs, dict):
            stats["structural_invalid"] += 1
            continue
        if 'id' not in obs:
            stats["structural_invalid"] += 1
            continue
        if 'value' not in obs or obs['value'] is None:
            stats["structural_invalid"] += 1
            continue
        
        obs_id = obs['id']
        obs_value = obs['value']
        
        # Check if ID exists in schema
        if obs_id not in schema_by_id:
            stats["invalid_id"] += 1
            continue
        
        schema_row = schema_by_id[obs_id]
        value_type = schema_row["value_type"]
        value_enum = schema_row["value_enum"]
        enum_lower_map = schema_row["enum_lower_map"]
        
        # For types with enums (SINGLE_SELECT, MULTI_SELECT)
        if value_type == "SINGLE_SELECT" and value_enum:
            if obs_value in value_enum:
                valid.append(obs)
            elif isinstance(obs_value, str) and obs_value.lower() in enum_lower_map:
                fixed_value = enum_lower_map[obs_value.lower()]
                fixed_obs = obs.copy()
                fixed_obs['value'] = fixed_value
                valid.append(fixed_obs)
                stats["case_fixed"] += 1
            else:
                stats["invalid_enum_removed"] += 1
        
        elif value_type == "MULTI_SELECT" and value_enum:
            if isinstance(obs_value, list):
                valid_values = []
                for v in obs_value:
                    if v in value_enum:
                        valid_values.append(v)
                    elif isinstance(v, str) and v.lower() in enum_lower_map:
                        valid_values.append(enum_lower_map[v.lower()])
                        stats["case_fixed"] += 1
                
                if valid_values:
                    fixed_obs = obs.copy()
                    fixed_obs['value'] = valid_values
                    valid.append(fixed_obs)
                else:
                    stats["invalid_enum_removed"] += 1
            else:
                if obs_value in value_enum:
                    fixed_obs = obs.copy()
                    fixed_obs['value'] = [obs_value]
                    valid.append(fixed_obs)
                elif isinstance(obs_value, str) and obs_value.lower() in enum_lower_map:
                    fixed_obs = obs.copy()
                    fixed_obs['value'] = [enum_lower_map[obs_value.lower()]]
                    valid.append(fixed_obs)
                    stats["case_fixed"] += 1
                else:
                    stats["invalid_enum_removed"] += 1
        
        else:
            valid.append(obs)
    
    return valid, stats


def save_results(config, all_predictions, all_gold, all_segments, eval_data, use_llm_segmentation=False, limit=None, data_split="train"):
    
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
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine subfolder based on temperature and segmentation
    if use_llm_segmentation:
        subfolder = "llm_segmentation"
    elif config.temperature > 0:
        subfolder = "self_consistency"
    else:
        subfolder = "baseline"
    
    # Create organized output directory with run-specific subfolder
    run_dir = config.output_dir / model_name / config_folder / subfolder / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load schema for validation
    schema_by_id = load_schema_for_validation(config.schema_path)
    
    # Save predictions (for error analysis) - raw predictions before validation
    predictions_path = run_dir / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        output = []
        for i, (preds, gold) in enumerate(zip(all_predictions, all_gold)):
            # Get segments for this transcript (if available)
            segments = all_segments[i] if all_segments and i < len(all_segments) else []
            output.append({
                "id": eval_data[i].get("id", str(i)),
                "predictions": preds,
                "gold": gold,
                "segments": segments  # Include segments for verification pass
            })
        json.dump(output, f, indent=2)
    
    # Save predictions in official JSONL format
    official_pred_path = run_dir / "submission.jsonl"
    
    # Aggregate validation stats across all predictions
    total_stats = {
        "total": 0,
        "structural_invalid": 0,
        "invalid_id": 0,
        "case_fixed": 0,
        "invalid_enum_removed": 0,
    }
    
    with open(official_pred_path, "w", encoding="utf-8") as f:
        for i, preds in enumerate(all_predictions):
            # Validate observations with schema
            validated_preds, stats = validate_observations_with_schema(preds, schema_by_id)
            
            # Aggregate stats
            for key in ["total", "structural_invalid", "invalid_id", "case_fixed", "invalid_enum_removed"]:
                total_stats[key] += stats[key]
            
            record = {
                "id": eval_data[i].get("id", str(i)),
                "observations": validated_preds
            }
            f.write(json.dumps(record) + "\n")
    
    # Report validation results
    print(f"\n{'='*60}")
    print("SCHEMA VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total observations processed: {total_stats['total']}")
    print(f"- Structural invalid (missing id/value): {total_stats['structural_invalid']}")
    print(f"- Invalid schema ID (hallucinated): {total_stats['invalid_id']}")
    print(f"- Case mismatches fixed: {total_stats['case_fixed']}")
    print(f"- Invalid enum values removed: {total_stats['invalid_enum_removed']}")
    
    valid_count = total_stats['total'] - total_stats['structural_invalid'] - total_stats['invalid_id'] - total_stats['invalid_enum_removed']
    print(f"Valid observations retained: {valid_count}")
    
    has_gold_data = any(len(gold) > 0 for gold in all_gold)
    official_result = None
    
    if has_gold_data:
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
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            # Run metadata
            "timestamp": timestamp,
            "run_folder": str(run_dir.relative_to(config.output_dir)),
            
            # Model settings
            "llm_model": config.llm_model,
            "llm_model_path": config.llm_model_path,
            "lora_path": config.lora_path,
            "model_display_name": model_name,
            "embedding_model": config.embedding_model,
            "temperature": config.temperature,
            "quantization": "4-bit" if getattr(config, 'load_in_4bit', False) else ("8-bit" if getattr(config, 'load_in_8bit', False) else None),
            
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
            
            # Schema validation stats
            "total_observations_raw": total_stats['total'],
            "structural_invalid": total_stats['structural_invalid'],
            "invalid_schema_id": total_stats['invalid_id'],
            "case_mismatches_fixed": total_stats['case_fixed'],
            "invalid_enum_removed": total_stats['invalid_enum_removed'],
            "valid_observations": valid_count,
            
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
    print(f"  predictions.json   (for error analysis)")
    print(f"  submission.jsonl   (official format)")
    print(f"  summary.json       (config + metrics)")


def main():
    parser = argparse.ArgumentParser(description="SYNUR Extraction Pipeline")
    parser.add_argument("--data", choices=["train", "dev", "test"], default="train",
                        help="Dataset to evaluate on")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Model name for display (e.g., gpt-4o-mini, llama-3.2-3b)")
    parser.add_argument("--model-path", default=None,
                        help="Path to local model folder. If provided, uses local model instead of OpenAI.")
    parser.add_argument("--lora-path", default=None,
                        help="Path to LoRA adapter folder (optional, for SFT models)")
    parser.add_argument("--base-model-path", default=None,
                        help="Override base model path for PEFT/LoRA models (use when adapter_config.json has wrong path)")
    parser.add_argument("--segmentation-model-path", default=None,
                        help="Path to separate model for LLM segmentation")
    parser.add_argument("--segmentation-model", default=None,
                        help="OpenAI model for segmentation")
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
    parser.add_argument("--shard", type=str, default=None,
                        help="Process a shard of data, format: 'N/M' (shard N of M total). E.g., '1/4' processes first quarter")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid search (BM25 + dense embeddings)")
    parser.add_argument("--hybrid-alpha", type=float, default=0.6,
                        help="Alpha for hybrid search (1.0=dense only, 0.0=BM25 only, default=0.6)")
    
    # Quantization options
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (uses bitsandbytes NF4). Reduces memory ~4x.")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization. Reduces memory ~2x.")
    
    # Parallel execution options
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs for parallel execution (enables data parallelism)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g., '2,3,4,5,6,7'). If not specified, uses GPUs 0 to num-gpus-1")
    
    # Self-consistency options
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for LLM sampling (0.0=deterministic, >0 for diversity). Use with --seed for self-consistency.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility when using temperature>0")
    
    args = parser.parse_args()
    
    # Determine few-shot setting (default to zero-shot)
    use_few_shot = args.few_shot
    
    # Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "outputs"
    
    # Create config with absolute paths (works from any directory)
    if args.data == "test":
        data_file = "SYNUR_testset_input.jsonl"
    else:
        data_file = f"{args.data}.jsonl"
    
    config = PipelineConfig(
        schema_path=PROJECT_ROOT / "Data/synur_schema.json",
        train_path=PROJECT_ROOT / "Data/train.jsonl",
        dev_path=PROJECT_ROOT / "Data" / data_file,
        llm_model=args.model,
        llm_model_path=args.model_path,
        lora_path=args.lora_path,
        base_model_path=args.base_model_path,
        embedding_model=args.embedding_model,
        use_few_shot=use_few_shot,
        num_few_shot_examples=args.num_examples,
        top_n_schema_rows=args.top_n,
        use_paper_prompts=not args.enhanced_prompts,  # Default to paper prompts
        output_dir=output_dir,
        temperature=args.temperature
    )
    
    # Add hybrid search settings
    config.use_hybrid_search = args.hybrid
    config.hybrid_alpha = args.hybrid_alpha
    
    # Add quantization settings
    config.load_in_4bit = args.load_in_4bit
    config.load_in_8bit = args.load_in_8bit
    
    # Add segmentation model path
    config.segmentation_model_path = args.segmentation_model_path
    
    # Determine display name for model
    model_display = config.get_model_display_name()
    
    print("=" * 60)
    print("SYNUR Extraction Pipeline")
    print(f"Model: {model_display}")
    if config.llm_model_path:
        print(f"Model path: {config.llm_model_path}")
    if config.load_in_4bit:
        print(f"Quantization: 4-bit (NF4)")
    elif config.load_in_8bit:
        print(f"Quantization: 8-bit")
    print(f"Embedding: {config.embedding_model}")
    print(f"Mode: {'Few-shot' if config.use_few_shot else 'Zero-shot'}")
    if config.use_few_shot:
        print(f"Examples: {config.num_few_shot_examples}")
    print(f"Prompts: {'Paper (Appendix A.1)' if config.use_paper_prompts else 'Enhanced'}")
    print(f"Top-N schema rows: {config.top_n_schema_rows}")
    print(f"Retrieval: {'Hybrid (BM25 + dense, alpha=' + str(config.hybrid_alpha) + ')' if config.use_hybrid_search else 'Dense only'}")
    print(f"Data: {args.data}")
    print(f"Temperature: {config.temperature}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Segmentation: {'LLM' if args.llm_segmentation else 'Simple rules'}")
    if args.llm_segmentation and args.segmentation_model_path:
        print(f"Segmentation model (local): {args.segmentation_model_path}")
    if args.llm_segmentation and args.segmentation_model:
        print(f"Segmentation model (OpenAI): {args.segmentation_model}")
    if args.limit:
        print(f"Limit: {args.limit} transcripts")
    if args.shard:
        print(f"Shard: {args.shard}")
    if args.num_gpus:
        print(f"Parallel: {args.num_gpus} GPUs")
    print("=" * 60)
    
    # Set random seed if specified (for reproducible sampling with temp>0)
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"\nRandom seed set to {args.seed}")
    
    if args.llm_segmentation and args.segmentation_model:
        print(f"\nUsing OpenAI for segmentation: {args.segmentation_model}")
        init_llm_segmentation_openai(args.segmentation_model)
    elif args.llm_segmentation and args.segmentation_model_path:
        print(f"\nLoading separate segmentation model")
        init_llm_segmentation(
            args.segmentation_model_path,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit
        )
    
    # Determine GPU list for parallel execution
    gpu_list = None
    if args.num_gpus:
        if args.gpus:
            gpu_list = [int(g) for g in args.gpus.split(",")]
        else:
            gpu_list = list(range(args.num_gpus))
        
        if len(gpu_list) != args.num_gpus:
            print(f"Warning: --gpus has {len(gpu_list)} GPUs but --num-gpus is {args.num_gpus}. Using {len(gpu_list)} GPUs.")
    
    # Run pipeline (parallel or single)
    if gpu_list:
        all_predictions, all_gold, all_segments = run_pipeline_parallel(
            config,
            gpus=gpu_list,
            use_llm_segmentation=args.llm_segmentation,
            verbose=True,
            limit=args.limit
        )
    else:
        all_predictions, all_gold, all_segments = run_pipeline(
            config,
            use_llm_segmentation=args.llm_segmentation,
            verbose=True,
            limit=args.limit,
            shard=args.shard
        )
    
    # Save results and run official evaluation
    eval_data = load_jsonl(config.dev_path)
    if args.limit:
        eval_data = eval_data[:args.limit]
    save_results(
        config, all_predictions, all_gold, all_segments, eval_data,
        use_llm_segmentation=args.llm_segmentation,
        limit=args.limit,
        data_split=args.data
    )


if __name__ == "__main__":
    main()