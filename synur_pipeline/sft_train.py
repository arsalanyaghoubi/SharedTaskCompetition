"""
Supervised Fine-Tuning (SFT) for Llama-3-8B-Instruct on SYNUR extraction task.

Usage:
    # Basic training
    python sft_train.py --output-dir ./sft_output
    
    # With custom settings
    python sft_train.py --output-dir ./sft_output --epochs 3 --batch-size 2 --lr 2e-5
    
    # Resume from checkpoint
    python sft_train.py --output-dir ./sft_output --resume-from-checkpoint
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Get paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent


def load_training_data(data_path, schema_path):
    
    # Load schema for reference
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    schema_by_id = {str(item['id']): item for item in schema}
    
    # Load training data
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
    
    return data, schema_by_id


def create_training_examples(data, schema_by_id, tokenizer, max_seq_length=4096, use_hybrid=True, top_n=60):

    from schema_rag import SchemaRAG, HybridSchemaRAG
    
    # Initialize RAG for schema retrieval
    schema_path = PROJECT_ROOT / "Data" / "synur_schema.json"
    if use_hybrid:
        rag = HybridSchemaRAG(schema_path, "BAAI/bge-base-en-v1.5", alpha=0.6)
        print("  Using HYBRID retrieval (BM25 + dense)")
    else:
        rag = SchemaRAG(schema_path, "BAAI/bge-base-en-v1.5")
        print("  Using dense-only retrieval")
    
    print(f"  Using top_n={top_n} for full-transcript retrieval")
    
    examples = []
    total_gold = 0
    retrieved_gold = 0
    
    for item in data:
        transcript_id = item.get('id', 'unknown')
        transcript = item['transcript']
        
        # Parse gold observations
        gold_obs = item['observations']
        if isinstance(gold_obs, str):
            gold_obs = json.loads(gold_obs)
        
        # Get relevant schema rows for the FULL transcript
        schema_rows = rag.retrieve(transcript, top_n=top_n)
        retrieved_ids = {str(row.id) for row in schema_rows}
        
        # Track retrieval recall
        for obs in gold_obs:
            total_gold += 1
            if str(obs['id']) in retrieved_ids:
                retrieved_gold += 1
        
        # Build schema context as JSON - EXACTLY like inference in extraction.py
        schema_json = json.dumps([row.to_dict() for row in schema_rows], indent=2)
        
        # Build expected output with ALL gold observations
        output_json = json.dumps({"observations": gold_obs}, indent=2)
        
        # Create the training example in chat format
        # MUST MATCH extraction.py EXTRACTION_SYSTEM_PROMPT_ENHANCED exactly
        system_prompt = """You are an expert at medical electronic health record (EHR) flowsheet analysis.

Your task is to extract clinical observations from nurse dictation transcripts and structure them according to a provided schema.

Guidelines:
1. Extract ONLY observations that are explicitly mentioned or clearly implied in the transcript
2. Match extracted values to the schema's value_enum when applicable (use exact matches)
3. For NUMERIC types, extract only the numeric value (no units)
4. For MULTI_SELECT types, return an array of applicable values
5. For STRING types, extract the relevant text as mentioned
6. For SINGLE_SELECT types, choose the single best matching option from value_enum
7. Do NOT fabricate or infer observations that are not in the transcript
8. If unsure, do NOT include the observation

Output Format:
Return a JSON object with key "observations" containing an array. Each observation must have:
- "id": The schema concept ID (string)
- "name": The schema concept name (string)  
- "value_type": The type from schema (string)
- "value": The extracted value (type depends on value_type)

For MULTI_SELECT, value should be an array of strings.
For NUMERIC, value should be a number.
For STRING and SINGLE_SELECT, value should be a string."""

        # MUST MATCH extraction.py EXTRACTION_USER_PROMPT_ENHANCED exactly
        # Note: schema comes BEFORE transcript in inference
        user_prompt = f"""SCHEMA:
{schema_json}

TRANSCRIPT:
{transcript}

Extract all clinical observations from the TRANSCRIPT that match concepts in the SCHEMA.
Return a JSON object with key "observations" containing an array of extracted observations.

OUTPUT:"""

        # Format as chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output_json}
        ]
        
        examples.append({
            "transcript_id": transcript_id,
            "messages": messages,
            "transcript": transcript,
            "gold_count": len(gold_obs)
        })
    
    # Report retrieval recall for training data
    if total_gold > 0:
        recall = retrieved_gold / total_gold * 100
        missed = total_gold - retrieved_gold
        print(f"  Gold observation coverage: {retrieved_gold}/{total_gold} ({recall:.1f}%) - {missed} missed")
    
    return examples


def format_for_training(examples, tokenizer):
    
    formatted = []
    
    for ex in examples:
        # Apply chat template to get full text
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Also get the prompt-only version (without assistant response)
        # to find where the response starts
        prompt_messages = ex["messages"][:-1]  # System + User only
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True  # This adds the assistant header
        )
        
        formatted.append({
            "text": text, 
            "prompt_text": prompt_text,
            "transcript_id": ex["transcript_id"]
        })
    
    return formatted


def main():
    parser = argparse.ArgumentParser(description="SFT for SYNUR extraction")
    
    # Model settings
    parser.add_argument("--model-path", type=str, 
                        default="/home1/shared/Models/Llama/Meta-Llama-3-8B-Instruct",
                        help="Path to base model")
    parser.add_argument("--output-dir", type=str, default="./sft_output",
                        help="Directory to save fine-tuned model")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio")
    
    # LoRA settings
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Quantization
    parser.add_argument("--use-4bit", action="store_true",
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use-8bit", action="store_true",
                        help="Use 8-bit quantization")
    
    # Other
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only prepare data, don't train")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"sft_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SYNUR SFT Training")
    print("=" * 60)
    print(f"Base model: {args.model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"Quantization: {'4-bit' if args.use_4bit else '8-bit' if args.use_8bit else 'None (fp16)'}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for training
    
    # Load model with optional quantization
    print("Loading model...")
    
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    elif args.use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare training data
    print("\nPreparing training data...")
    data_path = PROJECT_ROOT / "Data" / "train.jsonl"
    schema_path = PROJECT_ROOT / "Data" / "synur_schema.json"
    
    raw_data, schema_by_id = load_training_data(data_path, schema_path)
    print(f"Loaded {len(raw_data)} transcripts")
    
    # Limit samples for testing
    if args.max_samples is not None:
        raw_data = raw_data[:args.max_samples]
        print(f"Limiting to {len(raw_data)} samples for testing")
    
    # Create training examples (transcript-level, not segment-level)
    print("Creating training examples...")
    examples = create_training_examples(raw_data, schema_by_id, tokenizer, args.max_seq_length, use_hybrid=True)
    print(f"Created {len(examples)} training examples (1 per transcript)")
    
    # Format for training
    formatted_examples = format_for_training(examples, tokenizer)
    
    # Create dataset
    dataset = Dataset.from_list(formatted_examples)
    
    # Tokenize and create labels with proper masking
    def tokenize_and_mask(example):
        # Tokenize full text
        full_tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        
        # Tokenize prompt only to find where response starts
        prompt_tokens = tokenizer(
            example["prompt_text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        
        # Create labels: -100 for prompt tokens (ignored in loss), actual ids for response
        prompt_len = len(prompt_tokens["input_ids"])
        labels = [-100] * prompt_len + full_tokens["input_ids"][prompt_len:]
        
        # Ensure labels length matches input_ids
        labels = labels[:len(full_tokens["input_ids"])]
        
        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels
        }
    
    print("Tokenizing with loss masking...")
    tokenized_dataset = dataset.map(
        tokenize_and_mask,
        remove_columns=["text", "prompt_text", "transcript_id"],
        desc="Tokenizing"
    )
    
    print(f"Dataset size: {len(tokenized_dataset)}")
    print(f"Sample token length: {len(tokenized_dataset[0]['input_ids'])}")
    
    # Dry run - exit before training
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Data preparation complete, exiting before training")
        print("=" * 60)
        return
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=not args.use_4bit and not args.use_8bit,
        bf16=args.use_4bit or args.use_8bit,
        optim="paged_adamw_32bit" if args.use_4bit else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",  # Disable wandb/tensorboard
        seed=args.seed,
        dataloader_num_workers=4,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Changed from tokenizer= for newer transformers
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    final_output_dir = output_dir / "final"
    model.save_pretrained(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    # Save training config
    config = {
        "base_model": args.model_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_seq_length": args.max_seq_length,
        "num_examples": len(tokenized_dataset),
        "quantization": "4bit" if args.use_4bit else "8bit" if args.use_8bit else "none",
        "timestamp": timestamp,
    }
    
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {final_output_dir}")
    print(f"Config saved to: {output_dir / 'training_config.json'}")
    
    print("\nTo use the fine-tuned model:")
    print(f"python run_pipeline.py --model llama-3-8b-sft --model-path {final_output_dir} ...")


if __name__ == "__main__":
    main()
