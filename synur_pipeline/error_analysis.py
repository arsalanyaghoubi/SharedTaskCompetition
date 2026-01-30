import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_predictions(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_source_data(data_path):

    transcripts = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            transcripts[record["id"]] = record["transcript"]
    return transcripts


def load_schema(schema_path):

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    result = {}
    for item in schema:
        result[item["id"]] = item
        # Also store valid options for SELECT/MULTI_SELECT
        if "options" in item:
            result[item["id"]]["valid_options"] = set(
                opt.lower().strip() for opt in item["options"]
            )
    return result


def normalize_value(value):

    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return sorted([normalize_value(v) for v in value])
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if "value" in value:
            return normalize_value(value["value"])
        return str(value).lower()
    return value


def normalize_for_matching(text):
    """Remove filler words, punctuation, and extra whitespace for fuzzy matching."""
    if not text:
        return ""
    text = text.lower()
    # Remove common filler words/sounds
    fillers = [", um,", ", uh,", " um ", " uh ", "um,", "uh,", ", um", ", uh", "..."]
    for filler in fillers:
        text = text.replace(filler, " ")
    # Remove punctuation except apostrophes (for contractions)
    import re
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def check_text_in_transcript(text, transcript):
    """Check if text appears in transcript, accounting for filler words."""
    if not text or not transcript:
        return False
    
    # First try exact match
    if text.lower().strip() in transcript.lower():
        return True
    
    # Try with normalized versions (filler words removed)
    text_norm = normalize_for_matching(text)
    transcript_norm = normalize_for_matching(transcript)
    
    if text_norm in transcript_norm:
        return True
    
    # Check if most words from text appear in transcript (for fragmented matches)
    text_words = set(text_norm.split())
    transcript_words = set(transcript_norm.split())
    if len(text_words) >= 3:
        overlap = len(text_words & transcript_words) / len(text_words)
        if overlap >= 0.8:  # 80% of words match
            return True
    
    return False


def classify_string_error(pred_value, gold_value, transcript):

    pred_norm = str(pred_value).lower().strip() if pred_value is not None else ""
    gold_norm = str(gold_value).lower().strip() if gold_value is not None else ""
    
    # Check if gold is substring of prediction (over-extraction)
    if gold_norm in pred_norm and gold_norm != pred_norm:
        extra_text = pred_norm.replace(gold_norm, "").strip()
        # Check if the extra text is from the transcript
        if check_text_in_transcript(extra_text, transcript):
            return "over_extraction_from_transcript", {
                "extra_text": extra_text,
                "in_transcript": True
            }
        else:
            return "over_extraction_hallucinated", {
                "extra_text": extra_text,
                "in_transcript": False
            }
    
    # Check if prediction is substring of gold (under-extraction)
    if pred_norm in gold_norm and pred_norm != gold_norm:
        return "under_extraction", {
            "missing_text": gold_norm.replace(pred_norm, "").strip()
        }
    
    # Check if BOTH pred and gold are NOT in transcript - this is a semantic/format mismatch
    # (e.g., gold="Yes", pred="present" when transcript just mentions the condition exists)
    pred_in_transcript = check_text_in_transcript(pred_value, transcript)
    gold_in_transcript = check_text_in_transcript(gold_value, transcript)
    
    if not pred_in_transcript and not gold_in_transcript:
        # Neither is from transcript - both are interpretations, just formatted differently
        return "semantic_mismatch", {
            "note": "Both prediction and gold are interpretations, not literal extractions. Format/wording differs."
        }
    
    # Check if prediction is in transcript but different from gold
    if pred_in_transcript:
        return "wrong_span_from_transcript", {
            "note": "Predicted text exists in transcript but doesn't match gold"
        }
    
    # Prediction not in transcript but gold IS - true hallucination
    if not pred_in_transcript and gold_in_transcript:
        return "hallucinated_value", {
            "note": "Predicted text not found in transcript (gold value IS in transcript)"
        }
    
    return "wrong_value_other", {}


def classify_select_error(pred_value, gold_value, schema_item):

    valid_options = schema_item.get("valid_options", set())
    
    # Normalize values
    if isinstance(pred_value, list):
        pred_set = set(v.lower().strip() if isinstance(v, str) else str(v) for v in pred_value)
    else:
        pred_set = {pred_value.lower().strip() if isinstance(pred_value, str) else str(pred_value)}
    
    if isinstance(gold_value, list):
        gold_set = set(v.lower().strip() if isinstance(v, str) else str(v) for v in gold_value)
    else:
        gold_set = {gold_value.lower().strip() if isinstance(gold_value, str) else str(gold_value)}
    
    # Check for invalid options (not in schema)
    invalid_options = pred_set - valid_options if valid_options else set()
    
    if invalid_options:
        return "invalid_schema_option", {
            "invalid_options": list(invalid_options),
            "valid_options": list(valid_options)[:10]  # Show first 10
        }
    
    # Check for partial match
    overlap = pred_set & gold_set
    if overlap:
        return "partial_match", {
            "matched": list(overlap),
            "missed": list(gold_set - pred_set),
            "extra": list(pred_set - gold_set)
        }
    
    # Wrong but valid option
    return "wrong_valid_option", {
        "predicted": list(pred_set),
        "gold": list(gold_set)
    }


def categorize_errors(predictions, gold, schema, transcript=None, transcript_id=None):

    errors = {
        "correct": [],
        # STRING errors
        "over_extraction_from_transcript": [],
        "over_extraction_hallucinated": [],
        "under_extraction": [],
        "wrong_span_from_transcript": [],
        "hallucinated_value": [],
        "semantic_mismatch": [],
        "wrong_value_other": [],
        # SELECT/MULTI_SELECT errors
        "invalid_schema_option": [],
        "partial_match": [],
        "wrong_valid_option": [],
        # General errors
        "hallucinated_observation": [],  # ID not in gold
        "missed": [],                    # Gold observation not predicted
        "invalid_id": [],                # ID not in schema
    }
    
    # Build lookup maps
    gold_by_id = {str(g.get("id")): g for g in gold}
    pred_by_id = {str(p.get("id")): p for p in predictions}
    
    # Check each prediction
    for pred in predictions:
        pred_id = str(pred.get("id"))
        pred_value = pred.get("value")
        pred_name = pred.get("name", "")
        
        # Check if ID is valid (exists in schema)
        if pred_id not in schema:
            errors["invalid_id"].append({
                "transcript_id": transcript_id,
                "prediction": pred,
                "reason": f"ID {pred_id} not in schema"
            })
            continue
        
        schema_item = schema.get(pred_id, {})
        value_type = schema_item.get("value_type", "STRING")
        
        # Check if this ID exists in gold
        if pred_id not in gold_by_id:
            errors["hallucinated_observation"].append({
                "transcript_id": transcript_id,
                "prediction": pred,
                "value_type": value_type,
                "reason": "Observation ID not in gold standard"
            })
            continue
        
        gold_obs = gold_by_id[pred_id]
        gold_value = gold_obs.get("value")
        
        pred_norm = normalize_value(pred_value)
        gold_norm = normalize_value(gold_value)
        
        # Check for exact match
        if pred_norm == gold_norm:
            errors["correct"].append({
                "transcript_id": transcript_id,
                "prediction": pred,
                "gold": gold_obs
            })
            continue
        
        # Classify error based on value type
        if value_type == "STRING":
            error_type, details = classify_string_error(pred_value, gold_value, transcript)
            errors[error_type].append({
                "transcript_id": transcript_id,
                "prediction": pred,
                "gold": gold_obs,
                **details
            })
        elif value_type in ("SELECT", "MULTI_SELECT"):
            error_type, details = classify_select_error(pred_value, gold_value, schema_item)
            errors[error_type].append({
                "transcript_id": transcript_id,
                "prediction": pred,
                "gold": gold_obs,
                **details
            })
        else:
            # NUMERIC or other types
            errors["wrong_value_other"].append({
                "transcript_id": transcript_id,
                "prediction": pred,
                "gold": gold_obs,
                "reason": f"Value mismatch for {value_type}"
            })
    
    # Find missed observations (in gold but not predicted)
    for gold_obs in gold:
        gold_id = str(gold_obs.get("id"))
        if gold_id not in pred_by_id:
            errors["missed"].append({
                "transcript_id": transcript_id,
                "gold": gold_obs,
                "reason": "Not predicted"
            })
    
    return errors


def aggregate_errors(all_errors):

    totals = defaultdict(int)
    details = defaultdict(list)
    
    for transcript_errors in all_errors:
        for category, items in transcript_errors.items():
            totals[category] += len(items)
            details[category].extend(items)
    
    return dict(totals), dict(details)


def analyze_missed_observations(details, schema):

    missed_counts = defaultdict(int)
    
    for item in details.get("missed", []):
        obs_id = str(item["gold"].get("id"))
        obs_name = item["gold"].get("name", schema.get(obs_id, {}).get("name", "Unknown"))
        missed_counts[f"{obs_id}: {obs_name}"] += 1
    
    return dict(sorted(missed_counts.items(), key=lambda x: -x[1]))


def analyze_hallucinated_observations(details, schema):

    hallucinated_counts = defaultdict(int)
    
    for item in details.get("hallucinated_observation", []):
        obs_id = str(item["prediction"].get("id"))
        obs_name = item["prediction"].get("name", schema.get(obs_id, {}).get("name", "Unknown"))
        hallucinated_counts[f"{obs_id}: {obs_name}"] += 1
    
    return dict(sorted(hallucinated_counts.items(), key=lambda x: -x[1]))


def print_report(totals, details, schema):

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS REPORT")
    print("=" * 70)
    
    # Calculate totals
    correct = totals.get("correct", 0)
    
    string_errors = (
        totals.get("over_extraction_from_transcript", 0) +
        totals.get("over_extraction_hallucinated", 0) +
        totals.get("under_extraction", 0) +
        totals.get("wrong_span_from_transcript", 0) +
        totals.get("hallucinated_value", 0) +
        totals.get("semantic_mismatch", 0) +
        totals.get("wrong_value_other", 0)
    )
    
    select_errors = (
        totals.get("invalid_schema_option", 0) +
        totals.get("partial_match", 0) +
        totals.get("wrong_valid_option", 0)
    )
    
    hallucinated = totals.get("hallucinated_observation", 0)
    missed = totals.get("missed", 0)
    invalid = totals.get("invalid_id", 0)
    
    total_predictions = correct + string_errors + select_errors + hallucinated + invalid
    total_gold = correct + string_errors + select_errors + missed
    
    print(f"\nTotal Predictions: {total_predictions}")
    print(f"Total Gold Observations: {total_gold}")
    print(f"Correct: {correct} ({correct/total_predictions*100:.1f}% of predictions)")
    
    print(f"  Value Errors (STRING):        {string_errors}")
    print(f"  Value Errors (SELECT):        {select_errors}")
    print(f"  Hallucinated Observation:     {hallucinated}")
    print(f"  Missed Observation:           {missed}")
    if invalid > 0:
        print(f"  Invalid Schema ID:            {invalid}  (ID not even in schema)")
    
    # ERROR TYPE SUMMARY TABLE
    print("\n" + "-" * 70)
    print("ERROR TYPE BREAKDOWN")
    print("-" * 70)
    print(f"{'Category':<10} {'Error Type':<35} {'Count':>6}  Description")
    print("-" * 70)
    
    error_rows = [
        ("STRING", "over_extraction_from_transcript", "Correct + extra text from transcript"),
        ("STRING", "over_extraction_hallucinated", "Correct + made-up extra text"),
        ("STRING", "under_extraction", "Didn't pull enough text"),
        ("STRING", "wrong_span_from_transcript", "Wrong text from transcript"),
        ("STRING", "hallucinated_value", "Made-up text (gold IS in transcript)"),
        ("STRING", "semantic_mismatch", "Both pred & gold are interpretations"),
        ("STRING", "wrong_value_other", "Miscellaneous value mismatch"),
        ("SELECT", "invalid_schema_option", "Option not in schema"),
        ("SELECT", "partial_match", "Some options right, some wrong"),
        ("SELECT", "wrong_valid_option", "Valid option but not in gold"),
        ("GENERAL", "hallucinated_observation", "Predicted ID not in gold"),
        ("GENERAL", "missed", "Gold observation not predicted"),
        ("GENERAL", "invalid_id", "Predicted ID not in schema"),
    ]
    
    for cat, key, desc in error_rows:
        count = totals.get(key, 0)
        if count > 0:
            print(f"{cat:<10} {key:<35} {count:>6}  {desc}")
    
    # DETAILED ERRORS - Show all instances
    print("\n" + "=" * 70)
    print("SAMPLE ERRORS")
    print("=" * 70)
    
    # STRING errors
    string_error_types = [
        ("over_extraction_from_transcript", "OVER-EXTRACTION (from transcript)",
         "Model pulled correct answer + extra text from transcript"),
        ("over_extraction_hallucinated", "OVER-EXTRACTION (hallucinated)",
         "Model pulled correct answer + made-up extra text"),
        ("under_extraction", "UNDER-EXTRACTION",
         "Model didn't pull enough text"),
        ("wrong_span_from_transcript", "WRONG SPAN (from transcript)",
         "Wrong text, but it exists in transcript"),
        ("hallucinated_value", "HALLUCINATED VALUE",
         "Predicted text not found in transcript (but gold IS in transcript)"),
        ("semantic_mismatch", "SEMANTIC MISMATCH",
         "Both pred and gold are interpretations, not literal extractions - format differs"),
        ("wrong_value_other", "OTHER VALUE ERROR",
         "Miscellaneous value mismatch"),
    ]
    
    for key, title, desc in string_error_types:
        items = details.get(key, [])
        if not items:
            continue
        
        print(f"\n--- {title} ({len(items)} total) ---")
        print(f"    {desc}\n")
        
        # Show only first 5 examples
        for item in items[:5]:
            pred = item["prediction"]
            gold = item["gold"]
            tid = item.get("transcript_id", "?")
            print(f"  [{tid}] ID {pred.get('id')}: {pred.get('name')}")
            print(f"    Predicted: '{pred.get('value')}'")
            print(f"    Gold:      '{gold.get('value')}'")
            if "extra_text" in item:
                print(f"    Extra:     '{item['extra_text']}'")
            if "missing_text" in item:
                print(f"    Missing:   '{item['missing_text']}'")
            print()
        
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more\n")
    
    # SELECT errors
    select_error_types = [
        ("invalid_schema_option", "INVALID SCHEMA OPTION",
         "Predicted option doesn't exist in schema"),
        ("partial_match", "PARTIAL MATCH (MULTI_SELECT)",
         "Got some options right, missed/added others"),
        ("wrong_valid_option", "WRONG OPTION (valid but incorrect)",
         "Chose valid schema option that wasn't in gold"),
    ]
    
    for key, title, desc in select_error_types:
        items = details.get(key, [])
        if not items:
            continue
        
        print(f"\n--- {title} ({len(items)} total) ---")
        print(f"    {desc}\n")
        
        # Show only first 5 examples
        for item in items[:5]:
            pred = item["prediction"]
            gold = item.get("gold", {})
            if isinstance(gold, list):
                gold = gold[0] if gold else {}
            tid = item.get("transcript_id", "?")
            print(f"  [{tid}] ID {pred.get('id')}: {pred.get('name')}")
            print(f"    Predicted: {pred.get('value')}")
            print(f"    Gold:      {gold.get('value') if isinstance(gold, dict) else gold}")
            if "invalid_options" in item:
                print(f"    Invalid options: {item['invalid_options']}")
            if "matched" in item:
                print(f"    Matched: {item['matched']}, Missed: {item['missed']}, Extra: {item['extra']}")
            print()
        
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more\n")
    
    # Hallucinated observations (predicted ID not in gold)
    hallucinated_items = details.get("hallucinated_observation", [])
    if hallucinated_items:
        print(f"\n--- HALLUCINATED OBSERVATIONS ({len(hallucinated_items)} total) ---")
        print("    Model predicted an observation ID that exists in schema but wasn't in gold\n")
        
        # Show aggregated counts instead of every instance
        hallucinated_counts = defaultdict(int)
        for item in hallucinated_items:
            pred = item["prediction"]
            key = f"ID {pred.get('id')}: {pred.get('name')}"
            hallucinated_counts[key] += 1
        
        print("  Most common (top 15):")
        for key, count in sorted(hallucinated_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {count}x  {key}")
        print()
    
    # Missed observations
    missed_items = details.get("missed", [])
    if missed_items:
        print(f"\n--- MISSED OBSERVATIONS ({len(missed_items)} total) ---")
        print("    Gold observations that the model didn't predict at all\n")
        
        # Show aggregated counts instead of every instance
        missed_counts = defaultdict(int)
        for item in missed_items:
            gold = item["gold"]
            key = f"ID {gold.get('id')}: {gold.get('name')}"
            missed_counts[key] += 1
        
        print("  Most common (top 15):")
        for key, count in sorted(missed_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {count}x  {key}")
        print()
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze extraction errors")
    parser.add_argument("predictions_file", type=str, help="Path to predictions JSON file")
    parser.add_argument("--schema", type=str, default=None, help="Path to schema JSON file")
    parser.add_argument("--data", type=str, default="train", choices=["train", "dev"],
                       help="Which data split to use for transcript lookup")
    
    args = parser.parse_args()
    
    # Find schema
    script_dir = Path(__file__).parent
    
    if args.schema:
        schema_path = Path(args.schema)
    else:
        schema_path = script_dir.parent / "Data" / "synur_schema.json"
    
    if not schema_path.exists():
        print(f"Schema not found at {schema_path}. Use --schema to specify path.")
        return
    
    schema = load_schema(schema_path)
    print(f"Loaded schema with {len(schema)} concepts")
    
    # Load source data for transcript lookup
    data_path = script_dir.parent / "Data" / f"{args.data}.jsonl"
    if data_path.exists():
        transcripts = load_source_data(data_path)
        print(f"Loaded {len(transcripts)} transcripts from {args.data}.jsonl")
    else:
        transcripts = {}
        print(f"Warning: Could not load {data_path} - transcript-based analysis disabled")
    
    # Load predictions
    predictions_path = Path(args.predictions_file)
    if not predictions_path.exists():
        predictions_path = script_dir / "outputs" / args.predictions_file
    
    if not predictions_path.exists():
        print(f"Predictions file not found: {args.predictions_file}")
        return
    
    data = load_predictions(predictions_path)
    print(f"Loaded {len(data)} transcripts from {predictions_path.name}")
    
    # Analyze each transcript
    all_errors = []
    for item in data:
        preds = item.get("predictions", [])
        gold = item.get("gold", [])
        item_id = item.get("id")
        transcript = transcripts.get(item_id, "")
        
        errors = categorize_errors(preds, gold, schema, transcript, transcript_id=item_id)
        all_errors.append(errors)
    
    # Aggregate and report
    totals, details = aggregate_errors(all_errors)
    print_report(totals, details, schema)
    
    # Always save detailed output to file
    # Use the predictions filename as base, add _errors suffix
    predictions_stem = predictions_path.stem
    output_path = predictions_path.parent / f"{predictions_stem}_errors.json"
    
    # Build error type summary table
    error_type_summary = [
        {"error_type": "over_extraction_from_transcript", "count": totals.get("over_extraction_from_transcript", 0),
         "category": "STRING", "description": "Model pulled correct answer + extra text from transcript"},
        {"error_type": "over_extraction_hallucinated", "count": totals.get("over_extraction_hallucinated", 0),
         "category": "STRING", "description": "Model pulled correct answer + made-up extra text"},
        {"error_type": "under_extraction", "count": totals.get("under_extraction", 0),
         "category": "STRING", "description": "Model didn't pull enough text"},
        {"error_type": "wrong_span_from_transcript", "count": totals.get("wrong_span_from_transcript", 0),
         "category": "STRING", "description": "Wrong text, but it exists in transcript"},
        {"error_type": "hallucinated_value", "count": totals.get("hallucinated_value", 0),
         "category": "STRING", "description": "Predicted text not found in transcript (gold IS in transcript)"},
        {"error_type": "semantic_mismatch", "count": totals.get("semantic_mismatch", 0),
         "category": "STRING", "description": "Both pred and gold are interpretations, not literal extractions"},
        {"error_type": "wrong_value_other", "count": totals.get("wrong_value_other", 0),
         "category": "STRING", "description": "Miscellaneous value mismatch"},
        {"error_type": "invalid_schema_option", "count": totals.get("invalid_schema_option", 0),
         "category": "SELECT", "description": "Predicted option doesn't exist in schema"},
        {"error_type": "partial_match", "count": totals.get("partial_match", 0),
         "category": "SELECT", "description": "Got some options right, missed/added others"},
        {"error_type": "wrong_valid_option", "count": totals.get("wrong_valid_option", 0),
         "category": "SELECT", "description": "Chose valid schema option that wasn't in gold"},
        {"error_type": "hallucinated_observation", "count": totals.get("hallucinated_observation", 0),
         "category": "GENERAL", "description": "Model predicted an observation ID that wasn't in gold"},
        {"error_type": "missed", "count": totals.get("missed", 0),
         "category": "GENERAL", "description": "Gold observation that the model didn't predict"},
        {"error_type": "invalid_id", "count": totals.get("invalid_id", 0),
         "category": "GENERAL", "description": "Predicted ID doesn't exist in schema"},
    ]
    
    # Filter out "correct" from details - we only want errors
    errors_only = {k: v for k, v in details.items() if k != "correct"}
    
    # Calculate totals
    string_errors = sum(t["count"] for t in error_type_summary if t["category"] == "STRING")
    select_errors = sum(t["count"] for t in error_type_summary if t["category"] == "SELECT")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_predictions": totals.get("correct", 0) + string_errors + select_errors + 
                    totals.get("hallucinated_observation", 0) + totals.get("invalid_id", 0),
                "total_gold": totals.get("correct", 0) + string_errors + select_errors + totals.get("missed", 0),
                "correct": totals.get("correct", 0),
                "string_errors": string_errors,
                "select_errors": select_errors,
                "hallucinated_observations": totals.get("hallucinated_observation", 0),
                "missed_observations": totals.get("missed", 0),
            },
            "error_type_summary": error_type_summary,
            "missed_by_observation": analyze_missed_observations(details, schema),
            "hallucinated_by_observation": analyze_hallucinated_observations(details, schema),
            "errors": errors_only  # Only errors, not "correct"
        }, f, indent=2)
    
    print(f"\nFull error details saved to: {output_path.name}")


if __name__ == "__main__":
    main()
