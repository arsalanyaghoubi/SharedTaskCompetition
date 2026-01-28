"""
Evaluation Module

Computes F1 score for extracted observations against gold annotations.
"""

import json
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    true_positives: float  # Can be fractional with partial credit
    false_positives: float
    false_negatives: float
    precision: float
    recall: float
    f1: float
    
    def __str__(self):
        # Format TP/FP/FN as int if whole numbers, otherwise show decimal
        tp_str = f"{self.true_positives:.2f}" if self.true_positives % 1 else f"{int(self.true_positives)}"
        fp_str = f"{self.false_positives:.2f}" if self.false_positives % 1 else f"{int(self.false_positives)}"
        fn_str = f"{self.false_negatives:.2f}" if self.false_negatives % 1 else f"{int(self.false_negatives)}"
        return (
            f"TP: {tp_str}, FP: {fp_str}, FN: {fn_str}\n"
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}"
        )


def normalize_value(value):
    """Normalize a value for comparison (lowercase, sort lists, convert numeric strings)."""
    if value is None:
        return None
    
    if isinstance(value, str):
        normalized = value.strip().lower()
        # Try to convert to number if it looks numeric
        try:
            return float(normalized)
        except ValueError:
            return normalized
    
    if isinstance(value, dict):
        # LLM sometimes returns dicts like {"value": "X"} - extract the value
        if "value" in value:
            return normalize_value(value["value"])
        # Otherwise convert to string for comparison
        return str(value).lower()
    
    if isinstance(value, list):
        # Normalize and sort list elements
        return sorted([normalize_value(v) for v in value])
    
    if isinstance(value, (int, float)):
        return float(value)
    
    return value


def values_match(pred_value, gold_value, value_type, partial_credit=False):
    """Check if predicted and gold values match, accounting for value type."""
    pred_norm = normalize_value(pred_value)
    gold_norm = normalize_value(gold_value)
    
    if value_type == "NUMERIC":
        # For numeric, check if values are close (within small tolerance)
        if isinstance(pred_norm, (int, float)) and isinstance(gold_norm, (int, float)):
            match = abs(pred_norm - gold_norm) < 0.01
        else:
            match = pred_norm == gold_norm
        return 1.0 if match else 0.0 if partial_credit else match
    
    elif value_type == "MULTI_SELECT":
        # Ensure both are lists
        if not isinstance(pred_norm, list):
            pred_norm = [pred_norm]
        if not isinstance(gold_norm, list):
            gold_norm = [gold_norm]
        
        pred_set = set(pred_norm)
        gold_set = set(gold_norm)
        
        if partial_credit:
            # Compute F1 at the value level (Jaccard-like)
            intersection = len(pred_set & gold_set)
            if intersection == 0:
                return 0.0
            precision = intersection / len(pred_set)
            recall = intersection / len(gold_set)
            return 2 * precision * recall / (precision + recall)
        else:
            # Exact match
            return pred_set == gold_set
    
    else:
        # For SINGLE_SELECT and STRING, exact match after normalization
        match = pred_norm == gold_norm
        return 1.0 if match else 0.0 if partial_credit else match


def observation_matches(pred_obs, gold_obs, partial_credit=False):
    """Check if a predicted observation matches a gold observation."""
    # IDs must match
    if str(pred_obs.get("id")) != str(gold_obs.get("id")):
        return 0.0 if partial_credit else False
    
    # Get value type (prefer gold, fallback to pred)
    value_type = gold_obs.get("value_type") or pred_obs.get("value_type") or "STRING"
    
    # Check if values match
    return values_match(pred_obs.get("value"), gold_obs.get("value"), value_type, partial_credit)


def evaluate_single(predictions, gold, partial_credit=False):
    """Evaluate predictions against gold annotations for a single transcript."""
    # Track which gold observations have been matched
    matched_gold = [False] * len(gold)
    
    true_positives = 0.0  # Can be fractional with partial credit
    false_positives = 0.0
    
    # For each prediction, try to find a matching gold observation
    for pred in predictions:
        best_match_score = 0.0
        best_match_idx = -1
        
        for i, gold_obs in enumerate(gold):
            if matched_gold[i]:
                continue
                
            # Check if IDs match first
            if str(pred.get("id")) != str(gold_obs.get("id")):
                continue
            
            # Get match score
            if partial_credit:
                score = observation_matches(pred, gold_obs, partial_credit=True)
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = i
            else:
                if observation_matches(pred, gold_obs, partial_credit=False):
                    best_match_score = 1.0
                    best_match_idx = i
                    break
        
        if best_match_idx >= 0:
            true_positives += best_match_score
            matched_gold[best_match_idx] = True
        else:
            false_positives += 1.0
    
    # False negatives are unmatched gold observations
    false_negatives = sum(1.0 for matched in matched_gold if not matched)
    
    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return EvaluationResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1
    )


def evaluate_dataset(
    all_predictions,
    all_gold,
    partial_credit=False
):
    """Evaluate predictions across an entire dataset (micro-averaged)."""
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    
    for preds, gold in zip(all_predictions, all_gold):
        result = evaluate_single(preds, gold, partial_credit=partial_credit)
        total_tp += result.true_positives
        total_fp += result.false_positives
        total_fn += result.false_negatives
    
    # Micro-averaged metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return EvaluationResult(
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        precision=precision,
        recall=recall,
        f1=f1
    )


def compute_error_analysis(predictions, gold):
    """Compute detailed error analysis for a single transcript."""
    matched_gold = [False] * len(gold)
    
    correct = []
    false_positives = []
    
    for pred in predictions:
        found_match = False
        for i, gold_obs in enumerate(gold):
            if not matched_gold[i] and observation_matches(pred, gold_obs):
                correct.append({
                    "predicted": pred,
                    "gold": gold_obs
                })
                matched_gold[i] = True
                found_match = True
                break
        
        if not found_match:
            # Find closest gold by ID for analysis
            same_id_gold = [g for g in gold if str(g.get("id")) == str(pred.get("id"))]
            false_positives.append({
                "predicted": pred,
                "same_id_in_gold": same_id_gold[0] if same_id_gold else None
            })
    
    false_negatives = [
        gold[i] for i, matched in enumerate(matched_gold) if not matched
    ]
    
    return {
        "correct": correct,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


if __name__ == "__main__":
    # Test evaluation
    gold = [
        {"id": "10", "name": "Oxygen saturation", "value_type": "NUMERIC", "value": 88},
        {"id": "13", "name": "Breathing pattern", "value_type": "MULTI_SELECT", "value": ["labored"]},
        {"id": "38", "name": "Use of accessory muscles", "value_type": "SINGLE_SELECT", "value": "Yes"},
    ]
    
    # Test case 1: Perfect match
    predictions1 = [
        {"id": "10", "name": "Oxygen saturation", "value_type": "NUMERIC", "value": 88},
        {"id": "13", "name": "Breathing pattern", "value_type": "MULTI_SELECT", "value": ["labored"]},
        {"id": "38", "name": "Use of accessory muscles", "value_type": "SINGLE_SELECT", "value": "Yes"},
    ]
    
    print("Test 1 - Perfect match:")
    result1 = evaluate_single(predictions1, gold)
    print(result1)
    print()
    
    # Test case 2: Some mismatches
    predictions2 = [
        {"id": "10", "name": "Oxygen saturation", "value_type": "NUMERIC", "value": 88},  # Correct
        {"id": "13", "name": "Breathing pattern", "value_type": "MULTI_SELECT", "value": ["shallow"]},  # Wrong value
        {"id": "99", "name": "Made up", "value_type": "STRING", "value": "fake"},  # FP
    ]
    
    print("Test 2 - Partial match:")
    result2 = evaluate_single(predictions2, gold)
    print(result2)
    print()
    
    # Test case 3: Value normalization
    predictions3 = [
        {"id": "10", "name": "Oxygen saturation", "value_type": "NUMERIC", "value": "88"},  # String instead of int
        {"id": "38", "name": "Use of accessory muscles", "value_type": "SINGLE_SELECT", "value": "yes"},  # Lowercase
    ]
    
    print("Test 3 - Value normalization:")
    result3 = evaluate_single(predictions3, gold)
    print(result3)
