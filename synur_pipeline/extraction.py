# Extracts the observations from transcript segments using an LLM with the filtered schema.

import json
import time

from schema_rag import SchemaRow


def _call_llm(client, messages, temperature, json_mode=True):

    response = client.chat.completions.create(
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"} if json_mode else None
    )
    return response


# Paper's simplified prompt (Appendix A.1) - they couldn't share the full version
EXTRACTION_PROMPT_PAPER = """You are an expert at medical electronic health record flowsheet analysis.

Below is a TRANSCRIPT from a nurse dictation along with a flowsheet SCHEMA. Please extract the clinical observations from the TRANSCRIPT in strict, parsable JSON adhering to SCHEMA.

SCHEMA: {schema}

TRANSCRIPT: {transcript}

OUTPUT:"""

# Enhanced prompt - more detailed guidance
EXTRACTION_SYSTEM_PROMPT_ENHANCED = """You are an expert at medical electronic health record (EHR) flowsheet analysis.

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

EXTRACTION_USER_PROMPT_ENHANCED = """SCHEMA:
{schema}

TRANSCRIPT:
{transcript}

Extract all clinical observations from the TRANSCRIPT that match concepts in the SCHEMA.
Return a JSON object with key "observations" containing an array of extracted observations.

OUTPUT:"""

EXTRACTION_USER_PROMPT_ENHANCED_WITH_EXAMPLES = """SCHEMA:
{schema}

EXAMPLES:
{examples}

TRANSCRIPT:
{transcript}

Extract all clinical observations from the TRANSCRIPT that match concepts in the SCHEMA.
Use the EXAMPLES as guidance for how to format your extractions.
Return a JSON object with key "observations" containing an array of extracted observations.

OUTPUT:"""


def format_few_shot_examples(examples):

    formatted = []
    for i, ex in enumerate(examples, 1):
        transcript = ex['transcript']
        
        # Parse observations if string
        obs = ex['observations']
        if isinstance(obs, str):
            obs = json.loads(obs)
        
        formatted.append(f"Example {i}:")
        formatted.append(f"Transcript: {transcript}")
        formatted.append(f"Observations: {json.dumps(obs, indent=2)}")
        formatted.append("")
    
    return "\n".join(formatted)


def extract_observations(
    segment,
    schema_rows,
    call_llm,
    few_shot_examples=None,
    temperature=0.0,
    use_paper_prompt=True
):

    # Format schema for prompt
    schema_json = json.dumps([row.to_dict() for row in schema_rows], indent=2)
    
    if use_paper_prompt:
        # Paper's single-prompt approach (no system/user separation in their simplified version)
        # We adapt it to the chat API format
        user_prompt = EXTRACTION_PROMPT_PAPER.format(
            schema=schema_json,
            transcript=segment
        )
        
        # Call the LLM with paper's approach
        result = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            json_mode=True
        )
    else:
        # Enhanced prompt with system/user separation
        if few_shot_examples:
            examples_str = format_few_shot_examples(few_shot_examples)
            user_prompt = EXTRACTION_USER_PROMPT_ENHANCED_WITH_EXAMPLES.format(
                schema=schema_json,
                examples=examples_str,
                transcript=segment
            )
        else:
            user_prompt = EXTRACTION_USER_PROMPT_ENHANCED.format(
                schema=schema_json,
                transcript=segment
            )
        
        # Call the LLM with enhanced approach
        result = call_llm(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT_ENHANCED},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            json_mode=True
        )
    
    # Parse the response - handle various formats GPT might return
    observations = _parse_llm_response(result, schema_rows)
    
    return observations


def _clean_llm_response(result):
    """Strip markdown code blocks, comments, and repair common JSON issues from LLM responses."""
    import re
    
    result = result.strip()
    
    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    if result.startswith("```"):
        # Find the end of the first line (might be ```json or just ```)
        first_newline = result.find("\n")
        if first_newline != -1:
            result = result[first_newline + 1:]
        # Remove trailing ```
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()
    
    # Remove // comments (common with Llama models)
    # This regex removes // and everything after it until end of line, but not inside strings
    # Simple approach: remove lines that are just comments, and inline comments after values
    lines = result.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove inline comments: "value": "something" // comment
        # Be careful not to remove // inside string values
        # Look for // that appears after a quote or comma/bracket
        comment_match = re.search(r'("[^"]*"|[\[\]{},:])\s*//.*$', line)
        if comment_match:
            # Find where the // starts and remove from there
            comment_pos = line.rfind('//')
            if comment_pos > 0:
                line = line[:comment_pos].rstrip()
        cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines)
    
    # Try to fix truncated JSON by finding the last complete object/array
    # If JSON doesn't end with } or ], try to repair it
    result = result.rstrip()
    if result and not result.endswith('}') and not result.endswith(']'):
        # Find the last complete structure
        # Count braces to find where we can safely cut
        brace_count = 0
        bracket_count = 0
        last_valid_pos = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(result):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    last_valid_pos = i + 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    last_valid_pos = i + 1
        
        # If we found a valid cut point and it's not at the end, truncate there
        if last_valid_pos > 0 and last_valid_pos < len(result):
            result = result[:last_valid_pos]
    
    # Remove trailing commas before } or ] (common JSON error)
    result = re.sub(r',\s*([}\]])', r'\1', result)
    
    return result


def _parse_llm_response(result, schema_rows):

    # Build lookup maps from schema
    id_to_row = {row.id: row for row in schema_rows}
    name_to_row = {row.name.lower(): row for row in schema_rows}
    
    # Clean the response (strip markdown code blocks, etc.)
    result = _clean_llm_response(result)
    
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as e:
        print(f"  [WARNING] JSON parse error: {e}")
        print(f"  [WARNING] Raw response: {result[:500]}...")
        return []
    
    # Case 1: {"observations": [...]}
    if isinstance(parsed, dict) and "observations" in parsed:
        return parsed["observations"]
    
    # Case 2: Direct array [...]
    if isinstance(parsed, list):
        return parsed
    
    # Case 3: Flat dict with IDs or names as keys
    if isinstance(parsed, dict):
        observations = []
        
        for key, value in parsed.items():
            # Skip null/None/empty values
            if value is None:
                continue
            if value == "":
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            
            # Try to find matching schema row
            row = None
            
            # Check if key is a schema ID
            if key in id_to_row:
                row = id_to_row[key]
            # Check if key is a schema name (case-insensitive)
            elif key.lower() in name_to_row:
                row = name_to_row[key.lower()]
            
            if row:
                observations.append({
                    "id": row.id,
                    "name": row.name,
                    "value_type": row.value_type,
                    "value": value
                })
        
        return observations
    
    return []


def extract_from_full_transcript(
    transcript,
    segments,
    schema_rag,
    call_llm,
    top_n_schema=60,
    few_shot_examples=None,
    temperature=0.0,
    use_paper_prompt=True
):

    all_observations = []
    seen_ids = set()
    
    for segment in segments:
        # Retrieve relevant schema rows for this segment
        schema_rows = schema_rag.retrieve(segment, top_n=top_n_schema)
        
        # Extract observations from this segment
        segment_obs = extract_observations(
            segment=segment,
            schema_rows=schema_rows,
            call_llm=call_llm,
            few_shot_examples=few_shot_examples,
            temperature=temperature,
            use_paper_prompt=use_paper_prompt
        )
        
        # Deduplicate by ID (keep first occurrence)
        for obs in segment_obs:
            obs_id = obs.get("id")
            if obs_id and obs_id not in seen_ids:
                all_observations.append(obs)
                seen_ids.add(obs_id)
    
    return all_observations

