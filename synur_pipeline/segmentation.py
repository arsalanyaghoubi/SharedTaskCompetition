# Splits nurse dictation transcripts into medically coherent segments. Can use LLM-based or simple rule-based methods.

import json
from openai import OpenAI


# Paper's simplified prompt (Appendix A.1) - they couldn't share the full version
SEGMENTATION_PROMPT_PAPER = """Given an input TRANSCRIPT of a nurse's observations about a patient, your task is to divide the input TRANSCRIPT into contiguous SEGMENTS, based on clinical facts.

A clinical fact refers to specific, verifiable information related to the health of a patient.

TRANSCRIPT:
{transcript}

SEGMENTS:"""

# Enhanced prompt - more detailed guidance, better JSON formatting. **Had an LLM help write this. Prompt engineering could further improve performance.**
SEGMENTATION_SYSTEM_PROMPT_ENHANCED = """You are an expert clinical NLP system that segments nurse dictation transcripts.

Your task is to divide a nurse dictation TRANSCRIPT into contiguous, non-overlapping SEGMENTS based on clinical facts.

A clinical fact refers to specific, verifiable information related to the health of a patient, such as:
- Vital signs (blood pressure, heart rate, oxygen saturation, temperature)
- Physical examination findings (breath sounds, bowel sounds, edema, skin condition)
- Symptoms and observations (pain, nausea, confusion, mobility)
- Interventions and equipment (IV therapy, oxygen delivery, positioning)
- Assessments and scores (fall risk, Glasgow coma scale, pain severity)

Guidelines:
1. Each segment should contain one or more related clinical facts
2. Keep segments as focused as possible while maintaining context
3. Preserve the exact text - do not modify or summarize
4. Segments must be contiguous and non-overlapping
5. The concatenation of all segments should equal the original transcript

Output your response as a JSON object with key "segments" containing an array of segment strings."""

SEGMENTATION_USER_PROMPT_ENHANCED = """TRANSCRIPT:
{transcript}

Divide this transcript into medically coherent segments. Return a JSON object with key "segments" containing an array of strings.

SEGMENTS:"""

# Segment a nurse dictation transcript into medically coherent chunks.
def segment_transcript(transcript, client, model="gpt-4o", use_paper_prompt=True):

    if use_paper_prompt:
        # Paper's exact single-prompt approach
        messages = [
            {"role": "user", "content": SEGMENTATION_PROMPT_PAPER.format(transcript=transcript)}
        ]
    else:
        # Enhanced system+user prompt approach
        messages = [
            {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT_ENHANCED},
            {"role": "user", "content": SEGMENTATION_USER_PROMPT_ENHANCED.format(transcript=transcript)}
        ]
    
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=messages
    )
    
    result = response.choices[0].message.content
    
    try:
        parsed = json.loads(result)
        # Handle different possible JSON structures
        if isinstance(parsed, list):
            segments = parsed
        elif isinstance(parsed, dict) and "segments" in parsed:
            segments = parsed["segments"]
        else:
            # Fallback: treat entire transcript as one segment
            segments = [transcript]
    except json.JSONDecodeError:
        # If parsing fails, treat entire transcript as one segment
        segments = [transcript]
    
    # Ensure we have at least one segment
    if not segments:
        segments = [transcript]
    
    return segments

# Simple rule-based segmentation as fallback method that splits on [Clinician] markers or double newlines.
def segment_transcript_simple(transcript):
    
    if "[Clinician]" in transcript:
        parts = transcript.split("[Clinician]")
        segments = []
        for part in parts:
            part = part.strip()
            if part:
                segments.append(part)
        return segments if segments else [transcript]
    
    # Otherwise split on double newlines
    segments = [s.strip() for s in transcript.split("\n\n") if s.strip()]
    return segments if segments else [transcript]
