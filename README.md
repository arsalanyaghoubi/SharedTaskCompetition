# Lakefront_AI_Ramblers-MEDIQA_SYNUR_2026

This repository contains our submission for the **MEDIQA-SYNUR 2026 Shared Task** on clinical observation extraction from nursing transcripts.

Prepared by msabanluc on behalf of **Team Lakefront AI Ramblers**

## Pipeline Overview

Our approach uses a multi-stage pipeline, similar to the methods discussed in the base paper:

1. **Segmentation** - Split transcripts into manageable segments
2. **Schema RAG** - Hybrid BGE + BM25 retrieval to identify relevant schema items
3. **Extraction** - LLM-based observation extraction
4. **Self-Consistency Voting** - Aggregate predictions across multiple inference runs
5. **Verification** - LLM-based verification pass to filter hallucinations

See `synur_pipeline/` for implementation details.


