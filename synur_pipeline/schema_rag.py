# RAG for schema filtering. It embeds transcript segments and schema rows, computes cosine similarity, and returns top-N most relevant schema concepts.

import json
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from rank_bm25 import BM25Okapi


@dataclass
class SchemaRow:

    id: str
    name: str
    value_type: str
    value_enum: list = None
    
    # Create text representation for embedding
    def to_embedding_text(self):
        text = self.name
        if self.value_enum:
            text += f" (possible values: {', '.join(self.value_enum)})"
        return text
    
    # Convert to dictionary for JSON serialization
    def to_dict(self):
        d = {
            "id": self.id,
            "name": self.name,
            "value_type": self.value_type
        }
        if self.value_enum:
            d["value_enum"] = self.value_enum
        return d

# Pre-compute embeddings for schema rows and retrieve relevant ones for a segment
class SchemaRAG:
    
    OPENAI_MODELS = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
    
    def __init__(self, schema_path, embedding_model="text-embedding-3-small"):
        self.schema_path = schema_path
        self.embedding_model_name = embedding_model
        self.is_openai = embedding_model in self.OPENAI_MODELS
        
        if self.is_openai:
            from openai import OpenAI
            self.openai_client = OpenAI()
            self.model = None
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            self.openai_client = None
        
        # Load and parse schema
        self.schema_rows = self._load_schema()
        
        # Pre-compute embeddings for all schema rows
        self.schema_embeddings = self._embed_schema()
    
    def _embed_texts(self, texts):

        if self.is_openai:
            # OpenAI batch embedding
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=texts
            )
            embeddings = np.array([d.embedding for d in response.data])
        else:
            # sentence-transformers
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def _embed_single(self, text):

        if self.is_openai:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=[text]
            )
            embedding = np.array(response.data[0].embedding)
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
        
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
        
    def _load_schema(self):

        with open(self.schema_path, "r", encoding="utf-8") as f:
            raw_schema = json.load(f)
        
        rows = []
        for item in raw_schema:
            row = SchemaRow(
                id=item["id"],
                name=item["name"],
                value_type=item["value_type"],
                value_enum=item.get("value_enum")
            )
            rows.append(row)
        
        return rows
    
    def _embed_schema(self):

        texts = [row.to_embedding_text() for row in self.schema_rows]
        return self._embed_texts(texts)
    
    def retrieve(self, segment, top_n=60):

        # Embed the segment
        segment_embedding = self._embed_single(segment)
        
        # Compute cosine similarities
        similarities = np.dot(self.schema_embeddings, segment_embedding)
        
        # Get top-N indices
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # Return corresponding schema rows
        return [self.schema_rows[i] for i in top_indices]
    
    def retrieve_with_scores(self, segment, top_n=60):

        # Embed the segment
        segment_embedding = self._embed_single(segment)
        
        # Compute cosine similarities
        similarities = np.dot(self.schema_embeddings, segment_embedding)
        
        # Get top-N indices
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # Return corresponding schema rows with scores
        return [(self.schema_rows[i], similarities[i]) for i in top_indices]
    
    def get_schema_row_by_id(self, schema_id):

        for row in self.schema_rows:
            if row.id == schema_id:
                return row
        return None
    
    def format_schema_for_prompt(self, rows):

        schema_list = [row.to_dict() for row in rows]
        return json.dumps(schema_list, indent=2)


# Hybrid retrieval combining dense embeddings (semantic) with BM25 (lexical)
class HybridSchemaRAG(SchemaRAG):
    
    def __init__(self, schema_path, embedding_model="text-embedding-3-small", alpha=0.6):
        # Initialize dense retrieval from parent class
        super().__init__(schema_path, embedding_model)
        self.alpha = alpha
        
        # Build BM25 index for sparse retrieval
        self.tokenized_corpus = [
            self._tokenize(row.to_embedding_text()) 
            for row in self.schema_rows
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text):

        return text.lower().split()
    
    def _normalize_scores(self, scores):

        scores = np.array(scores)
        if scores.max() - scores.min() == 0:
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    def retrieve(self, segment, top_n=60):

        # Dense scores (cosine similarity)
        segment_embedding = self._embed_single(segment)
        dense_scores = np.dot(self.schema_embeddings, segment_embedding)
        
        # Sparse scores (BM25)
        tokenized_query = self._tokenize(segment)
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize and combine
        dense_norm = self._normalize_scores(dense_scores)
        sparse_norm = self._normalize_scores(sparse_scores)
        hybrid_scores = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm
        
        # Get top-N indices
        top_indices = np.argsort(hybrid_scores)[-top_n:][::-1]
        
        return [self.schema_rows[i] for i in top_indices]
    
    def retrieve_with_scores(self, segment, top_n=60):

        segment_embedding = self._embed_single(segment)
        dense_scores = np.dot(self.schema_embeddings, segment_embedding)
        
        tokenized_query = self._tokenize(segment)
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        dense_norm = self._normalize_scores(dense_scores)
        sparse_norm = self._normalize_scores(sparse_scores)
        hybrid_scores = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm
        
        top_indices = np.argsort(hybrid_scores)[-top_n:][::-1]
        
        return [(self.schema_rows[i], hybrid_scores[i]) for i in top_indices]


# Embeds the training transcripts and finds the most similar ones for few-shot example selection
class FewShotRAG:
    
    OPENAI_MODELS = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
    
    def __init__(self, train_data, embedding_model="text-embedding-3-small"):

        self.train_data = train_data
        self.embedding_model_name = embedding_model
        self.is_openai = embedding_model in self.OPENAI_MODELS
        
        if self.is_openai:
            from openai import OpenAI
            self.openai_client = OpenAI()
            self.model = None
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            self.openai_client = None
        
        # Pre-compute embeddings for all training transcripts
        self.transcript_embeddings = self._embed_transcripts()
    
    def _embed_texts(self, texts):

        if self.is_openai:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=texts
            )
            embeddings = np.array([d.embedding for d in response.data])
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def _embed_single(self, text):

        if self.is_openai:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=[text]
            )
            embedding = np.array(response.data[0].embedding)
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
        
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _embed_transcripts(self):

        transcripts = [item['transcript'] for item in self.train_data]
        return self._embed_texts(transcripts)
    
    def retrieve(self, transcript, top_n=2):

        query_embedding = self._embed_single(transcript)
        similarities = np.dot(self.transcript_embeddings, query_embedding)
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Collect top_n examples, skipping exact matches (similarity > 0.99)
        results = []
        for idx in sorted_indices:
            if similarities[idx] > 0.99:
                # Skip near-exact matches (likely the same transcript)
                continue
            results.append(self.train_data[idx])
            if len(results) >= top_n:
                break
        
        return results
