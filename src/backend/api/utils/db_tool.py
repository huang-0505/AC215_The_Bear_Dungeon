"""
db_tool.py
Utilities to create and manage embedding databases for action/target retrieval.
Uses Google GenAI for text embeddings and cosine similarity for semantic search.
"""

import os
import json
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from typing import List


# ========== Configuration ==========
class DBConfig:
    """Configuration for database and embedding settings."""

    def __init__(self):
        # GCP Settings (from environment variables)
        self.gcp_project = os.environ.get("GCP_PROJECT")
        if not self.gcp_project:
            raise ValueError("GCP_PROJECT environment variable must be set")

        self.gcp_location = os.environ.get("GCP_LOCATION", "us-central1")

        # Model Settings
        self.embedding_model = "text-embedding-004"
        self.embedding_dimension = 256


# ========== Embedding Generator ==========
class EmbeddingGenerator:
    """Handles text embedding generation using Google GenAI."""

    def __init__(self, config: DBConfig):
        self.config = config
        self.client = genai.Client(vertexai=True, project=config.gcp_project, location=config.gcp_location)

    def generate_single(self, query: str) -> np.ndarray:
        """Generate embedding for a single query string."""
        response = self.client.models.embed_content(
            model=self.config.embedding_model,
            contents=query,
            config=types.EmbedContentConfig(output_dimensionality=self.config.embedding_dimension),
        )
        return np.array(response.embeddings[0].values)


# ========== Similarity Calculator ==========
def cosine_similarity(query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and corpus embeddings."""
    query_norm = query_emb / np.linalg.norm(query_emb)
    corpus_norm = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)
    return np.dot(corpus_norm, query_norm)


# ========== Convenience Functions ==========
def retrieve_top_k(query: str, embeddings_file: str = "data/embeddings-actions.jsonl", k: int = 5) -> List[int]:
    """
    Convenience function for quick retrieval without config setup.

    Args:
        query: Natural language query
        embeddings_file: Path to embeddings database
        k: Number of results to return

    Returns:
        List of top-k IDs
    """
    config = DBConfig()
    generator = EmbeddingGenerator(config)

    # Load embeddings database
    df = pd.read_json(embeddings_file, lines=True)
    embeddings = np.vstack(df["embedding"].values)

    # Generate query embedding
    query_emb = generator.generate_single(query)

    # Calculate similarities
    sims = cosine_similarity(query_emb, embeddings)

    # Get top-k indices
    topk_idx = np.argsort(sims)[-k:][::-1]

    # Return top-k IDs
    results = df.iloc[topk_idx].copy()

    return results["id"].tolist()
