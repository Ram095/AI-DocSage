"""
Embeddings and Similarity Search Utilities

This module handles:
- Text chunking (token-based and paragraph-based)
- Creating embeddings using OpenAI Embeddings API
- Cosine similarity search for RAG retrieval

Designed to be explicit (no LangChain) and pluggable for future vector DB integration.
"""

import re
from typing import List, Tuple, Optional
import numpy as np
from openai import OpenAI


# Chunking configuration
DEFAULT_CHUNK_SIZE = 600  # Target tokens per chunk (500-800 range)
CHUNK_OVERLAP = 100  # Overlap tokens between chunks
MIN_CHUNK_SIZE = 50  # Minimum tokens to form a chunk

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.
    Uses a simple heuristic: ~4 characters per token for English text.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def chunk_text_by_paragraphs(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Chunk text by paragraphs, respecting natural document boundaries.
    
    Args:
        text: Full document text
        chunk_size: Target tokens per chunk
        
    Returns:
        List of text chunks
    """
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        para_tokens = estimate_tokens(paragraph)
        
        # If single paragraph exceeds chunk size, split it further
        if para_tokens > chunk_size:
            # First, add accumulated chunk if any
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sent_tokens = estimate_tokens(sentence)
                if current_size + sent_tokens > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                current_chunk.append(sentence)
                current_size += sent_tokens
        
        # Check if adding this paragraph would exceed chunk size
        elif current_size + para_tokens > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_size = para_tokens
        else:
            current_chunk.append(paragraph)
            current_size += para_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def chunk_text_by_tokens(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk text by approximate token count with overlap.
    
    Args:
        text: Full document text
        chunk_size: Target tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Convert to words for easier manipulation
    words = text.split()
    
    # Estimate words per chunk (roughly 0.75 words per token)
    words_per_chunk = int(chunk_size * 0.75)
    overlap_words = int(overlap * 0.75)
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        # Only add if chunk has meaningful content
        if estimate_tokens(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append(chunk_text)
        
        # Move start with overlap
        start = end - overlap_words if end < len(words) else len(words)
    
    return chunks


def chunk_text(text: str, method: str = 'paragraph', chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Main chunking function - splits text into manageable chunks.
    
    Args:
        text: Full document text
        method: 'paragraph' or 'token'
        chunk_size: Target tokens per chunk
        
    Returns:
        List of text chunks
    """
    if method == 'paragraph':
        return chunk_text_by_paragraphs(text, chunk_size)
    elif method == 'token':
        return chunk_text_by_tokens(text, chunk_size)
    else:
        raise ValueError(f"Unknown chunking method: {method}")


def create_embeddings(client: OpenAI, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Create embeddings for a list of texts using OpenAI API.
    
    Args:
        client: OpenAI client instance
        texts: List of text strings to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of embedding vectors
        
    Raises:
        Exception: If API call fails
    """
    if not texts:
        return []
    
    embeddings = []
    
    # Process in batches to avoid rate limits (OpenAI allows up to 2048 inputs)
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            # Don't log the actual texts (could contain sensitive info)
            raise Exception(f"Failed to create embeddings for batch {i // batch_size + 1}: {str(e)}")
    
    return embeddings


def create_single_embedding(client: OpenAI, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Create embedding for a single text (used for queries).
    
    Args:
        client: OpenAI client instance
        text: Text string to embed
        model: OpenAI embedding model to use
        
    Returns:
        Embedding vector as list of floats
    """
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def similarity_search(
    query_embedding: List[float],
    chunk_embeddings: List[List[float]],
    chunks: List[str],
    top_k: int = 3,
    similarity_threshold: float = 0.3
) -> List[Tuple[str, float, int]]:
    """
    Find most similar chunks to a query using cosine similarity.
    
    This function is designed to be replaceable with a vector database later.
    The interface would remain the same, but implementation would change.
    
    Args:
        query_embedding: Embedding vector of the user query
        chunk_embeddings: List of chunk embedding vectors
        chunks: List of text chunks (same order as embeddings)
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score to include
        
    Returns:
        List of tuples: (chunk_text, similarity_score, chunk_index)
        Sorted by similarity in descending order
    """
    if not chunk_embeddings or not chunks:
        return []
    
    # Calculate similarities for all chunks
    similarities = []
    for idx, chunk_emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_emb)
        similarities.append((chunks[idx], score, idx))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Filter by threshold and limit to top_k
    results = [
        (text, score, idx) 
        for text, score, idx in similarities[:top_k]
        if score >= similarity_threshold
    ]
    
    return results


# ============================================================
# VECTOR DATABASE INTEGRATION POINT
# ============================================================
# 
# To integrate a vector database (e.g., Pinecone, Weaviate, Chroma):
# 
# 1. Create a new class that implements these methods:
#    - store_embeddings(chunks, embeddings, metadata)
#    - search(query_embedding, top_k, filters)
#    - delete_collection()
# 
# 2. Replace the in-memory storage in session_state with DB calls
# 
# 3. The similarity_search function above can be replaced with:
#    def similarity_search_db(query_embedding, collection_name, top_k):
#        results = db_client.query(
#            collection_name=collection_name,
#            query_vector=query_embedding,
#            top_k=top_k
#        )
#        return [(r.text, r.score, r.id) for r in results]
# ============================================================
