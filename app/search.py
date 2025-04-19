from app.embedding_model import embedding_model as model
import numpy as np
import re
from nltk.corpus import stopwords
from typing import List, Dict
from app.storage import get_all_chunks
    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score between 0 and 1.
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

STOPWORDS = set(stopwords.words("english"))


def extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords by removing stopwords and short tokens.

    Args:
        text (str): User query text.

    Returns:
        List[str]: A list of extracted keywords.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]

def search_chunks(query: str, top_k: int = 5, alpha: float = 0.75) -> List[Dict]:
    """
    Perform hybrid search using a weighted combination of semantic similarity and keyword overlap.

    Args:
        query (str): User query string.
        top_k (int): Number of top results to return.
        alpha (float): Weight for semantic similarity (between 0 and 1).

    Returns:
        List[Dict]: Top-k most relevant chunks based on combined scoring.
    """
    query_vec = model.encode(query)
    keywords = extract_keywords(query)

    scored = []
    all_chunks = get_all_chunks()

    for chunk in all_chunks:
        chunk_vec = np.array(chunk["embedding"])
        semantic_score = cosine_similarity(query_vec, chunk_vec)
        keyword_overlap = sum(1 for kw in keywords if kw in chunk["text"].lower())

        # Normalize keyword score 
        keyword_score = keyword_overlap / (len(keywords) or 1)

        # Combine scores
        final_score = alpha * semantic_score + (1 - alpha) * keyword_score
        chunk["semantic_score"] = semantic_score
        chunk["keyword_score"] = keyword_score
        chunk["final_score"] = final_score
        scored.append((final_score, chunk))

    top_chunks = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    return [chunk for _, chunk in top_chunks]
