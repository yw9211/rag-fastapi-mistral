from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from app.storage import get_all_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search through stored chunks using semantic similarity.
    
    Returns top_k most similar chunks.
    """
    query_vec = model.encode(query)

    chunks = get_all_chunks()
    scored = []
    for chunk in chunks:
        chunk_vec = np.array(chunk["embedding"])
        score = cosine_similarity(query_vec, chunk_vec)
        scored.append((score, chunk))

    top_chunks = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    return [chunk for _, chunk in top_chunks]