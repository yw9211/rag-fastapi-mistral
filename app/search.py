from sentence_transformers import SentenceTransformer
import numpy as np
import re
from nltk.corpus import stopwords
from typing import List, Dict
from app.storage import get_all_chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
    
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

def filter_chunks_by_keywords(chunks: List[Dict], keywords: List[str]) -> List[Dict]:
    """
    Filter document chunks to retain only those containing at least one keyword.

    Args:
        chunks (List[Dict]): List of text chunks with metadata.
        keywords (List[str]): List of keywords to match.

    Returns:
        List[Dict]: Filtered list of chunks containing the keywords.
    """
    filtered = []
    for chunk in chunks:
        text = chunk["text"].lower()
        if any(keyword in text for keyword in keywords):
            filtered.append(chunk)
    return filtered


def search_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform hybrid search: filter chunks by keyword match, then re-rank using semantic similarity.

    Args:
        query (str): User query string.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict]: Top-k relevant chunks based on hybrid scoring.
    """
    query_vec = model.encode(query)
    keywords = extract_keywords(query)

    all_chunks = get_all_chunks()
    filtered_chunks = filter_chunks_by_keywords(all_chunks, keywords)
    candidate_chunks = filtered_chunks if filtered_chunks else all_chunks

    scored = []
    for chunk in candidate_chunks:
        chunk_vec = np.array(chunk["embedding"])
        score = cosine_similarity(query_vec, chunk_vec)
        scored.append((score, chunk))

    top_chunks = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    return [chunk for _, chunk in top_chunks]
