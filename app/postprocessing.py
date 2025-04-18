from typing import List, Dict
from app.search import extract_keywords

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Remove exact duplicate chunks based on their text content.

    Args:
        chunks (List[Dict]): List of chunks with 'text' field.

    Returns:
        List[Dict]: Chunks with duplicates removed.
    """
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        text = chunk["text"]
        if text not in seen:
            seen.add(text)
            unique_chunks.append(chunk)
    return unique_chunks

def keyword_score(text: str, keywords: List[str]) -> int:
    """
    Count how many query keywords appear in the given chunk text.

    Args:
        text (str): Chunk text.
        keywords (List[str]): List of keywords.

    Returns:
        int: Keyword overlap count.
    """
    return sum(1 for kw in keywords if kw in text.lower())

def rerank_chunks(chunks: List[Dict], query: str) -> List[Dict]:
    """
    Re-rank chunks based on a combination of keyword score and original semantic rank.

    Args:
        chunks (List[Dict]): Top-k chunks from initial search.
        query (str): Transformed query string.

    Returns:
        List[Dict]: Re-ranked list of chunks.
    """
    keywords = extract_keywords(query)
    ranked = []
    for i, chunk in enumerate(chunks):
        score = keyword_score(chunk["text"], keywords)
        final_score = score + (len(chunks) - i) * 0.01  # preserve semantic rank
        ranked.append((final_score, chunk))

    reranked = sorted(ranked, key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in reranked]

def truncate_chunks(chunks: List[Dict], max_chars: int = 3000) -> List[Dict]:
    """
    Truncate chunk list so combined text length stays under a limit.

    Args:
        chunks (List[Dict]): List of ranked chunks.
        max_chars (int): Maximum allowed combined character length.

    Returns:
        List[Dict]: Truncated list of chunks that fit within limit.
    """
    total = 0
    result = []
    for chunk in chunks:
        chunk_len = len(chunk["text"])
        if total + chunk_len <= max_chars:
            result.append(chunk)
            total += chunk_len
        else:
            break
    return result

