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

