from typing import List, Dict

CHUNK_STORE: List[Dict] = []

def add_chunks(filename: str, chunks: List[str], embeddings: List[List[float]]):
    """
    Store the given chunks in memory with filename and embedding.
    """
    for chunk, emb in zip(chunks, embeddings):
        CHUNK_STORE.append({
            "filename": filename,
            "text": chunk,
            "embedding": emb  # stored as list of floats
        })

def get_all_chunks() -> List[Dict]:
    """
    Return all stored text chunks.
    """
    return CHUNK_STORE