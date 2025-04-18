from app.ingestion import chunk_text

def test_chunk_text_basic():
    text = "abcdefghij" * 10  # 100 characters
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)

def test_chunk_text_overlap_correctness():
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert chunks[1].startswith(chunks[0][-20:])  # ensure overlap
