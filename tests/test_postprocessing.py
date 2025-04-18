from app.postprocessing import deduplicate_chunks, truncate_chunks

def test_deduplicate_chunks_removes_duplicates():
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "a"}]
    unique = deduplicate_chunks(chunks)
    assert len(unique) == 2
    assert {"text": "a"} in unique
    assert {"text": "b"} in unique

def test_truncate_chunks_limit_characters():
    chunks = [{"text": "a" * 1000}, {"text": "b" * 1000}, {"text": "c" * 1000}]
    result = truncate_chunks(chunks, max_chars=2000)
    assert len(result) == 2
