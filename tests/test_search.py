import numpy as np
import pytest
from app.search import cosine_similarity, extract_keywords

def test_cosine_similarity_basic():
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([1.0, 0.0])
    assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)

def test_cosine_similarity_zero_vector():
    vec1 = np.array([0.0, 0.0])
    vec2 = np.array([1.0, 0.0])
    assert cosine_similarity(vec1, vec2) == 0.0

def test_extract_keywords_removes_stopwords():
    text = "What is the benefit of Buddy Bites?"
    keywords = extract_keywords(text)
    assert "buddy" in keywords
    assert "bites" in keywords
    assert "what" not in keywords
    assert "is" not in keywords
