from sentence_transformers import SentenceTransformer

# Load and cache the model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")