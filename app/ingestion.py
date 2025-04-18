import os
from PyPDF2 import PdfReader

# Global config for chunking (characters)
CHUNK_SIZE = 500  
OVERLAP_SIZE = 100  

# Save uploaded PDF content to a temporary file on disk
def save_pdf_temp(filename, content):
    path = f"data/{filename}"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path

# Extract text from a PDF file given its path
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Split the full text into smaller chunks for processing
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE):
    assert overlap < chunk_size, "Overlap must be smaller than chunk size"
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap 
    return chunks

# Full ingestion pipeline: save file, extract text, and chunk it
def process_pdf_files(filename, content):
    path = save_pdf_temp(filename, content)
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    return chunks
