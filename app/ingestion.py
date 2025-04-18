import os
from PyPDF2 import PdfReader

CHUNK_SIZE = 500  # characters

def save_pdf_temp(filename, content):
    path = f"data/{filename}"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def process_pdf_files(filename, content):
    path = save_pdf_temp(filename, content)
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    return chunks
