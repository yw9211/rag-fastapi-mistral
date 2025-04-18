import os
from pypdf import PdfReader

def save_pdf_temp(filename, content):
    """
    Save uploaded PDF content to a local file in the 'data/' directory.

    Args:
        filename (str): The name of the file to save.
        content (bytes): The raw PDF content.

    Returns:
        str: The path to the saved file.
    """
    path = f"data/{filename}"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path

def extract_text_from_pdf(path):
    """
    Extract all text from a PDF file using pypdf.

    Args:
        path (str): File path to the PDF.

    Returns:
        str: Combined text from all pages.
    """
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split a string of text into overlapping chunks.

    Args:
        text (str): The full text to chunk.
        chunk_size (int): Number of characters per chunk.
        overlap (int): Number of characters to overlap between chunks.

    Returns:
        list[str]: List of overlapping text chunks.
    """
    assert overlap < chunk_size, "Overlap must be smaller than chunk size"
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap 
    return chunks

def process_pdf_files(filename, content, chunk_size=500, overlap=100):
    """
    Complete pipeline to process a PDF: save, extract text, and chunk.

    Args:
        filename (str): The name of the uploaded file.
        content (bytes): The raw PDF content.

    Returns:
        list[str]: List of text chunks extracted from the PDF.
    """
    path = save_pdf_temp(filename, content)
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    return chunks
