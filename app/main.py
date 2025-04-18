from fastapi import FastAPI, UploadFile, File
from app.ingestion import process_pdf_files

# Create FastAPI instance
app = FastAPI()

# Define a POST endpoint at /upload to receive PDF files
@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        # Read file content asynchronously
        content = await file.read()
        # Extract and chunk text from the PDF
        chunks = process_pdf_files(file.filename, content)
        # Store the filename and number of chunks for this file
        results.append({
            "filename": file.filename,
            "chunks_created": len(chunks)
        })
    # Return a JSON response with a summary of the operation
    return {"status": "success", "files": results}
