from fastapi import FastAPI, UploadFile, File
from app.ingestion import process_pdf_files

app = FastAPI()

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        chunks = process_pdf_files(file.filename, content)
        results.append({
            "filename": file.filename,
            "chunks_created": len(chunks)
        })
    return {"status": "success", "files": results}
