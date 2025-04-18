from fastapi import FastAPI, UploadFile, File, Form
from app.ingestion import process_pdf_files
from app.query_transform import transform_query
from app.mistral_utils import is_search_query_llm

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

@app.post("/query")
async def query_knowledge_base(question: str = Form(...)):
    # TODO: waiting on API key
    # Use LLM to determine if the query requires knowledge base search
    # if not is_search_query_llm(question):
    #     return {"response": "NO NEED to search knowledge base"}

    # TODO: Transform the query (e.g., normalize or improve it)
    transformed = transform_query(question)

    # TODO: Search + RAG + call Mistral (coming next)

    # Placeholder for semantic search and answer generation
    return {"response": f"(Simulated response for query: '{transformed}')"}
