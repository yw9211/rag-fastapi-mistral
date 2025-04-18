from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from sentence_transformers import SentenceTransformer
from app.ingestion import process_pdf_files
from app.mistral_utils import is_search_query_llm, transform_query
from app.storage import add_chunks  
from app.search import search_chunks

# Create FastAPI instance
app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a POST endpoint at /upload to receive PDF files
@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    chunk_size: int = Query(500, description="Number of characters per chunk"),
    overlap: int = Query(100, description="Number of overlapping characters per chunk"),
):
    results = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are supported."
            )
        # Read file content asynchronously
        content = await file.read()
        # Extract and chunk text from the PDF
        chunks = process_pdf_files(file.filename, content, chunk_size=chunk_size, overlap=overlap)
        # Embed all chunks at once
        embeddings = model.encode(chunks).tolist()
        add_chunks(file.filename, chunks, embeddings)
        # Store the filename and number of chunks for this file
        results.append({
            "filename": file.filename,
            "chunks_created": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap
        })
    # Return a JSON response with a summary of the operation
    return {"status": "success", "files": results}

@app.post("/query")
async def query_knowledge_base(question: str = Form(...)):
    # Step 1: Transform the query for clarity (TODO: waiting on API key)
    #transformed = transform_query(question)
    transformed = question

    # Step 2: Decide if KB search is needed (TODO: waiting on API key)
    # use_kb = is_search_query_llm(transformed)
    use_kb = True

    # Step 3: If no KB needed, just ask LLM directly (TODO: waiting on API key)
    if not use_kb:
        return {
            "response": f"(LLM would answer directly: '{transformed}')",
            "used_knowledge_base": False
        }

    # Step 4: Search the knowledge base
    top_chunks = search_chunks(transformed, top_k=3)
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    # Step 5: Post-processing

    # Step 6: Ask LLM with extra context (TODO: waiting on API key)
    # response = generate_response(query=transformed, context=context)
    return {
        "response": f"(LLM would answer using context: '{transformed}')",
        "used_knowledge_base": True,
        "context_preview": context[:1000],
        "sources": [chunk["filename"] for chunk in top_chunks]
    }





