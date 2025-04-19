from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from app.ingestion import process_pdf_files
from app.mistral_utils import is_search_query_llm, transform_query, generate_response, embed_chunks_mistral
from app.storage import add_chunks  
from app.search import search_chunks
from app.postprocessing import deduplicate_chunks, truncate_chunks

# Create FastAPI instance
app = FastAPI()

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
        embeddings = embed_chunks_mistral(chunks)
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
    # Step 1: Transform the query to improve retrieval
    transformed = transform_query(question)

    # Step 2: Determine if KB search is needed 
    use_kb = is_search_query_llm(transformed)

    # Step 3: If no KB needed, send the query directly to LLM
    if not use_kb:
        response = generate_response(query=question, context="")
        return response

    # Step 4: Retrieve top chunks from knowledge base
    top_chunks = search_chunks(transformed, top_k=5)

    # Step 5: Post-processing of chunks
    top_chunks = deduplicate_chunks(top_chunks)
    top_chunks = truncate_chunks(top_chunks, max_chars=3000)
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    # Step 6: Generate LLM response using context
    response = generate_response(query=transformed, context=context)
    return response

@app.post("/debug_query")
async def debug_query(question: str = Form(...)):
    debug = {}

    # Step 1: Transform the query to improve retrieval
    transformed = transform_query(question)
    debug["original_query"] = question
    debug["transformed_query"] = transformed

    # Step 2: Determine if KB search is needed 
    use_kb = is_search_query_llm(question) 
    debug["used_knowledge_base"] = use_kb

    # Step 3: If no KB needed, send the query directly to LLM
    if not use_kb:
        response = generate_response(query=question, context="")
        debug["response"] = response
        debug["context_used"] = None
        debug["top_chunks"] = []
        return debug

    # Step 4: Retrieve top chunks from knowledge base
    top_chunks = search_chunks(transformed, top_k=5)
    debug["initial_top_chunks"] = [
        {
            "filename": chunk["filename"],
            "text_preview": chunk["text"][:250]
        }
        for chunk in top_chunks
    ]

    # Step 5: Post-processing of chunks
    top_chunks = deduplicate_chunks(top_chunks)
    debug["after_deduplication"] = [chunk["text"][:250] for chunk in top_chunks]

    top_chunks = truncate_chunks(top_chunks, max_chars=3000)
    debug["after_truncation"] = [chunk["text"][:250] for chunk in top_chunks]

    # Step 6: Generate LLM response using context
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])
    debug["top_chunks_count"] = len(top_chunks)

    response = generate_response(query=transformed, context=context)
    debug["response"] = response

    return debug


