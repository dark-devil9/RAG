# main.py
import httpx
import pypdfium2 as pdfium
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastembed import TextEmbedding
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import asyncio # Import asyncio for concurrent operations
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Environment and API Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Asynchronous HTTP client for making API calls to the LLM
# This is more efficient in an async FastAPI application.
async_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hyper-Optimized PDF RAG Service",
    description="A highly optimized, single-endpoint RAG service that processes a PDF and answers multiple questions concurrently.",
)

# --- Pydantic Models for API I/O ---
class QuestionPayload(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOut(BaseModel):
    answers: List[str]

# --- Global In-Memory Cache ---
pdf_cache: Dict[str, Dict[str, Any]] = {}

# --- Model Loading ---
# This is a CPU-bound operation, so it's done once at startup.
print("[INFO] Initializing TextEmbedding model (fastembed)...")
embedding_model = TextEmbedding()
print("[INFO] Model loaded successfully.")


# --- Core Logic: Optimized PDF Processing ---
def process_and_index_pdf(pdf_url: str):
    """
    Downloads a PDF, extracts text, chunks it, and creates a FAISS index.
    This is a synchronous function designed to be run in a separate thread
    to avoid blocking the main asyncio event loop.
    """
    global pdf_cache
    print(f"[INFO] Starting background processing for PDF: {pdf_url}")
    total_start_time = time.time()

    try:
        # 1. Download PDF content
        download_start_time = time.time()
        with httpx.Client() as http_client:
            response = http_client.get(str(pdf_url), timeout=60.0)
            response.raise_for_status()
        pdf_bytes = response.content
        print(f"[TIMER] PDF downloaded in {time.time() - download_start_time:.2f}s")

        # 2. Extract text with pypdfium2
        extract_start_time = time.time()
        pdf_doc = pdfium.PdfDocument(pdf_bytes)
        full_text = "\n".join(page.get_textpage().get_text_range() for page in pdf_doc)
        print(f"[TIMER] Text extracted in {time.time() - extract_start_time:.2f}s")

        if not full_text.strip():
             pdf_cache[pdf_url] = {"index": None, "chunks": [], "timestamp": time.time()}
             return

        # 3. Chunk the text
        chunk_start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200, length_function=len
        )
        text_chunks = text_splitter.split_text(full_text)
        print(f"[TIMER] Text split into {len(text_chunks)} chunks in {time.time() - chunk_start_time:.2f}s")

        # 4. Generate embeddings
        embed_start_time = time.time()
        vectors = list(embedding_model.embed(text_chunks))
        vectors_np = np.array(vectors, dtype="float32")
        print(f"[TIMER] Embeddings generated in {time.time() - embed_start_time:.2f}s")
        
        # 5. Create and populate FAISS index
        index_start_time = time.time()
        index = faiss.IndexFlatL2(vectors_np.shape[1])
        index.add(vectors_np)
        print(f"[TIMER] FAISS index created in {time.time() - index_start_time:.2f}s")

        # 6. Store in cache
        pdf_cache[pdf_url] = {"index": index, "chunks": text_chunks, "timestamp": time.time()}
        print(f"[SUCCESS] PDF processed in {time.time() - total_start_time:.2f}s")

    except Exception as e:
        print(f"[ERROR] An error occurred during PDF processing: {e}")
        # Store a failure state to avoid retrying a broken PDF
        pdf_cache[pdf_url] = {"index": None, "chunks": [], "error": str(e)}


# --- Helper function for concurrent LLM calls ---
async def get_answer_for_question(question: str, index: faiss.Index, chunks: List[str]) -> str:
    """
    Asynchronously gets an answer for a single question.
    """
    try:
        # Embed the question
        question_embedding = np.array(list(embedding_model.embed([question])), dtype="float32")
        
        # Search the FAISS index
        k = 5
        _, indices = index.search(question_embedding, k)
        context = "\n\n---\n\n".join(chunks[i] for i in indices[0])

        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        models_to_try = ["moonshotai/kimi-k2:free", "mistralai/mistral-7b-instruct:free"]

        for model in models_to_try:
            try:
                # Use the async client for non-blocking API calls
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[WARNING] LLM API call failed for model {model}: {e}")
                continue
        
        return "Error: All LLM providers failed for this question."

    except Exception as e:
        print(f"[ERROR] Failed to process question '{question}': {e}")
        return "Error: Could not process this question."


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=AnswerOut)
async def run_rag(payload: QuestionPayload):
    """
    Receives a PDF URL and questions, processes the PDF in the background if not cached,
    and then fetches answers for all questions CONCURRENTLY.
    """
    if not api_key:
        raise HTTPException(status_code=503, detail="OpenRouter API key is not configured.")
        
    pdf_url_str = str(payload.documents)
    
    if pdf_url_str not in pdf_cache:
        # Run the synchronous, CPU-bound processing in a separate thread
        # to avoid blocking the main server process.
        await asyncio.to_thread(process_and_index_pdf, pdf_url_str)
    else:
        print(f"[INFO] Using cached index for {pdf_url_str}")

    # Wait a moment to ensure the cache is populated if processing just started
    await asyncio.sleep(0.1) 
    
    cached_data = pdf_cache.get(pdf_url_str)
    if not cached_data or "error" in cached_data:
        error_detail = cached_data.get("error", "Unknown processing error.") if cached_data else "Cache not found."
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {error_detail}")

    index = cached_data["index"]
    chunks = cached_data["chunks"]

    if not chunks or index is None:
        return AnswerOut(answers=["I could not extract any text from the provided PDF."] * len(payload.questions))

    tasks = [get_answer_for_question(q, index, chunks) for q in payload.questions]
    
    print(f"[INFO] Sending {len(tasks)} questions to LLM providers concurrently...")
    start_time = time.time()
    final_answers = await asyncio.gather(*tasks)
    print(f"[TIMER] All {len(tasks)} answers received in {time.time() - start_time:.2f} seconds.")

    return AnswerOut(answers=final_answers)

@app.get("/")
def health_check():
    """A simple health check endpoint."""
    return {"status": "ok", "cached_pdfs": list(pdf_cache.keys())}

# To run this app:
# 1. Install dependencies: pip install "fastapi[all]" "uvicorn[standard]" httpx pypdfium2 "fastembed>=0.2.0" faiss-cpu "openai>=1.0.0" python-dotenv langchain
# 2. Create a .env file with your OPENAI_API_KEY
# 3. Run with uvicorn: uvicorn main:app --reload
