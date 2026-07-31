"""
FastAPI backend for the RAG document search app.

Run from the project root (so `src` and `data` resolve correctly):
    uvicorn backend.app:app --reload --port 8000

Env vars (see .env.example):
    GROQ_API_KEY   - required, your Groq API key
"""
import os
import json
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.data_loader import LOADER_REGISTRY, load_all_documents
from src.search import RAGSearch

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG Document Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded lazily so the API can still start (e.g. for health checks) even if
# GROQ_API_KEY isn't set yet.
_rag: RAGSearch | None = None


def get_rag() -> RAGSearch:
    global _rag
    if _rag is None:
        _rag = RAGSearch()
    return _rag


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/api/health")
def health():
    return {"status": "ok", "supported_extensions": sorted(LOADER_REGISTRY.keys())}


@app.get("/api/files")
def list_files():
    rag = get_rag()
    indexed = rag.vectorstore.list_sources()
    return {
        "indexed_sources": [os.path.basename(s) for s in indexed],
        "total_chunks": len(rag.vectorstore.metadata),
    }


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    rag = get_rag()
    results = []
    new_documents = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in LOADER_REGISTRY:
            results.append({"filename": f.filename, "status": "skipped",
                             "reason": f"unsupported extension {ext}"})
            continue

        dest = UPLOAD_DIR / f.filename
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)

            loader_fn, label = LOADER_REGISTRY[ext]
            docs = loader_fn(dest)
            for d in docs:
                d.metadata.setdefault("source", str(dest))
                d.metadata.setdefault("file_type", label)
            new_documents.extend(docs)
            results.append({"filename": f.filename, "status": "loaded", "chunks_source_docs": len(docs)})
        except Exception as e:
            results.append({"filename": f.filename, "status": "error", "reason": str(e)})

    if new_documents:
        rag.ingest_new_documents(new_documents)

    return {"results": results, "total_indexed_chunks": len(rag.vectorstore.metadata)}


def _rebuild_from_disk():
    """IndexFlatL2 can't delete individual vectors, so removal = rebuild from
    whatever files remain under data/. Fine for small/medium corpora."""
    rag = get_rag()
    rag.vectorstore.index = None
    rag.vectorstore.metadata = []
    docs = load_all_documents("data")
    if docs:
        rag.vectorstore.build_from_documents(docs)
    else:
        rag.vectorstore.save()  # persist the now-empty index


@app.delete("/api/files/{filename}")
def delete_file(filename: str):
    rag = get_rag()
    matches = [p for p in Path("data").rglob(filename) if p.is_file()]
    if not matches:
        raise HTTPException(404, f"No file named '{filename}' found under data/")
    for p in matches:
        p.unlink()
    _rebuild_from_disk()
    return {"deleted": filename, "indexed_sources": rag.vectorstore.list_sources()}


@app.post("/api/reset")
def reset_library():
    """Wipe the index AND delete all uploaded files. Does not touch anything
    outside data/uploads."""
    rag = get_rag()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _rebuild_from_disk()
    return {"status": "reset", "indexed_sources": rag.vectorstore.list_sources()}


@app.post("/api/query")
def query(req: QueryRequest):
    rag = get_rag()
    if rag.vectorstore.index is None or rag.vectorstore.index.ntotal == 0:
        raise HTTPException(400, "No documents indexed yet. Upload files first.")
    result = rag.search_and_summarize(req.query, top_k=req.top_k)
    if result.get("error"):
        status = 429 if result["error"]["kind"] == "rate_limit" else 502
        raise HTTPException(status, detail=result["error"])
    return result


@app.post("/api/query/stream")
def query_stream(req: QueryRequest):
    rag = get_rag()
    if rag.vectorstore.index is None or rag.vectorstore.index.ntotal == 0:
        raise HTTPException(400, "No documents indexed yet. Upload files first.")

    def event_gen():
        for event in rag.stream_search_and_summarize(req.query, top_k=req.top_k):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# Serve the static frontend at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")