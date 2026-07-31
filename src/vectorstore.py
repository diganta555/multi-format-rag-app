import os
import re
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from src.embedding import EmbeddingPipeline

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.bm25 = None
        self._reranker = None
        self._reranker_model_name = reranker_model
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    @property
    def reranker(self) -> CrossEncoder:
        # Lazy-loaded: only pull the cross-encoder into memory once it's actually used.
        if self._reranker is None:
            print(f"[INFO] Loading re-ranker: {self._reranker_model_name}")
            self._reranker = CrossEncoder(self._reranker_model_name)
        return self._reranker

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------
    def build_from_documents(self, documents: List[Any]):
        """Full (re)build from a document set. Wipes any existing index."""
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        self.index = None
        self.metadata = []
        self.add_documents(documents, save=True)
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_documents(self, documents: List[Any], save: bool = True):
        if not documents:
            print("[INFO] No new documents to add.")
            return
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model,
                                      chunk_size=self.chunk_size,
                                      chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_document(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "file_type": chunk.metadata.get("file_type", "unknown"),
            }
            for chunk in chunks
        ]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self._rebuild_bm25()
        if save:
            self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        if embeddings.size == 0:
            return
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)

    def _rebuild_bm25(self):
        if not self.metadata:
            self.bm25 = None
            return
        tokenized = [_tokenize(m.get("text", "")) for m in self.metadata]
        self.bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            self._rebuild_bm25()
            print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")
        else:
            print(f"[INFO] No existing index at {self.persist_dir}; starting empty.")

    def list_sources(self) -> List[str]:
        return sorted({m.get("source", "unknown") for m in self.metadata})

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Pure vector search (kept for backward compatibility)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

    def hybrid_search(self, query_text: str, top_k: int = 5, fetch_k: int = 20, rerank: bool = True):
        """
        Vector search + BM25 keyword search, merged with reciprocal rank
        fusion, then optionally re-ranked by a cross-encoder for the final
        cut. This is the search path the RAG-search endpoint should use.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        n = self.index.ntotal
        fetch_k = min(fetch_k, n)

        # --- vector leg ---
        query_emb = self.model.encode([query_text]).astype("float32")
        vec_results = self.search(query_emb, top_k=fetch_k)
        vec_rank = {r["index"]: rank for rank, r in enumerate(vec_results)}

        # --- keyword leg ---
        bm25_rank = {}
        if self.bm25 is not None:
            scores = self.bm25.get_scores(_tokenize(query_text))
            top_bm25_idx = np.argsort(scores)[::-1][:fetch_k]
            bm25_rank = {int(idx): rank for rank, idx in enumerate(top_bm25_idx) if scores[idx] > 0}

        # --- reciprocal rank fusion ---
        k_const = 60
        all_idx = set(vec_rank) | set(bm25_rank)
        fused = []
        for idx in all_idx:
            score = 0.0
            if idx in vec_rank:
                score += 1.0 / (k_const + vec_rank[idx])
            if idx in bm25_rank:
                score += 1.0 / (k_const + bm25_rank[idx])
            fused.append((idx, score))
        fused.sort(key=lambda x: x[1], reverse=True)
        candidates = fused[:fetch_k]

        results = [
            {"index": idx, "fusion_score": score, "metadata": self.metadata[idx]}
            for idx, score in candidates
            if idx < len(self.metadata)
        ]

        if rerank and results:
            pairs = [[query_text, r["metadata"].get("text", "")] for r in results]
            ce_scores = self.reranker.predict(pairs)
            for r, s in zip(results, ce_scores):
                r["rerank_score"] = float(s)
            results.sort(key=lambda r: r["rerank_score"], reverse=True)

        return results[:top_k]