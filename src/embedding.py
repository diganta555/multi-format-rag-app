from typing import List, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import numpy as np

from src.data_loader import load_all_documents

# Row/record-based formats already produce one atomic Document per record
# (CSVLoader = one row, JSON loader = one record). Splitting those further
# just fragments a record across chunks for no benefit.
ATOMIC_TYPES = {"CSV", "JSON", "EXCEL"}


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_document(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        atomic = [d for d in documents if d.metadata.get("file_type") in ATOMIC_TYPES]
        splittable = [d for d in documents if d.metadata.get("file_type") not in ATOMIC_TYPES]

        split_chunks = splitter.split_documents(splittable) if splittable else []
        chunks = atomic + split_chunks
        print(f"[INFO] {len(atomic)} atomic record(s) kept whole, "
              f"{len(splittable)} document(s) split into {len(split_chunks)} chunk(s).")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks....")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embedding shape : {embeddings.shape}")
        return embeddings