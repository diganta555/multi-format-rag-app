import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.load()

        if self.vectorstore.index is None and os.path.isdir("data"):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            if docs:
                self.vectorstore.build_from_documents(docs)

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to a .env file (see .env.example) — "
                "never hardcode API keys in source."
            )
        self.llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

    def ingest_new_documents(self, documents):
        self.vectorstore.add_documents(documents, save=True)

    def _build_prompt(self, query: str, top_k: int):
        results = self.vectorstore.hybrid_search(query, top_k=top_k)
        texts, sources = [], []
        for r in results:
            meta = r.get("metadata") or {}
            if meta.get("text"):
                texts.append(meta["text"])
                sources.append(meta.get("source", "unknown"))

        if not texts:
            return None, []

        context = "\n\n".join(
            f"[Source: {os.path.basename(src)}]\n{text}" for src, text in zip(sources, texts)
        )
        prompt = (
            f"Answer the query using only the context below. If the context doesn't "
            f"contain the answer, say so explicitly rather than guessing. Cite sources "
            f"by filename where relevant.\n\nQuery: '{query}'\n\nContext:\n{context}"
        )
        return prompt, sorted(set(os.path.basename(s) for s in sources))

    def search_and_summarize(self, query: str, top_k: int = 5) -> dict:
        prompt, sources = self._build_prompt(query, top_k)
        if prompt is None:
            return {"answer": "No relevant documents found.", "sources": []}
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"answer": response.content, "sources": sources}

    def stream_search_and_summarize(self, query: str, top_k: int = 5):
        """Generator yielding answer text chunks, then a final sources list."""
        prompt, sources = self._build_prompt(query, top_k)
        if prompt is None:
            yield {"type": "answer_chunk", "text": "No relevant documents found."}
            yield {"type": "sources", "sources": []}
            return
        for chunk in self.llm.stream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield {"type": "answer_chunk", "text": chunk.content}
        yield {"type": "sources", "sources": sources}