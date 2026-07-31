import os
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from groq import RateLimitError, APIStatusError
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

_RETRY_PATTERN = re.compile(r"try again in ([\d.hms]+)", re.IGNORECASE)


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
        self.llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", max_tokens=1024)

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
            f"by filename where relevant. When presenting mathematical formulas, "
            f"equations, or expressions, format them using LaTeX syntax wrapped in "
            f"single dollar signs for inline math (e.g. $E = mc^2$) or double dollar "
            f"signs for standalone equations (e.g. $$\\text{{Attention}}(Q,K,V) = "
            f"\\text{{softmax}}(\\frac{{QK^T}}{{\\sqrt{{d_k}}}})V$$), rather than plain "
            f"text with carets or unicode symbols.\n\n"
            f"Query: '{query}'\n\nContext:\n{context}"
        )
        return prompt, sorted(set(os.path.basename(s) for s in sources))

    @staticmethod
    def _describe_error(e: Exception) -> dict:
        """Turn a Groq exception into a structured, frontend-friendly error dict."""
        if isinstance(e, RateLimitError):
            detail = getattr(e, "message", str(e))
            match = _RETRY_PATTERN.search(detail)
            retry_in = match.group(1) if match else None
            return {
                "kind": "rate_limit",
                "title": "Daily usage limit reached",
                "message": (
                    "Groq's free tier caps how much can be generated per day, and "
                    "today's limit has been hit."
                    + (f" Try again in {retry_in}." if retry_in else " Try again later.")
                ),
                "retry_in": retry_in,
                "raw": detail,
            }
        if isinstance(e, APIStatusError):
            return {
                "kind": "api_error",
                "title": "AI service error",
                "message": getattr(e, "message", str(e)),
                "retry_in": None,
                "raw": str(e),
            }
        return {
            "kind": "unknown",
            "title": "Unexpected error",
            "message": str(e),
            "retry_in": None,
            "raw": str(e),
        }

    def search_and_summarize(self, query: str, top_k: int = 5) -> dict:
        prompt, sources = self._build_prompt(query, top_k)
        if prompt is None:
            return {"answer": "No relevant documents found.", "sources": [], "error": None}
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
        except (RateLimitError, APIStatusError) as e:
            return {"answer": None, "sources": [], "error": self._describe_error(e)}
        return {"answer": response.content, "sources": sources, "error": None}

    def stream_search_and_summarize(self, query: str, top_k: int = 5):
        """Generator yielding {"type": "answer_chunk"|"sources"|"error", ...} events."""
        prompt, sources = self._build_prompt(query, top_k)
        if prompt is None:
            yield {"type": "answer_chunk", "text": "No relevant documents found."}
            yield {"type": "sources", "sources": []}
            return
        try:
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                if chunk.content:
                    yield {"type": "answer_chunk", "text": chunk.content}
        except (RateLimitError, APIStatusError) as e:
            yield {"type": "error", **self._describe_error(e)}
            return
        yield {"type": "sources", "sources": sources}