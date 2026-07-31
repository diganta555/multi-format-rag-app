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
        # llama-3.1-8b-instant uses noticeably fewer tokens per response than
        # llama-3.3-70b-versatile while still being solid for grounded RAG
        # answers (it's synthesizing retrieved passages, not reasoning from
        # scratch). Override with GROQ_MODEL in .env/secrets if you want the
        # larger model as default instead.
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.llm = ChatGroq(api_key=groq_api_key, model=model_name, max_tokens=1024)
        self.model_name = model_name
        self._query_cache: dict = {}

    def set_model(self, model_name: str):
        """Switch models at runtime (e.g. from a UI dropdown) without restarting."""
        if model_name != self.model_name:
            groq_api_key = os.getenv("GROQ_API_KEY")
            self.llm = ChatGroq(api_key=groq_api_key, model=model_name, max_tokens=1024)
            self.model_name = model_name
            self._query_cache.clear()

    def ingest_new_documents(self, documents):
        self.vectorstore.add_documents(documents, save=True)
        self._query_cache.clear()

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
            f"Answer using only the context below; say so if it's not covered. "
            f"Cite sources by filename. Format any math as LaTeX "
            f"($inline$ or $$block$$), not plain-text carets/unicode.\n\n"
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
        cache_key = (query.strip().lower(), top_k, self.vectorstore.index.ntotal if self.vectorstore.index else 0)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        prompt, sources = self._build_prompt(query, top_k)
        if prompt is None:
            return {"answer": "No relevant documents found.", "sources": [], "error": None}
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
        except (RateLimitError, APIStatusError) as e:
            return {"answer": None, "sources": [], "error": self._describe_error(e)}
        result = {"answer": response.content, "sources": sources, "error": None}
        self._query_cache[cache_key] = result
        return result

    def stream_search_and_summarize(self, query: str, top_k: int = 5):
        """Generator yielding {"type": "answer_chunk"|"sources"|"error", ...} events."""
        cache_key = (query.strip().lower(), top_k, self.vectorstore.index.ntotal if self.vectorstore.index else 0)
        if cache_key in self._query_cache:
            cached = self._query_cache[cache_key]
            yield {"type": "answer_chunk", "text": cached["answer"]}
            yield {"type": "sources", "sources": cached["sources"]}
            return

        prompt, sources = self._build_prompt(query, top_k)
        if prompt is None:
            yield {"type": "answer_chunk", "text": "No relevant documents found."}
            yield {"type": "sources", "sources": []}
            return
        full_answer = ""
        try:
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "answer_chunk", "text": chunk.content}
        except (RateLimitError, APIStatusError) as e:
            yield {"type": "error", **self._describe_error(e)}
            return
        self._query_cache[cache_key] = {"answer": full_answer, "sources": sources, "error": None}
        yield {"type": "sources", "sources": sources}