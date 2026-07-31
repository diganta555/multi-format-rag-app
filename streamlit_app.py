"""
Streamlit version of Stacks — no separate FastAPI backend needed.
Run locally with:
    streamlit run streamlit_app.py

Deploy free at https://share.streamlit.io by connecting this GitHub repo.
Set GROQ_API_KEY under the app's "Secrets" in Streamlit Cloud settings
(same key name as your local .env), formatted as:
    GROQ_API_KEY = "gsk_..."
"""
import os
import shutil
from pathlib import Path

import streamlit as st

# Streamlit Cloud injects secrets via st.secrets, not a .env file.
# Bridge it into os.environ so the existing src/ modules (which read
# os.getenv) work unmodified both locally and when deployed.
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from src.data_loader import LOADER_REGISTRY, load_all_documents  # noqa: E402
from src.search import RAGSearch  # noqa: E402

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Stacks — Document Search", page_icon="\U0001F4DA", layout="wide")


@st.cache_resource(show_spinner="Loading models and index…")
def get_rag() -> RAGSearch:
    return RAGSearch()


def rebuild_from_disk(rag: RAGSearch):
    rag.vectorstore.index = None
    rag.vectorstore.metadata = []
    docs = load_all_documents("data")
    if docs:
        rag.vectorstore.build_from_documents(docs)
    else:
        rag.vectorstore.save()


st.title("\U0001F4DA Stacks")
st.caption("Upload documents into the stacks, then ask the reading room a question. "
           "Answers are drawn only from what you've filed.")

try:
    rag = get_rag()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

col_upload, col_search = st.columns([1, 1.4], gap="large")

# ---------------------------------------------------------------------
# Left column: upload + indexed sources
# ---------------------------------------------------------------------
with col_upload:
    st.subheader("Accessions")
    st.caption("Supported: " + ", ".join(sorted(LOADER_REGISTRY.keys())))

    uploaded = st.file_uploader("Drop files here", accept_multiple_files=True, label_visibility="collapsed")

    if uploaded and st.button("File these documents", type="primary"):
        new_documents = []
        results = []
        with st.spinner(f"Filing {len(uploaded)} file(s)…"):
            for f in uploaded:
                ext = Path(f.name).suffix.lower()
                if ext not in LOADER_REGISTRY:
                    results.append((f.name, "skipped", f"unsupported extension {ext}"))
                    continue
                dest = UPLOAD_DIR / f.name
                dest.write_bytes(f.getbuffer())
                try:
                    loader_fn, label = LOADER_REGISTRY[ext]
                    docs = loader_fn(dest)
                    for d in docs:
                        d.metadata.setdefault("source", str(dest))
                        d.metadata.setdefault("file_type", label)
                    new_documents.extend(docs)
                    results.append((f.name, "filed", None))
                except Exception as e:
                    results.append((f.name, "error", str(e)))

            if new_documents:
                rag.ingest_new_documents(new_documents)

        for name, status, reason in results:
            if status == "filed":
                st.success(f"{name} — filed")
            else:
                st.error(f"{name} — {status}: {reason}")

    st.divider()
    st.subheader("Indexed sources")
    sources = rag.vectorstore.list_sources()

    if not sources:
        st.caption("Nothing filed yet.")
    else:
        for i, src in enumerate(sources, 1):
            name = os.path.basename(src)
            c1, c2 = st.columns([4, 1])
            c1.write(f"`{i:02d}` {name}")
            if name != "unknown" and c2.button("remove", key=f"del-{i}-{name}"):
                for p in Path("data").rglob(name):
                    if p.is_file():
                        p.unlink()
                rebuild_from_disk(rag)
                st.rerun()

        if st.button("Clear entire library"):
            if UPLOAD_DIR.exists():
                shutil.rmtree(UPLOAD_DIR)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            rebuild_from_disk(rag)
            st.rerun()

# ---------------------------------------------------------------------
# Right column: search
# ---------------------------------------------------------------------
with col_search:
    st.subheader("Reading Room")

    has_docs = rag.vectorstore.index is not None and rag.vectorstore.index.ntotal > 0
    if has_docs:
        st.success(f"{len(sources)} source(s) filed — ready to ask")
    else:
        st.info("Upload a document to get started")

    query = st.text_input("Ask something about your documents…", disabled=not has_docs)
    top_k = st.number_input("Passages to consider", min_value=1, max_value=20, value=5)

    if st.button("Search", disabled=not has_docs) and query:
        st.markdown("**Answer**")

        def token_stream():
            for event in rag.stream_search_and_summarize(query, top_k=top_k):
                if event["type"] == "answer_chunk":
                    yield event["text"]
                elif event["type"] == "sources":
                    st.session_state["_last_sources"] = event["sources"]

        st.write_stream(token_stream())

        found_sources = st.session_state.get("_last_sources", [])
        if found_sources:
            st.markdown("**Sources:** " + " ".join(f"`{s}`" for s in found_sources))