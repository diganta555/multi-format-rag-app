"""
Streamlit version of Stacks — no separate FastAPI backend needed.
Run locally with:
    streamlit run streamlit_app.py

Deploy free at https://share.streamlit.io by connecting this GitHub repo.
Set GROQ_API_KEY under the app's "Secrets" in Streamlit Cloud settings
(same key name as your local .env), formatted as:
    GROQ_API_KEY = "gsk_..."
"""
import streamlit as st

# set_page_config must be the very first Streamlit command to actually
# execute — not just first in the file. The imports below (torch,
# transformers, sentence-transformers via src.search) are slow enough
# that if set_page_config were placed after them, a race can occur before
# it ever runs. Calling it immediately, before any slow imports, avoids that.
st.set_page_config(page_title="Stacks — Document Search", page_icon="\U0001F4DA", layout="wide")

import os
import shutil
from pathlib import Path

# Streamlit Cloud injects secrets via st.secrets, not a .env file.
# Bridge it into os.environ so the existing src/ modules (which read
# os.getenv) work unmodified both locally and when deployed.
# Locally, st.secrets raises FileNotFoundError if no secrets.toml exists
# at all (rather than just being empty) — that's expected when relying on
# .env for local dev, so we catch it and fall through to .env.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass  # no secrets.toml locally — fine, src/search.py falls back to .env

from src.data_loader import LOADER_REGISTRY, load_all_documents  # noqa: E402
from src.search import RAGSearch  # noqa: E402

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink: #0f1a2b;
    --paper: #eee7d8;
    --paper-panel: #f6f1e4;
    --navy: #152238;
    --navy-2: #1d3050;
    --amber: #c7862b;
    --line: rgba(15,26,43,0.14);
    --line-strong: rgba(15,26,43,0.28);
}

.stApp { background-color: var(--paper); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; max-width: 1600px; }

/* Full-bleed navy hero banner, breaking out of Streamlit's centered container */
.stacks-hero {
    position: relative;
    left: 50%; right: 50%;
    margin-left: -50vw; margin-right: -50vw;
    width: 100vw;
    background: var(--navy);
    color: var(--paper);
    padding: 2.2rem max(2rem, calc(50vw - 800px)) 1.6rem;
    border-bottom: 4px solid var(--amber);
    margin-bottom: 2.8rem;
}
.stacks-hero .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #e0a53f;
    margin-bottom: 0.5rem;
}
.stacks-hero h1 {
    font-family: 'Source Serif 4', serif !important;
    font-weight: 700 !important;
    font-size: 2.1rem !important;
    color: var(--paper) !important;
    margin: 0 0 0.3rem !important;
}
.stacks-hero p {
    font-family: 'Inter', sans-serif;
    color: #d8cfba;
    font-size: 0.95rem;
    max-width: 60ch;
    margin: 0;
}

/* Section labels styled like archive drawer tabs */
.stacks-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--navy-2);
    border-bottom: 1px solid var(--line-strong);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}

h3 {
    font-family: 'Source Serif 4', serif !important;
    color: var(--navy) !important;
}

/* Bordered card panels (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--paper-panel) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 2px !important;
    padding: 0.5rem 0.4rem !important;
}

/* Buttons */
.stButton > button {
    background-color: var(--navy) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}
.stButton > button:hover { background-color: var(--navy-2) !important; }
.stButton > button[kind="secondary"] {
    background: none !important;
    color: var(--amber) !important;
    border: 1px solid var(--line-strong) !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input {
    background-color: var(--paper) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 2px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Alerts (success/info) recolored to fit the palette */
.stAlert {
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--paper) !important;
    border: 2px dashed var(--line-strong) !important;
    border-radius: 2px !important;
}

/* Monospace index numbers + filenames */
.stMarkdown code {
    background: none !important;
    color: var(--amber) !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 0 !important;
}

/* Source chips at the end of an answer */
.stacks-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.55rem;
    border: 1px solid var(--amber);
    color: var(--navy-2);
    border-radius: 2px;
    background: rgba(199,134,43,0.1);
    margin-right: 0.35rem;
}
.stacks-chip-ext {
    border-color: var(--line-strong) !important;
    color: var(--navy-2) !important;
    background: none !important;
    margin-bottom: 0.35rem;
}

/* Divider under each indexed-source row instead of a boxed look
   (scoped to rows inside bordered panels only, not the outer page columns) */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"] {
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.3rem;
    margin-bottom: 0.3rem;
}
/* Widget labels, captions, and uploader text — force dark, readable color */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stCaption, [data-testid="stCaptionContainer"],
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] small,
.stMarkdown p {
    color: var(--ink) !important;
    opacity: 1 !important;
}

/* Streamlit dims the whole app to ~70% opacity while a script run is in
   progress; that's what made everything look washed out mid-action.
   Force full opacity on the main view wrapper so text stays readable. */
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    opacity: 1 !important;
}
</style>

<div class="stacks-hero">
    <div class="eyebrow">Drawer 01 — Reference &amp; Retrieval</div>
    <h1>\U0001F4DA Stacks</h1>
    <p>Upload documents into the stacks, then ask the reading room a question.
    Answers are drawn only from what you've filed.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models and index...")
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
    panel = st.container(border=True)
    with panel:
        st.markdown('<div class="stacks-label">Accessions</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop files here", accept_multiple_files=True, label_visibility="collapsed")

        chips = "".join(f'<span class="stacks-chip stacks-chip-ext">{ext}</span>'
                         for ext in sorted(LOADER_REGISTRY.keys()))
        st.markdown(f'<div style="margin:0.6rem 0 0.4rem">{chips}</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="stacks-label">Indexed sources</div>', unsafe_allow_html=True)
        sources = rag.vectorstore.list_sources()

        if not sources:
            st.caption("Nothing filed yet.")
        else:
            for i, src in enumerate(sources, 1):
                name = os.path.basename(src)
                c1, c2 = st.columns([3.2, 1.3])
                c1.write(f"`{i:02d}` {name}")
                if name != "unknown" and c2.button("remove", key=f"del-{i}-{name}", type="secondary"):
                    for p in UPLOAD_DIR.glob(name):
                        if p.is_file():
                            p.unlink()
                    rebuild_from_disk(rag)
                    st.rerun()

            if st.button("Clear entire library", type="secondary"):
                if UPLOAD_DIR.exists():
                    shutil.rmtree(UPLOAD_DIR)
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                rebuild_from_disk(rag)
                st.rerun()

# ---------------------------------------------------------------------
# Right column: search
# ---------------------------------------------------------------------
with col_search:
    panel = st.container(border=True)
    with panel:
        st.markdown('<div class="stacks-label">Reading Room</div>', unsafe_allow_html=True)

        has_docs = rag.vectorstore.index is not None and rag.vectorstore.index.ntotal > 0
        if has_docs:
            st.success(f"{len(sources)} source(s) filed — ready to ask")
        else:
            st.info("Upload a document to get started")

        query = st.text_input("Ask something about your documents…", disabled=not has_docs)
        top_k = st.number_input("Passages to consider", min_value=1, max_value=20, value=5)

        if st.button("Search", disabled=not has_docs) and query:
            st.markdown("**Answer**")

            answer_placeholder = st.empty()
            full_answer = ""
            found_sources = []
            error_event = None

            for event in rag.stream_search_and_summarize(query, top_k=top_k):
                if event["type"] == "answer_chunk":
                    full_answer += event["text"]
                    # st.markdown (unlike st.write_stream) properly renders
                    # $...$ / $$...$$ LaTeX via KaTeX, which matters since the
                    # model is prompted to output formulas in LaTeX syntax.
                    answer_placeholder.markdown(full_answer)
                elif event["type"] == "sources":
                    found_sources = event["sources"]
                elif event["type"] == "error":
                    error_event = event

            if error_event:
                answer_placeholder.empty()
                if error_event["kind"] == "rate_limit":
                    st.warning(f"⏳ **{error_event['title']}** — {error_event['message']}")
                else:
                    st.error(f"⚠️ **{error_event['title']}** — {error_event['message']}")
            elif found_sources:
                chips = "".join(f'<span class="stacks-chip">{s}</span>' for s in found_sources)
                st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)