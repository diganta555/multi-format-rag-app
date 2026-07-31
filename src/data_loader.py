import os
import json
import logging
from pathlib import Path
from typing import List, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual file loaders — each returns List[Document] for a single file
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> List[Document]:
    return PyPDFLoader(str(path)).load()


def _load_text(path: Path) -> List[Document]:
    return TextLoader(str(path), encoding="utf-8").load()


def _load_csv(path: Path) -> List[Document]:
    return CSVLoader(str(path)).load()


def _load_excel(path: Path) -> List[Document]:
    return UnstructuredExcelLoader(str(path)).load()


def _load_docx(path: Path) -> List[Document]:
    return Docx2txtLoader(str(path)).load()


def _load_json(path: Path) -> List[Document]:
    """
    Smart JSON loader:
    - If the JSON is a list of objects, create one Document per record.
    - Otherwise, serialize the whole structure into a single Document.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    if isinstance(data, list):
        for i, record in enumerate(data):
            text = json.dumps(record, indent=2, ensure_ascii=False) if not isinstance(record, str) else record
            docs.append(
                Document(page_content=text, metadata={"source": str(path), "record_index": i})
            )
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
        docs.append(Document(page_content=text, metadata={"source": str(path)}))

    return docs


def _load_markdown(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(path)})]


# Extension -> (loader function, human-readable label)
LOADER_REGISTRY: dict[str, tuple[Callable[[Path], List[Document]], str]] = {
    ".pdf": (_load_pdf, "PDF"),
    ".txt": (_load_text, "TEXT"),
    ".csv": (_load_csv, "CSV"),
    ".xlsx": (_load_excel, "EXCEL"),
    ".xls": (_load_excel, "EXCEL"),
    ".docx": (_load_docx, "WORD"),
    ".json": (_load_json, "JSON"),
    ".md": (_load_markdown, "MARKDOWN"),
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_all_documents(data_dir: str, max_workers: int = 8) -> List[Any]:
    """
    Recursively load all supported files from data_dir and convert them to
    LangChain Document objects, loading files concurrently for speed.

    Supported: PDF, TEXT, CSV, Excel (.xlsx/.xls), Word (.docx), JSON, Markdown

    Each returned Document's metadata includes at least a "source" path and a
    "file_type" tag, so downstream code can filter/trace by origin.
    """
    data_path = Path(data_dir).resolve()
    logger.info(f"Scanning for documents in: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    # Discover all supported files in one pass
    files_by_type: dict[str, List[Path]] = {}
    for ext in LOADER_REGISTRY:
        matches = list(data_path.glob(f"**/*{ext}"))
        if matches:
            files_by_type[ext] = matches

    total_files = sum(len(v) for v in files_by_type.values())
    logger.info(f"Discovered {total_files} files across {len(files_by_type)} supported formats")

    documents: List[Any] = []
    failures: List[tuple[str, str]] = []

    def _process(path: Path, loader_fn: Callable, label: str) -> tuple[Path, str, List[Document] | None, Exception | None]:
        try:
            loaded = loader_fn(path)
            for doc in loaded:
                doc.metadata.setdefault("source", str(path))
                doc.metadata.setdefault("file_type", label)
            return path, label, loaded, None
        except Exception as e:
            return path, label, None, e

    # Load files concurrently — big speedup when you have many PDFs/docs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ext, files in files_by_type.items():
            loader_fn, label = LOADER_REGISTRY[ext]
            for path in files:
                futures.append(executor.submit(_process, path, loader_fn, label))

        for future in as_completed(futures):
            path, label, loaded, error = future.result()
            if error:
                logger.error(f"Failed to load {label} file {path}: {error}")
                failures.append((str(path), str(error)))
            else:
                logger.info(f"Loaded {len(loaded)} doc(s) from {label} file: {path.name}")
                documents.extend(loaded)

    logger.info(f"Finished loading: {len(documents)} documents loaded, {len(failures)} failures")
    if failures:
        logger.warning(f"Failed files: {[f[0] for f in failures]}")

    return documents