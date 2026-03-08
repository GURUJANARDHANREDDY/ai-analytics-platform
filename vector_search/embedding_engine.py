"""Vector Search Engine – FAISS-based semantic search over dataset metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VECTOR_DIR = DATA_DIR / "vectors"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    _model: SentenceTransformer | None = None
    ST_AVAILABLE = True
except ImportError:
    _model = None
    ST_AVAILABLE = False


def _get_model() -> Any:
    global _model
    if _model is None and ST_AVAILABLE:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _text_for_dataset(name: str, columns: list[str], dtypes: dict[str, str],
                      row_count: int = 0, description: str = "") -> str:
    parts = [f"Dataset: {name}"]
    if description:
        parts.append(f"Description: {description}")
    parts.append(f"Columns: {', '.join(columns[:30])}")
    parts.append(f"Types: {', '.join(f'{k}({v})' for k, v in list(dtypes.items())[:20])}")
    if row_count:
        parts.append(f"Rows: {row_count}")
    return ". ".join(parts)


class DatasetVectorStore:
    def __init__(self):
        self.index = None
        self.entries: list[dict[str, Any]] = []
        self._index_path = VECTOR_DIR / "faiss.index"
        self._meta_path = VECTOR_DIR / "entries.json"
        self._load()

    def _load(self):
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                self.entries = json.load(f)
        if FAISS_AVAILABLE and self._index_path.exists():
            self.index = faiss.read_index(str(self._index_path))

    def _save(self):
        with open(self._meta_path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(self._index_path))

    def add_dataset(self, name: str, columns: list[str], dtypes: dict[str, str],
                    row_count: int = 0, description: str = ""):
        if not ST_AVAILABLE or not FAISS_AVAILABLE:
            self.entries.append({"name": name, "columns": columns, "row_count": row_count, "description": description})
            self._save()
            return

        model = _get_model()
        text = _text_for_dataset(name, columns, dtypes, row_count, description)
        embedding = model.encode([text], normalize_embeddings=True).astype("float32")

        dim = embedding.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embedding)

        self.entries.append({"name": name, "columns": columns, "row_count": row_count,
                             "description": description, "text": text})
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not ST_AVAILABLE or not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return self._keyword_search(query)

        model = _get_model()
        q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.entries):
                entry = self.entries[idx].copy()
                entry["score"] = round(float(score), 4)
                results.append(entry)
        return results

    def _keyword_search(self, query: str) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for entry in self.entries:
            text = f"{entry.get('name', '')} {' '.join(entry.get('columns', []))} {entry.get('description', '')}"
            if any(word in text.lower() for word in query_lower.split()):
                results.append(entry)
        return results[:5]

    def list_all(self) -> list[dict[str, Any]]:
        return self.entries
