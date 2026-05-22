from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    title: str
    source: str
    excerpt: str
    score: float


class HashEmbeddings:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _hash(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for i, ch in enumerate(text.encode("utf-8", errors="ignore")):
            v[i % self.dim] += float((ch % 31) - 15)
        norm = sum(x * x for x in v) ** 0.5
        if norm > 0:
            v = [x / norm for x in v]
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash(text)


def _load_protocol_documents(protocol_dir: str, *, collection: str) -> list[Document]:
    docs: list[Document] = []
    if not os.path.isdir(protocol_dir):
        return docs

    jsonl_paths = glob.glob(os.path.join(protocol_dir, "*.jsonl"))
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                doc_id = str(item.get("id") or os.path.basename(path))
                title = str(item.get("title") or "Protocolo")
                text = str(item.get("text") or "")
                source = str(item.get("source") or os.path.basename(path))
                if not text.strip():
                    continue
                docs.append(Document(page_content=text, metadata={"doc_id": doc_id, "title": title, "source": source, "collection": collection}))

    txt_paths = glob.glob(os.path.join(protocol_dir, "*.txt"))
    for path in txt_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": os.path.basename(path),
                    "title": os.path.splitext(os.path.basename(path))[0],
                    "source": os.path.basename(path),
                    "collection": collection,
                },
            )
        )

    return docs


def _build_embeddings(embeddings_model: str):
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        import sentence_transformers

        return HuggingFaceEmbeddings(model_name=embeddings_model)
    except Exception:
        return HashEmbeddings(dim=384)


class MultiVectorStore:
    def __init__(self, *, internal, external) -> None:
        self._internal = internal
        self._external = external

    def _internal_is_good(self, internal_hits: list[tuple[Any, float]]) -> bool:
        if not internal_hits:
            return False
        best = internal_hits[0][1]
        try:
            best_score = float(best)
        except Exception:
            return False

        distance_max = float(os.environ.get("TC3_RAG_INTERNAL_DISTANCE_MAX", "0.35"))
        similarity_min = float(os.environ.get("TC3_RAG_INTERNAL_SIMILARITY_MIN", "0.75"))

        return (best_score <= distance_max) or (best_score >= similarity_min)

    def similarity_search_with_score(self, query: str, k: int = 4):
        internal_hits = self._internal.similarity_search_with_score(query, k=max(1, k))
        if self._internal_is_good(internal_hits):
            return internal_hits[:k]

        if self._external is None:
            return internal_hits[:k]

        k_internal = max(1, min(k - 1, 3))
        k_external = max(1, k - k_internal)
        external_hits = self._external.similarity_search_with_score(query, k=k_external)
        return list(internal_hits[:k_internal]) + list(external_hits)


def build_vectorstore(protocol_dir: str, vectorstore_dir: str, embeddings_model: str, protocol_external_dir: str | None = None):
    from langchain_community.vectorstores import Chroma

    os.makedirs(vectorstore_dir, exist_ok=True)

    embeddings = _build_embeddings(embeddings_model)

    vectorstore = Chroma(collection_name="hospital_protocols", embedding_function=embeddings, persist_directory=vectorstore_dir)

    existing = vectorstore._collection.count()
    if existing and existing > 0:
        internal = vectorstore
    else:
        raw_docs = _load_protocol_documents(protocol_dir, collection="interno")
        if raw_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
            chunks = splitter.split_documents(raw_docs)
            ids: list[str] = []
            for i, d in enumerate(chunks):
                base = str(d.metadata.get("doc_id") or "doc")
                ids.append(f"{base}::chunk::{i:05d}")
            vectorstore.add_documents(chunks, ids=ids)
            vectorstore.persist()
        internal = vectorstore

    external = None
    external_collection = "externo_pubmedqa"
    if protocol_external_dir and os.path.isdir(protocol_external_dir):
        ext_store = Chroma(collection_name="external_pubmedqa", embedding_function=embeddings, persist_directory=vectorstore_dir)
        if ext_store._collection.count() and ext_store._collection.count() > 0:
            external = ext_store
        else:
            ext_docs = _load_protocol_documents(protocol_external_dir, collection=external_collection)
            if ext_docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
                chunks = splitter.split_documents(ext_docs)
                ids: list[str] = []
                for i, d in enumerate(chunks):
                    base = str(d.metadata.get("doc_id") or "doc")
                    ids.append(f"{base}::chunk::{i:05d}")
                ext_store.add_documents(chunks, ids=ids)
                ext_store.persist()
                external = ext_store

    if external is None:
        return internal
    return MultiVectorStore(internal=internal, external=external)


def retrieve_chunks(vectorstore, query: str, k: int = 4) -> list[RetrievedChunk]:
    hits = vectorstore.similarity_search_with_score(query, k=k)
    out: list[RetrievedChunk] = []
    for doc, score in hits:
        doc_id = str(doc.metadata.get("doc_id") or "")
        title = str(doc.metadata.get("title") or "Documento")
        source = str(doc.metadata.get("source") or "")
        collection = str(doc.metadata.get("collection") or "").strip()
        if collection:
            source = f"{collection}:{source}" if source else collection
        excerpt = doc.page_content[:550].strip()
        out.append(RetrievedChunk(doc_id=doc_id, title=title, source=source, excerpt=excerpt, score=float(score)))
    return out


def chunks_to_citations(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    citations: list[dict[str, Any]] = []
    for c in chunks:
        key = (c.doc_id, c.excerpt)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "source": c.source,
                "excerpt": c.excerpt,
                "score": c.score,
            }
        )
    return citations
