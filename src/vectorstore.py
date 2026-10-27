"""Shared vector store setup and the rulebook retriever, used for both indexing and querying.

The embedding model comes from config, which guarantees that chunks and questions are
embedded with the same model.

Two search types:
- "vector": the chunks whose embeddings are closest to the question's.
- "hybrid": vector search + keyword search (BM25), merged with reciprocal rank fusion.
  Keyword search catches exact terms (e.g. "CP", "VP", keywords like Grenades) that
  embeddings can miss.
"""

import re
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi

from config import EMBEDDING_MODEL, PERSIST_DIR, SEARCH_TYPE, TOP_K

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

HYBRID_CANDIDATES = 2  # hybrid search fuses the top k * HYBRID_CANDIDATES results of each method
RRF_CONSTANT = 60  # standard value for reciprocal rank fusion


@cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@cache
def get_vector_db(persist_dir: Path = PERSIST_DIR) -> Chroma:
    return Chroma(persist_directory=str(persist_dir), embedding_function=get_embeddings())


def tokenize(text: str) -> list[str]:
    """Lowercase words without punctuation or Markdown, so 'Grenades**' matches 'grenades'."""
    return re.findall(r"\w+", text.lower())


class KeywordIndex:
    """BM25 keyword search over the same chunks as the vector store."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in docs])

    def search(self, question: str, k: int) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in best]


@cache
def get_keyword_index(persist_dir: Path = PERSIST_DIR) -> KeywordIndex:
    data = get_vector_db(persist_dir).get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {}) for text, meta in zip(data["documents"], data["metadatas"])
    ]
    return KeywordIndex(docs)


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int) -> list[Document]:
    """Merge several ranked lists: a chunk ranked high in any list ends up high in the result."""
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (RRF_CONSTANT + rank + 1)
            docs.setdefault(key, doc)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[key] for key in best]


def search(
    question: str,
    question_vector: list[float],
    k: int = TOP_K,
    search_type: str = SEARCH_TYPE,
    persist_dir: Path = PERSIST_DIR,
) -> list[Document]:
    """Search with an already-embedded question (lets the ablation embed each question only once)."""
    db = get_vector_db(persist_dir)
    if search_type == "vector":
        return db.similarity_search_by_vector(question_vector, k=k)
    if search_type == "hybrid":
        n_candidates = k * HYBRID_CANDIDATES
        vector_hits = db.similarity_search_by_vector(question_vector, k=n_candidates)
        keyword_hits = get_keyword_index(persist_dir).search(question, n_candidates)
        return reciprocal_rank_fusion([vector_hits, keyword_hits], k)
    raise ValueError(f"Unknown search type: {search_type!r} (use 'vector' or 'hybrid')")


def retrieve(
    question: str,
    k: int = TOP_K,
    search_type: str = SEARCH_TYPE,
    persist_dir: Path = PERSIST_DIR,
) -> list[Document]:
    """Return the k chunks most relevant to the question (the text-channel retriever)."""
    return search(question, get_embeddings().embed_query(question), k, search_type, persist_dir)