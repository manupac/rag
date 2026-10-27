"""Compare retrieval settings, changing one setting at a time from the current config (ablation).

Only retrieval is measured, so no chat model is called: each question is embedded once,
and every setting is tested on the same questions.

Metric: for each question, the share of its source chunk's sentences that appear in the
retrieved text ("coverage"). Unlike an exact-chunk match, this works across different
chunk sizes. A question counts as a hit if coverage >= COVERAGE_THRESHOLD.
The table also shows the average size of the retrieved text: more text costs more tokens
and makes the LLM's job harder, so a higher hit rate is not free.

Usage:
  python main.py ablation-generate   # once: write the questions used for the comparison
  python main.py ablation            # build missing indexes, run every setting, save a CSV

The first run builds one vector store per chunk size/overlap (a few minutes each on the
free tier). They are kept in data/, so later runs are fast.
"""

import csv
import json
import random
import re
from pathlib import Path

from chunker import create_chunks
from config import (
    ABLATION_N_QUESTIONS,
    ABLATION_QUESTIONS_PATH,
    ABLATION_RESULTS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EVAL_SEED,
    PERSIST_DIR,
    SEARCH_TYPE,
    TOP_K,
)
from eval_dataset import generate_rulebook_items
from indexer import build_index, is_complete
from llm import call_with_retry
from vectorstore import get_embeddings, search

COVERAGE_THRESHOLD = 0.5
MIN_SENTENCE_CHARS = 25  # shorter lines (headings, labels) are not counted

BASELINE = {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "k": TOP_K, "search_type": SEARCH_TYPE}
OTHER_SEARCH = "hybrid" if SEARCH_TYPE == "vector" else "vector"

# Each experiment changes ONE setting from the baseline (your current config).
EXPERIMENTS = [
    ("baseline", {}),
    ("k=3", {"k": 3}),
    ("k=8", {"k": 8}),
    ("chunk size 400", {"chunk_size": 400}),
    ("chunk size 1200", {"chunk_size": 1200}),
    ("overlap 0", {"chunk_overlap": 0}),
    (f"{OTHER_SEARCH} search", {"search_type": OTHER_SEARCH}),
]


# ---------- metric ----------


def normalize(text: str) -> str:
    """Lowercase, without Markdown stars and with single spaces, so formatting doesn't matter."""
    return re.sub(r"\s+", " ", text.replace("*", "")).strip().lower()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?:])\s+|\n+", text)
    return [s for s in (normalize(p) for p in parts) if len(s) >= MIN_SENTENCE_CHARS]


def coverage(source_text: str, retrieved) -> float | None:
    """Share of the source chunk's sentences found in the retrieved chunks."""
    source_sentences = sentences(source_text)
    if not source_sentences:
        return None
    retrieved_text = normalize(" ".join(doc.page_content for doc in retrieved))
    return sum(s in retrieved_text for s in source_sentences) / len(source_sentences)


# ---------- questions ----------


def generate_questions():
    """Write a larger set of rulebook questions, only for this comparison (no reference answers needed)."""
    rng = random.Random(EVAL_SEED)
    items = generate_rulebook_items(ABLATION_N_QUESTIONS, rng)
    questions = [{"question": item["question"], "source_text": item["source_text"]} for item in items]
    ABLATION_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_QUESTIONS_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(questions)} questions to {ABLATION_QUESTIONS_PATH}")


def load_questions() -> list[dict]:
    if not ABLATION_QUESTIONS_PATH.exists():
        raise SystemExit(f"{ABLATION_QUESTIONS_PATH} not found. Run 'python main.py ablation-generate' first.")
    return json.loads(ABLATION_QUESTIONS_PATH.read_text(encoding="utf-8"))


# ---------- running ----------


def store_dir(chunk_size: int, chunk_overlap: int) -> Path:
    """The baseline uses the normal vector store; other chunkings get their own folder."""
    if (chunk_size, chunk_overlap) == (CHUNK_SIZE, CHUNK_OVERLAP):
        return PERSIST_DIR
    return DATA_DIR / f"vectorstore_{chunk_size}_{chunk_overlap}"


def ensure_index(chunk_size: int, chunk_overlap: int) -> Path:
    persist_dir = store_dir(chunk_size, chunk_overlap)
    if not is_complete(persist_dir):
        print(f"\nBuilding index for chunk size {chunk_size}, overlap {chunk_overlap} in {persist_dir}")
        build_index(create_chunks(chunk_size, chunk_overlap), persist_dir)
    return persist_dir


def run_experiment(settings: dict, questions: list[dict], vectors: list[list[float]]) -> dict:
    persist_dir = store_dir(settings["chunk_size"], settings["chunk_overlap"])
    scores, context_chars = [], []
    for item, vector in zip(questions, vectors):
        retrieved = search(item["question"], vector, settings["k"], settings["search_type"], persist_dir)
        score = coverage(item["source_text"], retrieved)
        if score is not None:
            scores.append(score)
            context_chars.append(sum(len(doc.page_content) for doc in retrieved))
    n = len(scores)
    return {
        "hit_rate": sum(s >= COVERAGE_THRESHOLD for s in scores) / n if n else None,
        "mean_coverage": sum(scores) / n if n else None,
        "avg_context_chars": round(sum(context_chars) / n) if n else None,
        "n_questions": n,
    }


def print_table(rows: list[dict]) -> None:
    print(f"\n{'experiment':<18}{'hit rate':>10}{'coverage':>10}{'context chars':>15}")
    for row in rows:
        hit = "n/a" if row["hit_rate"] is None else f"{row['hit_rate']:.0%}"
        cov = "n/a" if row["mean_coverage"] is None else f"{row['mean_coverage']:.0%}"
        print(f"{row['experiment']:<18}{hit:>10}{cov:>10}{row['avg_context_chars'] or 0:>15}")
    print(f"\n{rows[0]['n_questions']} questions; hit = at least {COVERAGE_THRESHOLD:.0%} of the source text retrieved")


def main():
    questions = load_questions()
    experiments = [(name, {**BASELINE, **changes}) for name, changes in EXPERIMENTS]

    # 1. Build the vector stores that don't exist yet (one per chunk size/overlap)
    for chunking in sorted({(s["chunk_size"], s["chunk_overlap"]) for _, s in experiments}):
        ensure_index(*chunking)

    # 2. Embed each question once; every experiment reuses these vectors
    print(f"\nEmbedding {len(questions)} questions...")
    embed = get_embeddings().embed_query
    vectors = [call_with_retry(embed, item["question"]) for item in questions]

    # 3. Run every experiment on the same questions
    rows = []
    for name, settings in experiments:
        rows.append({"experiment": name, **settings, **run_experiment(settings, questions, vectors)})

    print_table(rows)
    ABLATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ABLATION_RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {ABLATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()