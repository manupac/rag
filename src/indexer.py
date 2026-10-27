"""Rebuild the vector store from scratch: chunk the rulebook, embed, store."""

import shutil
import time
from pathlib import Path

from chunker import create_chunks
from config import PERSIST_DIR
from vectorstore import get_vector_db

BATCH_SIZE = 90  # stay under the 100/minute free-tier limit
PAUSE_SECONDS = 62  # wait for the per-minute window to reset
DONE_MARKER = "index_complete"  # written at the end, so a half-built store is never reused


def is_complete(persist_dir: Path) -> bool:
    return (persist_dir / DONE_MARKER).exists()


def build_index(chunks, persist_dir: Path = PERSIST_DIR) -> None:
    """Embed the chunks into a fresh vector store in persist_dir."""
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    # start from an empty store, so re-running never duplicates chunks
    shutil.rmtree(persist_dir, ignore_errors=True)
    vector_db = get_vector_db(persist_dir)

    # embed and store in batches (respects the rate limit)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]
        while True:
            try:
                vector_db.add_documents(batch, ids=batch_ids)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" not in str(e):
                    raise
                if "PerDay" in str(e):  # daily quota used up: waiting won't help
                    raise SystemExit("Daily embedding quota used up. Run again tomorrow; finished indexes are kept.")
                print("Per-minute rate limit hit, waiting 30s and retrying...")
                time.sleep(30)
        print(f"Indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")
        if start + BATCH_SIZE < len(chunks):
            time.sleep(PAUSE_SECONDS)

    (persist_dir / DONE_MARKER).write_text("ok", encoding="utf-8")
    print(f"Done: {len(chunks)} chunks in '{persist_dir}'")


def main():
    build_index(create_chunks())


if __name__ == "__main__":
    main()