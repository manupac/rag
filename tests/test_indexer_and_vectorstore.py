import pytest

import indexer
import vectorstore
from config import TOP_K


class FakeDB:
    """Stands in for Chroma. `errors` are raised, one per add_documents call, before succeeding."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.batches = []  # (documents, ids) for every successful call

    def add_documents(self, documents, ids):
        if self.errors:
            raise self.errors.pop(0)
        self.batches.append((list(documents), list(ids)))


@pytest.fixture
def run_indexer(tmp_path, monkeypatch):
    """Runs indexer.main() on n fake chunks. Returns (db, sleeps, persist_dir)."""

    def run(n_chunks, errors=()):
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir()
        (persist_dir / "stale.txt").write_text("old store")

        db = FakeDB(errors)
        sleeps = []
        monkeypatch.setattr(indexer, "PERSIST_DIR", persist_dir)
        monkeypatch.setattr(indexer, "create_chunks", lambda: [f"chunk text {i}" for i in range(n_chunks)])
        monkeypatch.setattr(indexer, "get_vector_db", lambda: db)
        monkeypatch.setattr(indexer.time, "sleep", sleeps.append)

        indexer.main()
        return db, sleeps, persist_dir

    return run


# ---------- indexer ----------


def test_indexer_splits_chunks_into_batches_with_pauses_between(run_indexer):
    db, sleeps, _ = run_indexer(200)

    sizes = [len(docs) for docs, _ in db.batches]
    assert sizes == [indexer.BATCH_SIZE, indexer.BATCH_SIZE, 200 - 2 * indexer.BATCH_SIZE]
    assert sleeps == [indexer.PAUSE_SECONDS, indexer.PAUSE_SECONDS]  # no pause after the last batch


def test_indexer_gives_every_chunk_a_unique_sequential_id(run_indexer):
    db, _, _ = run_indexer(200)

    all_ids = [i for _, ids in db.batches for i in ids]
    assert all_ids == [f"chunk-{i}" for i in range(200)]


def test_indexer_single_batch_never_pauses(run_indexer):
    db, sleeps, _ = run_indexer(indexer.BATCH_SIZE)

    assert len(db.batches) == 1
    assert sleeps == []


def test_indexer_wipes_the_old_store_first(run_indexer):
    _, _, persist_dir = run_indexer(5)
    assert not persist_dir.exists()  # the stale folder was deleted before rebuilding


def test_indexer_waits_and_retries_on_rate_limit(run_indexer):
    db, sleeps, _ = run_indexer(10, errors=[Exception("429 RESOURCE_EXHAUSTED quota")])

    assert sleeps == [30]
    assert len(db.batches) == 1  # the same batch succeeded on the retry
    assert len(db.batches[0][0]) == 10


def test_indexer_raises_other_errors(run_indexer):
    with pytest.raises(ValueError):
        run_indexer(10, errors=[ValueError("something else broke")])


# ---------- vectorstore.retrieve ----------


class FakeSearchDB:
    def __init__(self):
        self.calls = []

    def similarity_search(self, question, k):
        self.calls.append((question, k))
        return ["chunk"] * k


def test_retrieve_uses_top_k_by_default(monkeypatch):
    db = FakeSearchDB()
    monkeypatch.setattr(vectorstore, "get_vector_db", lambda: db)

    assert len(vectorstore.retrieve("how do I score?")) == TOP_K
    assert db.calls == [("how do I score?", TOP_K)]


def test_retrieve_accepts_custom_k(monkeypatch):
    db = FakeSearchDB()
    monkeypatch.setattr(vectorstore, "get_vector_db", lambda: db)

    assert len(vectorstore.retrieve("question", k=2)) == 2
    assert db.calls == [("question", 2)]
