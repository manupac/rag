from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import indexer
import vectorstore
from config import TOP_K

# Tests call build_index with an explicit temporary folder, so they can never touch the real
# data/chroma_db_rulebook (build_index deletes the folder it is given).


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
def run_indexer(tmp_path, monkeypatch, sleeps):
    """Runs build_index on n fake chunks in a temp folder. Returns (db, sleeps, persist_dir)."""
    persist_dir = tmp_path / "chroma"

    def run(n_chunks, errors=()):
        persist_dir.mkdir(exist_ok=True)
        (persist_dir / "stale.txt").write_text("old store")
        db = FakeDB(errors)

        def fake_get_vector_db(path):
            path.mkdir(parents=True, exist_ok=True)  # like Chroma, which creates its folder
            return db

        monkeypatch.setattr(indexer, "get_vector_db", fake_get_vector_db)
        indexer.build_index([f"chunk text {i}" for i in range(n_chunks)], persist_dir)
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

    assert not (persist_dir / "stale.txt").exists()  # the stale content was deleted before rebuilding


def test_indexer_marks_the_store_complete_at_the_end(run_indexer):
    _, _, persist_dir = run_indexer(5)

    assert indexer.is_complete(persist_dir)
    assert (persist_dir / indexer.DONE_MARKER).read_text(encoding="utf-8") == "ok"


def test_is_complete_is_false_for_a_missing_or_unmarked_folder(tmp_path):
    assert indexer.is_complete(tmp_path / "missing") is False

    (tmp_path / "half_built").mkdir()
    assert indexer.is_complete(tmp_path / "half_built") is False


def test_indexer_waits_and_retries_on_rate_limit(run_indexer):
    db, sleeps, _ = run_indexer(10, errors=[Exception("429 RESOURCE_EXHAUSTED quota")])

    assert sleeps == [30]
    assert len(db.batches) == 1  # the same batch succeeded on the retry
    assert len(db.batches[0][0]) == 10


def test_indexer_stops_on_daily_quota_and_leaves_no_marker(run_indexer, tmp_path, sleeps):
    error = Exception("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(SystemExit, match="Daily embedding quota"):
        run_indexer(10, errors=[error])

    assert sleeps == []  # waiting would not help
    assert not indexer.is_complete(tmp_path / "chroma")  # a half-built store is never reused


def test_indexer_raises_other_errors_and_leaves_no_marker(run_indexer, tmp_path):
    with pytest.raises(ValueError):
        run_indexer(10, errors=[ValueError("something else broke")])

    assert not indexer.is_complete(tmp_path / "chroma")


def test_main_builds_the_index_from_the_chunks(monkeypatch):
    calls = []
    monkeypatch.setattr(indexer, "create_chunks", lambda: ["a", "b"])
    monkeypatch.setattr(indexer, "build_index", lambda chunks: calls.append(chunks))

    indexer.main()

    assert calls == [["a", "b"]]


# ---------- vectorstore: retrieve / search ----------


class FakeEmbeddings:
    def __init__(self):
        self.embedded = []

    def embed_query(self, text):
        self.embedded.append(text)
        return [0.1, 0.2]


class FakeSearchDB:
    def __init__(self, hits=None):
        self.hits = hits
        self.calls = []

    def similarity_search_by_vector(self, vector, k):
        self.calls.append((vector, k))
        if self.hits is None:
            return ["chunk"] * k
        return self.hits[:k]


class FakeKeywordIndex:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, question, k):
        self.calls.append((question, k))
        return self.hits[:k]


@pytest.fixture
def use_store(monkeypatch):
    def install(db):
        embeddings = FakeEmbeddings()
        monkeypatch.setattr(vectorstore, "get_vector_db", lambda persist_dir=None: db)
        monkeypatch.setattr(vectorstore, "get_embeddings", lambda: embeddings)
        return embeddings

    return install


def contents(docs):
    return [d.page_content for d in docs]


def test_retrieve_uses_top_k_by_default(use_store):
    db = FakeSearchDB()
    use_store(db)

    assert len(vectorstore.retrieve("how do I score?", search_type="vector")) == TOP_K
    assert db.calls == [([0.1, 0.2], TOP_K)]


def test_retrieve_accepts_custom_k(use_store):
    db = FakeSearchDB()
    use_store(db)

    assert len(vectorstore.retrieve("question", k=2, search_type="vector")) == 2
    assert db.calls == [([0.1, 0.2], 2)]


def test_retrieve_embeds_the_question_once(use_store):
    embeddings = use_store(FakeSearchDB())
    vectorstore.retrieve("how do I score?", search_type="vector")

    assert embeddings.embedded == ["how do I score?"]


def test_search_rejects_an_unknown_search_type(use_store):
    use_store(FakeSearchDB())

    with pytest.raises(ValueError, match="Unknown search type"):
        vectorstore.search("q", [0.1], k=3, search_type="fuzzy")


def test_hybrid_search_merges_vector_and_keyword_results(monkeypatch, use_store):
    a, b, c, d = (Document(page_content=name) for name in "ABCD")
    db = FakeSearchDB(hits=[a, b, c, d])
    use_store(db)
    keyword = FakeKeywordIndex(hits=[c, a])
    monkeypatch.setattr(vectorstore, "get_keyword_index", lambda persist_dir=None: keyword)

    result = vectorstore.search("q", [0.1], k=2, search_type="hybrid")

    n_candidates = 2 * vectorstore.HYBRID_CANDIDATES
    assert db.calls == [([0.1], n_candidates)]  # each method is asked for more than k candidates
    assert keyword.calls == [("q", n_candidates)]
    assert contents(result) == ["A", "C"]  # A and C are ranked high by both methods


# ---------- vectorstore: reciprocal_rank_fusion ----------


def test_rrf_puts_chunks_ranked_high_in_both_lists_first():
    a, b, c = (Document(page_content=name) for name in "ABC")

    result = vectorstore.reciprocal_rank_fusion([[a, b], [b, c]], k=2)

    assert contents(result) == ["B", "A"]


def test_rrf_returns_each_chunk_once_even_if_it_is_a_different_object():
    first = [Document(page_content="same"), Document(page_content="other")]
    second = [Document(page_content="same")]

    result = vectorstore.reciprocal_rank_fusion([first, second], k=10)

    assert contents(result) == ["same", "other"]


def test_rrf_of_nothing_is_empty():
    assert vectorstore.reciprocal_rank_fusion([[], []], k=3) == []


# ---------- vectorstore: keyword search ----------


def test_tokenize_lowercases_and_drops_punctuation_and_markdown():
    assert vectorstore.tokenize("**Grenades** Hit-Roll 2D6!") == ["grenades", "hit", "roll", "2d6"]


def test_keyword_index_finds_the_chunk_with_the_query_words():
    docs = [
        Document(page_content="grenades are thrown in the shooting phase"),
        Document(page_content="objective markers score victory points"),
        Document(page_content="charge moves use two dice"),
        Document(page_content="fight phase pile in and consolidate"),
    ]
    index = vectorstore.KeywordIndex(docs)

    assert index.search("victory points", k=1) == [docs[1]]
    assert len(index.search("victory points", k=3)) == 3


def test_get_keyword_index_is_built_from_the_vector_store(monkeypatch, tmp_path):
    class DB:
        def get(self, include):
            assert "documents" in include
            return {"documents": ["alpha text", "beta text"], "metadatas": [{"section": "A"}, None]}

    monkeypatch.setattr(vectorstore, "get_vector_db", lambda persist_dir=None: DB())
    vectorstore.get_keyword_index.cache_clear()
    try:
        index = vectorstore.get_keyword_index(tmp_path)
        assert [d.page_content for d in index.docs] == ["alpha text", "beta text"]
        assert [d.metadata for d in index.docs] == [{"section": "A"}, {}]  # missing metadata becomes {}
    finally:
        vectorstore.get_keyword_index.cache_clear()