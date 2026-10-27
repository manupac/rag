import csv
import json
from types import SimpleNamespace

import pytest

import retrieval_ablation as ra
from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, PERSIST_DIR


def doc(text):
    return SimpleNamespace(page_content=text)


SENTENCE = "This sentence is definitely long enough to count."

# ---------- normalize / sentences ----------


def test_normalize_lowercases_and_removes_markdown_stars_and_extra_spaces():
    assert ra.normalize("Hello **World**\n\n  again ") == "hello world again"


def test_sentences_splits_normalizes_and_drops_short_lines():
    text = "Short.\nThis is a long enough sentence to count. Another sentence that is also long enough."

    assert ra.sentences(text) == [
        "this is a long enough sentence to count.",
        "another sentence that is also long enough.",
    ]


def test_sentences_splits_on_newlines_and_colons():
    text = "Rule heading that is long enough here:\n- a bullet that is long enough to be kept"

    assert len(ra.sentences(text)) == 2


# ---------- coverage ----------


def test_coverage_is_the_share_of_source_sentences_found():
    source = "Alpha sentence that is certainly long enough. Beta sentence that is also certainly long enough."

    assert ra.coverage(source, [doc("... alpha sentence that is certainly long enough. ...")]) == 0.5
    assert ra.coverage(source, [doc(source)]) == 1.0
    assert ra.coverage(source, [doc("nothing relevant")]) == 0.0


def test_coverage_ignores_markdown_and_spacing_differences():
    source = "The unit may **shoot** even if engaged in melee."

    assert ra.coverage(source, [doc("the unit   may shoot even if\nengaged in melee.")]) == 1.0


def test_coverage_looks_across_all_retrieved_chunks():
    source = "Alpha sentence that is certainly long enough. Beta sentence that is also certainly long enough."
    retrieved = [doc("alpha sentence that is certainly long enough."), doc("beta sentence that is also certainly long enough.")]

    assert ra.coverage(source, retrieved) == 1.0


def test_coverage_is_none_when_the_source_has_no_countable_sentence():
    assert ra.coverage("Short", [doc("anything")]) is None


# ---------- store_dir / experiments ----------


def test_store_dir_uses_the_normal_store_for_the_configured_chunking():
    assert ra.store_dir(CHUNK_SIZE, CHUNK_OVERLAP) == PERSIST_DIR


def test_store_dir_gives_other_chunkings_their_own_folder():
    assert ra.store_dir(400, 0) == DATA_DIR / "vectorstore_400_0"
    assert ra.store_dir(400, 0) != ra.store_dir(400, 50)


def test_every_experiment_changes_at_most_one_known_setting():
    names = [name for name, _ in ra.EXPERIMENTS]

    assert names[0] == "baseline"
    assert len(names) == len(set(names))
    for _, changes in ra.EXPERIMENTS:
        assert len(changes) <= 1
        assert set(changes) <= set(ra.BASELINE)


# ---------- run_experiment ----------


def test_run_experiment_computes_hit_rate_coverage_and_context_size(monkeypatch):
    questions = [
        {"question": "q1", "source_text": SENTENCE},  # fully retrieved: coverage 1.0
        {
            "question": "q2",
            "source_text": "Alpha sentence that is certainly long enough. Beta sentence that is also certainly long enough.",
        },  # half retrieved: coverage 0.5, which still counts as a hit
        {"question": "q3", "source_text": "tiny"},  # nothing countable: excluded
        {"question": "q4", "source_text": "A completely different sentence that was never retrieved."},  # miss
    ]
    retrieved = {
        "q1": SENTENCE,
        "q2": "alpha sentence that is certainly long enough.",
        "q3": "whatever",
        "q4": "unrelated",
    }
    calls = []

    def fake_search(question, vector, k, search_type, persist_dir):
        calls.append((question, vector, k, search_type, persist_dir))
        return [doc(retrieved[question])]

    monkeypatch.setattr(ra, "search", fake_search)
    settings = dict(ra.BASELINE)

    result = ra.run_experiment(settings, questions, [[1.0], [2.0], [3.0], [4.0]])

    assert result["n_questions"] == 3  # q3 was not counted
    assert result["hit_rate"] == pytest.approx(2 / 3)
    assert result["mean_coverage"] == pytest.approx(0.5)
    expected_chars = round((len(retrieved["q1"]) + len(retrieved["q2"]) + len(retrieved["q4"])) / 3)
    assert result["avg_context_chars"] == expected_chars
    # each question is searched with its own vector and the experiment's settings
    assert calls[0] == ("q1", [1.0], settings["k"], settings["search_type"], ra.store_dir(CHUNK_SIZE, CHUNK_OVERLAP))
    assert [c[1] for c in calls] == [[1.0], [2.0], [3.0], [4.0]]


def test_run_experiment_without_questions_returns_none_metrics(monkeypatch):
    monkeypatch.setattr(ra, "search", lambda *args: [])

    result = ra.run_experiment(dict(ra.BASELINE), [], [])

    assert result == {"hit_rate": None, "mean_coverage": None, "avg_context_chars": None, "n_questions": 0}


# ---------- questions ----------


def test_generate_questions_saves_only_question_and_source_text(tmp_path, monkeypatch, capsys):
    path = tmp_path / "eval" / "retrieval_questions.json"
    monkeypatch.setattr(ra, "ABLATION_QUESTIONS_PATH", path)
    monkeypatch.setattr(
        ra,
        "generate_rulebook_items",
        lambda n, rng: [
            {"question": "Q?", "expected_route": "RULEBOOK", "reference_answer": "A", "source_text": SENTENCE}
        ],
    )

    ra.generate_questions()

    assert json.loads(path.read_text(encoding="utf-8")) == [{"question": "Q?", "source_text": SENTENCE}]


def test_load_questions_missing_file_exits_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "ABLATION_QUESTIONS_PATH", tmp_path / "missing.json")

    with pytest.raises(SystemExit, match="ablation-generate"):
        ra.load_questions()


def test_load_questions_reads_the_saved_file(tmp_path, monkeypatch):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "Q?", "source_text": "t"}]), encoding="utf-8")
    monkeypatch.setattr(ra, "ABLATION_QUESTIONS_PATH", path)

    assert ra.load_questions() == [{"question": "Q?", "source_text": "t"}]


# ---------- main ----------


def test_main_builds_each_chunking_once_and_saves_one_row_per_experiment(tmp_path, monkeypatch, capsys):
    results_path = tmp_path / "out" / "ablation_results.csv"
    built = []
    embedded = []

    def fake_embed(text):
        embedded.append(text)
        return [0.0]

    monkeypatch.setattr(ra, "ABLATION_RESULTS_PATH", results_path)
    monkeypatch.setattr(ra, "load_questions", lambda: [{"question": "What is it?", "source_text": SENTENCE}])
    monkeypatch.setattr(ra, "ensure_index", lambda size, overlap: built.append((size, overlap)))
    monkeypatch.setattr(ra, "get_embeddings", lambda: SimpleNamespace(embed_query=fake_embed))
    monkeypatch.setattr(ra, "search", lambda *args: [doc(SENTENCE)])

    ra.main()

    assert len(built) == len(set(built))  # one index per distinct chunking
    assert (CHUNK_SIZE, CHUNK_OVERLAP) in built
    assert (400, CHUNK_OVERLAP) in built
    assert embedded == ["What is it?"]  # the question is embedded once, not once per experiment

    with results_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["experiment"] for r in rows] == [name for name, _ in ra.EXPERIMENTS]
    assert all(float(r["hit_rate"]) == 1.0 for r in rows)
    assert "experiment" in capsys.readouterr().out