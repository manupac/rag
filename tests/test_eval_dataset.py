import json
import random
import sqlite3

import pytest
from langchain_core.documents import Document

import eval_dataset
from eval_dataset import call_with_retry, generate_rulebook_items, generate_sql_items, load_questions, parse_json

# ---------- parse_json ----------


def test_parse_json_plain():
    assert parse_json('{"question": "Q?", "answer": "A."}') == {"question": "Q?", "answer": "A."}


def test_parse_json_inside_code_fence():
    assert parse_json('```json\n{"question": "Q?", "answer": "A."}\n```') == {"question": "Q?", "answer": "A."}


def test_parse_json_with_extra_words():
    assert parse_json('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}


def test_parse_json_without_json_raises():
    with pytest.raises(ValueError):
        parse_json("no json here")


# ---------- call_with_retry ----------


def test_call_with_retry_waits_on_rate_limit_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(eval_dataset.time, "sleep", sleeps.append)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return "done"

    assert call_with_retry(flaky) == "done"
    assert sleeps == [eval_dataset.RETRY_WAIT_SECONDS] * 2


def test_call_with_retry_raises_other_errors_immediately(monkeypatch):
    monkeypatch.setattr(eval_dataset.time, "sleep", lambda s: None)

    def broken():
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError):
        call_with_retry(broken)


def test_call_with_retry_gives_up_eventually(monkeypatch):
    monkeypatch.setattr(eval_dataset.time, "sleep", lambda s: None)
    attempts = []

    def always_limited():
        attempts.append(1)
        raise Exception("RESOURCE_EXHAUSTED")

    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        call_with_retry(always_limited)
    assert len(attempts) == eval_dataset.MAX_RETRIES


# ---------- generate_rulebook_items ----------


def test_rulebook_items_skip_short_chunks_and_bad_json(monkeypatch):
    long_a = Document(page_content="A " * 200)
    long_b = Document(page_content="B " * 200)
    short = Document(page_content="too short")
    monkeypatch.setattr(eval_dataset, "create_chunks", lambda: [long_a, long_b, short])

    replies = iter(['{"question": "What is A?", "answer": "A is A."}', "this is not json"])
    monkeypatch.setattr(eval_dataset, "ask_llm", lambda system, user: next(replies))

    items = generate_rulebook_items(10, random.Random(0))

    assert len(items) == 1  # the short chunk was never used; one reply was unusable
    item = items[0]
    assert item["question"] == "What is A?"
    assert item["reference_answer"] == "A is A."
    assert item["expected_route"] == "RULEBOOK"
    assert item["source_text"] in (long_a.page_content, long_b.page_content)


def test_rulebook_items_respect_n(monkeypatch):
    chunks = [Document(page_content=f"{i} " * 200) for i in range(5)]
    monkeypatch.setattr(eval_dataset, "create_chunks", lambda: chunks)
    monkeypatch.setattr(eval_dataset, "ask_llm", lambda s, u: '{"question": "Q?", "answer": "A."}')

    assert len(generate_rulebook_items(3, random.Random(0))) == 3


# ---------- generate_sql_items ----------


@pytest.fixture
def small_db(tmp_path, monkeypatch):
    path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE units (name TEXT, toughness TEXT)")
    conn.executemany("INSERT INTO units VALUES (?, ?)", [("Carnifex", "9"), ("Ripper", "1"), ("Ghost", "4")])
    conn.execute("CREATE TABLE weapons (name TEXT, damage TEXT)")
    conn.execute("INSERT INTO weapons VALUES ('Bolter', '1')")
    conn.execute("CREATE TABLE empty_table (x TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(eval_dataset, "connect", lambda: sqlite3.connect(path))


def test_sql_items_are_built_from_table_rows(small_db, monkeypatch):
    prompts = []

    def fake_ask(system, user):
        prompts.append(user)
        return '{"question": "What is it?", "answer": "Something."}'

    monkeypatch.setattr(eval_dataset, "ask_llm", fake_ask)
    items = generate_sql_items(2, random.Random(0))

    assert all(i["expected_route"] == "DATABASE" for i in items)
    assert all("source_table" in i for i in items)
    assert all(p.startswith("Table: ") and "Columns: " in p for p in prompts)


def test_sql_items_skip_tables_without_rows(small_db, monkeypatch):
    monkeypatch.setattr(eval_dataset, "ask_llm", lambda s, u: '{"question": "Q?", "answer": "A."}')
    items = generate_sql_items(6, random.Random(1))

    assert items  # something was generated
    assert "empty_table" not in {i["source_table"] for i in items}


# ---------- main / load_questions ----------


def test_main_saves_questions_with_ids_and_they_can_be_loaded(tmp_path, monkeypatch, capsys):
    path = tmp_path / "eval" / "questions.json"
    monkeypatch.setattr(eval_dataset, "EVAL_QUESTIONS_PATH", path)
    monkeypatch.setattr(
        eval_dataset,
        "generate_rulebook_items",
        lambda n, rng: [{"question": "R?", "expected_route": "RULEBOOK", "reference_answer": "r", "source_text": "t"}],
    )
    monkeypatch.setattr(
        eval_dataset,
        "generate_sql_items",
        lambda n, rng: [{"question": "S?", "expected_route": "DATABASE", "reference_answer": "s", "source_table": "units"}],
    )

    eval_dataset.main()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [i["id"] for i in saved] == ["q1", "q2"]
    assert load_questions(path) == saved


def test_load_questions_missing_file_exits_with_hint(tmp_path):
    with pytest.raises(SystemExit, match="eval-generate"):
        load_questions(tmp_path / "missing.json")
