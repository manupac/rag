import json
import random
import sqlite3

import pytest
from langchain_core.documents import Document

import eval_dataset
from eval_dataset import (
    QAPair,
    generate_qa,
    generate_rulebook_items,
    generate_sql_items,
    load_manual_questions,
    load_questions,
    replace_questions,
)


def qa(question="Q?", answer="A."):
    return QAPair(question=question, answer=answer)


# ---------- generate_qa ----------


def test_generate_qa_returns_the_structured_reply(monkeypatch, fake_llm, sleeps):
    pair = qa("Q?", "A.")
    fake = fake_llm([pair])
    monkeypatch.setattr(eval_dataset, "get_llm", lambda: fake)

    assert generate_qa("system text", "user text") == pair

    system_message, human_message = fake.calls[0]
    assert system_message.content == "system text"
    assert human_message.content == "user text"
    assert fake.schema is QAPair


def test_generate_qa_passes_none_through(monkeypatch, fake_llm, sleeps):
    fake = fake_llm([None])
    monkeypatch.setattr(eval_dataset, "get_llm", lambda: fake)

    assert generate_qa("s", "u") is None


# ---------- generate_rulebook_items ----------


def test_rulebook_items_skip_short_chunks_unusable_replies_and_errors(monkeypatch):
    chunks = [Document(page_content=f"{name} " * 200) for name in "ABC"]
    short = Document(page_content="too short")
    monkeypatch.setattr(eval_dataset, "create_chunks", lambda: [*chunks, short])

    replies = iter([qa("What is A?", " A is A. "), None, ValueError("bad reply")])
    systems = []

    def fake_generate(system, user):
        systems.append(system)
        reply = next(replies)  # a 4th call (the short chunk) would raise StopIteration
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(eval_dataset, "generate_qa", fake_generate)

    items = generate_rulebook_items(10, random.Random(0))

    assert len(items) == 1  # the short chunk was never used; one reply was None, one raised
    item = items[0]
    assert item["question"] == "What is A?"
    assert item["reference_answer"] == "A is A."  # stripped
    assert item["expected_route"] == "RULEBOOK"
    assert item["source_text"] in [c.page_content for c in chunks]
    assert set(systems) == {eval_dataset.RULEBOOK_QA_SYSTEM}


def test_rulebook_items_respect_n(monkeypatch):
    chunks = [Document(page_content=f"{i} " * 200) for i in range(5)]
    monkeypatch.setattr(eval_dataset, "create_chunks", lambda: chunks)
    monkeypatch.setattr(eval_dataset, "generate_qa", lambda s, u: qa())

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

    def fake_generate(system, user):
        prompts.append(user)
        return qa("What is it?", "Something.")

    monkeypatch.setattr(eval_dataset, "generate_qa", fake_generate)
    items = generate_sql_items(2, random.Random(0))

    assert items
    assert all(i["expected_route"] == "DATABASE" for i in items)
    assert all("source_table" in i for i in items)
    assert all(p.startswith("Table: ") and "Columns: " in p for p in prompts)


def test_sql_items_skip_tables_without_rows(small_db, monkeypatch):
    monkeypatch.setattr(eval_dataset, "generate_qa", lambda s, u: qa())
    items = generate_sql_items(6, random.Random(1))

    assert items  # something was generated
    assert "empty_table" not in {i["source_table"] for i in items}


def test_sql_items_skip_unusable_replies(small_db, monkeypatch):
    monkeypatch.setattr(eval_dataset, "generate_qa", lambda s, u: None)

    assert generate_sql_items(3, random.Random(0)) == []


def test_sql_items_skip_the_tables_in_skip_tables(small_db, monkeypatch):
    monkeypatch.setattr(eval_dataset, "SKIP_TABLES", {"units", "weapons"})
    monkeypatch.setattr(eval_dataset, "generate_qa", lambda s, u: qa())
    items = generate_sql_items(4, random.Random(0))

    assert {i["source_table"] for i in items} <= {"empty_table"}


# ---------- replace_questions ----------


@pytest.fixture
def dataset(monkeypatch):
    """Two saved questions (one per type). Returns (items, saved) where saved collects save_questions calls."""
    items = [
        {
            "id": "q1",
            "question": "Old rulebook question?",
            "expected_route": "RULEBOOK",
            "reference_answer": "old",
            "source_text": "old text",
        },
        {
            "id": "q2",
            "question": "Old database question?",
            "expected_route": "DATABASE",
            "reference_answer": "old",
            "source_table": "units",
        },
    ]
    saved = []
    monkeypatch.setattr(eval_dataset, "load_questions", lambda: [dict(i) for i in items])
    monkeypatch.setattr(eval_dataset, "save_questions", saved.append)
    return items, saved


def new_rulebook_item(source_text="new text"):
    return {"question": "New?", "expected_route": "RULEBOOK", "reference_answer": "new", "source_text": source_text}


def test_replace_questions_replaces_only_the_given_id(dataset, monkeypatch):
    items, saved = dataset
    monkeypatch.setattr(eval_dataset, "generate_rulebook_items", lambda n, rng: [new_rulebook_item()])

    replace_questions(["q1"])

    result = saved[0]
    assert result[0]["id"] == "q1"  # keeps its id
    assert result[0]["question"] == "New?"
    assert result[1] == items[1]  # the other question is untouched


def test_replace_questions_uses_the_generator_of_the_same_type(dataset, monkeypatch):
    _, saved = dataset

    def not_called(n, rng):
        raise AssertionError("a database question must not be replaced by a rulebook one")

    new_sql = {"question": "New DB?", "expected_route": "DATABASE", "reference_answer": "n", "source_table": "weapons"}
    monkeypatch.setattr(eval_dataset, "generate_rulebook_items", not_called)
    monkeypatch.setattr(eval_dataset, "generate_sql_items", lambda n, rng: [new_sql])

    replace_questions(["q2"])

    assert saved[0][1]["question"] == "New DB?"
    assert saved[0][1]["id"] == "q2"


def test_replace_questions_keeps_the_old_one_if_only_duplicates_are_generated(dataset, monkeypatch, capsys):
    items, saved = dataset
    calls = []

    def duplicate(n, rng):
        calls.append(1)
        return [new_rulebook_item(source_text="old text")]  # same source as the existing q1

    monkeypatch.setattr(eval_dataset, "generate_rulebook_items", duplicate)

    replace_questions(["q1"])

    assert len(calls) == eval_dataset.MAX_REPLACE_TRIES
    assert saved[0] == items
    assert "Could not generate a new question for q1" in capsys.readouterr().out


def test_replace_questions_rejects_unknown_ids(dataset):
    _, saved = dataset

    with pytest.raises(SystemExit, match="Unknown question ids: q9"):
        replace_questions(["q1", "q9"])
    assert saved == []


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


def test_main_with_ids_only_replaces_those_questions(monkeypatch):
    replaced = []
    monkeypatch.setattr(eval_dataset, "replace_questions", replaced.append)

    eval_dataset.main(["q3", "q7"])

    assert replaced == [["q3", "q7"]]


def test_load_questions_missing_file_exits_with_hint(tmp_path):
    with pytest.raises(SystemExit, match="eval-generate"):
        load_questions(tmp_path / "missing.json")


def test_load_manual_questions_is_optional(tmp_path):
    assert load_manual_questions(tmp_path / "missing.json") == []


def test_load_manual_questions_reads_the_file(tmp_path):
    path = tmp_path / "manual.json"
    path.write_text(json.dumps([{"id": "t1"}]), encoding="utf-8")

    assert load_manual_questions(path) == [{"id": "t1"}]