import sqlite3

import pytest

import sql_retriever
from sql_retriever import (
    SQLResult,
    extract_sql,
    faction_names,
    fix_text,
    format_result,
    retrieve_sql,
    run_query,
    schema_description,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A tiny throwaway SQLite database standing in for wahadb.sqlite."""
    path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE units (name TEXT, toughness TEXT)")
    conn.executemany(
        "INSERT INTO units VALUES (?, ?)",
        [("Carnifex", "9"), ("Hormagaunt", "3"), ("Termagant", "3"), ("Genestealer", "4"), ("Ripper", "1")],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(sql_retriever, "SQLITE_PATH", path)
    schema_description.cache_clear()
    faction_names.cache_clear()
    yield path
    schema_description.cache_clear()
    faction_names.cache_clear()


# ---------- extract_sql ----------


def test_extract_sql_removes_sql_fence_and_semicolon():
    assert extract_sql("```sql\nSELECT * FROM units;\n```") == "SELECT * FROM units"


def test_extract_sql_removes_plain_fence():
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"


def test_extract_sql_leaves_unfenced_query_alone():
    assert extract_sql("  SELECT 1;  ") == "SELECT 1"


# ---------- fix_text ----------


def test_fix_text_repairs_mojibake():
    broken = "Opponent’s".encode("utf-8").decode("latin-1")
    assert broken != "Opponent’s"
    assert fix_text(broken) == "Opponent’s"


def test_fix_text_leaves_clean_text_alone():
    assert fix_text("plain ascii") == "plain ascii"
    assert fix_text("café") == "café"
    assert fix_text("Opponent’s") == "Opponent’s"


def test_fix_text_passes_non_strings_through():
    assert fix_text(5) == 5
    assert fix_text(None) is None


# ---------- run_query ----------


@pytest.mark.parametrize(
    "query", ["DELETE FROM units", "DROP TABLE units", "UPDATE units SET name='x'", "", "PRAGMA table_info(units)"]
)
def test_run_query_rejects_non_select(db, query):
    with pytest.raises(ValueError):
        run_query(query)


def test_run_query_returns_columns_and_rows(db):
    columns, rows, truncated = run_query("SELECT name FROM units WHERE toughness = '9'")
    assert columns == ["name"]
    assert rows == [("Carnifex",)]
    assert truncated is False


def test_run_query_accepts_lowercase_and_with(db):
    assert run_query("select name from units")[1]
    assert run_query("WITH t AS (SELECT name FROM units) SELECT * FROM t")[1]


def test_run_query_caps_rows_and_flags_truncation(db, monkeypatch):
    monkeypatch.setattr(sql_retriever, "SQL_MAX_ROWS", 3)
    _, rows, truncated = run_query("SELECT name FROM units")
    assert len(rows) == 3
    assert truncated is True


def test_run_query_not_truncated_when_exactly_at_the_limit(db, monkeypatch):
    monkeypatch.setattr(sql_retriever, "SQL_MAX_ROWS", 5)
    _, rows, truncated = run_query("SELECT name FROM units")
    assert len(rows) == 5
    assert truncated is False


def test_database_is_read_only_even_for_with_delete(db):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        run_query("WITH x AS (SELECT 1) DELETE FROM units")

    _, rows, _ = run_query("SELECT name FROM units")
    assert len(rows) == 5  # nothing was deleted


# ---------- schema_description / faction_names ----------


def test_schema_description_lists_columns_and_sample_rows(db):
    description = schema_description()
    assert "units(name, toughness)" in description
    assert "Carnifex | 9" in description
    assert "Hormagaunt | 3" in description
    assert "Genestealer" not in description  # only SAMPLE_ROWS (2) examples are shown


def test_faction_names_are_sorted_and_comma_separated(db):
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE factions (name TEXT)")
    conn.executemany("INSERT INTO factions VALUES (?)", [("Tyranids",), ("Necrons",)])
    conn.commit()
    conn.close()

    assert faction_names() == "Necrons, Tyranids"


# ---------- format_result ----------


def test_format_result_for_error():
    result = SQLResult(query="q", columns=[], rows=[], error="boom")
    assert format_result(result) == "The query failed: boom"


def test_format_result_for_no_rows():
    result = SQLResult(query="q", columns=["name"], rows=[])
    assert format_result(result) == "The query returned no rows."


def test_format_result_builds_a_text_table_and_blanks_none():
    result = SQLResult(query="q", columns=["name", "toughness"], rows=[("Carnifex", "9"), ("Ghost", None)])
    assert format_result(result) == "name | toughness\nCarnifex | 9\nGhost | "


def test_format_result_notes_truncation():
    result = SQLResult(query="q", columns=["name"], rows=[("a",)], truncated=True)
    assert f"only the first {sql_retriever.SQL_MAX_ROWS} rows are shown" in format_result(result)


def test_format_result_repairs_mojibake():
    broken = "Opponent’s".encode("utf-8").decode("latin-1")
    result = SQLResult(query="q", columns=["text"], rows=[(broken,)])
    assert "Opponent’s" in format_result(result)


# ---------- retrieve_sql (with a fake LLM) ----------


@pytest.fixture
def use_llm(monkeypatch, fake_llm):
    def install(replies):
        llm = fake_llm(replies)
        llm.models = []  # the model names retrieve_sql asked for

        def get_llm(model=None):
            llm.models.append(model)
            return llm

        monkeypatch.setattr(sql_retriever, "get_llm", get_llm)
        monkeypatch.setattr(sql_retriever, "system_prompt", lambda: "system prompt")
        return llm

    return install


def test_retrieve_sql_success_on_first_try(db, use_llm):
    llm = use_llm(["SELECT name FROM units WHERE toughness = '9'"])
    result = retrieve_sql("Which unit has toughness 9?")

    assert result.error is None
    assert result.rows == [("Carnifex",)]
    assert len(llm.calls) == 1


def test_retrieve_sql_uses_the_sql_model(db, use_llm):
    llm = use_llm(["SELECT 1"])
    retrieve_sql("question")

    assert llm.models == [sql_retriever.SQL_MODEL]


def test_retrieve_sql_sends_system_prompt_then_question(db, use_llm):
    llm = use_llm(["SELECT 1"])
    retrieve_sql("Which unit has toughness 9?")

    system_message, human_message = llm.calls[0]
    assert system_message.content == "system prompt"
    assert "Which unit has toughness 9?" in human_message.content


def test_retrieve_sql_handles_fenced_reply(db, use_llm):
    use_llm(["```sql\nSELECT name FROM units WHERE toughness = '9';\n```"])
    result = retrieve_sql("question")

    assert result.query == "SELECT name FROM units WHERE toughness = '9'"
    assert result.rows == [("Carnifex",)]


def test_retrieve_sql_retries_with_the_error_message(db, use_llm):
    llm = use_llm(["SELECT * FROM nope", "SELECT name FROM units WHERE toughness = '9'"])
    result = retrieve_sql("question")

    assert result.error is None
    assert result.rows == [("Carnifex",)]
    assert len(llm.calls) == 2
    assert "no such table" in llm.calls[1][-1].content  # the error was sent back to the LLM


def test_retrieve_sql_gives_up_after_max_attempts(db, use_llm):
    llm = use_llm(["SELECT * FROM nope", "SELECT * FROM nope2"])
    result = retrieve_sql("question")

    assert "no such table" in result.error
    assert result.rows == []
    assert result.query == "SELECT * FROM nope2"
    assert len(llm.calls) == sql_retriever.SQL_MAX_ATTEMPTS


def test_retrieve_sql_rejects_non_select_then_recovers(db, use_llm):
    llm = use_llm(["DROP TABLE units", "SELECT name FROM units"])
    result = retrieve_sql("question")

    assert result.error is None
    assert "Only SELECT queries are allowed." in llm.calls[1][-1].content
    assert len(result.rows) == 5