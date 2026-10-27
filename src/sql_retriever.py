"""The SQL channel: the LLM writes a query from the question, we run it on the database.

The database is opened read-only, so a generated query can never change or delete data.
If a query fails, the error is sent back to the LLM so it can correct itself.
"""

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import SQL_DICTIONARY_PATH, SQL_MAX_ATTEMPTS, SQL_MAX_ROWS, SQL_MODEL, SQLITE_PATH
from llm import get_llm, text_of
from prompts import SQL_FIX_USER, SQL_SYSTEM, SQL_USER

SAMPLE_ROWS = 2  # example rows per table shown to the LLM
SAMPLE_CHARS = 60  # long text values in the examples are cut to this length


@dataclass
class SQLResult:
    query: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False  # True if there were more than SQL_MAX_ROWS rows
    error: str | None = None  # set if every attempt failed


def fix_text(value):
    """Repair text that was saved with the wrong encoding (e.g. 'Opponentâs' -> 'Opponent’s')."""
    if isinstance(value, str):
        try:
            return value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value  # already fine
    return value


def connect() -> sqlite3.Connection:
    # mode=ro opens the file read-only: INSERT, UPDATE, DELETE and DROP all fail
    return sqlite3.connect(f"{SQLITE_PATH.as_uri()}?mode=ro", uri=True)


@lru_cache(maxsize=1)
def schema_description() -> str:
    """Every table with its real columns and a few example rows, read from the database itself."""
    parts = []
    with closing(connect()) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for table in tables:
            columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            rows = conn.execute(f'SELECT * FROM "{table}" LIMIT {SAMPLE_ROWS}').fetchall()
            examples = "\n".join(
                "  " + " | ".join(str(fix_text(v))[:SAMPLE_CHARS].replace("\n", " ") for v in row) for row in rows
            )
            parts.append(f"{table}({', '.join(columns)})\n{examples}")
    return "\n\n".join(parts)


@lru_cache(maxsize=1)
def faction_names() -> str:
    """All faction names, so the LLM recognizes when a question is about a faction."""
    with closing(connect()) as conn:
        return ", ".join(str(fix_text(r[0])) for r in conn.execute("SELECT name FROM factions ORDER BY name"))
    

@lru_cache(maxsize=1)
def system_prompt() -> str:
    return SQL_SYSTEM.format(
        dictionary=SQL_DICTIONARY_PATH.read_text(encoding="utf-8"),
        schema=schema_description(),
        factions=faction_names(),
        max_rows=SQL_MAX_ROWS,
    )


def extract_sql(text: str) -> str:
    """Remove the ```sql fences the LLM sometimes adds around the query."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip().rstrip(";").strip()


def run_query(query: str) -> tuple[list[str], list[tuple], bool]:
    if not re.match(r"^\s*(SELECT|WITH)\b", query, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    with closing(connect()) as conn:
        cursor = conn.execute(query)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(SQL_MAX_ROWS + 1)
    return columns, rows[:SQL_MAX_ROWS], len(rows) > SQL_MAX_ROWS


def retrieve_sql(question: str) -> SQLResult:
    """Generate a query for the question and run it, retrying with the error if it fails."""
    messages = [SystemMessage(system_prompt()), HumanMessage(SQL_USER.format(question=question))]
    query, error = "", None

    for _ in range(SQL_MAX_ATTEMPTS):
        query = extract_sql(text_of(get_llm(SQL_MODEL).invoke(messages)))
        try:
            columns, rows, truncated = run_query(query)
            return SQLResult(query=query, columns=columns, rows=rows, truncated=truncated)
        except (sqlite3.Error, ValueError) as e:
            error = str(e)
            messages += [AIMessage(query), HumanMessage(SQL_FIX_USER.format(error=error))]

    return SQLResult(query=query, columns=[], rows=[], error=error)


def format_result(result: SQLResult) -> str:
    """Turn the result into text for the answer prompt."""
    if result.error:
        return f"The query failed: {result.error}"
    if not result.rows:
        return "The query returned no rows."
    lines = [" | ".join(result.columns)]
    lines += [" | ".join("" if v is None else str(fix_text(v)) for v in row) for row in result.rows]
    if result.truncated:
        lines.append(f"(only the first {SQL_MAX_ROWS} rows are shown)")
    return "\n".join(lines)


def main():
    """Ask data questions and see the generated SQL and its result, without the chatbot."""
    print("Ask a question about the database ('quit' to exit).")
    while (question := input("\nQuestion: ").strip()).lower() != "quit":
        if not question:
            continue
        result = retrieve_sql(question)
        print(f"\nSQL:\n{result.query}\n")
        print(format_result(result))


if __name__ == "__main__":
    main()