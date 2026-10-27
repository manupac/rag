"""Generate the evaluation dataset (a small set of questions with reference answers).

Rulebook questions: random chunks of the rulebook are shown to the LLM, which writes a
question that chunk can answer, plus a short reference answer.
Database questions: same idea, but with a few random rows of a random table.

The result is saved to eval/questions.json. Read it before you evaluate: the LLM
sometimes writes a weak question. To replace weak ones, pass their ids:
    python main.py eval-generate q3 q7
Only those questions are regenerated (same type, new random chunk or row); the others are kept.
Hand-written questions go in eval/manual_questions.json instead, which this script
never overwrites.

If you change the chunking, run this again: rulebook questions store the exact text
of their chunk, which no longer matches once the chunks change.
"""

import json
import random
from contextlib import closing

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from chunker import create_chunks
from config import EVAL_MANUAL_PATH, EVAL_N_RULEBOOK, EVAL_N_SQL, EVAL_QUESTIONS_PATH, EVAL_SEED
from llm import call_with_retry, get_llm
from router import DATABASE, RULEBOOK
from sql_retriever import connect, fix_text

MIN_CHUNK_CHARS = 300  # shorter chunks (headings, fragments) make poor questions
SAMPLE_ROWS = 3  # rows of a table shown to the LLM
MAX_CELL_CHARS = 300  # long text values are cut to this length
# metadata tables, and link tables that only contain ids
SKIP_TABLES = {
    "last_update",
    "source",
    "detachments_chapter_dp",
    "datasheets_leader",
    "datasheets_stratagems",
    "datasheets_enhancements",
    "datasheets_detachment_abilities",
}
MAX_REPLACE_TRIES = 5  # attempts to find a replacement that isn't already in the dataset

RULEBOOK_QA_SYSTEM = """You write test questions for a chatbot that answers from the Warhammer 40,000 Core Rules.
You are given one excerpt of the rulebook. Write ONE question that can be answered
using only that excerpt, and a short reference answer (1-3 sentences) based on it.

- The question must make sense on its own. Never mention "the excerpt" or "the text".
- Do not ask about a specific faction or unit.
- Phrase the question the way a player would, in your own words: do not copy the excerpt's wording."""

SQL_QA_SYSTEM = """You write test questions for a chatbot that answers from a database of
Warhammer 40,000 factions and units.
You are given a table's columns and a few of its rows. Write ONE question about ONE
specific row, and a short reference answer based on that row.

- Name the specific unit, weapon, faction or item the question is about, the way a player would.
- Ask only for facts in that single row (e.g. a stat, a cost, a rule text).
  Never ask for lists, counts or comparisons ("which units...", "how many..."):
  you only see a few rows, so your answer would be incomplete.
- The question must make sense on its own. Never mention "the table" or "the rows"."""


class QAPair(BaseModel):
    question: str
    answer: str


def generate_qa(system: str, user: str) -> QAPair | None:
    """Ask the LLM for a question/answer pair, returned as a typed object (no JSON parsing)."""
    generator = get_llm().with_structured_output(QAPair)
    return call_with_retry(generator.invoke, [SystemMessage(system), HumanMessage(user)])


def generate_rulebook_items(n: int, rng: random.Random) -> list[dict]:
    chunks = [c for c in create_chunks() if len(c.page_content) >= MIN_CHUNK_CHARS]
    items = []
    for chunk in rng.sample(chunks, min(n, len(chunks))):
        try:
            qa = generate_qa(RULEBOOK_QA_SYSTEM, chunk.page_content)
        except ValueError as e:
            print(f"Skipped a rulebook chunk: {e}")
            continue
        if qa is None:
            print("Skipped a rulebook chunk: no valid answer from the model")
            continue
        items.append(
            {
                "question": qa.question.strip(),
                "expected_route": RULEBOOK,
                "reference_answer": qa.answer.strip(),
                "source_text": chunk.page_content,  # used to check retrieval found it
            }
        )
    return items


def sample_table_rows(table: str, rng: random.Random) -> tuple[list[str], list[tuple]]:
    """Pick random rows from the whole table (not just the first ones)."""
    with closing(connect()) as conn:
        n_rows = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        columns = [c[0] for c in conn.execute(f'SELECT * FROM "{table}" LIMIT 1').description]
        offsets = rng.sample(range(n_rows), min(SAMPLE_ROWS, n_rows))
        rows = [conn.execute(f'SELECT * FROM "{table}" LIMIT 1 OFFSET {o}').fetchone() for o in offsets]
    return columns, rows


def generate_sql_items(n: int, rng: random.Random) -> list[dict]:
    with closing(connect()) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    tables = [t for t in tables if t not in SKIP_TABLES]

    order = []
    while len(order) < n and tables:  # spread the questions over the tables
        shuffled = tables[:]
        rng.shuffle(shuffled)
        order += shuffled

    items = []
    for table in order[:n]:
        columns, rows = sample_table_rows(table, rng)
        if not rows:
            continue
        lines = [" | ".join(str(fix_text(v))[:MAX_CELL_CHARS].replace("\n", " ") for v in row) for row in rows]
        prompt = f"Table: {table}\nColumns: {', '.join(columns)}\nRows:\n" + "\n".join(lines)
        try:
            qa = generate_qa(SQL_QA_SYSTEM, prompt)
        except ValueError as e:
            print(f"Skipped table {table}: {e}")
            continue
        if qa is None:
            print(f"Skipped table {table}: no valid answer from the model")
            continue
        items.append(
            {
                "question": qa.question.strip(),
                "expected_route": DATABASE,
                "reference_answer": qa.answer.strip(),
                "source_table": table,
            }
        )
    return items


def load_questions(path=EVAL_QUESTIONS_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run 'python main.py eval-generate' first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_manual_questions(path=EVAL_MANUAL_PATH) -> list[dict]:
    """Hand-written questions (multi-turn, unanswerable, tricky routing). Optional."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_questions(items: list[dict]) -> None:
    EVAL_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_QUESTIONS_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(items)} questions to {EVAL_QUESTIONS_PATH}")
    for item in items:
        print(f"  {item['id']} [{item['expected_route']}] {item['question']}")


def replace_questions(ids: list[str]) -> None:
    """Regenerate only the given questions, keeping the others. The new ones are of the same type."""
    items = load_questions()
    positions = {item["id"]: i for i, item in enumerate(items)}
    unknown = [qid for qid in ids if qid not in positions]
    if unknown:
        raise SystemExit(f"Unknown question ids: {', '.join(unknown)}")

    rng = random.Random()  # not seeded, so the replacements differ from the deleted questions
    for qid in ids:
        old = items[positions[qid]]
        used = {item.get("source_text") for item in items} | {item["question"] for item in items}
        for _ in range(MAX_REPLACE_TRIES):
            generate = generate_rulebook_items if old["expected_route"] == RULEBOOK else generate_sql_items
            new = generate(1, rng)
            if new and new[0].get("source_text", new[0]["question"]) not in used:
                break
        else:
            print(f"Could not generate a new question for {qid}; kept the old one.")
            continue
        new[0]["id"] = qid
        items[positions[qid]] = new[0]
        print(f"Replaced {qid}: {new[0]['question']}")
    save_questions(items)


def main(replace_ids: list[str] | None = None):
    if replace_ids:
        replace_questions(replace_ids)
        return
    rng = random.Random(EVAL_SEED)  # fixed seed: the same chunks and rows are picked each time
    items = generate_rulebook_items(EVAL_N_RULEBOOK, rng) + generate_sql_items(EVAL_N_SQL, rng)
    for i, item in enumerate(items, start=1):
        item["id"] = f"q{i}"
    save_questions(items)


if __name__ == "__main__":
    main()