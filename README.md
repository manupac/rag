# Warhammer 40,000 RAG Chatbot

A multi-turn chatbot that answers questions about **Warhammer 40,000** by combining two knowledge sources:

- **The Core Rules PDF**, answered with retrieval-augmented generation (RAG) over a vector store.
- **A SQLite database of faction and unit data** (datasheets, weapons, abilities, points, stratagems...), answered by having an LLM write and run read-only SQL.

A small LLM-based router decides whether each question needs the rulebook, the database, or **both** (decomposing cross-source questions into targeted sub-questions for each channel). Every answer is grounded in what was retrieved, so the bot says "I couldn't find it" instead of guessing.

## How it works

```text
                       ┌─────────────────────┐
user question ───────► │1. Rewrite follow-up │  → fix uncontextualized questions
                       └──────────┬──────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ 2. Router (LLM)     │  RULEBOOK or DATABASE
                       └───────┬─────┬───────┘
                RULEBOOK       │     │       DATABASE
             ┌─────────────────┘     └──────────────────┐
             ▼                                          ▼
  Vector (or hybrid) search                   LLM writes a SELECT query,
  over rulebook chunks (Chroma)               runs it read-only on SQLite
             │                                (retries once with the error)
             └──────────────────┬───────────────────────┘
                                ▼
              3. LLM answers using only the retrieved context
```

**Key design choices**

- **Hybrid PDF parsing.** The rulebook is converted to Markdown with both `marker-pdf` and `PyMuPDF`, then merged page by page. Marker's output is preferred, but PyMuPDF is used for any page where marker captured less than 85% of the words.

- **Structure-aware chunking.** The Markdown is split on headers first, then into 800-character chunks (100 overlap). The header path (e.g. `Section > Subsection`) is prepended to every chunk so each piece knows where it comes from, and the bot can cite sections in its answers.

- **Vector or hybrid retrieval.** Dense search with Gemini embeddings in ChromaDB, optionally fused with BM25 keyword search via reciprocal rank fusion (helps with exact terms like `CP`, `VP` or keywords such as *Grenades*).

- **Safe text-to-SQL.** The database is opened in read-only mode, only `SELECT`/`WITH` statements are accepted, results are capped at 50 rows, and failed queries are sent back to the LLM once for self-correction. The prompt includes the data dictionary, the live schema with sample rows, and the list of factions.

- **Cross-source routing (`BOTH`).** Questions that combine unit stats/keywords with core game mechanics (e.g. *"What roll do I need to wound a Carnifex with Strength 5?"*) are decomposed into a rulebook sub-question and a database sub-question, retrieved from both sources, and answered in a single prompt.

- **Multi-turn memory.** The last 5 question/answer pairs are kept, and follow-ups are rewritten into standalone questions before routing and retrieval.

- **Built-in evaluation.** Routing accuracy (overall and per-route), retrieval hit rate, SQL success rate and LLM-judged answer correctness (judged by a *different* model than the one that answers), plus a retrieval ablation study.

## Project structure

```text
main.py                  CLI entry point: python main.py
src/
  config.py              All paths, model names and tunable settings
  prompts.py             All prompt templates
  parser.py              PDF → Markdown (marker + PyMuPDF, merged)
  chunker.py             Markdown cleaning + header-aware chunking
  indexer.py              Embeds chunks into ChromaDB (rate-limit aware)
  vectorstore.py         Vector store, BM25 index, hybrid search (RRF)
  router.py              RULEBOOK / DATABASE / BOTH routing (structured output)
  sql_retriever.py       Text-to-SQL channel
  chatbot.py             The full pipeline + conversation history
  chat.py                Terminal chat UI (rich)
  llm.py                 Gemini models + retry helper
  eval_dataset.py        Generates evaluation questions
  evaluation.py          Runs the evaluation and writes metrics
  retrieval_ablation.py  Compares chunk size / overlap / k / search type
  inspect_retrieval.py   Quickly check what retrieval returns
eval/                    Question sets and saved results (committed)
tests/                   pytest suite
```

## Getting started

### Prerequisites

- [Pixi](https://pixi.sh) (manages Python 3.11 and all dependencies)
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini). The free tier works; the indexer and evaluation include rate-limit handling.
- The data files below (not included in the repo, see note)

### Installation

```bash
git clone https://github.com/manupac/rag.git
cd rag
pixi install
```

### Data

The `data/` folder is git-ignored. Place the following files in it:

```text
data/
└── raw/
    ├── Core Rules.pdf                  # the Warhammer 40k Core Rules
    ├── wahadb.sqlite                   # faction / unit database
    └── SQL_AGENT_DATA_DICTIONARY.md    # description of the database tables
```

### Build the knowledge base

```bash
pixi run python main.py parse     # PDF → Markdown (slow the first time: runs marker)
pixi run python main.py chunk     # optional: preview chunk statistics
pixi run python main.py index     # embed chunks into ChromaDB
```

### Chat

```bash
pixi run python main.py chat
```

In the chat, type `debug` to toggle a view of the rewritten question, the chosen route, and the retrieved chunks or generated SQL; `reset` starts a new conversation; `quit` exits.

Example questions:

- "What is the maximum percentage of points that can go into Strategic Reserves?" → rulebook
- "How does the Grenade stratagem work?" → rulebook
- "Which units in have the highest toughness?" → database
- "Can a Carnifex shoot while in Engagement Range of an enemy unit, and is there a penalty?" → both (unit keywords from database + Big Guns Never Tire from rulebook)
- "And what about their weapons?" → follow-up, rewritten using the conversation

## Commands

| Command | What it does |
|---|---|
| `parse` | Convert the rulebook PDF to Markdown |
| `chunk` | Show chunking statistics and the first chunk |
| `index` | Build the ChromaDB vector store |
| `inspect` | Check what retrieval returns for a question |
| `chat` | Talk to the bot in the terminal |
| `sql` | Test the text-to-SQL channel on its own |
| `eval-generate [ids]` | Generate evaluation questions (pass IDs like `q3 q7` to regenerate only those) |
| `eval` | Run the full evaluation |
| `ablation-generate` | Generate the questions for the retrieval ablation |
| `ablation` | Compare retrieval settings and save a CSV |

## Evaluation

Run `python main.py eval` to send every question in `eval/questions.json` (LLM-generated) and `eval/manual_questions.json` (hand-written, including multi-turn and cross-source BOTH questions) through the bot.

Metrics:

- **Routing accuracy:** did the router pick the right source (RULEBOOK, DATABASE, or BOTH), tracked overall and per route?
- **Retrieval hit rate:** was the chunk the question came from among the retrieved ones?
- **SQL success rate:** did the generated query run and return rows?
- **Answer correctness:** an LLM judge compares the answer to a reference answer.
- **Multi-turn correctness:** questions can carry a history to test follow-ups.

Saved runs with `k=5` and `k=8` both scored 100% on routing, retrieval, SQL success and answer correctness on the initial 17-question single-source set, and the suite now includes cross-source (BOTH) questions in `eval/manual_questions.json`. That is still a small set, so treat it as a sanity check rather than a benchmark.

## Retrieval ablation

`python main.py ablation` changes one setting at a time and measures how much of each question's source chunk is covered by the retrieved text (40 questions, no chat-model calls). Results from `eval/ablation_results.csv`:

| Experiment | Hit rate | Avg. context (chars) |
|---|---:|---:|
| baseline (800 / 100 / k=5 / vector) | 0.95 | 2,691 |
| `k=30` | 0.92 | 1,591 |
| `k=8` | 1.00 | 4,235 |
| chunk size 400 | 0.92 | 1,619 |
| chunk size 1200 | 0.97 | 3,272 |
| overlap 0 | 0.95 | 2,687 |
| hybrid search | 0.95 | 2,850 |

The default in `config.py` is `TOP_K = 8`, the best-performing setting, at the cost of more context tokens.

## Configuration

Everything tunable lives in `src/config.py`: model names, chunk size and overlap, `TOP_K`, `SEARCH_TYPE` (`"vector"` or `"hybrid"`), history length, SQL row/attempt limits, and evaluation settings.

## Development

```bash
pixi run test        # run the pytest suite
```

Code style is enforced with Ruff (line length 120) via pre-commit.

## Tech stack

Python 3.11 · LangChain · Google Gemini (chat + embeddings) · ChromaDB · BM25 (`rank-bm25`) · SQLite · marker-pdf · PyMuPDF · Rich · pytest · Pixi
