# Warhammer 40,000 RAG Chatbot

A multi-turn chatbot that answers questions about **Warhammer 40,000** from two sources:

- **The Core Rules PDF**, through retrieval-augmented generation (RAG) over a vector store.
- **A SQLite database of factions and units** (datasheets, weapons, abilities, points, stratagems...), through LLM-written, read-only SQL.

A router decides for each question whether it needs the rulebook, the database, or **both**, so text-only, database-only and combined questions are all supported. Answers are grounded in what was retrieved: if nothing relevant is found, the bot says so instead of guessing.

LLM backend: Google AI Studio (Gemini), through LangChain.

## Authors

- Manuela Pacheco
- Sara Graça

## Quick start

**Requirements:** [Pixi](https://pixi.sh) and a [Google AI Studio](https://aistudio.google.com/) API key (needed even with the prebuilt index, because every question is embedded and answered through Gemini).

```bash
git clone <repo-url>
cd <repo>
pixi install
cp .env.example .env        # then put your key in .env: GOOGLE_API_KEY=...
```

**Data.** The parsed rulebook (`data/parsed/rulebook_final.md`) and the ready-built vector index (`data/chroma_db_rulebook/`) are committed, so you can skip the `parse` and `index` steps. The two database files are not in the repo and must be placed here:

```text
data/raw/wahadb.sqlite
data/raw/SQL_AGENT_DATA_DICTIONARY.md
```

**Chat:**

```bash
pixi run python main.py chat
```

In the chat: `debug` shows the rewritten question, the route, and the retrieved chunks or SQL; `reset` starts a new conversation; `quit` exits.

**Rebuilding (optional).** Only needed if you change the parsing or chunking settings:

```bash
pixi run python main.py parse    # PDF -> Markdown (slow, runs marker-pdf; needs data/raw/Core Rules.pdf)
pixi run python main.py index    # re-embeds the chunks; takes several minutes because of free-tier rate limits
```

The committed index was built from the committed Markdown with chunk size 800 and overlap 100 (`src/config.py`). The evaluation questions store the exact text of their source chunk, so if you rebuild with other settings, regenerate them with `eval-generate` and `ablation-generate`.

## How it works

```text
user question
     │
     ▼
1. Rewrite follow-ups into standalone questions (only if there is history)
     │
     ▼
2. Router (LLM, structured output): RULEBOOK | DATABASE | BOTH
     │            │                       │
 RULEBOOK     DATABASE                  BOTH (one sub-question for each source)
     │            │                       │
 vector/hybrid   LLM writes a        both channels run
 search (Chroma) read-only SELECT,
     │           runs it on SQLite
     │           (1 retry on error)
     └────────────┴───────────┬───────────┘
                              ▼
3. LLM answers using only the retrieved context
```

## Technical choices and why

**Parsing.** The rulebook is converted with both `marker-pdf` and `PyMuPDF`, then merged page by page. Marker gives better structure (headers, tables) but sometimes drops text, so for any page where it captured less than 85% of PyMuPDF's words, PyMuPDF's version is used instead.

**Chunking.** The Markdown is split on headers first, then into chunks of 800 characters with 100 of overlap, so a chunk never mixes two sections. The header path (`Section > Subsection`) is added at the start of each chunk, which keeps long sections identifiable and lets the bot cite the section it answers from.

**Retrieval.** Chunks are embedded with Gemini and stored in ChromaDB, and the top 8 are retrieved. We picked `k=8` because it had the best coverage in the ablation below. Hybrid search (vector plus BM25, merged with reciprocal rank fusion) is implemented and can be switched on with `SEARCH_TYPE`, but it gave no gain on our questions, so vector search is the default.

**Router.** A single LLM call with structured output picks RULEBOOK, DATABASE or BOTH, and also writes a standalone sub-question for each source it needs. BOTH exists for questions like "What roll do I need to wound a Carnifex with Strength 5?", which needs the Toughness from the database and the wound-roll rule from the rulebook. If the router returns nothing valid, the question goes to the rulebook.

**Text-to-SQL.** The prompt includes the data dictionary, the live schema with a few sample rows, and the list of factions. The generated query can never change the data: the database is opened read-only, only `SELECT` and `WITH` are accepted, and results are capped at 50 rows. If a query fails, the error is sent back to the LLM for one correction.

**Multi-turn.** The bot keeps the last 5 question/answer pairs. Follow-ups such as "and its save?" are rewritten into standalone questions before routing and retrieval, because the router and retrievers only see one question at a time.

**Models.** Rewriting, routing and answering use `gemini-3.5-flash-lite`, and SQL generation uses `gemini-3.6-flash` because it needs more reasoning. Answers are judged by a different model (`gemini-3.7-flash`), so the bot does not grade itself.

## Evaluation

```bash
pixi run python main.py eval-generate   # generated questions (rulebook chunks and database rows)
pixi run python main.py eval            # run everything; resumable
```

**Dataset** (`eval/`): LLM-generated questions with reference answers (`questions.json`) and hand-written ones (`manual_questions.json`). The manual set covers the four benchmark questions from the course, multi-turn follow-ups, an unanswerable question, and 8 cross-source (BOTH) questions.

**Metrics:**

- **Routing accuracy:** did the router pick the expected source? Reported overall and per route.
- **Retrieval hit rate:** was the chunk the question came from among the retrieved ones? (generated rulebook questions only)
- **SQL success rate:** did the generated query run and return rows?
- **Answer correctness:** an LLM judge compares the answer with the reference answer.
- **Multi-turn correctness:** same, for questions with a history.

**Why custom metrics and not RAGAS:** the checks above are mapped to this system's own failure points (router, retriever, SQL, answer), and RAGAS metrics make several LLM calls per question, which is not practical on the free-tier quota (a few dozen requests per day per model). The evaluation script is resumable across days for the same reason.

### Results

<!-- TODO: fill in from eval/results.json once the final run is complete -->

| Metric | Overall | RULEBOOK | DATABASE | BOTH |
|---|---|---|---|---|
| Routing accuracy | x/x | x/x | x/x | x/x |
| Retrieval hit rate | x/x | x/x | n/a | x/x |
| SQL success rate | x/x | n/a | x/x | x/x |
| Answer correctness | x/x | x/x | x/x | x/x |
| Multi-turn correctness | x/x | | | |

**Failures and analysis:** <!-- TODO: which questions failed and why -->

**Limitations:** the question set is small, so these numbers are a sanity check and not a benchmark. Gemini is run at its default temperature, so scores vary slightly between runs. The judge is an LLM and can be wrong.

### Retrieval ablation

```bash
pixi run python main.py ablation-generate
pixi run python main.py ablation
```

One setting is changed at a time on 40 generated rulebook questions, with no chat-model calls. A question counts as a hit if at least 50% of its source chunk's sentences appear in the retrieved text. Average context size is shown because more context costs more tokens.

| Experiment | Hit rate | Avg. context (chars) |
|---|---:|---:|
| baseline (800 / 100 / k=5 / vector) | 0.95 | 2,691 |
| k=3 | 0.93 | 1,591 |
| k=8 | 1.00 | 4,235 |
| chunk size 400 | 0.93 | 1,619 |
| chunk size 1200 | 0.98 | 3,272 |
| overlap 0 | 0.95 | 2,687 |
| hybrid search | 0.95 | 2,850 |

`TOP_K = 8` is the best-performing setting, at the cost of about 60% more context than `k=5`. Small chunks (400) lose context, and hybrid search does not help on this question set.

## Configuration

Everything tunable is in `src/config.py`: model names, chunk size and overlap, `TOP_K`, `SEARCH_TYPE` (`"vector"` or `"hybrid"`), history length, SQL limits and evaluation settings.

## Project structure

```text
main.py                  CLI entry point (python main.py <command>)
src/
  config.py              Paths, model names, settings
  prompts.py             All prompt templates
  parser.py              PDF -> Markdown (marker + PyMuPDF merged)
  chunker.py             Markdown cleaning and header-aware chunking
  indexer.py             Embeds chunks into ChromaDB (rate-limit aware)
  vectorstore.py         Vector store, BM25 index, hybrid search
  router.py              RULEBOOK / DATABASE / BOTH routing
  sql_retriever.py       Text-to-SQL channel
  chatbot.py             Full pipeline and conversation history
  chat.py                Terminal chat UI
  llm.py                 Gemini models and retry helper
  eval_dataset.py        Generates evaluation questions
  evaluation.py          Runs the evaluation
  retrieval_ablation.py  Compares chunking and retrieval settings
data/parsed/             Parsed rulebook Markdown (committed)
data/chroma_db_rulebook/ Ready-built vector index (committed)
data/raw/                PDF, SQLite database and data dictionary (not committed)
eval/                    Question sets and results
tests/                   pytest suite (LLM calls are faked)
```

## Development

```bash
pixi run test
```

Code style: Ruff (line length 120) through pre-commit.

## Known limitations

- The free-tier quota limits how fast indexing and evaluation can run.
- Text-to-SQL can still write a wrong query for unusual questions. The bot then answers from a wrong or empty result, so check the SQL with `debug` on.
- Images and diagrams in the rulebook are not indexed (only text).
