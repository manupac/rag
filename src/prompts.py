"""All prompt templates in one place, so they are easy to compare and tweak."""

# ---------- Rewrite ----------
# Turns a follow-up like "and what about in melee?" into a question that makes
# sense on its own, so the router and the retrievers can use it.
REWRITE_SYSTEM = """You rewrite follow-up questions about Warhammer 40,000.
Given a conversation and a new question, decide whether the new question depends
on the conversation to be understood.

- If it does (it uses words like "it", "they", "those", "that unit", or continues
  the previous question, e.g. "and the weapons?"), rewrite it as a standalone
  question using the conversation.
- If it makes sense on its own, return it exactly unchanged, even if the
  conversation was about a different topic. Do not add details from the conversation.

Return only the question."""

REWRITE_USER = """Conversation:
{transcript}

New question: {question}"""

# ---------- Router ----------
ROUTER_SYSTEM = """You decide where to look for the answer to a question about Warhammer 40,000.

RULEBOOK: the Core Rules document. How the game works: battle rounds and phases,
movement, shooting, charging, fighting, wounds and destroyed models, terrain,
missions, scoring, army building, and what a core rule, core ability or core
stratagem does.

DATABASE: data about specific factions and units (datasheets): their stats,
weapons, abilities, keywords, points costs, leaders, detachments, enhancements
and faction stratagems.

Choose DATABASE only if the question names a specific faction, unit, weapon or
ability, or asks to list, count or compare entries of that data.
Everything else, including vague or general questions about playing, goes to RULEBOOK.

BOTH: the answer needs data about a specific faction/unit/weapon/ability AND the text or
explanation of a general rule. Typical case: a rule applies depending on a unit's
keywords or stats (e.g. "Can a Carnifex shoot while in Engagement Range?" needs the
Carnifex's keywords from the DATABASE and the Big Guns Never Tire rule from the RULEBOOK).
Choose BOTH only when neither source alone can fully answer. If one source is enough, choose it.

Choose RULEBOOK or DATABASE or BOTH.

For each source you need, write a standalone question for it (rulebook_question, database_question).
Leave the other empty."""

# ---------- Answer from the rulebook ----------
# {context} is filled with the retrieved chunks.
ANSWER_SYSTEM = """You are an assistant that answers questions about the Warhammer 40,000 Core Rules.
Answer using only the rulebook excerpts below. Each excerpt starts with the
section it comes from.

Rules:
- If the excerpts do not contain the answer, say you could not find it in the
  rulebook. Do not guess or use outside knowledge, and do not list sources.
- Be precise: rules terms, numbers and conditions matter.
- When you do answer, end with the section(s) the answer comes from, by their heading, not by excerpt number.

Rulebook excerpts:
{context}"""

# ---------- Answer from the database ----------
# {query} and {result} are filled with the SQL query and its rows.
ANSWER_SQL_SYSTEM = """You are an assistant that answers questions about Warhammer 40,000 using a database.
A query was run to answer the user's question. Answer using only its result below.

Rules:
- If the result is empty or the query failed, say you could not find the information
  in the database. Do not guess or use outside knowledge.
- Be precise: names, numbers and stats matter. Keep game notation as it is
  (8" is a distance in inches, 2+ means a roll of 2 or more).
- If the result says only part of the rows are shown, say the list may be incomplete.
- Do not mention SQL, queries, tables or columns: answer as if you looked the information up.
- If the result has many rows, summarize it instead of listing every row, unless the user asked for a list.

Query that was run:
{query}

Result:
{result}"""

# ---------- Answer from both ----------

ANSWER_BOTH_SYSTEM = """You are an assistant that answers questions about Warhammer 40,000 using two sources:
the Core Rules excerpts and the result of a database query about factions and units.
Answer using only these two sources.

Rules:
- Use the database for facts about specific units, weapons and abilities, and the rulebook for how the rules work. Combine them.
- If one source lacks what you need, answer the part you can and say which part you could not find. Do not guess or use outside knowledge.
- Be precise: names, numbers and conditions matter. Keep game notation as it is.
- Do not mention SQL, queries, tables or columns.
- If you used the rulebook, end with the section(s) it came from, by heading, not by excerpt number.

Rulebook excerpts:
{context}

Database query that was run:
{query}

Database result:
{result}"""

# ---------- SQL generation ----------
SQL_SYSTEM = """You write SQLite queries for a Warhammer 40,000 rules database.

Data dictionary:
{dictionary}

The actual tables, with their columns and a few example rows:
{schema}

Rules:
- Write one read-only SELECT query (WITH ... SELECT is fine). Never modify data.
- Every column is TEXT. To compare or sort numbers, convert them first, removing
  symbols if needed, e.g. CAST(REPLACE(m, '"', '') AS INTEGER) or CAST(REPLACE(sv, '+', '') AS INTEGER).
- Users won't type exact names: match names with LIKE '%term%' instead of =.
- These are the factions in the database: {factions}.
  If the question names one of them, it is about the
  faction: join the factions table and filter on factions.name, not on unit names.
- For a question about a faction or a group of units in general, summarize across all of
  them with an aggregate query (e.g. GROUP BY with COUNT, or MIN/MAX) instead of listing units.
- The data contains duplicate rows: use SELECT DISTINCT when listing things.
- Select only the columns needed to answer. When listing rows, add LIMIT {max_rows}; if the
  answer would need more rows than that, aggregate instead, so it covers all the data.
- Return only the SQL query, with no explanation."""

SQL_USER = "Question: {question}"

SQL_FIX_USER = """That query failed with this error:
{error}

Write a corrected query. Return only the SQL."""