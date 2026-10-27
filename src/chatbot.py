"""The chatbot: rewrite the question, route it, retrieve from the rulebook or the database,
build the prompt and call the LLM.

The Chatbot object keeps the conversation history, which is what makes it multi-turn.
"""

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config import MAX_HISTORY_TURNS
from llm import get_llm, text_of
from prompts import (
    ANSWER_BOTH_SYSTEM,
    ANSWER_SQL_SYSTEM,
    ANSWER_SYSTEM,
    REWRITE_SYSTEM,
    REWRITE_USER,
)
from router import BOTH, DATABASE, RULEBOOK, route
from sql_retriever import SQLResult, format_result, retrieve_sql
from vectorstore import retrieve


@dataclass
class Response:
    answer: str
    standalone_question: str  # the question actually used for routing and retrieval
    route: str  # RULEBOOK, DATABASE, or BOTH
    sources: list[Document] = field(default_factory=list)  # rulebook chunks
    sql: SQLResult | None = None  # query and rows


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, start=1))


class Chatbot:
    def __init__(self):
        self.history: list[BaseMessage] = []

    def reset(self) -> None:
        self.history = []

    def _recent_history(self) -> list[BaseMessage]:
        return self.history[-2 * MAX_HISTORY_TURNS :]  # one turn = question + answer

    def _rewrite(self, question: str) -> str:
        """Make a follow-up question understandable without the conversation."""
        if not self.history:
            return question
        transcript = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in self._recent_history()
        )
        messages = [
            SystemMessage(REWRITE_SYSTEM),
            HumanMessage(REWRITE_USER.format(transcript=transcript, question=question)),
        ]
        return text_of(get_llm().invoke(messages)).strip()

    def ask(self, question: str) -> Response:
        # 1. Rewrite follow-ups into standalone questions (multi-turn)
        standalone = self._rewrite(question)

        # 2. Route: rulebook, database, or both?
        plan = route(standalone)
        use_rules = plan.route in (RULEBOOK, BOTH)
        use_db = plan.route in (DATABASE, BOTH)

        # 3. Retrieve from the active channels
        docs = (
            retrieve(getattr(plan, "rulebook_question", None) or standalone)
            if use_rules
            else []
        )
        sql = (
            retrieve_sql(getattr(plan, "database_question", None) or standalone)
            if use_db
            else None
        )

        # Build prompt instructions based on the retrieved sources
        if use_rules and use_db and sql is not None:
            system = ANSWER_BOTH_SYSTEM.format(
                context=format_context(docs),
                query=sql.query,
                result=format_result(sql),
            )
        elif use_db and sql is not None:
            system = ANSWER_SQL_SYSTEM.format(
                query=sql.query,
                result=format_result(sql),
            )
        else:
            system = ANSWER_SYSTEM.format(context=format_context(docs))

        # 4. Prompt = instructions + context, then history, then the question
        messages = [
            SystemMessage(system),
            *self._recent_history(),
            HumanMessage(question),
        ]

        # 5. Generate the answer and remember the turn
        answer = text_of(get_llm().invoke(messages))
        self.history += [HumanMessage(question), AIMessage(answer)]

        return Response(
            answer=answer,
            standalone_question=standalone,
            route=plan.route,
            sources=docs,
            sql=sql,
        )