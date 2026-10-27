import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import chatbot
from chatbot import Chatbot, format_context
from prompts import REWRITE_SYSTEM
from router import Route
from sql_retriever import SQLResult


@pytest.fixture
def make_bot(monkeypatch, fake_llm):
    """Builds a Chatbot whose LLM, router and retrievers are all fakes.

    Returns (bot, llm, routed_questions, retrieved_questions).
    """

    def build(replies, route_result="RULEBOOK"):
        llm = fake_llm(replies)
        routed, retrieved = [], []
        plan = Route(route=route_result) if isinstance(route_result, str) else route_result

        monkeypatch.setattr(chatbot, "get_llm", lambda: llm)
        monkeypatch.setattr(chatbot, "route", lambda q: routed.append(q) or plan)

        def fake_retrieve(q):
            retrieved.append(q)
            return [Document(page_content="Shooting Phase > Hit Roll\n\nRoll a D6.")]

        monkeypatch.setattr(chatbot, "retrieve", fake_retrieve)
        monkeypatch.setattr(
            chatbot,
            "retrieve_sql",
            lambda q: SQLResult(query=f"SELECT 1 -- {q}", columns=["a"], rows=[(1,)]),
        )
        return Chatbot(), llm, routed, retrieved

    return build


# ---------- format_context ----------


def test_format_context_numbers_the_chunks():
    docs = [Document(page_content="first"), Document(page_content="second")]
    assert format_context(docs) == "[1] first\n\n[2] second"


def test_format_context_of_nothing_is_empty():
    assert format_context([]) == ""


# ---------- history ----------


def test_recent_history_keeps_only_the_last_turns():
    bot = Chatbot()
    bot.history = [HumanMessage(f"m{i}") if i % 2 == 0 else AIMessage(f"m{i}") for i in range(20)]

    recent = bot._recent_history()
    assert len(recent) == 2 * chatbot.MAX_HISTORY_TURNS
    assert recent == bot.history[-2 * chatbot.MAX_HISTORY_TURNS :]


def test_reset_clears_history():
    bot = Chatbot()
    bot.history = [HumanMessage("q"), AIMessage("a")]
    bot.reset()
    assert bot.history == []


# ---------- rewrite ----------


def test_rewrite_returns_question_untouched_without_history(fake_llm, monkeypatch):
    llm = fake_llm([])  # any LLM call would fail the test
    monkeypatch.setattr(chatbot, "get_llm", lambda: llm)

    assert Chatbot()._rewrite("How does charging work?") == "How does charging work?"
    assert llm.calls == []


# ---------- ask: first question, rulebook route ----------


def test_first_question_skips_rewrite_and_uses_rulebook(make_bot):
    bot, llm, routed, retrieved = make_bot(["The answer."])
    response = bot.ask("How does charging work?")

    assert len(llm.calls) == 1  # just the answer, no rewrite call
    assert routed == ["How does charging work?"]
    assert retrieved == ["How does charging work?"]
    assert response.answer == "The answer."
    assert response.route == "RULEBOOK"
    assert response.standalone_question == "How does charging work?"
    assert len(response.sources) == 1
    assert response.sql is None


def test_answer_prompt_contains_chunks_and_original_question(make_bot):
    bot, llm, _, _ = make_bot(["ok"])
    bot.ask("How does charging work?")

    messages = llm.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert "Roll a D6." in messages[0].content
    assert messages[-1].content == "How does charging work?"


def test_turn_is_remembered(make_bot):
    bot, _, _, _ = make_bot(["The answer."])
    bot.ask("question")

    assert [type(m) for m in bot.history] == [HumanMessage, AIMessage]
    assert bot.history[0].content == "question"
    assert bot.history[1].content == "The answer."


# ---------- ask: follow-up question ----------


def test_follow_up_is_rewritten_before_routing_but_answered_as_asked(make_bot):
    bot, llm, routed, retrieved = make_bot(
        ["First answer.", "What weapons does a Carnifex have?", "Second answer."]
    )
    bot.ask("Tell me about the Carnifex")
    response = bot.ask("and its weapons?")

    # routing and retrieval use the rewritten question
    assert routed[1] == "What weapons does a Carnifex have?"
    assert retrieved[1] == "What weapons does a Carnifex have?"
    assert response.standalone_question == "What weapons does a Carnifex have?"

    # call 1 is the rewrite: its instructions plus the conversation transcript
    rewrite_messages = llm.calls[1]
    assert rewrite_messages[0].content == REWRITE_SYSTEM
    assert "User: Tell me about the Carnifex" in rewrite_messages[1].content
    assert "Assistant: First answer." in rewrite_messages[1].content

    # call 2 is the answer: history, then the original wording of the question
    answer_messages = llm.calls[2]
    assert answer_messages[1].content == "Tell me about the Carnifex"
    assert answer_messages[2].content == "First answer."
    assert answer_messages[-1].content == "and its weapons?"


# ---------- ask: database route ----------


def test_database_route_uses_sql_and_not_the_vector_store(make_bot):
    bot, llm, _, retrieved = make_bot(["The toughness is 9."], route_result="DATABASE")
    response = bot.ask("What is the toughness of a Carnifex?")

    assert response.route == "DATABASE"
    assert "SELECT 1" in response.sql.query
    assert response.sources == []
    assert retrieved == []  # the rulebook retriever was never called
    assert "SELECT 1" in llm.calls[0][0].content  # the query is part of the answer prompt


# ---------- ask: both route ----------


def test_both_route_uses_both_retrievers_and_sub_questions(make_bot):
    plan = Route(
        route="BOTH",
        rulebook_question="How does the wound roll work?",
        database_question="What is the toughness of a Carnifex?",
    )
    bot, llm, _, retrieved = make_bot(["You need a 5+."], route_result=plan)
    response = bot.ask("What roll do I need to wound a Carnifex with Strength 5?")

    assert response.route == "BOTH"
    assert retrieved == ["How does the wound roll work?"]
    assert "What is the toughness of a Carnifex?" in response.sql.query
    assert len(response.sources) == 1
    assert "Roll a D6." in llm.calls[0][0].content
    assert "SELECT 1" in llm.calls[0][0].content