from types import SimpleNamespace

import router
from llm import text_of
from prompts import ROUTER_SYSTEM
from router import DATABASE, RULEBOOK, route

# ---------- llm.text_of ----------


def test_text_of_returns_plain_string_content():
    assert text_of(SimpleNamespace(content="hello")) == "hello"


def test_text_of_joins_list_of_parts():
    message = SimpleNamespace(content=[{"text": "Hello "}, {"text": "world"}])
    assert text_of(message) == "Hello world"


def test_text_of_ignores_non_text_parts():
    message = SimpleNamespace(content=[{"text": "kept"}, {"type": "thinking"}, "stray string"])
    assert text_of(message) == "kept"


# ---------- router.route ----------


def use_fake_router_llm(monkeypatch, fake_llm, reply):
    llm = fake_llm([reply])
    monkeypatch.setattr(router, "get_llm", lambda: llm)
    return llm


def test_route_database(monkeypatch, fake_llm):
    use_fake_router_llm(monkeypatch, fake_llm, "DATABASE")
    assert route("What is the toughness of a Carnifex?") == DATABASE


def test_route_rulebook(monkeypatch, fake_llm):
    use_fake_router_llm(monkeypatch, fake_llm, "RULEBOOK")
    assert route("How does charging work?") == RULEBOOK


def test_route_is_case_and_whitespace_insensitive(monkeypatch, fake_llm):
    use_fake_router_llm(monkeypatch, fake_llm, "  database\n")
    assert route("anything") == DATABASE


def test_route_defaults_to_rulebook_on_unclear_reply(monkeypatch, fake_llm):
    use_fake_router_llm(monkeypatch, fake_llm, "I am not sure what you mean")
    assert route("anything") == RULEBOOK


def test_route_handles_list_of_parts_reply(monkeypatch, fake_llm):
    llm = fake_llm([[{"text": "DATABASE"}]])
    monkeypatch.setattr(router, "get_llm", lambda: llm)
    assert route("anything") == DATABASE


def test_route_sends_router_prompt_then_the_question(monkeypatch, fake_llm):
    llm = use_fake_router_llm(monkeypatch, fake_llm, "RULEBOOK")
    route("How do I score points?")

    system_message, human_message = llm.calls[0]
    assert system_message.content == ROUTER_SYSTEM
    assert human_message.content == "How do I score points?"
