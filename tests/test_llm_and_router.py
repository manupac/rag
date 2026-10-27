from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import llm
import router
from llm import DailyQuotaError, call_with_retry, text_of
from prompts import ROUTER_SYSTEM
from router import BOTH, DATABASE, RULEBOOK, Route, route

# ---------- llm.text_of ----------


def test_text_of_returns_plain_string_content():
    assert text_of(SimpleNamespace(content="hello")) == "hello"


def test_text_of_joins_list_of_parts():
    message = SimpleNamespace(content=[{"text": "Hello "}, {"text": "world"}])
    assert text_of(message) == "Hello world"


def test_text_of_ignores_non_text_parts():
    message = SimpleNamespace(content=[{"text": "kept"}, {"type": "thinking"}, "stray string"])
    assert text_of(message) == "kept"


# ---------- llm.call_with_retry ----------


def test_call_with_retry_passes_arguments_through(sleeps):
    assert call_with_retry(lambda a, b=0: a + b, 1, b=2) == 3
    assert sleeps == []


def test_call_with_retry_waits_on_rate_limit_then_succeeds(sleeps):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return "done"

    assert call_with_retry(flaky) == "done"
    assert sleeps == [llm.RETRY_WAIT_SECONDS] * 2


def test_call_with_retry_raises_other_errors_immediately(sleeps):
    attempts = []

    def broken():
        attempts.append(1)
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError):
        call_with_retry(broken)
    assert len(attempts) == 1
    assert sleeps == []


def test_call_with_retry_gives_up_after_max_retries(sleeps):
    attempts = []

    def always_limited():
        attempts.append(1)
        raise Exception("RESOURCE_EXHAUSTED per minute")

    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        call_with_retry(always_limited)
    assert len(attempts) == llm.MAX_RETRIES
    assert len(sleeps) == llm.MAX_RETRIES - 1  # no wait after the last attempt


def test_call_with_retry_raises_daily_quota_at_once(sleeps):
    attempts = []

    def daily_limit():
        attempts.append(1)
        raise Exception("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(DailyQuotaError):
        call_with_retry(daily_limit)
    assert len(attempts) == 1  # waiting would not help
    assert sleeps == []


# ---------- router.route ----------


def use_fake_router_llm(monkeypatch, fake_llm, reply):
    fake = fake_llm([reply])
    monkeypatch.setattr(router, "get_llm", lambda: fake)
    return fake


def test_route_returns_the_structured_plan(monkeypatch, fake_llm):
    plan = Route(route="BOTH", rulebook_question="How does wounding work?", database_question="Carnifex toughness?")
    fake = use_fake_router_llm(monkeypatch, fake_llm, plan)

    result = route("What do I need to wound a Carnifex?")

    assert result == plan
    assert result.route == BOTH
    assert fake.schema is Route  # the model is asked for a Route, not free text


def test_route_database(monkeypatch, fake_llm):
    use_fake_router_llm(monkeypatch, fake_llm, Route(route="DATABASE"))
    assert route("What is the toughness of a Carnifex?").route == DATABASE


def test_route_rulebook(monkeypatch, fake_llm):
    use_fake_router_llm(monkeypatch, fake_llm, Route(route="RULEBOOK"))
    assert route("How does charging work?").route == RULEBOOK


def test_route_defaults_to_rulebook_when_the_model_returns_nothing(monkeypatch, fake_llm):
    use_fake_router_llm(monkeypatch, fake_llm, None)
    result = route("anything")

    assert result.route == RULEBOOK
    assert result.rulebook_question == ""
    assert result.database_question == ""


def test_route_sends_router_prompt_then_the_question(monkeypatch, fake_llm):
    fake = use_fake_router_llm(monkeypatch, fake_llm, Route(route="RULEBOOK"))
    route("How do I score points?")

    system_message, human_message = fake.calls[0]
    assert system_message.content == ROUTER_SYSTEM
    assert human_message.content == "How do I score points?"


def test_route_model_only_accepts_the_three_routes():
    with pytest.raises(ValidationError):
        Route(route="MAYBE")


def test_route_sub_questions_default_to_empty():
    plan = Route(route="RULEBOOK")
    assert plan.rulebook_question == ""
    assert plan.database_question == ""