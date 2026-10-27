"""Shared test setup: import paths, a fake LLM so no test ever calls Gemini, and a no-wait sleep."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:  # modules import each other as the app does: `from config import ...`
    sys.path.insert(0, str(SRC))


class FakeLLM:
    """Stands in for the Gemini chat model.

    Give it the replies it should return, in order. Every call is recorded in
    `calls` (a copy of the messages), and an unexpected extra call fails the test.

    - `invoke(messages)` returns an object with `.content` (a normal chat reply).
    - `with_structured_output(schema).invoke(messages)` returns the reply itself
      (e.g. a `Route` or `QAPair` object, or None), like LangChain does.
    """

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []
        self.schema = None

    def _next(self, messages):
        self.calls.append(list(messages))
        assert self.replies, "the LLM was called more times than the test expected"
        return self.replies.pop(0)

    def invoke(self, messages):
        return SimpleNamespace(content=self._next(messages))

    def with_structured_output(self, schema):
        self.schema = schema
        return _StructuredFake(self)


class _StructuredFake:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, messages):
        return self.parent._next(messages)


@pytest.fixture
def fake_llm():
    """Returns the FakeLLM class: fake_llm(["reply 1", "reply 2"])."""
    return FakeLLM


@pytest.fixture
def sleeps(monkeypatch):
    """Replaces time.sleep everywhere (tests never wait) and returns the list of requested waits."""
    waits = []
    monkeypatch.setattr(time, "sleep", waits.append)
    return waits