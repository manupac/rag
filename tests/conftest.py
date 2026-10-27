"""Shared test setup: import paths and a fake LLM so no test ever calls Gemini."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):  # ROOT lets parser.py's `from src.config` work
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class FakeLLM:
    """Stands in for the Gemini chat model.

    Give it the replies it should return, in order. Every call is recorded in
    `calls` (a copy of the messages), and an unexpected extra call fails the test.
    """

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        assert self.replies, "the LLM was called more times than the test expected"
        return SimpleNamespace(content=self.replies.pop(0))


@pytest.fixture
def fake_llm():
    """Returns the FakeLLM class: fake_llm(["reply 1", "reply 2"])."""
    return FakeLLM
