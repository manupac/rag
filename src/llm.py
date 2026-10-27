"""The chat models used in the project, and helpers to call them."""

import time
from functools import cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config import JUDGE_MODEL, LLM_MODEL

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 30


class DailyQuotaError(Exception):
    """The daily quota is used up: waiting won't help, try again tomorrow."""


@cache
def get_llm(model: str = LLM_MODEL) -> ChatGoogleGenerativeAI:
    # Temperature is not set: Google recommends keeping the default (1.0) for Gemini 3
    # models, as lower values can make answers worse. Answers are therefore not fully
    # deterministic, so evaluation scores can vary slightly between runs.
    return ChatGoogleGenerativeAI(
        model=model,
        thinking_level="low",  # less hidden reasoning = faster answers
    )


def get_judge_llm() -> ChatGoogleGenerativeAI:
    """A different model for judging answers, so the bot is not graded by itself."""
    return get_llm(JUDGE_MODEL)


def text_of(message) -> str:
    """Return a model reply as plain text, whether it's a string or a list of parts."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def call_with_retry(fn, *args, **kwargs):
    """Call fn, waiting and retrying on a per-minute rate limit.

    A daily limit is raised at once as DailyQuotaError, because waiting 30 seconds
    cannot fix a quota that resets in hours.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            message = str(e)
            if "RESOURCE_EXHAUSTED" not in message:
                raise
            if "PerDay" in message:
                raise DailyQuotaError(message) from e
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"Rate limit hit, waiting {RETRY_WAIT_SECONDS}s and retrying...")
            time.sleep(RETRY_WAIT_SECONDS)