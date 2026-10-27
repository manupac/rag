"""The router: decides if a question is answered from the rulebook or from the database."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from llm import get_llm
from prompts import ROUTER_SYSTEM

RULEBOOK = "RULEBOOK"
DATABASE = "DATABASE"


class Route(BaseModel):
    """The model must answer with one of these two values: no text to parse."""

    route: Literal["RULEBOOK", "DATABASE"] = Field(description="Where to look for the answer")


def route(question: str) -> str:
    """Return RULEBOOK or DATABASE. If the model gives no valid answer, use the rulebook."""
    router = get_llm().with_structured_output(Route)
    result = router.invoke([SystemMessage(ROUTER_SYSTEM), HumanMessage(question)])
    return result.route if result else RULEBOOK


def main():
    """Type questions and see which channel the router picks."""
    print("Type a question to see where it would be routed ('quit' to exit).")
    while (question := input("\nQuestion: ").strip()).lower() != "quit":
        if question:
            print(route(question))


if __name__ == "__main__":
    main()