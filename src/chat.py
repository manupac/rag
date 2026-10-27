"""Chat with the bot in the terminal.

Commands: 'quit' to exit, 'reset' to start a new conversation, 'debug' to toggle
showing the rewritten question, the route, and the chunks or SQL used.
"""

from rich.console import Console
from rich.markdown import Markdown

from chatbot import Chatbot
from router import DATABASE

console = Console()


def show_debug(response) -> None:
    console.print(f"\nsearched for: {response.standalone_question}", style="dim", markup=False)
    console.print(f"route: {response.route}", style="dim", markup=False)
    if response.route == DATABASE:
        sql = response.sql
        console.print(f"SQL:\n{sql.query}", style="dim", markup=False)
        if sql.error:
            console.print(f"error: {sql.error}", style="dim red", markup=False)
        else:
            more = " (truncated)" if sql.truncated else ""
            console.print(f"{len(sql.rows)} rows{more}", style="dim", markup=False)
    else:
        for i, doc in enumerate(response.sources, start=1):
            preview = doc.page_content[:150].replace("\n", " ")
            console.print(f"[{i}] {preview}...", style="dim", markup=False)


def main():
    bot = Chatbot()
    debug = False
    console.print("Ask a question about the rules or the units ('quit', 'reset', 'debug').", style="bold")

    while True:
        question = console.input("\n[bold cyan]You:[/] ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            bot.reset()
            console.print("Conversation reset.", style="yellow")
            continue
        if question.lower() == "debug":
            debug = not debug
            console.print(f"Debug {'on' if debug else 'off'}.", style="yellow")
            continue

        try:
            with console.status("Thinking..."):
                response = bot.ask(question)
        except Exception as e:  # noqa: BLE001 - keep the chat alive on any error
            console.print(f"Error: {e}", style="red", markup=False)
            continue

        if debug:
            show_debug(response)

        console.print("\n[bold green]Bot:[/]")
        console.print(Markdown(response.answer))


if __name__ == "__main__":
    main()