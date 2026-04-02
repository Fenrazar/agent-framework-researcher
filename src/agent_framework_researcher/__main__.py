"""CLI entry point for the deep research agent.

Run with: uv run -m agent_framework_researcher
"""

import asyncio
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent_framework_researcher.client_factory import create_client
from agent_framework_researcher.configuration import Configuration
from agent_framework_researcher.models import HumanClarificationRequest
from agent_framework_researcher.workflow import build_workflow

console = Console()


async def main() -> None:
    config = Configuration()
    client = create_client(config)

    console.print(
        Panel(
            "[bold]Agent Framework Researcher[/bold]\n"
            "Deep research agent powered by Microsoft Agent Framework",
            style="blue",
        )
    )

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = console.input("[bold green]Research topic:[/bold green] ")

    if not query.strip():
        console.print("[red]No research topic provided. Exiting.[/red]")
        return

    workflow = build_workflow(client, config)
    final_report = ""
    pending_responses: dict[str, str] | None = None

    while True:
        if pending_responses is not None:
            stream = workflow.run(stream=True, responses=pending_responses)
        else:
            stream = workflow.run(query, stream=True)

        pending_responses = None
        has_request = False

        async for event in stream:
            if event.type == "progress":
                console.print(f"  [cyan]▸[/cyan] [dim]{event.data}[/dim]")
            elif event.type == "request_info" and isinstance(event.data, HumanClarificationRequest):
                console.print(f"\n  [yellow]?[/yellow] {event.data.question}")
                answer = console.input("[bold green]Your answer:[/bold green] ")
                pending_responses = {event.request_id: answer}
                has_request = True
            elif event.type == "output":
                final_report = str(event.data)

        if not has_request:
            break

    console.print("\n")
    console.print(Panel("[bold]Final Report[/bold]", style="green"))
    console.print(Markdown(final_report))


def cli() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        console.print("\n\n[yellow]Research cancelled.[/yellow]")


if __name__ == "__main__":
    cli()
