"""CLI entry point for the deep research agent.

Run with: uv run -m agent_framework_researcher
"""

import asyncio
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent_framework_researcher.client_factory import create_client
from agent_framework_researcher.configuration import Configuration
from agent_framework_researcher.workflow import build_deep_research_workflow

console = Console()


async def main() -> None:
    load_dotenv()
    config = Configuration.from_env()
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

    console.print(f"\n[dim]Researching: {query}[/dim]\n")

    agent = build_deep_research_workflow(client, config)
    result = await agent.run(query)

    console.print("\n")
    console.print(Panel("[bold]Final Report[/bold]", style="green"))
    console.print(Markdown(str(result)))


def cli() -> None:
    """Synchronous entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
