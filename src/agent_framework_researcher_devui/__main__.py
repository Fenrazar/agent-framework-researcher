"""DevUI server for the deep research agent.

Run with: uv run -m agent_framework_researcher_devui

Launches the Agent Framework DevUI at http://localhost:8080 with:
- DeepResearchWorkflow: The full research pipeline (clarify → brief → supervisor → report)
- Supervisor: The standalone research supervisor agent (for quick testing)
"""

import sys

from agent_framework.devui import serve

from agent_framework_researcher.agents import create_supervisor_agent
from agent_framework_researcher.client_factory import create_client
from agent_framework_researcher.configuration import Configuration
from agent_framework_researcher.workflow import build_workflow


def main() -> None:
    config = Configuration()
    client = create_client(config)

    # Build entities for DevUI
    workflow = build_workflow(client, config)
    supervisor = create_supervisor_agent(client, config)

    # Parse CLI args
    port = 8080
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--port", "-p") and i < len(sys.argv) - 1:
            port = int(sys.argv[i + 1])

    serve(
        entities=[workflow, supervisor],
        port=port,
        auto_open=True,
        instrumentation_enabled=True,
    )


if __name__ == "__main__":
    main()
