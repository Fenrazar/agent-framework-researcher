# Agent Framework Researcher

Deep research agent — a port of [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) to [microsoft/agent-framework](https://github.com/microsoft/agent-framework) (Python).

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env

# Run (CLI)
uv run -m agent_framework_researcher

# Run (DevUI — web interface at http://localhost:8080)
uv run -m agent_framework_researcher_devui
```

## Development

```bash
# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_tools.py::test_think_tool -v

# Lint
uv run ruff check .

# Type check
uv run mypy src/
```