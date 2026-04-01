# Agent Framework Researcher

Port of [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) to [microsoft/agent-framework](https://github.com/microsoft/agent-framework) (Python). Idiomatic AF port — no LangChain compatibility shims. Uses `uv` for all Python tooling.

## Build, Test, Lint

```bash
# Install all dependencies
uv sync

# Run the agent
uv run -m agent_framework_researcher

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_tools.py::test_think_tool -v

# Lint
uv run ruff check src/ tests/

# Lint with auto-fix
uv run ruff check --fix src/ tests/

# Type check
uv run mypy src/
```

## Architecture

Deep research agent as a multi-step workflow. The original LangGraph 3-layer nested graph is flattened into AF primitives:

```
WorkflowBuilder (deep_research_workflow)
├── ClarifyExecutor          → HITL via ctx.request_info() / @response_handler
├── WriteBriefExecutor       → Agent call to generate a structured research brief
├── SupervisorAgent          → Agent with tools: conduct_research, research_complete, think
│   └── conduct_research()   → Spawns ResearcherAgent internally (Agent.run())
│       └── Researcher       → Agent with search tools (Tavily, MCP, etc.)
└── FinalReportExecutor      → Agent call to write the final report from findings
```

Key design decisions:
- **AF's `Agent` IS the ReAct loop.** Do not hand-roll tool-call loops. Use `Agent(tools=[...]).run()` — it handles tool execution, retries, and message threading internally.
- **Researchers are spawned inside tool functions**, not as workflow nodes. The supervisor's `conduct_research` tool creates an `Agent` instance and calls `.run()`. This avoids needing dynamic fan-out at the workflow level.
- **State sharing** between executors uses `ctx.set_state("key", value)` / `ctx.get_state("key")`, not LangGraph-style TypedDict reducers.
- **No `init_chat_model()`** — AF doesn't have a universal model factory. Use `OpenAIChatClient(model=...)` directly.

## Key Mapping: LangGraph → Agent Framework

| LangGraph | Agent Framework | Notes |
|-----------|----------------|-------|
| `StateGraph` + `builder.compile()` | `WorkflowBuilder(...).build()` | |
| Node functions | `Executor` subclasses with `@handler` | |
| `Command[Literal["next"]]` routing | `add_switch_case_edge_group` with `Case`/`Default` | |
| `.with_structured_output(Model)` | Parse JSON from `agent.run()` + Pydantic validation | AF has no built-in structured output binding |
| `.bind_tools()` | `Agent(tools=[...])` | Pass plain functions — AF auto-wraps them |
| `interrupt()` for HITL | `ctx.request_info()` + `@response_handler` | |
| `init_chat_model("openai:gpt-4.1")` | `OpenAIChatClient(model="gpt-4.1")` | No provider prefix |
| MCP via `langchain-mcp-adapters` | `client.get_mcp_tool(url=..., approval_mode=...)` | Native AF MCP |
| Tavily search tool | `OpenAIChatClient.get_web_search_tool()` | Native model web search, no API key |
| `WorkflowAgent` | `workflow.as_agent()` | Wraps a Workflow as an Agent |

## Conventions

### Imports
All AF workflow primitives are exported from `agent_framework` directly:
```python
from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, handler, response_handler
from agent_framework.openai import OpenAIChatClient
```

### Executors
- Subclass `Executor`, implement `@handler` methods.
- `ctx.send_message(msg)` passes data to the next executor.
- `ctx.yield_output(result)` emits final output (terminal nodes only).
- `ctx.set_state()` / `ctx.get_state()` for cross-executor state.

### Tools
- Define as plain async functions with `Annotated` type hints. AF auto-wraps them.
- Docstrings become tool descriptions.
- Web search uses `OpenAIChatClient.get_web_search_tool()` — returns a dict `{'type': 'web_search', ...}` passed directly in `Agent(tools=[...])`. No external API key needed.
- Read API keys from `os.environ` directly.

### Structured Output
AF doesn't have `.with_structured_output()`. Instead:
1. Include "respond in valid JSON matching this schema: ..." in the agent instructions.
2. Parse: `json.loads(result.text)`.
3. Validate: `Model.model_validate(parsed)`.

### Configuration
- `Configuration.from_env()` loads from environment variables.
- Model strings use AF format (`"gpt-4.1"`) not LangChain format (`"openai:gpt-4.1"`).
- Supports both OpenAI and Azure OpenAI via `LLM_PROVIDER` env var.

### Prompts
- `prompts.py` is copied verbatim from open_deep_research. Do not modify unless fixing a bug.

## Source References

- **Original**: https://github.com/langchain-ai/open_deep_research (cloned in `_reference/`)
- **Target framework**: https://github.com/microsoft/agent-framework/tree/main/python
- **AF workflow samples**: https://github.com/microsoft/agent-framework/tree/main/python/samples/03-workflows
