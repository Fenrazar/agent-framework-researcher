"""Tests for configuration and models."""

from agent_framework_researcher.configuration import Configuration, MCPConfig, SearchAPI
from agent_framework_researcher.models import (
    ClarifyWithUser,
    CompressedResearch,
    HumanClarificationRequest,
    ResearchBriefMessage,
    ResearchQuestion,
    SupervisorInput,
)


def test_configuration_defaults(monkeypatch):
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("RESEARCH_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    config = Configuration(_env_file=None)
    assert config.search_api == SearchAPI.WEB_SEARCH
    assert config.allow_clarification is True
    assert config.max_concurrent_research_units == 5
    assert config.default_model == "gpt-4.1"
    assert config.research_model == "gpt-4.1"  # falls back to default_model
    assert config.llm_provider == "openai"


def test_configuration_from_env(monkeypatch):
    monkeypatch.setenv("ALLOW_CLARIFICATION", "false")
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    config = Configuration()
    assert config.allow_clarification is False
    assert config.llm_provider == "azure"


def test_configuration_llm_fields(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_ENDPOINT", "https://test.openai.azure.com/")
    config = Configuration()
    assert config.llm_api_key == "sk-test"
    assert config.llm_endpoint == "https://test.openai.azure.com/"


def test_configuration_default_model_fallback():
    config = Configuration(default_model="o3")
    assert config.research_model == "o3"
    assert config.compression_model == "o3"
    assert config.final_report_model == "o3"


def test_configuration_per_task_model_override():
    config = Configuration(default_model="o3", research_model="gpt-4o")
    assert config.research_model == "gpt-4o"
    assert config.compression_model == "o3"


def test_configuration_with_overrides():
    config = Configuration(max_researcher_iterations=10)
    assert config.max_researcher_iterations == 10


def test_mcp_config():
    mcp = MCPConfig(url="https://example.com/mcp", tools=["search"])
    assert mcp.url == "https://example.com/mcp"
    assert mcp.tools == ["search"]


def test_clarify_with_user_model():
    c = ClarifyWithUser(need_clarification=True, question="What scope?", verification="")
    assert c.need_clarification is True


def test_research_question_model():
    rq = ResearchQuestion(research_brief="Compare X vs Y")
    assert "Compare" in rq.research_brief


def test_research_brief_message():
    msg = ResearchBriefMessage(messages_text="user query")
    assert msg.messages_text == "user query"


def test_supervisor_input():
    inp = SupervisorInput(research_brief="brief")
    assert inp.research_brief == "brief"


def test_human_clarification_request():
    req = HumanClarificationRequest(question="What do you mean?")
    assert req.question == "What do you mean?"
    assert req.verification == ""


def test_compressed_research():
    cr = CompressedResearch(compressed_research="findings", raw_notes=["note1"])
    assert cr.compressed_research == "findings"
    assert len(cr.raw_notes) == 1
