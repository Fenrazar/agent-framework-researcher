"""Tests for configuration and models."""

from agent_framework_researcher.configuration import Configuration, MCPConfig, SearchAPI
from agent_framework_researcher.models import (
    ClarifyWithUser,
    CompressedResearch,
    HumanClarificationRequest,
    ResearchBriefMessage,
    ResearchQuestion,
    Summary,
    SupervisorInput,
)


def test_configuration_defaults():
    config = Configuration()
    assert config.search_api == SearchAPI.TAVILY
    assert config.allow_clarification is True
    assert config.max_concurrent_research_units == 5
    assert config.research_model == "gpt-4.1"
    assert config.llm_provider == "openai"


def test_configuration_from_env(monkeypatch):
    monkeypatch.setenv("ALLOW_CLARIFICATION", "false")
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    config = Configuration.from_env()
    assert config.allow_clarification is False
    assert config.llm_provider == "azure"


def test_configuration_from_env_with_overrides():
    config = Configuration.from_env(overrides={"max_researcher_iterations": 10})
    assert config.max_researcher_iterations == 10


def test_mcp_config():
    mcp = MCPConfig(url="https://example.com/mcp", tools=["search"])
    assert mcp.url == "https://example.com/mcp"
    assert mcp.tools == ["search"]


def test_summary_model():
    s = Summary(summary="test summary", key_excerpts="key info")
    assert s.summary == "test summary"


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
