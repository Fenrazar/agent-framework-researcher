"""Tests for tools module."""

from agent_framework_researcher.tools import (
    get_model_token_limit,
    get_search_tools,
    get_today_str,
    is_token_limit_exceeded,
    think,
)


def test_think_tool():
    result = think("This is my reflection")
    assert "Reflection recorded" in result
    assert "This is my reflection" in result


def test_get_today_str():
    today = get_today_str()
    assert len(today) > 0
    # Should contain a year
    assert "202" in today


def test_get_model_token_limit_known():
    assert get_model_token_limit("gpt-4.1") == 1_047_576
    assert get_model_token_limit("gpt-4.1-mini") == 1_047_576
    assert get_model_token_limit("gpt-4o") == 128_000


def test_get_model_token_limit_unknown():
    assert get_model_token_limit("unknown-model-xyz") is None


def test_is_token_limit_exceeded_true():
    exc = Exception("maximum context length exceeded")
    assert is_token_limit_exceeded(exc, "gpt-4.1") is True


def test_is_token_limit_exceeded_false():
    exc = Exception("connection timeout")
    assert is_token_limit_exceeded(exc, "gpt-4.1") is False


def test_get_search_tools_tavily(sample_config):
    tools = get_search_tools(sample_config)
    assert len(tools) == 1


def test_get_search_tools_none(sample_config):
    from agent_framework_researcher.configuration import SearchAPI

    sample_config.search_api = SearchAPI.NONE
    tools = get_search_tools(sample_config)
    assert len(tools) == 0
