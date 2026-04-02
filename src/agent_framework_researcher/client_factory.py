"""Chat client factory supporting both OpenAI and Azure OpenAI providers."""

from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential

from agent_framework_researcher.configuration import Configuration


def _is_azure(config: Configuration) -> bool:
    """Detect Azure mode from explicit config or configured endpoint."""
    if config.llm_provider == "azure":
        return True
    return "openai.azure.com" in (config.llm_endpoint or "")


def create_client(config: Configuration, model: str | None = None) -> OpenAIChatClient:
    """Create an AF chat client based on configuration.

    Provider detection order:
    1. Explicit ``config.llm_provider == "azure"``
    2. ``config.llm_endpoint`` is set → Azure
    3. Otherwise → OpenAI

    For Azure, authentication resolves as:
    1. ``config.llm_api_key``
    2. DefaultAzureCredential (supports managed identity, az login, etc.)

    Args:
        config: Application configuration with provider and model settings.
        model: Optional model override. Falls back to config.default_model.

    Returns:
        Configured OpenAIChatClient.
    """
    resolved_model = model or config.default_model
    kwargs: dict = {"model": resolved_model}

    if _is_azure(config):
        endpoint = config.llm_endpoint or ""
        # AF expects the base endpoint, not a sub-path
        endpoint = endpoint.split("/openai")[0].rstrip("/")
        kwargs["azure_endpoint"] = endpoint

        if config.llm_api_key:
            kwargs["api_key"] = config.llm_api_key
        else:
            kwargs["credential"] = DefaultAzureCredential()
    else:
        if config.llm_api_key:
            kwargs["api_key"] = config.llm_api_key

    return OpenAIChatClient(**kwargs)
