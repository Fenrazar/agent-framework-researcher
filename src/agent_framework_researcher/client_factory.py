"""Chat client factory supporting both OpenAI and Azure OpenAI providers."""

import os

from agent_framework.openai import OpenAIChatClient

from agent_framework_researcher.configuration import Configuration


def _is_azure(config: Configuration) -> bool:
    """Detect Azure mode from explicit config or environment variables."""
    if config.llm_provider == "azure":
        return True
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))


def create_client(config: Configuration, model: str | None = None) -> OpenAIChatClient:
    """Create an AF chat client based on configuration.

    Provider detection order:
    1. Explicit ``config.llm_provider == "azure"``
    2. ``AZURE_OPENAI_ENDPOINT`` env var present → Azure
    3. Otherwise → OpenAI

    For Azure, authentication resolves as:
    1. ``AZURE_OPENAI_API_KEY`` env var
    2. ``az login`` credential (AzureCliCredential)

    Args:
        config: Application configuration with provider and model settings.
        model: Optional model override. Falls back to config.research_model.

    Returns:
        Configured OpenAIChatClient.
    """
    resolved_model = model or config.research_model
    kwargs: dict = {"model": resolved_model}

    if _is_azure(config):
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        # AF expects the base endpoint, not a sub-path
        azure_endpoint = azure_endpoint.split("/openai")[0].rstrip("/")
        kwargs["azure_endpoint"] = azure_endpoint

        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        if deployment:
            kwargs["model"] = deployment

        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if azure_key:
            kwargs["api_key"] = azure_key
        else:
            # Fall back to az login credential
            from azure.identity import AzureCliCredential

            kwargs["credential"] = AzureCliCredential()
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

    return OpenAIChatClient(**kwargs)
