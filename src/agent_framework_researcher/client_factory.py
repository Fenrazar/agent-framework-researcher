"""Chat client factory supporting both OpenAI and Azure OpenAI providers."""

import os

from agent_framework.openai import OpenAIChatClient

from agent_framework_researcher.configuration import Configuration


def create_client(config: Configuration, model: str | None = None) -> OpenAIChatClient:
    """Create an AF chat client based on configuration.

    Args:
        config: Application configuration with provider and model settings.
        model: Optional model override. Falls back to config.research_model.

    Returns:
        Configured OpenAIChatClient (works for both OpenAI and Azure OpenAI).
    """
    resolved_model = model or config.research_model
    kwargs: dict = {"model": resolved_model}

    if config.llm_provider == "azure":
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        if azure_endpoint:
            kwargs["azure_endpoint"] = azure_endpoint
        if azure_key:
            kwargs["api_key"] = azure_key
        if deployment:
            kwargs["model"] = deployment
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

    return OpenAIChatClient(**kwargs)
