"""Workflow wiring — connects executors and agents via WorkflowBuilder."""

from agent_framework import Workflow, WorkflowAgent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

from agent_framework_researcher.agents import create_supervisor_agent
from agent_framework_researcher.configuration import Configuration
from agent_framework_researcher.executors import (
    ClarifyExecutor,
    FinalReportExecutor,
    SupervisorExecutor,
    WriteBriefExecutor,
)


def build_workflow(client: OpenAIChatClient, config: Configuration) -> Workflow:
    """Build the deep research workflow.

    Pipeline: clarify → write_brief → supervisor → final_report
    """
    clarify = ClarifyExecutor(client, config)
    write_brief = WriteBriefExecutor(client, config)
    supervisor = SupervisorExecutor(create_supervisor_agent(client, config))
    final_report = FinalReportExecutor(client, config)

    return (
        WorkflowBuilder(start_executor=clarify, output_executors=[final_report])
        .add_edge(clarify, write_brief)
        .add_edge(write_brief, supervisor)
        .add_edge(supervisor, final_report)
        .build()
    )


def build_deep_research_workflow(client: OpenAIChatClient, config: Configuration) -> WorkflowAgent:
    """Build the workflow and wrap it as an Agent for CLI usage."""
    return build_workflow(client, config).as_agent(name="DeepResearcher")
