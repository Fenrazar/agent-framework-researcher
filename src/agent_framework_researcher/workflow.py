"""Workflow wiring — connects executors and agents via WorkflowBuilder."""

from agent_framework import WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

from agent_framework_researcher.agents import create_supervisor_agent
from agent_framework_researcher.configuration import Configuration
from agent_framework_researcher.executors import (
    ClarifyExecutor,
    FinalReportExecutor,
    SupervisorExecutor,
    WriteBriefExecutor,
)


def build_deep_research_workflow(client: OpenAIChatClient, config: Configuration):
    """Build the full deep research workflow and return it as an Agent.

    Pipeline: clarify → write_brief → supervisor → final_report

    The supervisor Agent is wrapped in SupervisorExecutor to bridge
    typed messages between workflow stages. Researchers are spawned
    inside the supervisor's `conduct_research` tool.
    """
    clarify = ClarifyExecutor(client, config)
    write_brief = WriteBriefExecutor(client, config)
    supervisor = SupervisorExecutor(create_supervisor_agent(client, config))
    final_report = FinalReportExecutor(client, config)

    workflow = (
        WorkflowBuilder(start_executor=clarify, output_executors=[final_report])
        .add_edge(clarify, write_brief)
        .add_edge(write_brief, supervisor)
        .add_edge(supervisor, final_report)
        .build()
    )

    return workflow.as_agent(name="DeepResearcher")
