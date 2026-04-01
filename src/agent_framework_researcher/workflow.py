"""Workflow wiring — connects executors and agents via WorkflowBuilder."""

from agent_framework import WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

from agent_framework_researcher.agents import create_supervisor_agent
from agent_framework_researcher.configuration import Configuration
from agent_framework_researcher.executors import ClarifyExecutor, FinalReportExecutor, WriteBriefExecutor


def build_deep_research_workflow(client: OpenAIChatClient, config: Configuration):
    """Build the full deep research workflow and return it as an Agent.

    Pipeline: clarify → write_brief → supervisor → final_report

    The supervisor Agent is passed directly to WorkflowBuilder and auto-wrapped
    as an AgentExecutor. Researchers are spawned inside the supervisor's
    `conduct_research` tool — no dynamic fan-out at the workflow level.
    """
    clarify = ClarifyExecutor(client, config)
    write_brief = WriteBriefExecutor(client, config)
    supervisor = create_supervisor_agent(client, config)
    final_report = FinalReportExecutor(client, config)

    workflow = (
        WorkflowBuilder(start_executor=clarify)
        .add_edge(clarify, write_brief)
        .add_edge(write_brief, supervisor)
        .add_edge(supervisor, final_report)
        .with_output_from(final_report)
        .build()
    )

    return workflow.as_agent(name="DeepResearcher")
