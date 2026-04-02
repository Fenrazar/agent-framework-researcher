"""Workflow executors for the deep research pipeline.

Each executor corresponds to a stage in the research workflow:
ClarifyExecutor → WriteBriefExecutor → (SupervisorAgent) → FinalReportExecutor
"""

import json
import logging
from typing import Never

from agent_framework import (
    Agent,
    BaseChatClient,
    Executor,
    Message,
    WorkflowContext,
    WorkflowEvent,
    handler,
    response_handler,
)

from agent_framework_researcher.client_factory import create_client
from agent_framework_researcher.configuration import Configuration
from agent_framework_researcher.models import (
    ClarifyWithUser,
    HumanClarificationRequest,
    ResearchBriefMessage,
    ResearchQuestion,
    SupervisorInput,
)
from agent_framework_researcher.prompts import (
    clarify_with_user_instructions,
    final_report_generation_prompt,
    transform_messages_into_research_topic_prompt,
)
from agent_framework_researcher.tools import get_model_token_limit, get_today_str, is_token_limit_exceeded

logger = logging.getLogger(__name__)


class ClarifyExecutor(Executor):
    """Analyze user input and optionally ask clarifying questions (HITL).

    If clarification is disabled or not needed, passes input directly downstream.
    """

    def __init__(self, client: BaseChatClient, config: Configuration):
        super().__init__(id="clarify")
        self._client = client
        self._config = config

    @handler
    async def handle_str(self, text: str, ctx: WorkflowContext[ResearchBriefMessage, Never]) -> None:
        await self._clarify(text, ctx)

    @handler
    async def handle_message(self, message: Message, ctx: WorkflowContext[ResearchBriefMessage, Never]) -> None:
        await self._clarify(message.text or "", ctx)

    @handler
    async def handle_messages(self, messages: list[Message], ctx: WorkflowContext[ResearchBriefMessage, Never]) -> None:
        messages_text = "\n".join(m.text for m in messages if m.text)
        await self._clarify(messages_text, ctx)

    async def _clarify(self, messages_text: str, ctx: WorkflowContext[ResearchBriefMessage, Never]) -> None:
        await ctx.add_event(WorkflowEvent(type="progress", data="Analyzing query..."))

        if not self._config.allow_clarification:
            await ctx.add_event(WorkflowEvent(type="progress", data="Skipping clarification — proceeding to research"))
            await ctx.send_message(ResearchBriefMessage(messages_text=messages_text))
            return

        prompt_content = clarify_with_user_instructions.format(
            messages=messages_text,
            date=get_today_str(),
        )

        agent = Agent(
            client=self._client,
            instructions=(
                "You are a clarification assistant. "
                "Always respond with valid JSON matching: "
                '{"need_clarification": bool, "question": "...", "verification": "..."}'
            ),
        )
        result = await agent.run(prompt_content)

        try:
            parsed = json.loads(result.text)
            clarification = ClarifyWithUser.model_validate(parsed)
        except Exception:
            # If parsing fails, skip clarification and proceed
            await ctx.send_message(ResearchBriefMessage(messages_text=messages_text))
            return

        if clarification.need_clarification:
            await ctx.add_event(WorkflowEvent(type="progress", data=f"Asking clarifying question: {clarification.question}"))
            ctx.set_state("original_query", messages_text)
            await ctx.request_info(
                HumanClarificationRequest(
                    question=clarification.question,
                    verification=clarification.verification,
                ),
                str,
            )
        else:
            await ctx.add_event(WorkflowEvent(type="progress", data="No clarification needed — proceeding to research"))
            ctx.set_state("verification", clarification.verification)
            await ctx.send_message(ResearchBriefMessage(messages_text=messages_text))

    @response_handler
    async def on_clarification(
        self,
        original: HumanClarificationRequest,
        response: str,
        ctx: WorkflowContext[ResearchBriefMessage, Never],
    ) -> None:
        """Handle user's clarification response, then proceed to research."""
        original_query = ctx.get_state("original_query") or ""
        messages_text = f"Original query: {original_query}\n\nClarification question: {original.question}\nUser's answer: {response}"
        await ctx.send_message(ResearchBriefMessage(messages_text=messages_text))


class WriteBriefExecutor(Executor):
    """Transform user messages into a structured research brief."""

    def __init__(self, client: BaseChatClient, config: Configuration):
        super().__init__(id="write_brief")
        self._client = client
        self._config = config

    @handler
    async def handle(self, msg: ResearchBriefMessage, ctx: WorkflowContext[SupervisorInput, Never]) -> None:
        await ctx.add_event(WorkflowEvent(type="progress", data="Generating research brief..."))

        prompt_content = transform_messages_into_research_topic_prompt.format(
            messages=msg.messages_text,
            date=get_today_str(),
        )

        agent = Agent(
            client=self._client,
            instructions=(
                "You are a research brief writer. "
                "Always respond with valid JSON matching: "
                '{"research_brief": "..."}'
            ),
        )
        result = await agent.run(prompt_content)

        try:
            parsed = json.loads(result.text)
            question = ResearchQuestion.model_validate(parsed)
            research_brief = question.research_brief
        except Exception:
            # Fallback: use raw response as the brief
            research_brief = result.text

        ctx.set_state("research_brief", research_brief)
        ctx.set_state("messages_text", msg.messages_text)
        await ctx.add_event(WorkflowEvent(type="progress", data=f"Research brief ready: {research_brief[:200]}..."))
        await ctx.send_message(SupervisorInput(research_brief=research_brief))


class SupervisorExecutor(Executor):
    """Wraps the supervisor Agent as a workflow executor.

    Accepts SupervisorInput, runs the supervisor agent, and passes
    the research findings (as a string) to the next executor.
    """

    def __init__(self, supervisor_agent: Agent):
        super().__init__(id="supervisor")
        self._agent = supervisor_agent

    @handler
    async def handle(self, msg: SupervisorInput, ctx: WorkflowContext[str, Never]) -> None:
        await ctx.add_event(WorkflowEvent(type="progress", data="Supervisor starting research..."))
        result = await self._agent.run(msg.research_brief)
        findings = result.text or ""
        ctx.set_state("findings", findings)
        await ctx.add_event(WorkflowEvent(type="progress", data="Research complete — compiling findings"))
        await ctx.send_message(findings)


class FinalReportExecutor(Executor):
    """Generate the final comprehensive research report."""

    def __init__(self, client: BaseChatClient, config: Configuration):
        super().__init__(id="final_report")
        self._client = client
        self._config = config

    @handler
    async def handle(self, trigger: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.add_event(WorkflowEvent(type="progress", data="Writing final report..."))

        research_brief = ctx.get_state("research_brief") or ""
        findings = ctx.get_state("findings") or trigger
        messages_text = ctx.get_state("messages_text") or ""

        report_client = create_client(self._config, model=self._config.final_report_model)

        max_retries = 3
        current_retry = 0
        findings_limit: int | None = None

        while current_retry <= max_retries:
            try:
                truncated_findings = findings[:findings_limit] if findings_limit else findings
                prompt = final_report_generation_prompt.format(
                    research_brief=research_brief,
                    messages=messages_text,
                    findings=truncated_findings,
                    date=get_today_str(),
                )

                agent = Agent(client=report_client, instructions="You are a research report writer.")
                result = await agent.run(prompt)

                await ctx.add_event(WorkflowEvent(type="progress", data="Final report complete"))
                await ctx.yield_output(result.text)
                return

            except Exception as e:
                if is_token_limit_exceeded(e, self._config.final_report_model):
                    current_retry += 1
                    if current_retry == 1:
                        token_limit = get_model_token_limit(self._config.final_report_model)
                        findings_limit = (token_limit or 100_000) * 4
                    else:
                        findings_limit = int(findings_limit * 0.9)
                    continue
                else:
                    await ctx.yield_output(f"Error generating final report: {e}")
                    return

        await ctx.yield_output("Error generating final report: Maximum retries exceeded")
