"""Workflow executors for the deep research pipeline.

Each executor corresponds to a stage in the research workflow:
ClarifyExecutor → WriteBriefExecutor → (SupervisorAgent) → FinalReportExecutor
"""

import json
import logging

from agent_framework import Agent, BaseChatClient, Executor, Message, WorkflowContext, handler, response_handler

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
    async def handle(self, messages: list[Message], ctx: WorkflowContext) -> None:
        messages_text = "\n".join(m.text for m in messages if m.text)

        if not self._config.allow_clarification:
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
            await ctx.request_info(
                HumanClarificationRequest(
                    question=clarification.question,
                    verification=clarification.verification,
                ),
                str,
            )
        else:
            ctx.set_state("verification", clarification.verification)
            await ctx.send_message(ResearchBriefMessage(messages_text=messages_text))

    @response_handler
    async def on_clarification(
        self,
        original: HumanClarificationRequest,
        response: str,
        ctx: WorkflowContext,
    ) -> None:
        """Handle user's clarification response, then proceed to research."""
        messages_text = f"User query (with clarification):\n{response}"
        await ctx.send_message(ResearchBriefMessage(messages_text=messages_text))


class WriteBriefExecutor(Executor):
    """Transform user messages into a structured research brief."""

    def __init__(self, client: BaseChatClient, config: Configuration):
        super().__init__(id="write_brief")
        self._client = client
        self._config = config

    @handler
    async def handle(self, msg: ResearchBriefMessage, ctx: WorkflowContext) -> None:
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
        await ctx.send_message(SupervisorInput(research_brief=research_brief))


class FinalReportExecutor(Executor):
    """Generate the final comprehensive research report."""

    def __init__(self, client: BaseChatClient, config: Configuration):
        super().__init__(id="final_report")
        self._client = client
        self._config = config

    @handler
    async def handle(self, trigger: str, ctx: WorkflowContext) -> None:
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
