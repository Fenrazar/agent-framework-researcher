"""Pydantic data models and inter-executor message types.

Structured output models are ported from open_deep_research/state.py.
Dataclasses are new AF-specific message types for workflow communication.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# Structured output models (used for JSON parsing from LLM responses)


class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


# Inter-executor message dataclasses


@dataclass
class ResearchBriefMessage:
    """Passed from ClarifyExecutor → WriteBriefExecutor."""

    messages_text: str


@dataclass
class SupervisorInput:
    """Passed from WriteBriefExecutor → SupervisorAgent."""

    research_brief: str


@dataclass
class HumanClarificationRequest:
    """Sent to the user via ctx.request_info() for HITL clarification."""

    question: str
    verification: str = ""


@dataclass
class CompressedResearch:
    """Result from a single researcher's compressed findings."""

    compressed_research: str
    raw_notes: list[str] = field(default_factory=list)
