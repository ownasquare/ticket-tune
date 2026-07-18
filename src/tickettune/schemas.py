"""Canonical data contracts for TicketTune support-ticket triage."""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .prompts import SYSTEM_PROMPT
from .strict_json import StrictJSONError, loads_strict

Category = Literal[
    "account_access",
    "billing",
    "bug",
    "cancellation",
    "feature_request",
    "shipping",
    "security",
]
Priority = Literal["low", "medium", "high", "urgent"]
Sentiment = Literal["positive", "neutral", "frustrated", "angry", "worried"]
MessageRole = Literal["system", "user", "assistant"]
PiiPlaceholder = Annotated[str, Field(pattern=r"^\[[A-Z][A-Z0-9_]*\]$")]

CATEGORY_LABELS: tuple[Category, ...] = (
    "account_access",
    "billing",
    "bug",
    "cancellation",
    "feature_request",
    "shipping",
    "security",
)
PRIORITY_LABELS: tuple[Priority, ...] = ("low", "medium", "high", "urgent")
SENTIMENT_LABELS: tuple[Sentiment, ...] = (
    "positive",
    "neutral",
    "frustrated",
    "angry",
    "worried",
)

_PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z][A-Z0-9_]*\]")


class StrictSchema(BaseModel):
    """Shared strictness for source records and generated predictions."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class ChatMessage(StrictSchema):
    """One message in the conversational fine-tuning record."""

    role: MessageRole
    content: str = Field(min_length=1, max_length=8000)


class TriageOutput(StrictSchema):
    """The exact JSON object TicketTune is trained to produce."""

    category: Category
    priority: Priority
    sentiment: Sentiment
    response: str = Field(min_length=12, max_length=1200)
    next_action: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")


class Provenance(StrictSchema):
    """Auditable origin statement carried by every source example."""

    source: Literal["synthetic"] = "synthetic"
    created_by: str = Field(min_length=3, max_length=120)
    license: Literal["CC0-1.0"] = "CC0-1.0"
    contains_real_customer_data: Literal[False] = False


class TicketExample(StrictSchema):
    """One validated source conversation plus its structured gold output."""

    id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{2,63}$")
    messages: list[ChatMessage] = Field(min_length=3, max_length=3)
    expected: TriageOutput
    provenance: Provenance
    pii_placeholders: list[PiiPlaceholder] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_conversation(self) -> Self:
        roles = [message.role for message in self.messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError("messages must be ordered exactly as system, user, assistant")
        if self.messages[0].content != SYSTEM_PROMPT:
            raise ValueError("system message must exactly match the TicketTune task contract")

        try:
            loads_strict(self.messages[-1].content)
            assistant_output = TriageOutput.model_validate_json(self.messages[-1].content)
        except (ValidationError, json.JSONDecodeError, StrictJSONError) as exc:
            raise ValueError(
                "assistant message must be one valid TriageOutput JSON object"
            ) from exc
        if assistant_output != self.expected:
            raise ValueError("assistant JSON must exactly match expected")

        if len(self.pii_placeholders) != len(set(self.pii_placeholders)):
            raise ValueError("pii_placeholders must be unique")
        combined = "\n".join(message.content for message in self.messages)
        combined += f"\n{self.expected.response}"
        discovered = set(_PLACEHOLDER_PATTERN.findall(combined))
        declared = set(self.pii_placeholders)
        if discovered != declared:
            missing = sorted(discovered - declared)
            unused = sorted(declared - discovered)
            details: list[str] = []
            if missing:
                details.append(f"undeclared placeholders: {missing}")
            if unused:
                details.append(f"unused placeholders: {unused}")
            raise ValueError("pii placeholder inventory mismatch; " + "; ".join(details))
        if not _PLACEHOLDER_PATTERN.search(self.messages[1].content):
            raise ValueError("user message must use at least one explicit PII placeholder")
        return self

    @property
    def prompt_messages(self) -> list[ChatMessage]:
        """System and user messages supplied to TRL as the prompt."""

        return self.messages[:2]

    @property
    def completion_messages(self) -> list[ChatMessage]:
        """Assistant-only completion used for completion-only loss."""

        return self.messages[2:]
