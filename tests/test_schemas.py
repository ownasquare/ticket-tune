from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tickettune.schemas import TicketExample, TriageOutput
from tickettune.strict_json import NonFiniteJSONConstantError, loads_strict

FIXTURE = Path(__file__).parent / "fixtures" / "tickets.jsonl"


def _record() -> dict[str, object]:
    first_line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    value = json.loads(first_line)
    assert isinstance(value, dict)
    return value


def test_expected_output_uses_known_labels_and_required_fields() -> None:
    value = TriageOutput(
        category="billing",
        priority="high",
        sentiment="frustrated",
        response="I can help review the invoice safely.",
        next_action="review_invoice",
    )

    assert value.priority == "high"
    assert set(value.model_dump()) == {
        "category",
        "priority",
        "sentiment",
        "response",
        "next_action",
    }


def test_unknown_output_label_is_rejected() -> None:
    with pytest.raises(ValidationError, match="priority"):
        TriageOutput(
            category="billing",
            priority="critical",  # type: ignore[arg-type]
            sentiment="worried",
            response="I can help review this invoice safely.",
            next_action="review_invoice",
        )


def test_unknown_source_key_is_rejected() -> None:
    record = _record()
    record["customer_email"] = "not-allowed"

    with pytest.raises(ValidationError, match="customer_email"):
        TicketExample.model_validate(record)


def test_message_role_order_is_exact() -> None:
    record = _record()
    messages = record["messages"]
    assert isinstance(messages, list)
    messages[0], messages[1] = messages[1], messages[0]

    with pytest.raises(ValidationError, match="ordered exactly"):
        TicketExample.model_validate(record)


def test_system_message_must_match_shared_task_contract() -> None:
    record = _record()
    messages = record["messages"]
    assert isinstance(messages, list)
    system = messages[0]
    assert isinstance(system, dict)
    system["content"] = "Return something helpful."

    with pytest.raises(ValidationError, match="TicketTune task contract"):
        TicketExample.model_validate(record)


def test_assistant_json_must_match_expected() -> None:
    record = _record()
    messages = record["messages"]
    assert isinstance(messages, list)
    assistant = messages[-1]
    assert isinstance(assistant, dict)
    payload = json.loads(str(assistant["content"]))
    payload["priority"] = "urgent"
    assistant["content"] = json.dumps(payload)

    with pytest.raises(ValidationError, match="exactly match expected"):
        TicketExample.model_validate(record)


def test_assistant_json_rejects_duplicate_category_when_later_value_matches() -> None:
    record = _record()
    messages = record["messages"]
    expected = record["expected"]
    assert isinstance(messages, list)
    assert isinstance(expected, dict)
    assistant = messages[-1]
    assert isinstance(assistant, dict)
    expected_category = str(expected["category"])
    shadow_category = "billing" if expected_category != "billing" else "bug"
    remainder = {key: value for key, value in expected.items() if key != "category"}
    assistant["content"] = (
        f'{{"category":{json.dumps(shadow_category)},'
        f'"category":{json.dumps(expected_category)},'
        f"{json.dumps(remainder, sort_keys=True)[1:]}"
    )

    with pytest.raises(ValidationError, match="valid TriageOutput JSON"):
        TicketExample.model_validate(record)


def test_assistant_message_must_be_plain_valid_json() -> None:
    record = _record()
    messages = record["messages"]
    assert isinstance(messages, list)
    assistant = messages[-1]
    assert isinstance(assistant, dict)
    assistant["content"] = "```json\n{}\n```"

    with pytest.raises(ValidationError, match="valid TriageOutput JSON"):
        TicketExample.model_validate(record)


def test_placeholder_inventory_must_be_complete() -> None:
    record = _record()
    placeholders = record["pii_placeholders"]
    assert isinstance(placeholders, list)
    placeholders.pop()

    with pytest.raises(ValidationError, match="placeholder inventory mismatch"):
        TicketExample.model_validate(record)


def test_placeholder_inventory_rejects_duplicates_and_unused_values() -> None:
    duplicate_record = _record()
    duplicate_placeholders = duplicate_record["pii_placeholders"]
    assert isinstance(duplicate_placeholders, list)
    duplicate_placeholders.append(duplicate_placeholders[0])
    with pytest.raises(ValidationError, match="must be unique"):
        TicketExample.model_validate(duplicate_record)

    unused_record = _record()
    unused_placeholders = unused_record["pii_placeholders"]
    assert isinstance(unused_placeholders, list)
    unused_placeholders.append("[UNUSED]")
    with pytest.raises(ValidationError, match="unused placeholders"):
        TicketExample.model_validate(unused_record)


def test_user_message_itself_must_contain_a_placeholder() -> None:
    record = _record()
    messages = record["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    assert isinstance(user_message, dict)
    content = str(user_message["content"])
    for placeholder in record["pii_placeholders"]:
        content = content.replace(str(placeholder), "the customer")
    user_message["content"] = content

    with pytest.raises(ValidationError, match="user message must use"):
        TicketExample.model_validate(record)


def test_prompt_and_completion_projection_boundaries() -> None:
    example = TicketExample.model_validate(_record())

    assert [message.role for message in example.prompt_messages] == ["system", "user"]
    assert [message.role for message in example.completion_messages] == ["assistant"]


def test_source_contracts_are_immutable_after_validation() -> None:
    example = TicketExample.model_validate(_record())

    with pytest.raises(ValidationError, match="frozen"):
        example.id = "TT-MUTATED"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_non_finite_constants(constant: str) -> None:
    with pytest.raises(NonFiniteJSONConstantError, match="non-finite JSON constant"):
        loads_strict(f'{{"metrics":{{"loss":{constant}}}}}')
