"""Send one deterministic support-ticket request through `uv run python` locally."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin, urlparse

from tickettune.prompts import SYSTEM_PROMPT
from tickettune.schemas import TriageOutput
from tickettune.strict_json import StrictJSONError, loads_strict

DEFAULT_TICKET = "I was charged twice for invoice [INVOICE_ID]. Please help."


def _require_safe_url(url: str, *, allow_remote: bool) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an absolute HTTP(S) URL")
    if parsed.hostname not in {"127.0.0.1", "::1", "localhost"} and not allow_remote:
        raise ValueError("remote requests require --allow-remote")
    return url.rstrip("/")


def _response_content(value: dict[str, Any]) -> str:
    try:
        content = value["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response did not contain choices[0].message.content") from exc
    if not isinstance(content, str):
        raise ValueError("response content was not a string")
    return content


def _validated_response(value: dict[str, Any]) -> str:
    """Return canonical task JSON or fail without printing untrusted model text."""

    content = _response_content(value)
    try:
        decoded = loads_strict(content)
    except (json.JSONDecodeError, StrictJSONError) as exc:
        raise ValueError("model response was not valid JSON") from exc
    try:
        validated = TriageOutput.model_validate(decoded)
    except ValueError as exc:
        raise ValueError("model response did not match the TicketTune output schema") from exc
    return validated.model_dump_json()


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-apply the URL policy to every HTTP redirect target."""

    def __init__(self, *, allow_remote: bool) -> None:
        super().__init__()
        self._allow_remote = allow_remote

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        safe_target = _require_safe_url(
            urljoin(req.full_url, newurl), allow_remote=self._allow_remote
        )
        return super().redirect_request(req, fp, code, msg, headers, safe_target)


def _request_payload(*, model: str, ticket: str) -> dict[str, Any]:
    """Build the same canonical system-plus-user contract used by evaluation."""

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="tickettune")
    parser.add_argument("--ticket", default=DEFAULT_TICKET)
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()
    base_url = _require_safe_url(args.base_url, allow_remote=args.allow_remote)

    body = json.dumps(_request_payload(model=args.model, ticket=args.ticket)).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(_SafeRedirectHandler(allow_remote=args.allow_remote))
    try:
        # Initial and redirected URLs are policy-checked and default to loopback.
        with opener.open(request, timeout=60) as response:  # nosec B310
            payload = loads_strict(response.read())
    except (urllib.error.URLError, UnicodeDecodeError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else str(exc)
        print(json.dumps({"error": str(reason)}, sort_keys=True))
        return 1
    if not isinstance(payload, dict):
        print(json.dumps({"error": "server response was not a JSON object"}, sort_keys=True))
        return 2
    try:
        output = _validated_response(payload)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
