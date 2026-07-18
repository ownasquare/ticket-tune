"""Check a local TicketTune-compatible HTTP health endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin, urlparse


def _require_safe_url(url: str, *, allow_remote: bool) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an absolute HTTP(S) URL")
    if parsed.hostname not in {"127.0.0.1", "::1", "localhost"} and not allow_remote:
        raise ValueError("remote health checks require --allow-remote")
    return url.rstrip("/")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()
    base_url = _require_safe_url(args.base_url, allow_remote=args.allow_remote)

    request = urllib.request.Request(f"{base_url}/health", method="GET")
    opener = urllib.request.build_opener(_SafeRedirectHandler(allow_remote=args.allow_remote))
    try:
        # Initial and redirected URLs are policy-checked and default to loopback.
        with opener.open(request, timeout=5) as response:  # nosec B310
            status = response.status
    except (urllib.error.URLError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else str(exc)
        print(json.dumps({"healthy": False, "error": str(reason)}, sort_keys=True))
        return 1
    healthy = status == 200
    print(json.dumps({"healthy": healthy, "status": status}, sort_keys=True))
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
