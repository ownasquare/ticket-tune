"""Command wrapper for TicketTune's tracked-tree and Git-history secret scan."""

from __future__ import annotations

from tickettune.secret_scan import main

if __name__ == "__main__":
    raise SystemExit(main())
