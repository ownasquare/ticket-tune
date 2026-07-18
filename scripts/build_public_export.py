"""Command wrapper for TicketTune's deterministic public exporter."""

from __future__ import annotations

from tickettune.public_export import main

if __name__ == "__main__":
    raise SystemExit(main())
