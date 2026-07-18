"""Verify the built wheel's self-contained newcomer workflow."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("tickettune-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one TicketTune wheel; found {len(wheels)}")
    wheel = wheels[0]
    sys.path.insert(0, str(wheel))

    import tickettune
    from tickettune.onboarding import init_project, run_quickstart, starter_config_path

    if str(wheel) not in str(tickettune.__file__):
        raise RuntimeError(
            f"TicketTune was not imported from the built wheel: {tickettune.__file__}"
        )
    if str(wheel) not in str(starter_config_path()):
        raise RuntimeError("starter config fell back to checkout-owned files")

    with tempfile.TemporaryDirectory(prefix="tickettune-wheel-proof-") as temporary:
        proof_root = Path(temporary)
        initialized = init_project(proof_root / "project")
        if len(initialized.dataset_path.read_text(encoding="utf-8").splitlines()) != 56:
            raise RuntimeError("wheel starter dataset does not contain 56 records")
        result = run_quickstart(proof_root / "quickstart")
        if not result.training_plan_ready or not result.evaluation_passed:
            raise RuntimeError("wheel quickstart did not pass")

    print("Wheel install proof passed: bundled starter and offline quickstart are self-contained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
