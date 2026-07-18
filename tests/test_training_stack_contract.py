from __future__ import annotations

from pathlib import Path

import pytest

from tickettune.config import load_config

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket.:UserWarning")
def test_pinned_training_stack_constructs_public_trl_and_peft_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    pytest.importorskip("transformers")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from tickettune.training import (
        _build_peft_config,
        _build_sft_config,
        _load_training_libraries,
        _model_init_kwargs,
        build_quantization_config,
    )

    libraries = _load_training_libraries()
    cpu_config = load_config(ROOT / "configs" / "cpu-smoke.yaml")
    sft_config = _build_sft_config(
        cpu_config,
        libraries,
        has_evaluation=True,
    )
    peft_config = _build_peft_config(cpu_config, libraries)
    model_kwargs = _model_init_kwargs(cpu_config, libraries, allow_download=False)

    assert sft_config.use_cpu is True
    assert sft_config.warmup_steps == 0
    assert sft_config.max_length == 256
    assert peft_config.base_model_name_or_path == cpu_config.model.name_or_path
    assert peft_config.revision == cpu_config.model.revision
    assert model_kwargs["device_map"] is None
    assert model_kwargs["local_files_only"] is True

    qlora_config = load_config(ROOT / "configs" / "qwen-7b-qlora.yaml")
    quantization = build_quantization_config(qlora_config, libraries)
    assert quantization is not None
    assert quantization.load_in_4bit is True
    assert quantization.bnb_4bit_quant_type == "nf4"
