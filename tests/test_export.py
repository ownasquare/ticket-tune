from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import types
import urllib.request
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
import yaml

from tickettune import export as export_module
from tickettune.export import (
    LLAMA_CPP_REVISION,
    ExportExecutionError,
    ExportValidationError,
    materialize_ollama_plan,
    merge_adapter,
    render_ollama_modelfile,
    run_argv,
    write_ollama_export_provenance,
)
from tickettune.export import (
    build_merge_plan as _build_merge_plan,
)
from tickettune.export import (
    build_ollama_export_plan as _build_ollama_export_plan,
)
from tickettune.export import (
    build_vllm_argv as _build_vllm_argv,
)
from tickettune.export import (
    build_vllm_plan as _build_vllm_plan,
)
from tickettune.prompts import SYSTEM_PROMPT
from tickettune.run_manifest import ArtifactDigest, RunManifest, canonical_json_bytes

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROJECT_ROOT = Path(__file__).parents[1]

# The long-standing unit fixtures below intentionally omit full training-run
# artifacts. Keep them on the explicit non-release path; strict-default tests
# call the functions through ``export_module`` instead.
build_merge_plan = partial(_build_merge_plan, allow_unqualified_local_smoke=True)
build_vllm_plan = partial(_build_vllm_plan, allow_unqualified_local_smoke=True)
build_vllm_argv = partial(_build_vllm_argv, allow_unqualified_local_smoke=True)
build_ollama_export_plan = partial(
    _build_ollama_export_plan,
    allow_unqualified_local_smoke=True,
)


def _adapter(
    tmp_path: Path,
    *,
    base_model: str = BASE_MODEL,
    rank: int = 16,
    model_revision: str | None = None,
) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config: dict[str, object] = {"base_model_name_or_path": base_model, "r": rank}
    if model_revision is not None:
        config["revision"] = model_revision
    (adapter / "adapter_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"offline-test-placeholder")
    return adapter


def _attach_training_manifest(
    adapter: Path,
    *,
    model_revision: str = MODEL_REVISION,
    qualified: bool = True,
) -> Path:
    config = {"profile": "quality" if qualified else "smoke"}
    dataset_sha256 = {
        "source": "1" * 64,
        "manifest": "2" * 64,
        "train": "3" * 64,
        "validation": "4" * 64,
    }
    if qualified:
        dataset_sha256.update(
            {
                "qualification_review_manifest": "5" * 64,
                "qualification_report": "6" * 64,
            }
        )
    artifacts = tuple(
        ArtifactDigest(
            path=f"{adapter.name}/{path.name}",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
        for path in (
            adapter / "adapter_config.json",
            adapter / "adapter_model.safetensors",
        )
    )
    manifest = RunManifest(
        run_id=adapter.parent.name,
        created_at=datetime.now(UTC),
        status="completed",
        project_name="ticket-tune",
        model_name_or_path=BASE_MODEL,
        model_revision=model_revision,
        method="qlora",
        seed=42,
        config_sha256=hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        config=config,
        dataset_sha256=dataset_sha256,
        packages={},
        runtime={},
        artifacts=artifacts,
    )
    manifest_path = adapter.parent / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _merged_model(
    tmp_path: Path,
    *,
    model_type: str = "qwen2",
    qualified_lineage: bool = False,
) -> Path:
    merged = tmp_path / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text(
        json.dumps({"model_type": model_type}),
        encoding="utf-8",
    )
    (merged / "model.safetensors").write_bytes(b"offline-test-placeholder")
    artifact_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(merged.iterdir())
        if path.is_file()
    }
    provenance: dict[str, object] = {
        "schema_version": 1,
        "operation": "peft_safe_merge",
        "base_model": BASE_MODEL,
        "model_revision": MODEL_REVISION,
        "adapter_base_model": BASE_MODEL,
        "adapter_revision": MODEL_REVISION,
        "adapter_config_sha256": hashlib.sha256(b"offline-test-adapter-config").hexdigest(),
        "adapter_weight_files": ["adapter_model.safetensors"],
        "adapter_weight_sha256": [hashlib.sha256(b"offline-test-adapter-weights").hexdigest()],
        "load_in_4bit": False,
        "load_in_8bit": False,
        "safe_merge": True,
        "safe_serialization": True,
        "trust_remote_code": False,
        "artifact_sha256": artifact_hashes,
    }
    if qualified_lineage:
        provenance.update(
            {
                "training_manifest_sha256": "7" * 64,
                "training_config_sha256": "8" * 64,
                "training_dataset_sha256": {
                    "source": "1" * 64,
                    "manifest": "2" * 64,
                    "train": "3" * 64,
                    "validation": "4" * 64,
                },
                "qualification_sha256": {
                    "qualification_review_manifest": "5" * 64,
                    "qualification_report": "6" * 64,
                },
                "lineage_boundary": "qualified_release_lineage",
            }
        )
    (merged / "tickettune-merge-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return merged


def test_release_planners_require_completed_sibling_training_manifest(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ExportValidationError, match="completed sibling training manifest"):
        export_module.build_merge_plan(
            BASE_MODEL,
            adapter,
            tmp_path / "merged",
            model_revision=MODEL_REVISION,
        )
    with pytest.raises(ExportValidationError, match="completed sibling training manifest"):
        export_module.build_vllm_plan(
            BASE_MODEL,
            adapter,
            model_revision=MODEL_REVISION,
        )


def test_release_planners_bind_configured_revision_when_adapter_omits_it(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    _attach_training_manifest(adapter, model_revision="0" * 40)

    with pytest.raises(ExportValidationError, match="configured model revision"):
        export_module.build_merge_plan(
            BASE_MODEL,
            adapter,
            tmp_path / "merged",
            model_revision=MODEL_REVISION,
        )
    with pytest.raises(ExportValidationError, match="configured model revision"):
        export_module.build_vllm_plan(
            BASE_MODEL,
            adapter,
            model_revision=MODEL_REVISION,
        )


def test_local_smoke_override_rejects_broken_training_manifest_symlink(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    (tmp_path / "manifest.json").symlink_to(tmp_path / "missing-manifest.json")

    with pytest.raises(ExportValidationError, match="regular non-symlink"):
        export_module.build_merge_plan(
            BASE_MODEL,
            adapter,
            tmp_path / "merged",
            model_revision=MODEL_REVISION,
            allow_unqualified_local_smoke=True,
        )


def test_release_planners_require_qualification_lineage_and_mark_smoke_override(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    _attach_training_manifest(adapter, qualified=False)

    with pytest.raises(ExportValidationError, match="qualification lineage"):
        export_module.build_merge_plan(
            BASE_MODEL,
            adapter,
            tmp_path / "strict-merged",
            model_revision=MODEL_REVISION,
        )
    with pytest.raises(ExportValidationError, match="qualification lineage"):
        export_module.build_vllm_plan(
            BASE_MODEL,
            adapter,
            model_revision=MODEL_REVISION,
        )

    merge_plan = export_module.build_merge_plan(
        BASE_MODEL,
        adapter,
        tmp_path / "smoke-merged",
        model_revision=MODEL_REVISION,
        allow_unqualified_local_smoke=True,
    )
    vllm_plan = export_module.build_vllm_plan(
        BASE_MODEL,
        adapter,
        model_revision=MODEL_REVISION,
        allow_unqualified_local_smoke=True,
    )
    expected = "unqualified_local_smoke_override_not_release_evidence"
    assert merge_plan.lineage_boundary == expected
    assert vllm_plan.lineage_boundary == expected


def test_release_planners_accept_qualified_revision_bound_lineage(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    _attach_training_manifest(adapter)

    merge_plan = export_module.build_merge_plan(
        BASE_MODEL,
        adapter,
        tmp_path / "merged",
        model_revision=MODEL_REVISION,
    )
    vllm_plan = export_module.build_vllm_plan(
        BASE_MODEL,
        adapter,
        model_revision=MODEL_REVISION,
    )

    assert merge_plan.lineage_boundary == "qualified_release_lineage"
    assert vllm_plan.lineage_boundary == "qualified_release_lineage"
    assert vllm_plan.training_manifest_sha256 == merge_plan.training_manifest_sha256
    assert dict(vllm_plan.qualification_sha256) == {
        "qualification_report": "6" * 64,
        "qualification_review_manifest": "5" * 64,
    }


def test_merge_release_boundary_requires_configured_revision(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    _attach_training_manifest(adapter)

    with pytest.raises(ExportValidationError, match="exact configured model revision"):
        export_module.build_merge_plan(BASE_MODEL, adapter, tmp_path / "strict-merged")

    smoke_plan = export_module.build_merge_plan(
        BASE_MODEL,
        adapter,
        tmp_path / "smoke-merged",
        allow_unqualified_local_smoke=True,
    )
    assert smoke_plan.lineage_boundary == "unqualified_local_smoke_override_not_release_evidence"


def test_ollama_release_plan_requires_qualified_merge_lineage(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)

    with pytest.raises(ExportValidationError, match="qualified merge lineage"):
        export_module.build_ollama_export_plan(merged, tmp_path / "strict-ollama")

    smoke_plan = export_module.build_ollama_export_plan(
        merged,
        tmp_path / "smoke-ollama",
        allow_unqualified_local_smoke=True,
    )
    assert smoke_plan.lineage_boundary == "unqualified_local_smoke_override_not_release_evidence"


def test_ollama_release_plan_accepts_qualified_merge_lineage(tmp_path: Path) -> None:
    plan = export_module.build_ollama_export_plan(
        _merged_model(tmp_path, qualified_lineage=True),
        tmp_path / "ollama",
    )

    assert plan.lineage_boundary == "qualified_release_lineage"
    assert dict(plan.qualification_sha256) == {
        "qualification_report": "6" * 64,
        "qualification_review_manifest": "5" * 64,
    }


def test_merge_plan_requires_matching_base_and_non_quantized_reload(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    plan = build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")

    assert plan.base_model == BASE_MODEL
    assert plan.adapter_base_model == BASE_MODEL
    assert plan.adapter_revision is None
    assert len(plan.adapter_config_sha256) == 64
    assert plan.adapter_weight_files == ("adapter_model.safetensors",)
    assert len(plan.adapter_weight_sha256[0]) == 64
    assert plan.dtype == "bfloat16"
    assert plan.allow_download is False
    assert plan.safe_merge is True
    assert plan.load_in_4bit is False
    assert plan.trust_remote_code is False
    assert plan.to_dict()["adapter_path"] == str(adapter.resolve())
    assert plan.model_dump(mode="json")["safe_merge"] is True


def test_merge_plan_rejects_adapter_for_another_base(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, base_model="Qwen/Qwen2.5-1.5B-Instruct")

    with pytest.raises(ExportValidationError, match="base model mismatch"):
        build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")


def test_merge_plan_rejects_empty_base_model(tmp_path: Path) -> None:
    with pytest.raises(ExportValidationError, match="cannot be empty"):
        build_merge_plan(" ", _adapter(tmp_path), tmp_path / "merged")


def test_merge_plan_rejects_missing_adapter_weights(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    (adapter / "adapter_model.safetensors").unlink()

    with pytest.raises(ExportValidationError, match="adapter weights"):
        build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")


def test_merge_execution_is_offline_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path)
    model_calls: list[dict[str, object]] = []
    tokenizer_calls: list[dict[str, object]] = []

    class FakeMerged:
        def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
            assert safe_serialization is True
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"merged")

    class FakeAdapted:
        def merge_and_unload(self, *, safe_merge: bool) -> FakeMerged:
            assert safe_merge is True
            return FakeMerged()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, base: object, path: str, *, is_trainable: bool) -> FakeAdapted:
            assert base is not None
            assert path == str(adapter.resolve())
            assert is_trainable is False
            return FakeAdapted()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> object:
            assert name == BASE_MODEL
            model_calls.append(kwargs)
            return object()

    class FakeTokenizer:
        def save_pretrained(self, path: Path) -> None:
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> FakeTokenizer:
            assert name == BASE_MODEL
            tokenizer_calls.append(kwargs)
            return FakeTokenizer()

    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = object()  # type: ignore[attr-defined]
    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = FakePeftModel  # type: ignore[attr-defined]
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoModelForCausalLM = FakeAutoModel  # type: ignore[attr-defined]
    transformers_module.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    plan = build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")
    result = merge_adapter(plan)

    assert result.output_dir == str((tmp_path / "merged").resolve())
    merge_provenance = json.loads(Path(result.provenance_path).read_text(encoding="utf-8"))
    assert (
        merge_provenance["lineage_boundary"]
        == "unqualified_local_smoke_override_not_release_evidence"
    )
    assert model_calls[0]["local_files_only"] is True
    assert tokenizer_calls[0]["local_files_only"] is True

    original_metadata = export_module._adapter_metadata(adapter)
    metadata_calls = 0

    def changing_metadata(
        path: Path,
        *,
        expected_model_revision: str | None = None,
    ) -> export_module.AdapterMetadata:
        nonlocal metadata_calls
        assert path == adapter.resolve()
        assert expected_model_revision is None
        metadata_calls += 1
        if metadata_calls == 1:
            return original_metadata
        return replace(original_metadata, weight_sha256=("0" * 64,))

    monkeypatch.setattr(export_module, "_adapter_metadata", changing_metadata)
    changed_plan = replace(plan, output_dir=str((tmp_path / "merged-changed").resolve()))
    with pytest.raises(ExportExecutionError, match="changed while the merge"):
        merge_adapter(changed_plan)


def test_merge_download_requires_immutable_revision(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ExportValidationError, match="full 40-character commit SHA"):
        build_merge_plan(
            BASE_MODEL,
            adapter,
            tmp_path / "merged",
            model_revision="main",
            allow_download=True,
        )


def test_merge_download_accepts_full_commit_revision(tmp_path: Path) -> None:
    plan = build_merge_plan(
        BASE_MODEL,
        _adapter(tmp_path),
        tmp_path / "merged",
        model_revision="a" * 40,
        allow_download=True,
    )

    assert plan.allow_download is True
    assert plan.model_revision == "a" * 40


def test_merge_plan_and_execution_reject_declared_revision_mismatch(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path, model_revision=MODEL_REVISION)

    with pytest.raises(ExportValidationError, match="revision mismatch"):
        build_merge_plan(
            BASE_MODEL,
            adapter,
            tmp_path / "merged",
            model_revision="b" * 40,
        )

    plan = build_merge_plan(
        BASE_MODEL,
        adapter,
        tmp_path / "merged",
        model_revision=MODEL_REVISION,
    )
    config_path = adapter / "adapter_config.json"
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["revision"] = "b" * 40
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ExportValidationError, match="revision mismatch"):
        merge_adapter(plan)


@pytest.mark.parametrize("dtype", ["auto", "int8", "float64"])
def test_merge_plan_rejects_unsafe_dtype(tmp_path: Path, dtype: str) -> None:
    with pytest.raises(ExportValidationError, match="Merge dtype"):
        build_merge_plan(BASE_MODEL, _adapter(tmp_path), tmp_path / "merged", dtype=dtype)


def test_merge_plan_rejects_adapter_as_output(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    for output in (adapter, adapter / "merged", tmp_path):
        with pytest.raises(ExportValidationError, match="overwrite the adapter"):
            build_merge_plan(BASE_MODEL, adapter, output)


def test_merge_plan_and_execution_reject_local_base_output_overlap(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base"
    base.mkdir(parents=True)
    adapter_root = tmp_path / "adapter-root"
    adapter_root.mkdir()
    adapter = _adapter(adapter_root, base_model=str(base))

    for output in (base / "merged", base.parent):
        with pytest.raises(ExportValidationError, match="local base-model"):
            build_merge_plan(str(base), adapter, output)

    plan = build_merge_plan(str(base), adapter, tmp_path / "safe-output")
    unsafe_plan = replace(plan, output_dir=str((base / "merged").resolve()))
    with pytest.raises(ExportExecutionError, match="local base-model"):
        merge_adapter(unsafe_plan)


def test_merge_execution_rejects_adapter_changed_after_planning(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    plan = build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")
    (adapter / "adapter_model.safetensors").write_bytes(b"changed-after-plan")

    with pytest.raises(ExportExecutionError, match="changed after the merge plan"):
        merge_adapter(plan)


def test_merge_plan_supports_matching_local_base_path(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter(tmp_path, base_model=str(base))

    plan = build_merge_plan(str(base), adapter, tmp_path / "merged")

    assert plan.base_model == str(base)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("not-json", "Invalid PEFT adapter config"),
        ("[]", "expected a JSON object"),
        (json.dumps({"r": 16}), "base_model_name_or_path"),
        (json.dumps({"base_model_name_or_path": BASE_MODEL, "r": True}), "positive integer r"),
    ],
)
def test_merge_plan_rejects_invalid_adapter_config(
    tmp_path: Path, config: str, message: str
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(config, encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    with pytest.raises(ExportValidationError, match=message):
        build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")


def test_merge_and_vllm_plans_reject_symlinked_adapter_config(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    target = tmp_path / "real-adapter-config.json"
    target.write_text(
        json.dumps({"base_model_name_or_path": BASE_MODEL, "r": 16}),
        encoding="utf-8",
    )
    (adapter / "adapter_config.json").symlink_to(target)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    with pytest.raises(ExportValidationError, match="non-symlink"):
        build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")
    with pytest.raises(ExportValidationError, match="non-symlink"):
        build_vllm_plan(BASE_MODEL, adapter, model_revision=MODEL_REVISION)


def test_merge_and_vllm_plans_reject_symlinked_adapter_root(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    linked = tmp_path / "linked-adapter"
    linked.symlink_to(adapter, target_is_directory=True)

    with pytest.raises(ExportValidationError, match="directory cannot be a symlink"):
        build_merge_plan(BASE_MODEL, linked, tmp_path / "merged")
    with pytest.raises(ExportValidationError, match="directory cannot be a symlink"):
        build_vllm_plan(BASE_MODEL, linked, model_revision=MODEL_REVISION)


def test_merge_and_vllm_plans_reject_symlinked_adapter_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": BASE_MODEL, "r": 16}),
        encoding="utf-8",
    )
    target = tmp_path / "real-adapter-model.safetensors"
    target.write_bytes(b"weights")
    (adapter / "adapter_model.safetensors").symlink_to(target)

    with pytest.raises(ExportValidationError, match="must not be symlinks"):
        build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")
    with pytest.raises(ExportValidationError, match="must not be symlinks"):
        build_vllm_plan(BASE_MODEL, adapter, model_revision=MODEL_REVISION)


def test_merge_plan_rejects_duplicate_adapter_config_keys(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        f'{{"base_model_name_or_path":"wrong","base_model_name_or_path":"{BASE_MODEL}","r":16}}',
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    with pytest.raises(ExportValidationError, match="duplicate JSON object key"):
        build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")


def test_merge_plan_rejects_missing_adapter_directory_and_config(tmp_path: Path) -> None:
    with pytest.raises(ExportValidationError, match="does not exist"):
        build_merge_plan(BASE_MODEL, tmp_path / "missing", tmp_path / "merged")

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(ExportValidationError, match="Missing PEFT adapter config"):
        build_merge_plan(BASE_MODEL, adapter, tmp_path / "merged")


def test_merge_dry_run_and_existing_output_refusal(tmp_path: Path) -> None:
    output = tmp_path / "merged"
    plan = build_merge_plan(BASE_MODEL, _adapter(tmp_path), output)

    assert merge_adapter(plan, dry_run=True) is plan
    output.mkdir()
    with pytest.raises(ExportExecutionError, match="Refusing to overwrite"):
        merge_adapter(plan)


def test_merge_refuses_quantized_or_unsafe_plan(tmp_path: Path) -> None:
    plan = build_merge_plan(BASE_MODEL, _adapter(tmp_path), tmp_path / "merged")

    with pytest.raises(ExportExecutionError, match="quantized or unsafe"):
        merge_adapter(replace(plan, load_in_4bit=True))
    with pytest.raises(ExportExecutionError, match="quantized or unsafe"):
        merge_adapter(replace(plan, safe_merge=False))


def test_merge_reports_missing_training_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_merge_plan(BASE_MODEL, _adapter(tmp_path), tmp_path / "merged")
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ExportExecutionError, match="training extra"):
        merge_adapter(plan)


def test_merge_rejects_unavailable_torch_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_merge_plan(BASE_MODEL, _adapter(tmp_path), tmp_path / "merged")
    torch_module = types.ModuleType("torch")
    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = object  # type: ignore[attr-defined]
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoModelForCausalLM = object  # type: ignore[attr-defined]
    transformers_module.AutoTokenizer = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    with pytest.raises(ExportExecutionError, match="does not expose dtype"):
        merge_adapter(plan)


def test_failed_merge_cleans_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_merge_plan(BASE_MODEL, _adapter(tmp_path), tmp_path / "merged")

    class FailingAutoModel:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> object:
            del cls, name, kwargs
            raise RuntimeError("simulated model load failure")

    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = object()  # type: ignore[attr-defined]
    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = object  # type: ignore[attr-defined]
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoModelForCausalLM = FailingAutoModel  # type: ignore[attr-defined]
    transformers_module.AutoTokenizer = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    with pytest.raises(RuntimeError, match="simulated model load failure"):
        merge_adapter(plan)

    assert not list(tmp_path.glob(".merged-merge-*"))


def test_vllm_argv_uses_static_json_descriptor_and_loopback(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, rank=32)

    argv = build_vllm_argv(
        BASE_MODEL,
        adapter,
        model_revision=MODEL_REVISION,
        served_model_name="tickettune",
    )

    assert argv[:3] == ["vllm", "serve", BASE_MODEL]
    assert argv[argv.index("--revision") + 1] == MODEL_REVISION
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--enable-lora" in argv
    assert argv[argv.index("--generation-config") + 1] == "vllm"
    descriptor = json.loads(argv[argv.index("--lora-modules") + 1])
    assert descriptor == {
        "base_model_name": BASE_MODEL,
        "name": "tickettune",
        "path": str(adapter.resolve()),
    }
    assert all("=" not in item for item in argv if item.startswith("tickettune"))


def test_vllm_argv_rejects_public_bind_without_explicit_opt_in(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ExportValidationError, match="loopback"):
        build_vllm_argv(BASE_MODEL, adapter, model_revision=MODEL_REVISION, host="0.0.0.0")  # noqa: S104


def test_vllm_argv_rejects_rank_above_server_limit(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, rank=64)

    with pytest.raises(ExportValidationError, match="max_lora_rank"):
        build_vllm_argv(BASE_MODEL, adapter, model_revision=MODEL_REVISION, max_lora_rank=32)


def test_vllm_remote_bind_requires_and_honors_opt_in(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ExportValidationError, match="loopback"):
        build_vllm_argv(BASE_MODEL, adapter, model_revision=MODEL_REVISION, host="example.com")

    argv = build_vllm_argv(
        BASE_MODEL,
        adapter,
        model_revision=MODEL_REVISION,
        host="example.com",
        allow_remote=True,
    )
    assert argv[argv.index("--host") + 1] == "example.com"


def test_vllm_accepts_localhost_name(tmp_path: Path) -> None:
    argv = build_vllm_argv(
        BASE_MODEL,
        _adapter(tmp_path),
        model_revision=MODEL_REVISION,
        host="localhost",
    )

    assert argv[argv.index("--host") + 1] == "localhost"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"port": 0}, "port"),
        ({"port": True}, "port"),
        ({"max_lora_rank": 0}, "max_lora_rank"),
        ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 1.1}, "gpu_memory_utilization"),
        ({"dtype": "int8"}, "dtype"),
        ({"max_model_len": 64}, "max_model_len"),
        ({"max_model_len": True}, "max_model_len"),
        ({"served_model_name": "bad name"}, "served model name"),
    ],
)
def test_vllm_rejects_invalid_server_settings(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ExportValidationError, match=message):
        build_vllm_argv(
            BASE_MODEL,
            _adapter(tmp_path),
            model_revision=MODEL_REVISION,
            **kwargs,  # type: ignore[arg-type]
        )


def test_vllm_rejects_adapter_for_different_base(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, base_model="Qwen/Qwen2.5-1.5B-Instruct")

    with pytest.raises(ExportValidationError, match="base model mismatch"):
        build_vllm_argv(BASE_MODEL, adapter, model_revision=MODEL_REVISION)


def test_vllm_remote_model_requires_immutable_revision(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    with pytest.raises(ExportValidationError, match="40-character commit SHA"):
        build_vllm_plan(BASE_MODEL, adapter)
    with pytest.raises(ExportValidationError, match="40-character commit SHA"):
        build_vllm_plan(BASE_MODEL, adapter, model_revision="main", allow_download=True)


def test_vllm_plan_is_offline_by_default_and_download_is_explicit(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    offline = build_vllm_plan(BASE_MODEL, adapter, model_revision=MODEL_REVISION)
    online = build_vllm_plan(
        BASE_MODEL,
        adapter,
        model_revision=MODEL_REVISION,
        allow_download=True,
    )

    assert offline.allow_download is False
    assert dict(offline.environment_overrides) == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert offline.argv[offline.argv.index("--revision") + 1] == MODEL_REVISION
    assert "revision_not_embedded_in_adapter" in offline.provenance_boundary
    assert online.allow_download is True
    assert dict(online.environment_overrides) == {
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
    }


def test_vllm_plan_validates_declared_adapter_revision(tmp_path: Path) -> None:
    matching = _adapter(tmp_path, model_revision=MODEL_REVISION)
    plan = build_vllm_plan(BASE_MODEL, matching, model_revision=MODEL_REVISION)
    assert plan.adapter_revision == MODEL_REVISION

    other_root = tmp_path / "other"
    other_root.mkdir()
    mismatching = _adapter(other_root, model_revision="b" * 40)
    with pytest.raises(ExportValidationError, match="revision mismatch"):
        build_vllm_plan(BASE_MODEL, mismatching, model_revision=MODEL_REVISION)


def test_ollama_plan_is_merged_hf_to_pinned_llama_cpp_to_gguf(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)

    plan = build_ollama_export_plan(merged, tmp_path / "ollama")

    assert plan.llama_cpp_revision == LLAMA_CPP_REVISION
    assert len(LLAMA_CPP_REVISION) == 40
    assert plan.source_kind == "merged_hf"
    assert plan.merge_provenance_path.endswith("tickettune-merge-provenance.json")
    assert len(plan.merge_provenance_sha256) == 64
    assert dict(plan.merged_artifact_sha256)
    assert plan.direct_adapter_supported is False
    assert Path(plan.conversion_argv[0]).resolve() == Path(sys.executable).resolve()
    assert plan.conversion_argv[-2:] == ("--outtype", "f16")
    assert plan.quantize_argv[-1] == "Q4_K_M"
    assert plan.f16_gguf_path in plan.checksum_argv
    assert plan.gguf_path in plan.checksum_argv
    assert plan.modelfile_path in plan.checksum_argv
    assert plan.ollama_create_argv[:2] == ("ollama", "create")
    assert "ADAPTER" not in plan.modelfile
    assert plan.to_dict()["commands"][0] == plan.clone_argv


def test_ollama_plan_rejects_source_output_overlap(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)

    for output in (merged, merged / "ollama", tmp_path):
        with pytest.raises(ExportValidationError, match="must be isolated"):
            build_ollama_export_plan(merged, output)


def test_ollama_plan_rejects_direct_qwen_adapter(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)
    adapter = _adapter(tmp_path)

    with pytest.raises(ExportValidationError, match="Direct Qwen adapter"):
        build_ollama_export_plan(
            merged,
            tmp_path / "ollama",
            adapter_path=adapter,
            model_family="qwen2",
        )


def test_ollama_f16_plan_skips_quantizer_build(tmp_path: Path) -> None:
    plan = build_ollama_export_plan(
        _merged_model(tmp_path),
        tmp_path / "ollama",
        quantization="F16",
    )

    assert plan.configure_argv == ()
    assert plan.build_argv == ()
    assert plan.quantize_argv == ()
    assert plan.gguf_path.endswith("tickettune-f16.gguf")
    assert all(command for command in plan.commands)


def test_ollama_plan_rejects_direct_non_qwen_adapter(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path, model_type="llama")
    adapter = _adapter(tmp_path)

    with pytest.raises(ExportValidationError, match="outside TicketTune"):
        build_ollama_export_plan(
            merged,
            tmp_path / "ollama",
            adapter_path=adapter,
            model_family="llama",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_family": "llama"}, "does not match"),
        ({"quantization": "Q2_K"}, "quantization"),
        ({"llama_cpp_revision": "main"}, "40-character commit SHA"),
        ({"model_name": "bad name"}, "Ollama model name"),
    ],
)
def test_ollama_plan_rejects_invalid_options(
    tmp_path: Path, kwargs: dict[str, str], message: str
) -> None:
    with pytest.raises(ExportValidationError, match=message):
        build_ollama_export_plan(
            _merged_model(tmp_path),
            tmp_path / "ollama",
            **kwargs,  # type: ignore[arg-type]
        )


def test_ollama_plan_rejects_invalid_merged_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ExportValidationError, match="does not exist"):
        build_ollama_export_plan(tmp_path / "missing", tmp_path / "ollama")

    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(ExportValidationError, match=r"missing config\.json"):
        build_ollama_export_plan(merged, tmp_path / "ollama")

    (merged / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ExportValidationError, match="safetensors"):
        build_ollama_export_plan(merged, tmp_path / "ollama")

    (merged / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ExportValidationError, match="model_type"):
        build_ollama_export_plan(merged, tmp_path / "ollama")


def test_ollama_plan_requires_verified_merge_provenance(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)
    provenance = merged / "tickettune-merge-provenance.json"
    provenance.unlink()
    with pytest.raises(ExportValidationError, match="requires a regular"):
        build_ollama_export_plan(merged, tmp_path / "ollama")

    merged = _merged_model(tmp_path / "tampered")
    (merged / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ExportValidationError, match="hash mismatch"):
        build_ollama_export_plan(merged, tmp_path / "ollama-tampered")


def test_ollama_plan_rejects_symlinked_merged_root(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)
    linked = tmp_path / "linked-merged"
    linked.symlink_to(merged, target_is_directory=True)

    with pytest.raises(ExportValidationError, match="directory cannot be a symlink"):
        build_ollama_export_plan(linked, tmp_path / "ollama")


def test_ollama_plan_rejects_merge_without_safe_non_quantized_invariants(
    tmp_path: Path,
) -> None:
    merged = _merged_model(tmp_path)
    provenance = merged / "tickettune-merge-provenance.json"
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["safe_merge"] = False
    provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="non-quantized, safe merge"):
        build_ollama_export_plan(merged, tmp_path / "ollama")


def test_ollama_plan_rejects_untracked_merged_file(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)
    (merged / "untracked.txt").write_text("not in provenance", encoding="utf-8")

    with pytest.raises(ExportValidationError, match="untracked"):
        build_ollama_export_plan(merged, tmp_path / "ollama")


def test_ollama_plan_rejects_untracked_nested_merged_file(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)
    untracked = merged / "untracked-dir"
    untracked.mkdir()
    (untracked / "payload").write_bytes(b"not in provenance")

    with pytest.raises(ExportValidationError, match="untracked"):
        build_ollama_export_plan(merged, tmp_path / "ollama")


def test_ollama_plan_rejects_nested_merged_symlink(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)
    nested = merged / "nested"
    nested.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    (nested / "payload.bin").symlink_to(target)

    with pytest.raises(ExportValidationError, match="symbolic links"):
        build_ollama_export_plan(merged, tmp_path / "ollama")


def test_render_ollama_modelfile_rejects_directive_injection() -> None:
    with pytest.raises(ExportValidationError, match="line breaks"):
        render_ollama_modelfile(Path("model.gguf\nADAPTER ./untrusted"))


def test_render_ollama_modelfile_is_deterministic() -> None:
    rendered = render_ollama_modelfile(Path("tickettune-q4_k_m.gguf"))

    assert rendered == (
        "FROM ./tickettune-q4_k_m.gguf\n"
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 2048\n"
        f'SYSTEM """{SYSTEM_PROMPT}"""\n'
    )


def test_chat_example_uses_canonical_system_prompt() -> None:
    namespace = runpy.run_path(str(PROJECT_ROOT / "deploy/examples/chat.py"))
    request_payload = namespace["_request_payload"]
    assert callable(request_payload)
    payload = request_payload(model="tickettune", ticket="Example ticket")

    assert payload["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Example ticket"},
    ]


def test_chat_example_help_runs_with_project_interpreter() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(PROJECT_ROOT / "deploy/examples/chat.py"), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--base-url" in result.stdout


def test_chat_example_rejects_redirects_off_loopback() -> None:
    namespace = runpy.run_path(str(PROJECT_ROOT / "deploy/examples/chat.py"))
    handler_type = namespace["_SafeRedirectHandler"]
    handler = handler_type(allow_remote=False)
    request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions")

    with pytest.raises(ValueError, match="remote requests require"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://example.com/collect",
        )


def test_chat_example_validates_model_output_before_printing() -> None:
    namespace = runpy.run_path(str(PROJECT_ROOT / "deploy/examples/chat.py"))
    validate = namespace["_validated_response"]
    valid_output = {
        "category": "billing",
        "priority": "high",
        "sentiment": "frustrated",
        "response": "I can help investigate the duplicate charge.",
        "next_action": "investigate_duplicate_charge",
    }

    rendered = validate({"choices": [{"message": {"content": json.dumps(valid_output)}}]})
    assert json.loads(rendered) == valid_output
    with pytest.raises(ValueError, match="not valid JSON"):
        validate({"choices": [{"message": {"content": "untrusted raw text"}}]})
    duplicate = json.dumps(valid_output).replace(
        '"category": "billing"',
        '"category": "shadowed", "category": "billing"',
        1,
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        validate({"choices": [{"message": {"content": duplicate}}]})


@pytest.mark.parametrize(
    ("path", "kwargs", "message"),
    [
        (Path("model.bin"), {}, "end in .gguf"),
        (Path("bad model.gguf"), {}, "whitespace"),
        (Path("model.gguf"), {"system_prompt": 'bad """ prompt'}, "triple quotes"),
        (Path("model.gguf"), {"context_length": 0}, "context_length"),
        (Path("model.gguf"), {"temperature": 3}, "temperature"),
    ],
)
def test_render_ollama_modelfile_rejects_invalid_values(
    path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ExportValidationError, match=message):
        render_ollama_modelfile(path, **kwargs)  # type: ignore[arg-type]


def test_materialize_ollama_plan_is_idempotent_and_refuses_conflict(tmp_path: Path) -> None:
    plan = build_ollama_export_plan(_merged_model(tmp_path), tmp_path / "ollama")

    first = materialize_ollama_plan(plan)
    second = materialize_ollama_plan(plan)
    assert first == second
    assert first.read_text(encoding="utf-8") == plan.modelfile

    first.write_text("FROM ./different.gguf\n", encoding="utf-8")
    with pytest.raises(ExportExecutionError, match="Refusing to overwrite"):
        materialize_ollama_plan(plan)


def test_materialize_ollama_plan_refuses_nonempty_output(tmp_path: Path) -> None:
    plan = build_ollama_export_plan(_merged_model(tmp_path), tmp_path / "ollama")
    output = Path(plan.output_dir)
    output.mkdir()
    (output / "unexpected.txt").write_text("untrusted", encoding="utf-8")

    with pytest.raises(ExportExecutionError, match="non-empty Ollama output"):
        materialize_ollama_plan(plan)


def test_local_smoke_plan_cannot_be_relabelled_as_release_evidence(tmp_path: Path) -> None:
    plan = build_ollama_export_plan(_merged_model(tmp_path), tmp_path / "ollama")
    relabelled = replace(plan, lineage_boundary="qualified_release_lineage")

    with pytest.raises(ExportExecutionError, match="export-plan snapshot"):
        materialize_ollama_plan(relabelled)


def test_ollama_execution_revalidates_merged_source(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)
    plan = build_ollama_export_plan(merged, tmp_path / "ollama")
    (merged / "model.safetensors").write_bytes(b"changed-after-plan")

    with pytest.raises(ExportExecutionError, match="export-plan snapshot"):
        materialize_ollama_plan(plan)


def test_ollama_provenance_revalidates_source_after_conversion(tmp_path: Path) -> None:
    merged = _merged_model(tmp_path)
    plan = build_ollama_export_plan(merged, tmp_path / "ollama")
    materialize_ollama_plan(plan)
    Path(plan.f16_gguf_path).write_bytes(b"f16")
    Path(plan.gguf_path).write_bytes(b"quantized")
    (merged / "model.safetensors").write_bytes(b"changed-during-conversion")

    with pytest.raises(ExportExecutionError, match="export-plan snapshot"):
        write_ollama_export_provenance(plan)


def test_ollama_export_writes_immutable_durable_provenance(tmp_path: Path) -> None:
    plan = build_ollama_export_plan(_merged_model(tmp_path), tmp_path / "ollama")
    materialize_ollama_plan(plan)
    Path(plan.f16_gguf_path).write_bytes(b"f16")
    Path(plan.gguf_path).write_bytes(b"quantized")

    first = write_ollama_export_provenance(plan)
    second = write_ollama_export_provenance(plan)
    payload = json.loads(Path(first.provenance_path).read_text(encoding="utf-8"))

    assert first == second
    assert payload["merge_provenance_sha256"] == plan.merge_provenance_sha256
    assert payload["lineage_boundary"] == plan.lineage_boundary
    assert set(payload["artifact_sha256"]) == {
        Path(plan.f16_gguf_path).name,
        Path(plan.gguf_path).name,
        "Modelfile",
    }
    Path(plan.gguf_path).write_bytes(b"changed")
    with pytest.raises(ExportExecutionError, match="Refusing to overwrite"):
        write_ollama_export_provenance(plan)


@pytest.mark.parametrize("argv", [[], ["python3", "bad\x00argument"]])
def test_run_argv_rejects_empty_or_nul_values(argv: list[str]) -> None:
    with pytest.raises(ExportValidationError, match="argv"):
        run_argv(argv)


def test_run_argv_invokes_process_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], *, cwd: Path | None, check: bool, shell: bool) -> None:
        captured.update(argv=argv, cwd=cwd, check=check, shell=shell)

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)

    run_argv(["program", "literal argument"], cwd=tmp_path)

    assert captured == {
        "argv": ["program", "literal argument"],
        "cwd": tmp_path,
        "check": True,
        "shell": False,
    }


def test_run_argv_applies_offline_overrides_and_routes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(
        argv: list[str],
        *,
        cwd: Path | None,
        check: bool,
        shell: bool,
        env: dict[str, str],
        stdout: object,
        stderr: object,
    ) -> None:
        captured.update(
            argv=argv,
            cwd=cwd,
            check=check,
            shell=shell,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    run_argv(
        ["program", "argument"],
        environment_overrides={"HF_HUB_OFFLINE": "1"},
        route_output_to_stderr=True,
    )

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert captured["stdout"] is sys.stderr
    assert captured["stderr"] is sys.stderr
    assert captured["shell"] is False


def test_run_argv_wraps_failed_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise export_module.subprocess.CalledProcessError(17, ["program"])

    monkeypatch.setattr(export_module.subprocess, "run", fail)

    with pytest.raises(ExportExecutionError, match="status 17"):
        run_argv(["program"])


def test_vllm_compose_pins_image_and_host_loopback() -> None:
    compose = (PROJECT_ROOT / "deploy/vllm/compose.yaml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(compose)
    service = parsed["services"]["vllm"]

    assert service["image"] == (
        "vllm/vllm-openai:v0.24.0@"
        "sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8"
    )
    assert service["platform"] == "linux/amd64"
    assert service["healthcheck"]["test"][0] == "CMD"
    assert "127.0.0.1:${VLLM_PORT:-8000}:8000" in compose
    assert "CMD-SHELL" not in compose
    assert '"name":"${SERVED_MODEL_NAME:-tickettune}"' in compose
    assert "--revision" in service["command"]
    assert any("MODEL_REVISION" in item for item in service["command"])
    assert service["environment"]["HF_HUB_OFFLINE"] == "${HF_HUB_OFFLINE:-1}"
    assert service["environment"]["TRANSFORMERS_OFFLINE"] == "${TRANSFORMERS_OFFLINE:-1}"
    assert service["volumes"][0]["source"] == (
        "${ADAPTER_PATH:?Set ADAPTER_PATH to an immutable runs/<run-id>/adapter directory}"
    )
    assert "artifacts/adapter" not in compose

    env_example = (PROJECT_ROOT / "deploy/vllm/.env.example").read_text(encoding="utf-8")
    assert f"MODEL_REVISION={MODEL_REVISION}" not in env_example
    assert "MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28" in env_example
    assert "HF_HUB_OFFLINE=1" in env_example


def test_ollama_template_is_gguf_only() -> None:
    modelfile = (PROJECT_ROOT / "deploy/ollama/Modelfile.template").read_text(encoding="utf-8")

    assert "FROM ./tickettune-q4_k_m.gguf" in modelfile
    assert "\nADAPTER " not in modelfile
