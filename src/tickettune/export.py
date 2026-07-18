"""Safe, deterministic export and local-serving plans.

The pure builders in this module are intentionally usable without importing
PyTorch, Transformers, PEFT, vLLM, or Ollama. Heavy libraries are loaded only
by :func:`merge_adapter`, and every external command is represented as an argv
sequence so callers never need a shell.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NoReturn, cast

from pydantic import ValidationError

from tickettune.prompts import SYSTEM_PROMPT
from tickettune.run_manifest import RunManifest, canonical_json_bytes
from tickettune.strict_json import DuplicateJSONKeyError, StrictJSONError, loads_strict

# Security: subprocess is confined to run_argv, which validates argv and disables the shell.

LLAMA_CPP_REVISION = "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3"
"""Pinned llama.cpp ``b9637`` commit used by the GGUF conversion plan."""

DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
"""Backward-compatible alias for TicketTune's canonical task contract."""

_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9._-]+)?$")
_SUPPORTED_MERGE_DTYPES = frozenset({"bfloat16", "float16", "float32"})
_SUPPORTED_QUANTIZATIONS = frozenset({"F16", "Q4_K_M", "Q5_K_M", "Q8_0"})
_REQUIRED_QUALIFICATION_HASHES = frozenset(
    {"qualification_review_manifest", "qualification_report"}
)

QUALIFIED_RELEASE_LINEAGE = "qualified_release_lineage"
UNQUALIFIED_LOCAL_SMOKE_LINEAGE = "unqualified_local_smoke_override_not_release_evidence"
LineageBoundary = Literal[
    "qualified_release_lineage",
    "unqualified_local_smoke_override_not_release_evidence",
]
_LINEAGE_BOUNDARIES = frozenset({QUALIFIED_RELEASE_LINEAGE, UNQUALIFIED_LOCAL_SMOKE_LINEAGE})


class ExportValidationError(ValueError):
    """Raised when an export request is unsafe or internally inconsistent."""


class ExportExecutionError(RuntimeError):
    """Raised when a requested export cannot be executed safely."""


class _SerializablePlan:
    """Small compatibility layer for JSON-oriented CLI rendering."""

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], asdict(self))  # type: ignore[call-overload]

    def model_dump(self, *, mode: Literal["python", "json"] = "python") -> dict[str, Any]:
        del mode
        return self.to_dict()


@dataclass(frozen=True, slots=True)
class AdapterMetadata(_SerializablePlan):
    base_model_name_or_path: str
    model_revision: str | None
    rank: int
    config_path: str
    weight_paths: tuple[str, ...]
    config_sha256: str
    weight_sha256: tuple[str, ...]
    training_manifest_path: str | None
    training_manifest_sha256: str | None
    training_config_sha256: str | None
    training_dataset_sha256: tuple[tuple[str, str], ...]
    qualification_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class MergePlan(_SerializablePlan):
    base_model: str
    model_revision: str | None
    adapter_path: str
    adapter_base_model: str
    adapter_revision: str | None
    adapter_rank: int
    adapter_config_sha256: str
    adapter_weight_files: tuple[str, ...]
    adapter_weight_sha256: tuple[str, ...]
    training_manifest_path: str | None
    training_manifest_sha256: str | None
    training_config_sha256: str | None
    training_dataset_sha256: tuple[tuple[str, str], ...]
    qualification_sha256: tuple[tuple[str, str], ...]
    lineage_boundary: LineageBoundary
    output_dir: str
    dtype: str
    allow_download: bool = False
    safe_merge: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    safe_serialization: bool = True
    parity_checks: tuple[str, ...] = (
        "Run the fixed evaluation prompts through the base model plus adapter.",
        "Run the same prompts and decoding settings through the merged model.",
        "Require schema-valid JSON from both runtimes.",
        "Require exact category, priority, sentiment, next_action, and routing parity.",
        "Treat response-text exactness as diagnostic rather than a release gate.",
    )


@dataclass(frozen=True, slots=True)
class MergeResult(_SerializablePlan):
    output_dir: str
    provenance_path: str
    artifact_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class VerifiedMergedModel(_SerializablePlan):
    """Read-only, byte-verified identity of a TicketTune safe merge."""

    merged_model: str
    model_family: str
    provenance_path: str
    provenance_sha256: str
    merge_dtype: str | None
    base_model: str
    model_revision: str | None
    adapter_base_model: str
    adapter_revision: str | None
    adapter_config_sha256: str
    adapter_weight_files: tuple[str, ...]
    adapter_weight_sha256: tuple[str, ...]
    training_manifest_sha256: str | None
    training_config_sha256: str | None
    training_dataset_sha256: tuple[tuple[str, str], ...]
    qualification_sha256: tuple[tuple[str, str], ...]
    lineage_boundary: LineageBoundary | None
    artifact_sha256: tuple[tuple[str, str], ...]
    safe_merge: Literal[True] = True
    safe_serialization: Literal[True] = True


@dataclass(frozen=True, slots=True)
class OllamaExportPlan(_SerializablePlan):
    source_kind: Literal["merged_hf"]
    merged_model: str
    merge_provenance_path: str
    merge_provenance_sha256: str
    merged_artifact_sha256: tuple[tuple[str, str], ...]
    training_manifest_sha256: str | None
    training_config_sha256: str | None
    training_dataset_sha256: tuple[tuple[str, str], ...]
    qualification_sha256: tuple[tuple[str, str], ...]
    lineage_boundary: LineageBoundary
    output_dir: str
    model_name: str
    model_family: str
    llama_cpp_revision: str
    direct_adapter_supported: bool
    clone_argv: tuple[str, ...]
    checkout_argv: tuple[str, ...]
    configure_argv: tuple[str, ...]
    build_argv: tuple[str, ...]
    conversion_argv: tuple[str, ...]
    quantize_argv: tuple[str, ...]
    checksum_argv: tuple[str, ...]
    f16_gguf_path: str
    gguf_path: str
    modelfile_path: str
    export_provenance_path: str
    modelfile: str
    ollama_create_argv: tuple[str, ...]
    ollama_run_argv: tuple[str, ...]

    @property
    def commands(self) -> tuple[tuple[str, ...], ...]:
        """Return the ordered external build commands as immutable argv arrays."""

        return tuple(
            command
            for command in (
                self.clone_argv,
                self.checkout_argv,
                self.configure_argv,
                self.build_argv,
                self.conversion_argv,
                self.quantize_argv,
                self.checksum_argv,
                self.ollama_create_argv,
            )
            if command
        )

    def to_dict(self) -> dict[str, Any]:
        value = _SerializablePlan.to_dict(self)
        value["commands"] = self.commands
        return value


@dataclass(frozen=True, slots=True)
class OllamaExportResult(_SerializablePlan):
    """Durable provenance written after conversion outputs exist."""

    provenance_path: str
    provenance_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class VllmServePlan(_SerializablePlan):
    """Validated, offline-aware vLLM foreground execution plan."""

    base_model: str
    model_revision: str | None
    adapter_path: str
    adapter_base_model: str
    adapter_revision: str | None
    adapter_config_sha256: str
    adapter_weight_sha256: tuple[str, ...]
    training_manifest_path: str | None
    training_manifest_sha256: str | None
    training_config_sha256: str | None
    training_dataset_sha256: tuple[tuple[str, str], ...]
    qualification_sha256: tuple[tuple[str, str], ...]
    lineage_boundary: LineageBoundary
    allow_download: bool
    environment_overrides: tuple[tuple[str, str], ...]
    argv: tuple[str, ...]
    runtime: Literal["native_foreground"] = "native_foreground"
    provenance_boundary: str = "adapter_config_and_safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_regular_file_snapshot(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExportValidationError(f"Invalid {label} at {path}: not a regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExportValidationError(f"Invalid {label} at {path}: not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        current = os.lstat(path)
    except OSError as exc:
        raise ExportValidationError(f"Invalid {label} at {path}: file changed") from exc
    finally:
        os.close(descriptor)
    if _file_stat_identity(before) != _file_stat_identity(after) or (
        current.st_dev,
        current.st_ino,
    ) != (before.st_dev, before.st_ino):
        raise ExportValidationError(f"Invalid {label} at {path}: file changed")
    return payload


def _decode_json_object(payload: bytes, path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = loads_strict(payload.decode("utf-8"))
    except DuplicateJSONKeyError as exc:
        raise ExportValidationError(
            f"Invalid {label} at {path}: duplicate JSON object key: {exc.key!r}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError) as exc:
        raise ExportValidationError(f"Invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportValidationError(f"Invalid {label} at {path}: expected a JSON object")
    return value


def _read_json_snapshot(path: Path, *, label: str) -> tuple[dict[str, Any], bytes, str]:
    payload = _read_regular_file_snapshot(path, label=label)
    return (
        _decode_json_object(payload, path, label=label),
        payload,
        hashlib.sha256(payload).hexdigest(),
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value, _payload, _sha256_digest = _read_json_snapshot(path, label=label)
    return value


def _safe_text(value: str, *, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ExportValidationError(f"{label} cannot be empty")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ExportValidationError(f"{label} cannot contain NUL bytes or line breaks")
    return cleaned


def _normalized_base(value: str) -> str:
    cleaned = _safe_text(value, label="base model")
    candidate = Path(cleaned).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return cleaned.rstrip("/")


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""

    left = first.expanduser().resolve()
    right = second.expanduser().resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _raise_artifact_tree_error(message: str, *, execution: bool) -> NoReturn:
    if execution:
        raise ExportExecutionError(message)
    raise ExportValidationError(message)


def _scan_regular_artifact_tree(
    root: Path,
    *,
    excluded: frozenset[str] = frozenset(),
    execution: bool = False,
) -> tuple[tuple[tuple[str, Path], ...], tuple[str, ...]]:
    """Inventory every regular file and directory without following symlinks."""

    files: list[tuple[str, Path]] = []
    directories: list[str] = []
    entries = sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    for entry in entries:
        relative = entry.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if entry.is_symlink():
            _raise_artifact_tree_error(
                f"Merged model artifacts cannot be symbolic links: {relative}",
                execution=execution,
            )
        if entry.is_dir():
            directories.append(relative)
            continue
        if not entry.is_file():
            _raise_artifact_tree_error(
                f"Merged model artifacts must be regular files or directories: {relative}",
                execution=execution,
            )
        files.append((relative, entry))
    return tuple(files), tuple(directories)


def _artifact_parent_directories(names: set[str]) -> set[str]:
    parents: set[str] = set()
    for name in names:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            parents.add(parent.as_posix())
            parent = parent.parent
    return parents


def _normalized_artifact_name(value: str) -> str | None:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return None
    return path.as_posix()


def _adapter_training_lineage(
    adapter: Path,
    *,
    base_model: str,
    model_revision: str | None,
    config_sha256: str,
    weight_sha256: tuple[tuple[str, str], ...],
) -> tuple[
    str | None,
    str | None,
    str | None,
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    manifest_path = adapter.parent / "manifest.json"
    if manifest_path.is_symlink():
        raise ExportValidationError(
            f"Training manifest must be a regular non-symlink file: {manifest_path}"
        )
    if not manifest_path.exists():
        return None, None, None, (), ()
    if not manifest_path.is_file():
        raise ExportValidationError(
            f"Training manifest must be a regular non-symlink file: {manifest_path}"
        )
    decoded, _payload, manifest_sha256 = _read_json_snapshot(
        manifest_path,
        label="training manifest",
    )
    try:
        manifest = RunManifest.model_validate(decoded)
    except ValidationError as exc:
        raise ExportValidationError(f"Invalid training manifest at {manifest_path}") from exc
    if manifest.status != "completed" or manifest.run_id != adapter.parent.name:
        raise ExportValidationError("Adapter is not bound to a completed sibling training run")
    if _normalized_base(manifest.model_name_or_path) != _normalized_base(base_model):
        raise ExportValidationError("Training manifest base model does not match adapter config")
    if model_revision is not None and manifest.model_revision != model_revision:
        raise ExportValidationError(
            "Training manifest model revision does not match configured model revision"
        )
    if hashlib.sha256(canonical_json_bytes(manifest.config)).hexdigest() != manifest.config_sha256:
        raise ExportValidationError("Training manifest config hash is invalid")
    dataset_hashes = tuple(sorted(manifest.dataset_sha256.items()))
    required_dataset_hashes = {"source", "manifest", "train", "validation"}
    if not required_dataset_hashes.issubset(dict(dataset_hashes)) or any(
        re.fullmatch(r"[0-9a-f]{64}", digest) is None for _name, digest in dataset_hashes
    ):
        raise ExportValidationError(
            "Training manifest lacks valid source, prepared-manifest, train, or validation hashes"
        )
    artifact_hashes = {artifact.path: artifact.sha256 for artifact in manifest.artifacts}
    expected = {
        f"{adapter.name}/adapter_config.json": config_sha256,
        **{f"{adapter.name}/{name}": digest for name, digest in weight_sha256},
    }
    if any(artifact_hashes.get(name) != digest for name, digest in expected.items()):
        raise ExportValidationError(
            "Training manifest adapter artifacts do not match current adapter bytes"
        )
    qualification_hashes = tuple(
        (name, digest) for name, digest in dataset_hashes if name.startswith("qualification_")
    )
    return (
        str(manifest_path.resolve()),
        manifest_sha256,
        manifest.config_sha256,
        dataset_hashes,
        qualification_hashes,
    )


def _adapter_metadata(
    adapter_path: Path,
    *,
    expected_model_revision: str | None = None,
) -> AdapterMetadata:
    expanded = adapter_path.expanduser()
    if expanded.is_symlink():
        raise ExportValidationError("Adapter directory cannot be a symlink")
    adapter = expanded.resolve()
    if not adapter.is_dir():
        raise ExportValidationError(f"Adapter directory does not exist: {adapter}")

    config_path = adapter / "adapter_config.json"
    if config_path.is_symlink():
        raise ExportValidationError(
            f"PEFT adapter config must be a regular non-symlink file: {config_path}"
        )
    if not config_path.is_file():
        raise ExportValidationError(f"Missing PEFT adapter config: {config_path}")
    config = _read_json_object(config_path, label="PEFT adapter config")

    raw_base = config.get("base_model_name_or_path")
    if not isinstance(raw_base, str):
        raise ExportValidationError(
            "PEFT adapter config must contain string base_model_name_or_path"
        )
    raw_rank = config.get("r")
    if isinstance(raw_rank, bool) or not isinstance(raw_rank, int) or raw_rank < 1:
        raise ExportValidationError("PEFT adapter config must contain a positive integer r")

    raw_revision = config.get("revision")
    if raw_revision is not None and not isinstance(raw_revision, str):
        raise ExportValidationError("PEFT adapter config revision must be a string or null")
    revision = (
        _safe_text(raw_revision, label="adapter model revision")
        if isinstance(raw_revision, str)
        else None
    )

    weight_candidates = tuple(sorted(adapter.glob("*.safetensors")))
    if any(path.is_symlink() for path in weight_candidates):
        raise ExportValidationError("PEFT adapter safetensors weights must not be symlinks")
    weights = tuple(path for path in weight_candidates if path.is_file())
    if len(weights) != len(weight_candidates):
        raise ExportValidationError("PEFT adapter safetensors weights must be regular files")
    if not weights:
        raise ExportValidationError(
            f"Missing safetensors adapter weights in {adapter}; pickle-based weights are rejected"
        )
    config_sha256 = _sha256(config_path)
    named_weight_sha256 = tuple((path.name, _sha256(path)) for path in weights)
    (
        training_manifest_path,
        training_manifest_sha256,
        training_config_sha256,
        training_dataset_sha256,
        qualification_sha256,
    ) = _adapter_training_lineage(
        adapter,
        base_model=raw_base,
        model_revision=(
            expected_model_revision if expected_model_revision is not None else revision
        ),
        config_sha256=config_sha256,
        weight_sha256=named_weight_sha256,
    )
    return AdapterMetadata(
        base_model_name_or_path=raw_base,
        model_revision=revision,
        rank=raw_rank,
        config_path=str(config_path),
        weight_paths=tuple(str(path) for path in weights),
        config_sha256=config_sha256,
        weight_sha256=tuple(digest for _name, digest in named_weight_sha256),
        training_manifest_path=training_manifest_path,
        training_manifest_sha256=training_manifest_sha256,
        training_config_sha256=training_config_sha256,
        training_dataset_sha256=training_dataset_sha256,
        qualification_sha256=qualification_sha256,
    )


def _adapter_lineage_boundary(
    metadata: AdapterMetadata,
    *,
    configured_model_revision: str | None,
    allow_unqualified_local_smoke: bool,
) -> LineageBoundary:
    """Classify verified adapter lineage or fail closed for release-capable use."""

    if not isinstance(allow_unqualified_local_smoke, bool):
        raise ExportValidationError("allow_unqualified_local_smoke must be a boolean")
    qualification_hashes = dict(metadata.qualification_sha256)
    has_training_lineage = all(
        value is not None
        for value in (
            metadata.training_manifest_path,
            metadata.training_manifest_sha256,
            metadata.training_config_sha256,
        )
    ) and {"source", "manifest", "train", "validation"}.issubset(
        dict(metadata.training_dataset_sha256)
    )
    if (
        configured_model_revision is not None
        and has_training_lineage
        and _REQUIRED_QUALIFICATION_HASHES.issubset(qualification_hashes)
    ):
        return cast(LineageBoundary, QUALIFIED_RELEASE_LINEAGE)
    if allow_unqualified_local_smoke:
        return cast(LineageBoundary, UNQUALIFIED_LOCAL_SMOKE_LINEAGE)
    if configured_model_revision is None:
        raise ExportValidationError(
            "Release-capable export and serving require an exact configured model revision"
        )
    if not has_training_lineage:
        raise ExportValidationError(
            "Release-capable export and serving require a completed sibling training manifest"
        )
    raise ExportValidationError(
        "Release-capable export and serving require qualification lineage for the "
        "review manifest and qualification report"
    )


def _require_matching_base(base_model: str, metadata: AdapterMetadata) -> None:
    expected = _normalized_base(base_model)
    actual = _normalized_base(metadata.base_model_name_or_path)
    if expected != actual:
        raise ExportValidationError(
            "PEFT adapter/base model mismatch: "
            f"requested {expected!r}, adapter declares {actual!r}. "
            "Merging or serving against another base can silently corrupt outputs."
        )


def _require_matching_revision(model_revision: str | None, metadata: AdapterMetadata) -> None:
    """Reject a declared adapter revision that conflicts with the serving revision."""

    if metadata.model_revision is None:
        return
    if model_revision is None:
        raise ExportValidationError(
            "PEFT adapter declares a model revision but the serving plan does not"
        )
    if metadata.model_revision != model_revision:
        raise ExportValidationError(
            "PEFT adapter/model revision mismatch: "
            f"requested {model_revision!r}, adapter declares {metadata.model_revision!r}."
        )


def build_merge_plan(
    base_model: str,
    adapter_path: Path,
    output_dir: Path,
    *,
    dtype: str = "bfloat16",
    model_revision: str | None = None,
    allow_download: bool = False,
    allow_unqualified_local_smoke: bool = False,
) -> MergePlan:
    """Validate an adapter and describe a pristine, non-quantized PEFT merge.

    Release-capable plans require completed, revision-bound qualification
    lineage. ``allow_unqualified_local_smoke`` is an explicit non-release
    escape hatch for local fixtures and smoke checks only.
    """

    normalized_base = _safe_text(base_model, label="base model")
    if dtype not in _SUPPORTED_MERGE_DTYPES:
        allowed = ", ".join(sorted(_SUPPORTED_MERGE_DTYPES))
        raise ExportValidationError(f"Merge dtype must be one of: {allowed}")
    if model_revision is not None:
        model_revision = _safe_text(model_revision, label="model revision")
    if allow_download and (
        model_revision is None or re.fullmatch(r"[0-9a-f]{40}", model_revision) is None
    ):
        raise ExportValidationError(
            "allow_download requires model_revision to be a full 40-character commit SHA"
        )

    metadata = _adapter_metadata(
        adapter_path,
        expected_model_revision=model_revision,
    )
    _require_matching_base(normalized_base, metadata)
    _require_matching_revision(model_revision, metadata)
    lineage_boundary = _adapter_lineage_boundary(
        metadata,
        configured_model_revision=model_revision,
        allow_unqualified_local_smoke=allow_unqualified_local_smoke,
    )
    resolved_output = output_dir.expanduser().resolve()
    resolved_adapter = Path(metadata.config_path).parent
    if _paths_overlap(resolved_output, resolved_adapter):
        raise ExportValidationError(
            "Merged output directory cannot overwrite the adapter or overlap "
            "its directory tree in either direction"
        )
    base_candidate = Path(normalized_base).expanduser()
    if base_candidate.exists() and _paths_overlap(resolved_output, base_candidate):
        raise ExportValidationError(
            "Merged output directory cannot overlap the existing local base-model "
            "source tree in either direction"
        )

    return MergePlan(
        base_model=normalized_base,
        model_revision=model_revision,
        adapter_path=str(adapter_path.expanduser().resolve()),
        adapter_base_model=metadata.base_model_name_or_path,
        adapter_revision=metadata.model_revision,
        adapter_rank=metadata.rank,
        adapter_config_sha256=metadata.config_sha256,
        adapter_weight_files=tuple(Path(path).name for path in metadata.weight_paths),
        adapter_weight_sha256=metadata.weight_sha256,
        training_manifest_path=metadata.training_manifest_path,
        training_manifest_sha256=metadata.training_manifest_sha256,
        training_config_sha256=metadata.training_config_sha256,
        training_dataset_sha256=metadata.training_dataset_sha256,
        qualification_sha256=metadata.qualification_sha256,
        lineage_boundary=lineage_boundary,
        output_dir=str(resolved_output),
        dtype=dtype,
        allow_download=allow_download,
    )


def _require_planned_adapter_snapshot(plan: MergePlan, metadata: AdapterMetadata) -> None:
    """Reject adapter bytes or inventory that changed after planning."""

    observed = (
        metadata.config_sha256,
        tuple(Path(path).name for path in metadata.weight_paths),
        metadata.weight_sha256,
        metadata.training_manifest_sha256,
        metadata.training_config_sha256,
        metadata.training_dataset_sha256,
        metadata.qualification_sha256,
    )
    expected = (
        plan.adapter_config_sha256,
        plan.adapter_weight_files,
        plan.adapter_weight_sha256,
        plan.training_manifest_sha256,
        plan.training_config_sha256,
        plan.training_dataset_sha256,
        plan.qualification_sha256,
    )
    if observed != expected:
        raise ExportExecutionError(
            "Adapter bytes or file inventory changed after the merge plan was created"
        )


def merge_adapter(plan: MergePlan, *, dry_run: bool = False) -> MergePlan | MergeResult:
    """Merge a PEFT adapter into a freshly loaded, non-quantized base model.

    The destination must not already exist. The merge is built in a sibling
    temporary directory and renamed into place only after all artifacts and the
    provenance manifest have been written.
    """

    if dry_run:
        return plan
    if plan.load_in_4bit or plan.load_in_8bit or not plan.safe_merge:
        raise ExportExecutionError("Refusing a quantized or unsafe adapter merge plan")

    output_dir = Path(plan.output_dir).expanduser().resolve()
    adapter_dir = Path(plan.adapter_path).expanduser().resolve()
    if _paths_overlap(output_dir, adapter_dir):
        raise ExportExecutionError("Merged output directory cannot overlap the adapter source tree")
    base_candidate = Path(plan.base_model).expanduser()
    if base_candidate.exists() and _paths_overlap(output_dir, base_candidate):
        raise ExportExecutionError(
            "Merged output directory cannot overlap the existing local base-model source tree"
        )
    if output_dir.exists():
        raise ExportExecutionError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    metadata = _adapter_metadata(
        Path(plan.adapter_path),
        expected_model_revision=plan.model_revision,
    )
    _require_matching_base(plan.base_model, metadata)
    _require_matching_revision(plan.model_revision, metadata)
    observed_lineage_boundary = _adapter_lineage_boundary(
        metadata,
        configured_model_revision=plan.model_revision,
        allow_unqualified_local_smoke=(plan.lineage_boundary == UNQUALIFIED_LOCAL_SMOKE_LINEAGE),
    )
    if observed_lineage_boundary != plan.lineage_boundary:
        raise ExportExecutionError(
            "Adapter lineage boundary changed after the merge plan was created"
        )
    _require_planned_adapter_snapshot(plan, metadata)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ExportExecutionError(
            "Merging requires the training extra: install with `uv sync --extra train`."
        ) from exc

    torch_dtype = getattr(torch, plan.dtype, None)
    if torch_dtype is None:
        raise ExportExecutionError(f"PyTorch does not expose dtype {plan.dtype!r}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-merge-", dir=str(output_dir.parent))
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(
            plan.base_model,
            dtype=torch_dtype,
            local_files_only=not plan.allow_download,
            low_cpu_mem_usage=True,
            revision=plan.model_revision,
            trust_remote_code=plan.trust_remote_code,
        )
        adapted = PeftModel.from_pretrained(base, plan.adapter_path, is_trainable=False)
        merged = adapted.merge_and_unload(safe_merge=True)
        merged.save_pretrained(temporary, safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained(
            plan.base_model,
            local_files_only=not plan.allow_download,
            revision=plan.model_revision,
            trust_remote_code=plan.trust_remote_code,
        )
        tokenizer.save_pretrained(temporary)

        metadata_after_merge = _adapter_metadata(
            Path(plan.adapter_path),
            expected_model_revision=plan.model_revision,
        )
        if metadata_after_merge != metadata:
            raise ExportExecutionError(
                "Adapter bytes or metadata changed while the merge was in progress"
            )
        base_candidate = Path(plan.base_model).expanduser()
        if base_candidate.exists() and _paths_overlap(output_dir, base_candidate):
            raise ExportExecutionError(
                "Merged output directory began overlapping the local base-model source tree "
                "while the merge was in progress"
            )

        artifact_files, artifact_directories = _scan_regular_artifact_tree(
            temporary,
            execution=True,
        )
        artifact_hashes = tuple((relative, _sha256(path)) for relative, path in artifact_files)
        untracked_directories = sorted(
            set(artifact_directories)
            - _artifact_parent_directories({name for name, _digest in artifact_hashes})
        )
        if untracked_directories:
            raise ExportExecutionError(
                "Merged model produced directories with no inventoried artifacts: "
                f"{untracked_directories}"
            )
        provenance = {
            "schema_version": 1,
            "operation": "peft_safe_merge",
            "base_model": plan.base_model,
            "model_revision": plan.model_revision,
            "adapter_base_model": metadata.base_model_name_or_path,
            "adapter_revision": metadata.model_revision,
            "adapter_rank": metadata.rank,
            "adapter_config_sha256": metadata.config_sha256,
            "adapter_weight_files": [Path(path).name for path in metadata.weight_paths],
            "adapter_weight_sha256": list(metadata.weight_sha256),
            "training_manifest_sha256": metadata.training_manifest_sha256,
            "training_config_sha256": metadata.training_config_sha256,
            "training_dataset_sha256": dict(metadata.training_dataset_sha256),
            "qualification_sha256": dict(metadata.qualification_sha256),
            "lineage_boundary": plan.lineage_boundary,
            "dtype": plan.dtype,
            "allow_download": plan.allow_download,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "safe_merge": True,
            "trust_remote_code": False,
            "safe_serialization": True,
            "artifact_sha256": dict(artifact_hashes),
            "parity_checks": list(plan.parity_checks),
        }
        provenance_path = temporary / "tickettune-merge-provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return MergeResult(
        output_dir=str(output_dir),
        provenance_path=str(output_dir / "tickettune-merge-provenance.json"),
        artifact_sha256=artifact_hashes,
    )


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_served_model_name(value: str, *, label: str) -> str:
    cleaned = _safe_text(value, label=label)
    if not _SAFE_MODEL_NAME.fullmatch(cleaned):
        raise ExportValidationError(
            f"{label} may contain only letters, digits, '.', '_', '-', '/', and one optional tag"
        )
    return cleaned


def build_vllm_plan(
    base_model: str,
    adapter_path: Path,
    *,
    model_revision: str | None = None,
    allow_download: bool = False,
    allow_unqualified_local_smoke: bool = False,
    served_model_name: str = "tickettune",
    host: str = "127.0.0.1",
    port: int = 8000,
    max_lora_rank: int = 64,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    max_model_len: int = 2048,
    allow_remote: bool = False,
) -> VllmServePlan:
    """Build an immutable, offline-aware vLLM 0.24 static-LoRA plan.

    Static registration deliberately avoids enabling vLLM's runtime LoRA
    management endpoints. Public binds require an explicit opt-in; callers are
    still expected to add authentication and TLS at a reverse proxy. Remote
    Hugging Face identifiers require a full commit revision. Executions default
    to ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` unless the caller
    deliberately sets ``allow_download=True``.

    Release-capable plans also require completed qualification lineage. The
    local-smoke override is serialized as an explicit non-release boundary.
    """

    normalized_base = _safe_text(base_model, label="base model")
    local_base = Path(normalized_base).expanduser().exists()
    if model_revision is not None:
        model_revision = _safe_text(model_revision, label="model revision")
    if not local_base and (
        model_revision is None or re.fullmatch(r"[0-9a-f]{40}", model_revision) is None
    ):
        raise ExportValidationError(
            "Remote vLLM base models require model_revision to be a full 40-character commit SHA"
        )
    normalized_name = _validate_served_model_name(served_model_name, label="served model name")
    normalized_host = _safe_text(host, label="host")
    if not _is_loopback(normalized_host) and not allow_remote:
        raise ExportValidationError(
            "vLLM defaults to a loopback bind; set allow_remote=True only behind authenticated TLS"
        )
    if isinstance(port, bool) or not 1 <= port <= 65535:
        raise ExportValidationError("port must be between 1 and 65535")
    if isinstance(max_lora_rank, bool) or max_lora_rank < 1:
        raise ExportValidationError("max_lora_rank must be positive")
    if isinstance(tensor_parallel_size, bool) or tensor_parallel_size < 1:
        raise ExportValidationError("tensor_parallel_size must be positive")
    if not 0 < gpu_memory_utilization <= 1:
        raise ExportValidationError("gpu_memory_utilization must be greater than 0 and at most 1")
    if dtype not in {"auto", "half", "bfloat16", "float", "float16", "float32"}:
        raise ExportValidationError("unsupported vLLM dtype")
    if isinstance(max_model_len, bool) or not 128 <= max_model_len <= 131_072:
        raise ExportValidationError("max_model_len must be between 128 and 131072")

    metadata = _adapter_metadata(
        adapter_path,
        expected_model_revision=model_revision,
    )
    _require_matching_base(normalized_base, metadata)
    _require_matching_revision(model_revision, metadata)
    lineage_boundary = _adapter_lineage_boundary(
        metadata,
        configured_model_revision=model_revision,
        allow_unqualified_local_smoke=allow_unqualified_local_smoke,
    )
    if metadata.rank > max_lora_rank:
        raise ExportValidationError(
            f"Adapter rank {metadata.rank} exceeds max_lora_rank {max_lora_rank}"
        )

    descriptor = json.dumps(
        {
            "base_model_name": normalized_base,
            "name": normalized_name,
            "path": str(adapter_path.expanduser().resolve()),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    argv = [
        "vllm",
        "serve",
        normalized_base,
        "--host",
        normalized_host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        format(gpu_memory_utilization, ".3g"),
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--enable-lora",
        "--max-lora-rank",
        str(max_lora_rank),
        "--lora-modules",
        descriptor,
        "--generation-config",
        "vllm",
        "--disable-log-requests",
    ]
    if model_revision is not None:
        argv[3:3] = ["--revision", model_revision]

    network_flag = "0" if allow_download else "1"
    environment_overrides = (
        ("HF_HUB_OFFLINE", network_flag),
        ("TRANSFORMERS_OFFLINE", network_flag),
    )
    return VllmServePlan(
        base_model=normalized_base,
        model_revision=model_revision,
        adapter_path=str(adapter_path.expanduser().resolve()),
        adapter_base_model=metadata.base_model_name_or_path,
        adapter_revision=metadata.model_revision,
        adapter_config_sha256=metadata.config_sha256,
        adapter_weight_sha256=metadata.weight_sha256,
        training_manifest_path=metadata.training_manifest_path,
        training_manifest_sha256=metadata.training_manifest_sha256,
        training_config_sha256=metadata.training_config_sha256,
        training_dataset_sha256=metadata.training_dataset_sha256,
        qualification_sha256=metadata.qualification_sha256,
        lineage_boundary=lineage_boundary,
        allow_download=allow_download,
        environment_overrides=environment_overrides,
        argv=tuple(argv),
        provenance_boundary=(
            f"{lineage_boundary};adapter_declared_base_and_revision_plus_safetensors"
            if metadata.model_revision is not None
            else (
                f"{lineage_boundary};adapter_declared_base_plus_safetensors; "
                "configured_revision_bound_by_training_manifest; "
                "revision_not_embedded_in_adapter"
            )
        ),
    )


def build_vllm_argv(
    base_model: str,
    adapter_path: Path,
    *,
    model_revision: str | None = None,
    allow_download: bool = False,
    allow_unqualified_local_smoke: bool = False,
    served_model_name: str = "tickettune",
    host: str = "127.0.0.1",
    port: int = 8000,
    max_lora_rank: int = 64,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    max_model_len: int = 2048,
    allow_remote: bool = False,
) -> list[str]:
    """Return only the argv portion of :func:`build_vllm_plan`."""

    plan = build_vllm_plan(
        base_model,
        adapter_path,
        model_revision=model_revision,
        allow_download=allow_download,
        allow_unqualified_local_smoke=allow_unqualified_local_smoke,
        served_model_name=served_model_name,
        host=host,
        port=port,
        max_lora_rank=max_lora_rank,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        allow_remote=allow_remote,
    )
    return list(plan.argv)


def _verify_merge_provenance(
    merged: Path,
) -> tuple[Path, str, tuple[tuple[str, str], ...], dict[str, Any]]:
    provenance_path = merged / "tickettune-merge-provenance.json"
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise ExportValidationError(
            "Merged model requires a regular tickettune-merge-provenance.json from safe merge"
        )
    provenance, _payload, provenance_sha256 = _read_json_snapshot(
        provenance_path,
        label="merge provenance",
    )
    if provenance.get("schema_version") != 1 or provenance.get("operation") != "peft_safe_merge":
        raise ExportValidationError(
            "Merged model provenance has an unsupported schema or operation"
        )
    declared = provenance.get("artifact_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ExportValidationError("Merged model provenance must declare artifact_sha256")

    declared_hashes: dict[str, str] = {}
    for raw_name, raw_digest in declared.items():
        normalized_name = _normalized_artifact_name(raw_name) if isinstance(raw_name, str) else None
        if (
            normalized_name is None
            or not isinstance(raw_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_digest) is None
        ):
            raise ExportValidationError(
                "Merged model provenance contains an unsafe artifact name or SHA-256"
            )
        declared_hashes[normalized_name] = raw_digest

    artifact_paths, artifact_directories = _scan_regular_artifact_tree(
        merged,
        excluded=frozenset({provenance_path.name}),
    )
    actual_names = {name for name, _path in artifact_paths}
    declared_names = set(declared_hashes)
    untracked_directories = set(artifact_directories) - _artifact_parent_directories(declared_names)
    if actual_names != declared_names or untracked_directories:
        missing = sorted(set(declared_hashes) - actual_names)
        untracked = sorted(
            (actual_names - declared_names) | {f"{name}/" for name in untracked_directories}
        )
        raise ExportValidationError(
            "Merged model entries do not match provenance inventory: "
            f"missing={missing}, untracked={untracked}"
        )

    verified: list[tuple[str, str]] = []
    for relative, artifact in artifact_paths:
        actual_digest = _sha256(artifact)
        if actual_digest != declared_hashes[relative]:
            raise ExportValidationError(f"Merged model artifact hash mismatch: {relative}")
        verified.append((relative, actual_digest))
    return provenance_path, provenance_sha256, tuple(verified), provenance


def _validate_merged_hf_model(
    path: Path,
) -> tuple[Path, str, Path, str, tuple[tuple[str, str], ...], dict[str, Any]]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ExportValidationError("Merged Hugging Face model directory cannot be a symlink")
    merged = expanded.resolve()
    if not merged.is_dir():
        raise ExportValidationError(f"Merged Hugging Face model directory does not exist: {merged}")
    config_path = merged / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ExportValidationError(f"Merged Hugging Face model is missing {config_path.name}")
    if not any(item.is_file() for item in merged.glob("*.safetensors")):
        raise ExportValidationError(
            "Merged Hugging Face model must contain safetensors weights; "
            "pickle weights are rejected"
        )
    config, config_payload, _config_sha256 = _read_json_snapshot(
        config_path,
        label="merged model config",
    )
    family = config.get("model_type")
    if not isinstance(family, str) or not family:
        raise ExportValidationError("Merged model config must contain a string model_type")
    provenance_path, provenance_sha256, artifact_sha256, provenance = _verify_merge_provenance(
        merged
    )
    declared_config_sha256 = dict(artifact_sha256).get("config.json")
    if hashlib.sha256(config_payload).hexdigest() != declared_config_sha256:
        raise ExportValidationError("Merged model config changed during provenance verification")
    return merged, family, provenance_path, provenance_sha256, artifact_sha256, provenance


def verify_merged_model(
    path: Path,
    *,
    expected_base_model: str | None = None,
    expected_model_revision: str | None = None,
) -> VerifiedMergedModel:
    """Verify a safe-merged Hugging Face directory without loading model code.

    The public verifier is deliberately stricter than format-only GGUF planning:
    it requires the full adapter lineage and all safe-merge invariants written by
    :func:`merge_adapter`.  Callers may additionally bind the result to an exact
    configured base model and revision before performing local inference.
    """

    (
        merged,
        family,
        provenance_path,
        provenance_sha256,
        artifact_sha256,
        provenance,
    ) = _validate_merged_hf_model(path)

    invariant_values = {
        "safe_merge": True,
        "safe_serialization": True,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "trust_remote_code": False,
    }
    invalid_invariants = [
        name for name, required in invariant_values.items() if provenance.get(name) is not required
    ]
    if invalid_invariants:
        raise ExportValidationError(
            "Merged model provenance does not prove a non-quantized, safe merge: "
            + ", ".join(invalid_invariants)
        )

    base_model = provenance.get("base_model")
    adapter_base_model = provenance.get("adapter_base_model")
    model_revision = provenance.get("model_revision")
    adapter_revision = provenance.get("adapter_revision")
    adapter_config_sha256 = provenance.get("adapter_config_sha256")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ExportValidationError("Merged model provenance requires a string base_model")
    if not isinstance(adapter_base_model, str) or not adapter_base_model.strip():
        raise ExportValidationError("Merged model provenance requires a string adapter_base_model")
    if model_revision is not None and not isinstance(model_revision, str):
        raise ExportValidationError(
            "Merged model provenance model_revision must be a string or null"
        )
    if adapter_revision is not None and not isinstance(adapter_revision, str):
        raise ExportValidationError(
            "Merged model provenance adapter_revision must be a string or null"
        )
    if (
        not isinstance(adapter_config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", adapter_config_sha256) is None
    ):
        raise ExportValidationError("Merged model provenance requires an adapter_config_sha256")

    normalized_base = _normalized_base(base_model)
    normalized_adapter_base = _normalized_base(adapter_base_model)
    if normalized_base != normalized_adapter_base:
        raise ExportValidationError("Merged model provenance base and adapter lineage do not match")
    if adapter_revision is not None and adapter_revision != model_revision:
        raise ExportValidationError(
            "Merged model provenance model and adapter revisions do not match"
        )
    if expected_base_model is not None and _normalized_base(expected_base_model) != normalized_base:
        raise ExportValidationError(
            "Merged model/configured base model mismatch: "
            f"expected {_normalized_base(expected_base_model)!r}, provenance declares "
            f"{normalized_base!r}"
        )
    if expected_model_revision is not None and model_revision != expected_model_revision:
        raise ExportValidationError(
            "Merged model/configured revision mismatch: "
            f"expected {expected_model_revision!r}, provenance declares {model_revision!r}"
        )

    raw_weight_files = provenance.get("adapter_weight_files")
    raw_weight_hashes = provenance.get("adapter_weight_sha256")
    if not isinstance(raw_weight_hashes, list) or not raw_weight_hashes:
        raise ExportValidationError("Merged model provenance requires adapter weight hashes")
    adapter_hashes: list[str] = []
    for raw_digest in raw_weight_hashes:
        if not isinstance(raw_digest, str) or re.fullmatch(r"[0-9a-f]{64}", raw_digest) is None:
            raise ExportValidationError(
                "Merged model provenance contains an invalid adapter weight SHA-256"
            )
        adapter_hashes.append(raw_digest)

    # Early schema-v1 manifests recorded the complete ordered digest list but
    # not the corresponding basenames. Preserve compatibility with those
    # immutable artifacts while validating filenames whenever they are present.
    if raw_weight_files is None:
        adapter_weight_files: tuple[str, ...] = ()
    elif not isinstance(raw_weight_files, list) or len(raw_weight_files) != len(adapter_hashes):
        raise ExportValidationError(
            "Merged model provenance adapter weight files do not match its hashes"
        )
    else:
        validated_weight_files: list[str] = []
        seen_weight_files: set[str] = set()
        for raw_name in raw_weight_files:
            if (
                not isinstance(raw_name, str)
                or Path(raw_name).name != raw_name
                or _normalized_artifact_name(raw_name) != raw_name
                or raw_name in seen_weight_files
                or not raw_name.endswith(".safetensors")
            ):
                raise ExportValidationError(
                    "Merged model provenance contains an invalid adapter weight identity"
                )
            seen_weight_files.add(raw_name)
            validated_weight_files.append(raw_name)
        adapter_weight_files = tuple(validated_weight_files)

    raw_training_manifest_sha256 = provenance.get("training_manifest_sha256")
    raw_training_config_sha256 = provenance.get("training_config_sha256")
    raw_training_dataset_sha256 = provenance.get("training_dataset_sha256", {})
    raw_qualification_sha256 = provenance.get("qualification_sha256", {})
    training_values_present = any(
        value not in (None, {})
        for value in (
            raw_training_manifest_sha256,
            raw_training_config_sha256,
            raw_training_dataset_sha256,
            raw_qualification_sha256,
        )
    )
    if training_values_present:
        if (
            not isinstance(raw_training_manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_training_manifest_sha256) is None
            or not isinstance(raw_training_config_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_training_config_sha256) is None
            or not isinstance(raw_training_dataset_sha256, dict)
            or not raw_training_dataset_sha256
            or not isinstance(raw_qualification_sha256, dict)
        ):
            raise ExportValidationError("Merged model provenance has incomplete training lineage")
        training_dataset_items = tuple(sorted(raw_training_dataset_sha256.items()))
        qualification_items = tuple(sorted(raw_qualification_sha256.items()))
        if any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in (*training_dataset_items, *qualification_items)
        ):
            raise ExportValidationError("Merged model provenance has invalid training hashes")
        required_dataset_hashes = {"source", "manifest", "train", "validation"}
        if not required_dataset_hashes.issubset(dict(training_dataset_items)):
            raise ExportValidationError("Merged model provenance lacks required dataset lineage")
        training_manifest_sha256 = raw_training_manifest_sha256
        training_config_sha256 = raw_training_config_sha256
        training_dataset_sha256 = training_dataset_items
        qualification_sha256 = qualification_items
    else:
        training_manifest_sha256 = None
        training_config_sha256 = None
        training_dataset_sha256 = ()
        qualification_sha256 = ()

    raw_lineage_boundary = provenance.get("lineage_boundary")
    if raw_lineage_boundary is None:
        lineage_boundary: LineageBoundary | None = None
    elif (
        not isinstance(raw_lineage_boundary, str) or raw_lineage_boundary not in _LINEAGE_BOUNDARIES
    ):
        raise ExportValidationError(
            "Merged model provenance contains an unsupported lineage boundary"
        )
    else:
        lineage_boundary = cast(LineageBoundary, raw_lineage_boundary)
    if lineage_boundary == QUALIFIED_RELEASE_LINEAGE and not (
        training_manifest_sha256 is not None
        and training_config_sha256 is not None
        and _REQUIRED_QUALIFICATION_HASHES.issubset(dict(qualification_sha256))
    ):
        raise ExportValidationError(
            "Merged model provenance claims qualified release lineage without qualification hashes"
        )

    raw_merge_dtype = provenance.get("dtype")
    if raw_merge_dtype is None:
        # Early schema-v1 manifests did not bind merge precision. Keep them
        # readable for format-only consumers; live parity rejects the missing
        # precision proof before loading either model.
        merge_dtype: str | None = None
    elif not isinstance(raw_merge_dtype, str) or raw_merge_dtype not in _SUPPORTED_MERGE_DTYPES:
        allowed = ", ".join(sorted(_SUPPORTED_MERGE_DTYPES))
        raise ExportValidationError(f"Merged model provenance dtype must be one of: {allowed}")
    else:
        merge_dtype = raw_merge_dtype

    return VerifiedMergedModel(
        merged_model=str(merged),
        model_family=family,
        provenance_path=str(provenance_path),
        provenance_sha256=provenance_sha256,
        merge_dtype=merge_dtype,
        base_model=base_model,
        model_revision=model_revision,
        adapter_base_model=adapter_base_model,
        adapter_revision=adapter_revision,
        adapter_config_sha256=adapter_config_sha256,
        adapter_weight_files=adapter_weight_files,
        adapter_weight_sha256=tuple(adapter_hashes),
        training_manifest_sha256=training_manifest_sha256,
        training_config_sha256=training_config_sha256,
        training_dataset_sha256=training_dataset_sha256,
        qualification_sha256=qualification_sha256,
        lineage_boundary=lineage_boundary,
        artifact_sha256=artifact_sha256,
    )


def _verified_merge_lineage_boundary(
    verified: VerifiedMergedModel,
    *,
    allow_unqualified_local_smoke: bool,
) -> LineageBoundary:
    """Classify verified merge lineage for a downstream release-capable export."""

    if not isinstance(allow_unqualified_local_smoke, bool):
        raise ExportValidationError("allow_unqualified_local_smoke must be a boolean")
    qualified = (
        verified.lineage_boundary == QUALIFIED_RELEASE_LINEAGE
        and verified.training_manifest_sha256 is not None
        and verified.training_config_sha256 is not None
        and {"source", "manifest", "train", "validation"}.issubset(
            dict(verified.training_dataset_sha256)
        )
        and _REQUIRED_QUALIFICATION_HASHES.issubset(dict(verified.qualification_sha256))
    )
    if qualified:
        return cast(LineageBoundary, QUALIFIED_RELEASE_LINEAGE)
    if allow_unqualified_local_smoke:
        return cast(LineageBoundary, UNQUALIFIED_LOCAL_SMOKE_LINEAGE)
    raise ExportValidationError(
        "Release-capable Ollama export requires qualified merge lineage; "
        "use allow_unqualified_local_smoke=True only for local smoke tests or fixtures"
    )


def render_ollama_modelfile(
    gguf_path: Path,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0,
    context_length: int = 2048,
) -> str:
    """Render a deterministic GGUF-backed Modelfile without an ADAPTER directive."""

    raw_path = str(gguf_path)
    _safe_text(raw_path, label="GGUF path")
    name = gguf_path.name
    if gguf_path.suffix.lower() != ".gguf":
        raise ExportValidationError("GGUF path must end in .gguf")
    if any(character.isspace() for character in name):
        raise ExportValidationError("GGUF filename cannot contain whitespace")
    if '"""' in system_prompt or "\x00" in system_prompt:
        raise ExportValidationError("System prompt cannot contain NUL bytes or triple quotes")
    if isinstance(context_length, bool) or context_length < 1:
        raise ExportValidationError("context_length must be positive")
    if not 0 <= temperature <= 2:
        raise ExportValidationError("temperature must be between 0 and 2")
    temperature_text = format(temperature, "g")
    return (
        f"FROM ./{name}\n"
        f"PARAMETER temperature {temperature_text}\n"
        f"PARAMETER num_ctx {context_length}\n"
        f'SYSTEM """{system_prompt}"""\n'
    )


def build_ollama_export_plan(
    merged_model: Path,
    output_dir: Path,
    *,
    model_name: str = "tickettune",
    model_family: str | None = None,
    quantization: str = "Q4_K_M",
    llama_cpp_revision: str = LLAMA_CPP_REVISION,
    adapter_path: Path | None = None,
    allow_unqualified_local_smoke: bool = False,
    context_length: int = 2048,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> OllamaExportPlan:
    """Plan merged-HF -> pinned llama.cpp -> GGUF -> Ollama creation.

    A direct Qwen Safetensors adapter route is intentionally rejected. Ollama's
    documented direct-adapter architecture list does not include Qwen, and its
    import guide recommends non-QLoRA adapters even for supported families.
    Qualified merge lineage is required unless the caller explicitly marks the
    plan as a non-release local smoke or fixture operation.
    """

    verified_merge = verify_merged_model(merged_model)
    lineage_boundary = _verified_merge_lineage_boundary(
        verified_merge,
        allow_unqualified_local_smoke=allow_unqualified_local_smoke,
    )
    merged = Path(verified_merge.merged_model)
    detected_family = verified_merge.model_family
    merge_provenance_path = Path(verified_merge.provenance_path)
    merge_provenance_sha256 = verified_merge.provenance_sha256
    merged_artifact_sha256 = verified_merge.artifact_sha256
    family = (model_family or detected_family).strip().lower()
    if adapter_path is not None:
        if family.startswith("qwen"):
            raise ExportValidationError(
                "Direct Qwen adapter import is unsupported; merge into a pristine "
                "non-quantized base, then convert the merged Hugging Face model to GGUF."
            )
        raise ExportValidationError(
            "Direct adapter import is outside TicketTune's verified Ollama path; "
            "use a merged Hugging Face model and GGUF conversion."
        )
    if family != detected_family.lower():
        raise ExportValidationError(
            f"Requested model_family {family!r} does not match config model_type "
            f"{detected_family!r}"
        )
    if quantization not in _SUPPORTED_QUANTIZATIONS:
        allowed = ", ".join(sorted(_SUPPORTED_QUANTIZATIONS))
        raise ExportValidationError(f"quantization must be one of: {allowed}")
    revision = _safe_text(llama_cpp_revision, label="llama.cpp revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ExportValidationError("llama.cpp revision must be a full 40-character commit SHA")
    ollama_name = _validate_served_model_name(model_name, label="Ollama model name")

    output = output_dir.expanduser().resolve()
    if _paths_overlap(merged, output):
        raise ExportValidationError(
            "Ollama output and merged-model source must be isolated; "
            "neither may contain or overwrite the other"
        )
    llama_cpp = output / ".build" / "llama.cpp"
    build_dir = llama_cpp / "build"
    f16_gguf = output / f"{ollama_name.replace('/', '-')}-f16.gguf"
    quantized_gguf = (
        f16_gguf
        if quantization == "F16"
        else output / f"{ollama_name.replace('/', '-')}-{quantization.lower()}.gguf"
    )
    modelfile_path = output / "Modelfile"
    export_provenance_path = output / "tickettune-ollama-export-provenance.json"
    modelfile = render_ollama_modelfile(
        quantized_gguf,
        system_prompt=system_prompt,
        context_length=context_length,
    )

    clone_argv = (
        "git",
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        "https://github.com/ggml-org/llama.cpp.git",
        str(llama_cpp),
    )
    checkout_argv = (
        "git",
        "-C",
        str(llama_cpp),
        "checkout",
        "--detach",
        revision,
    )
    configure_argv = (
        ()
        if quantization == "F16"
        else (
            "cmake",
            "-S",
            str(llama_cpp),
            "-B",
            str(build_dir),
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        )
    )
    build_argv = (
        ()
        if quantization == "F16"
        else (
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-quantize",
            "-j",
            "2",
        )
    )
    conversion_argv = (
        sys.executable,
        str(llama_cpp / "convert_hf_to_gguf.py"),
        str(merged),
        "--outfile",
        str(f16_gguf),
        "--outtype",
        "f16",
    )
    quantize_argv = (
        ()
        if quantization == "F16"
        else (
            str(build_dir / "bin" / "llama-quantize"),
            str(f16_gguf),
            str(quantized_gguf),
            quantization,
        )
    )
    checksum_targets = [str(f16_gguf)]
    if quantized_gguf != f16_gguf:
        checksum_targets.append(str(quantized_gguf))
    checksum_targets.append(str(modelfile_path))
    checksum_argv = (
        "shasum",
        "-a",
        "256",
        *checksum_targets,
    )
    return OllamaExportPlan(
        source_kind="merged_hf",
        merged_model=str(merged),
        merge_provenance_path=str(merge_provenance_path),
        merge_provenance_sha256=merge_provenance_sha256,
        merged_artifact_sha256=merged_artifact_sha256,
        training_manifest_sha256=verified_merge.training_manifest_sha256,
        training_config_sha256=verified_merge.training_config_sha256,
        training_dataset_sha256=verified_merge.training_dataset_sha256,
        qualification_sha256=verified_merge.qualification_sha256,
        lineage_boundary=lineage_boundary,
        output_dir=str(output),
        model_name=ollama_name,
        model_family=family,
        llama_cpp_revision=revision,
        direct_adapter_supported=False,
        clone_argv=clone_argv,
        checkout_argv=checkout_argv,
        configure_argv=configure_argv,
        build_argv=build_argv,
        conversion_argv=conversion_argv,
        quantize_argv=quantize_argv,
        checksum_argv=checksum_argv,
        f16_gguf_path=str(f16_gguf),
        gguf_path=str(quantized_gguf),
        modelfile_path=str(modelfile_path),
        export_provenance_path=str(export_provenance_path),
        modelfile=modelfile,
        ollama_create_argv=("ollama", "create", ollama_name, "-f", str(modelfile_path)),
        ollama_run_argv=("ollama", "run", ollama_name),
    )


def _require_planned_merged_snapshot(plan: OllamaExportPlan) -> None:
    """Reject merged-model bytes or provenance that changed after planning."""

    try:
        verified_merge = verify_merged_model(Path(plan.merged_model))
        observed_lineage_boundary = _verified_merge_lineage_boundary(
            verified_merge,
            allow_unqualified_local_smoke=(
                plan.lineage_boundary == UNQUALIFIED_LOCAL_SMOKE_LINEAGE
            ),
        )
    except ExportValidationError as exc:
        raise ExportExecutionError(
            "Merged model no longer matches its verified export-plan snapshot"
        ) from exc
    observed = (
        verified_merge.merged_model,
        verified_merge.model_family.lower(),
        verified_merge.provenance_path,
        verified_merge.provenance_sha256,
        verified_merge.artifact_sha256,
        verified_merge.training_manifest_sha256,
        verified_merge.training_config_sha256,
        verified_merge.training_dataset_sha256,
        verified_merge.qualification_sha256,
        observed_lineage_boundary,
    )
    expected = (
        plan.merged_model,
        plan.model_family,
        plan.merge_provenance_path,
        plan.merge_provenance_sha256,
        plan.merged_artifact_sha256,
        plan.training_manifest_sha256,
        plan.training_config_sha256,
        plan.training_dataset_sha256,
        plan.qualification_sha256,
        plan.lineage_boundary,
    )
    if observed != expected:
        raise ExportExecutionError(
            "Merged-model bytes, inventory, or provenance changed after export planning"
        )


def materialize_ollama_plan(plan: OllamaExportPlan) -> Path:
    """Create only the output directory and Modelfile described by a plan."""

    _require_planned_merged_snapshot(plan)
    output = Path(plan.output_dir)
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ExportExecutionError(f"Ollama output must be a regular directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    modelfile_path = Path(plan.modelfile_path)
    resolved_output = output.resolve()
    if modelfile_path.resolve().parent != resolved_output:
        raise ExportExecutionError("Ollama Modelfile escaped the isolated output directory")
    unexpected = tuple(path for path in output.iterdir() if path != modelfile_path)
    if unexpected:
        names = ", ".join(sorted(path.name for path in unexpected))
        raise ExportExecutionError(
            f"Refusing non-empty Ollama output directory; unexpected entries: {names}"
        )
    if modelfile_path.is_symlink():
        raise ExportExecutionError(f"Ollama Modelfile cannot be a symbolic link: {modelfile_path}")
    if modelfile_path.exists() and modelfile_path.read_text(encoding="utf-8") != plan.modelfile:
        raise ExportExecutionError(f"Refusing to overwrite a different Modelfile: {modelfile_path}")
    modelfile_path.write_text(plan.modelfile, encoding="utf-8")
    return modelfile_path


def write_ollama_export_provenance(plan: OllamaExportPlan) -> OllamaExportResult:
    """Verify conversion outputs and atomically write their immutable provenance."""

    _require_planned_merged_snapshot(plan)
    output = Path(plan.output_dir)
    if not output.is_dir() or output.is_symlink():
        raise ExportExecutionError(f"Ollama output must be a regular directory: {output}")
    resolved_output = output.resolve()
    ordered_paths = (
        Path(plan.f16_gguf_path),
        Path(plan.gguf_path),
        Path(plan.modelfile_path),
    )
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in ordered_paths:
        resolved = path.resolve()
        if resolved.parent != resolved_output:
            raise ExportExecutionError(
                f"Cannot record Ollama export provenance outside output directory: {path}"
            )
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file() or path.is_symlink():
            raise ExportExecutionError(
                f"Cannot record Ollama export provenance; missing regular artifact: {path}"
            )
        unique_paths.append(path)

    artifact_hashes = tuple((path.name, _sha256(path)) for path in unique_paths)
    payload = {
        "schema_version": 1,
        "operation": "merged_hf_to_gguf",
        "merged_model": plan.merged_model,
        "merge_provenance_path": plan.merge_provenance_path,
        "merge_provenance_sha256": plan.merge_provenance_sha256,
        "merged_artifact_sha256": dict(plan.merged_artifact_sha256),
        "training_manifest_sha256": plan.training_manifest_sha256,
        "training_config_sha256": plan.training_config_sha256,
        "training_dataset_sha256": dict(plan.training_dataset_sha256),
        "qualification_sha256": dict(plan.qualification_sha256),
        "lineage_boundary": plan.lineage_boundary,
        "llama_cpp_revision": plan.llama_cpp_revision,
        "model_family": plan.model_family,
        "model_name": plan.model_name,
        "artifact_sha256": dict(artifact_hashes),
    }
    serialized = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    destination = Path(plan.export_provenance_path)
    if destination.resolve().parent != resolved_output or destination.is_symlink():
        raise ExportExecutionError(
            "Ollama export provenance path escaped the isolated output directory"
        )
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == serialized
        ):
            return OllamaExportResult(
                provenance_path=str(destination),
                provenance_sha256=_sha256(destination),
                artifact_sha256=artifact_hashes,
            )
        raise ExportExecutionError(
            f"Refusing to overwrite different Ollama export provenance: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return OllamaExportResult(
        provenance_path=str(destination),
        provenance_sha256=_sha256(destination),
        artifact_sha256=artifact_hashes,
    )


def run_argv(
    argv: tuple[str, ...] | list[str],
    *,
    cwd: Path | None = None,
    environment_overrides: Mapping[str, str] | None = None,
    route_output_to_stderr: bool = False,
) -> None:
    """Execute validated argv directly, optionally isolating machine JSON stdout."""

    if not argv or any(not item or "\x00" in item for item in argv):
        raise ExportValidationError("argv must contain non-empty, NUL-free strings")
    if environment_overrides and any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key
        or "=" in key
        or "\x00" in key
        or "\x00" in value
        for key, value in environment_overrides.items()
    ):
        raise ExportValidationError(
            "environment overrides must contain non-empty, NUL-free keys and values"
        )
    environment = None
    if environment_overrides:
        environment = os.environ.copy()
        environment.update(environment_overrides)

    kwargs: dict[str, Any] = {}
    if environment is not None:
        kwargs["env"] = environment
    if route_output_to_stderr:
        kwargs["stdout"] = sys.stderr
        kwargs["stderr"] = sys.stderr
    # The builder rejects empty/NUL-bearing values, callers pass an argv list,
    # and shell evaluation is explicitly disabled.
    try:
        subprocess.run(  # noqa: S603  # nosec B603
            list(argv), cwd=cwd, check=True, shell=False, **kwargs
        )
    except subprocess.CalledProcessError as exc:
        raise ExportExecutionError(
            f"command exited unsuccessfully with status {exc.returncode}"
        ) from exc


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LLAMA_CPP_REVISION",
    "QUALIFIED_RELEASE_LINEAGE",
    "UNQUALIFIED_LOCAL_SMOKE_LINEAGE",
    "AdapterMetadata",
    "ExportExecutionError",
    "ExportValidationError",
    "LineageBoundary",
    "MergePlan",
    "MergeResult",
    "OllamaExportPlan",
    "OllamaExportResult",
    "VerifiedMergedModel",
    "VllmServePlan",
    "build_merge_plan",
    "build_ollama_export_plan",
    "build_vllm_argv",
    "build_vllm_plan",
    "materialize_ollama_plan",
    "merge_adapter",
    "render_ollama_modelfile",
    "run_argv",
    "verify_merged_model",
    "write_ollama_export_provenance",
]
