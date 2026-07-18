"""Shared chat-template generation for baseline and adapter evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .data import DatasetIntegrityError, DatasetVerification, verify_prepared_dataset
from .export import ExportValidationError, VerifiedMergedModel, verify_merged_model
from .run_manifest import RunManifest, canonical_json_bytes
from .strict_json import StrictJSONError, loads_strict
from .training import QWEN_EOS_TOKEN

if TYPE_CHECKING:
    from .config import FineTuneConfig


class AdapterCompatibilityError(ValueError):
    """A PEFT adapter declares a different base model or revision."""


class GenerationOutputError(ValueError):
    """A generated artifact cannot be written without mutating prior evidence."""


class GenerationInputError(ValueError):
    """A verified local model input cannot be snapshotted safely for loading."""


Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class AdapterProvenance(BaseModel):
    """Verified lightweight adapter metadata carried into predictions."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    adapter_path: Path
    config_sha256: Sha256Digest
    weight_sha256: dict[str, Sha256Digest]
    training_manifest_path: Path | None = None
    training_manifest_sha256: Sha256Digest | None = None
    training_config_sha256: Sha256Digest | None = None
    training_dataset_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    qualification_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)


class GeneratedPrediction(BaseModel):
    """One model response paired with held-out ground truth."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    id: str
    expected: dict[str, Any]
    prediction: str
    latency_ms: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    model_name_or_path: str
    model_revision: str | None = None
    dataset_manifest_sha256: Sha256Digest | None = None
    dataset_split_sha256: Sha256Digest | None = None
    generation_config_sha256: Sha256Digest | None = None
    training_manifest_sha256: Sha256Digest | None = None
    training_config_sha256: Sha256Digest | None = None
    training_dataset_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    qualification_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    adapter_path: str | None = None
    adapter_config_sha256: Sha256Digest | None = None
    adapter_weight_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    merged_model_path: str | None = None
    merge_provenance_sha256: Sha256Digest | None = None
    merged_artifact_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    merged_adapter_config_sha256: Sha256Digest | None = None
    merged_adapter_weight_files: tuple[str, ...] = ()
    merged_adapter_weight_sha256: tuple[Sha256Digest, ...] = ()


@dataclass(frozen=True)
class GenerationLibraries:
    """Injected optional libraries for deterministic offline tests."""

    torch: Any
    AutoModelForCausalLM: Any
    AutoTokenizer: Any
    PeftModel: Any


@dataclass(frozen=True)
class GenerationCohort:
    """One stable, verified held-out cohort used by generation and scoring."""

    verification: DatasetVerification
    records: tuple[dict[str, Any], ...]
    ordered_ids: tuple[str, ...]
    expected_json: tuple[bytes, ...]


REQUIRED_TRAINING_DATASET_HASHES = frozenset({"source", "manifest", "train", "validation", "test"})


def _load_generation_libraries() -> GenerationLibraries:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "Generation dependencies are not installed. Run `uv sync --extra train` first."
        ) from exc
    return GenerationLibraries(
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        PeftModel=PeftModel,
    )


def _model_identity(value: str, *, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise AdapterCompatibilityError(f"{label} cannot be empty")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise AdapterCompatibilityError(f"{label} cannot contain NUL bytes or line breaks")
    candidate = Path(cleaned).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return cleaned.rstrip("/")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_regular_snapshot(path: Path, *, label: str) -> bytes:
    """Read one regular file without following its final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AdapterCompatibilityError(
            f"{label} must be a readable non-symlink file: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdapterCompatibilityError(f"{label} must be a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        current = os.lstat(path)
    except OSError as exc:
        raise AdapterCompatibilityError(f"{label} changed while it was being read: {path}") from exc
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after) or (
        current.st_dev,
        current.st_ino,
    ) != (before.st_dev, before.st_ino):
        raise AdapterCompatibilityError(f"{label} changed while it was being read: {path}")
    return payload


def _sha256_regular_snapshot(path: Path, *, label: str) -> str:
    """Hash one stable regular file without following its final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AdapterCompatibilityError(
            f"{label} must be a readable non-symlink file: {path}"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdapterCompatibilityError(f"{label} must be a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
    except OSError as exc:
        raise AdapterCompatibilityError(
            f"{label} changed while it was being hashed: {path}"
        ) from exc
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after) or (
        current.st_dev,
        current.st_ino,
    ) != (before.st_dev, before.st_ino):
        raise AdapterCompatibilityError(f"{label} changed while it was being hashed: {path}")
    return digest.hexdigest()


def _verified_training_lineage(
    adapter: Path,
    *,
    model_name_or_path: str,
    model_revision: str | None,
    adapter_config_sha256: str,
    adapter_weight_sha256: dict[str, str],
    required: bool,
) -> tuple[Path | None, str | None, str | None, dict[str, str], dict[str, str]]:
    manifest_path = adapter.parent / "manifest.json"
    if not manifest_path.exists():
        if required:
            raise AdapterCompatibilityError(
                f"release adapter requires its sibling training manifest: {manifest_path}"
            )
        return None, None, None, {}, {}
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AdapterCompatibilityError(
            f"training manifest must be a regular non-symlink file: {manifest_path}"
        )
    try:
        payload = _read_regular_snapshot(manifest_path, label="training manifest")
        decoded = loads_strict(payload.decode("utf-8"))
        manifest = RunManifest.model_validate(decoded)
    except (
        AdapterCompatibilityError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        StrictJSONError,
        ValidationError,
    ) as exc:
        raise AdapterCompatibilityError(f"invalid training manifest at {manifest_path}") from exc
    if manifest.status != "completed":
        raise AdapterCompatibilityError("adapter training manifest is not completed")
    if manifest.run_id != adapter.parent.name:
        raise AdapterCompatibilityError("adapter directory is not bound to its training run ID")
    if _model_identity(manifest.model_name_or_path, label="training base model") != _model_identity(
        model_name_or_path,
        label="configured base model",
    ):
        raise AdapterCompatibilityError("training manifest base model does not match generation")
    if manifest.model_revision != model_revision:
        raise AdapterCompatibilityError(
            "training manifest model revision does not match generation"
        )
    computed_config_sha256 = hashlib.sha256(canonical_json_bytes(manifest.config)).hexdigest()
    if computed_config_sha256 != manifest.config_sha256:
        raise AdapterCompatibilityError("training manifest config hash is invalid")

    dataset_hashes = dict(sorted(manifest.dataset_sha256.items()))
    if not REQUIRED_TRAINING_DATASET_HASHES.issubset(dataset_hashes) or any(
        re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in dataset_hashes.values()
    ):
        raise AdapterCompatibilityError(
            "training manifest lacks valid source, prepared-manifest, train, validation, "
            "or test hashes"
        )

    artifact_hashes = {artifact.path: artifact.sha256 for artifact in manifest.artifacts}
    expected_adapter_artifacts = {
        f"{adapter.name}/adapter_config.json": adapter_config_sha256,
        **{f"{adapter.name}/{name}": digest for name, digest in adapter_weight_sha256.items()},
    }
    if any(
        artifact_hashes.get(name) != digest for name, digest in expected_adapter_artifacts.items()
    ):
        raise AdapterCompatibilityError(
            "training manifest adapter artifacts do not match current adapter bytes"
        )
    qualification_hashes = {
        name: digest for name, digest in dataset_hashes.items() if name.startswith("qualification_")
    }
    return (
        manifest_path.resolve(),
        hashlib.sha256(payload).hexdigest(),
        manifest.config_sha256,
        dataset_hashes,
        qualification_hashes,
    )


def validate_adapter_compatibility(
    adapter_path: Path,
    *,
    model_name_or_path: str,
    model_revision: str | None,
    require_training_manifest: bool = False,
) -> AdapterProvenance:
    """Validate lightweight PEFT provenance before importing or attaching PEFT."""

    adapter = adapter_path.expanduser().resolve()
    if not adapter.is_dir():
        raise AdapterCompatibilityError(f"adapter directory does not exist: {adapter}")
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise AdapterCompatibilityError(
            f"adapter_config.json must be a regular non-symlink file: {config_path}"
        )
    try:
        config_payload = _read_regular_snapshot(config_path, label="PEFT adapter config")
        config = loads_strict(config_payload.decode("utf-8"))
    except (
        AdapterCompatibilityError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        StrictJSONError,
    ) as exc:
        raise AdapterCompatibilityError(
            f"invalid PEFT adapter config at {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise AdapterCompatibilityError(
            f"invalid PEFT adapter config at {config_path}: expected a JSON object"
        )
    raw_base = config.get("base_model_name_or_path")
    if not isinstance(raw_base, str):
        raise AdapterCompatibilityError(
            "PEFT adapter config must contain string base_model_name_or_path"
        )
    expected_base = _model_identity(model_name_or_path, label="configured base model")
    actual_base = _model_identity(raw_base, label="adapter base model")
    if expected_base != actual_base:
        raise AdapterCompatibilityError(
            "PEFT adapter/base model mismatch: "
            f"configured {expected_base!r}, adapter declares {actual_base!r}"
        )

    raw_revision = config.get("revision")
    if raw_revision is not None:
        if not isinstance(raw_revision, str):
            raise AdapterCompatibilityError("PEFT adapter config revision must be a string or null")
        actual_revision = _model_identity(raw_revision, label="adapter model revision")
        if model_revision is None:
            raise AdapterCompatibilityError(
                "PEFT adapter declares a model revision but the generation config does not"
            )
        expected_revision = _model_identity(
            model_revision,
            label="configured model revision",
        )
        if actual_revision != expected_revision:
            raise AdapterCompatibilityError(
                "PEFT adapter/model revision mismatch: "
                f"configured {expected_revision!r}, adapter declares {actual_revision!r}"
            )
    weights = tuple(sorted(path for path in adapter.glob("*.safetensors") if path.is_file()))
    if not weights:
        raise AdapterCompatibilityError(
            f"adapter must contain at least one Safetensors weight file: {adapter}"
        )
    if any(path.is_symlink() for path in weights):
        raise AdapterCompatibilityError("adapter Safetensors weights must not be symlinks")
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    weight_sha256 = {
        path.name: _sha256_regular_snapshot(path, label="adapter Safetensors weight")
        for path in weights
    }
    (
        training_manifest_path,
        training_manifest_sha256,
        training_config_sha256,
        training_dataset_sha256,
        qualification_sha256,
    ) = _verified_training_lineage(
        adapter,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        adapter_config_sha256=config_sha256,
        adapter_weight_sha256=weight_sha256,
        required=require_training_manifest,
    )
    return AdapterProvenance(
        adapter_path=adapter,
        config_sha256=config_sha256,
        weight_sha256=weight_sha256,
        training_manifest_path=training_manifest_path,
        training_manifest_sha256=training_manifest_sha256,
        training_config_sha256=training_config_sha256,
        training_dataset_sha256=training_dataset_sha256,
        qualification_sha256=qualification_sha256,
    )


def _regular_tree_inventory(root: Path, *, label: str) -> dict[str, str]:
    """Hash a complete regular-file tree while rejecting symbolic links."""

    try:
        root_before = os.lstat(root)
    except OSError as exc:
        raise GenerationInputError(f"{label} directory is not readable: {root}") from exc
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        raise GenerationInputError(f"{label} must be a non-symlink directory: {root}")

    inventory: dict[str, str] = {}
    for current_text, directory_names, filenames in os.walk(root, followlinks=False):
        current = Path(current_text)
        try:
            current_stat = os.lstat(current)
        except OSError as exc:
            raise GenerationInputError(f"{label} directory changed during inventory") from exc
        if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(current_stat.st_mode):
            raise GenerationInputError(f"{label} contains a non-directory path: {current}")
        directory_names.sort()
        filenames.sort()
        for directory_name in directory_names:
            directory = current / directory_name
            try:
                directory_stat = os.lstat(directory)
            except OSError as exc:
                raise GenerationInputError(f"{label} directory changed during inventory") from exc
            if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
                raise GenerationInputError(
                    f"{label} contains a symbolic-link directory: {directory}"
                )
        for filename in filenames:
            artifact = current / filename
            try:
                artifact_stat = os.lstat(artifact)
            except OSError as exc:
                raise GenerationInputError(f"{label} file changed during inventory") from exc
            if stat.S_ISLNK(artifact_stat.st_mode) or not stat.S_ISREG(artifact_stat.st_mode):
                raise GenerationInputError(f"{label} contains a non-regular file: {artifact}")
            relative = artifact.relative_to(root).as_posix()
            try:
                inventory[relative] = _sha256_regular_snapshot(
                    artifact,
                    label=f"{label} artifact",
                )
            except AdapterCompatibilityError as exc:
                raise GenerationInputError(str(exc)) from exc

    try:
        root_after = os.lstat(root)
    except OSError as exc:
        raise GenerationInputError(f"{label} directory changed during inventory") from exc
    if (root_before.st_dev, root_before.st_ino) != (root_after.st_dev, root_after.st_ino):
        raise GenerationInputError(f"{label} directory changed during inventory")
    if not inventory:
        raise GenerationInputError(f"{label} directory contains no regular files")
    return dict(sorted(inventory.items()))


def _link_regular_snapshot(source: Path, destination: Path, *, label: str) -> None:
    """Hard-link one held regular inode and prove the new name targets that inode."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune targets POSIX hosts.
        raise GenerationInputError("model input snapshots require O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise GenerationInputError(f"{label} must remain a regular file: {source}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GenerationInputError(f"{label} must remain a regular file: {source}")
        os.link(source, destination, follow_symlinks=False)
        linked = os.lstat(destination)
        after = os.fstat(descriptor)
        current = os.lstat(source)
    except OSError as exc:
        with suppress(OSError):
            destination.unlink()
        raise GenerationInputError(f"{label} changed while its snapshot was created") from exc
    finally:
        os.close(descriptor)
    held_identity = (before.st_dev, before.st_ino)
    before_content_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_content_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        before_content_identity != after_content_identity
        or (linked.st_dev, linked.st_ino) != held_identity
        or (current.st_dev, current.st_ino) != held_identity
    ):
        with suppress(OSError):
            destination.unlink()
        raise GenerationInputError(f"{label} changed while its snapshot was created")


@contextmanager
def _verified_input_snapshot(
    source: Path,
    *,
    expected_inventory: Mapping[str, str],
    label: str,
) -> Iterator[Path]:
    """Expose a private hard-link tree whose complete bytes match verified evidence."""

    actual_inventory = _regular_tree_inventory(source, label=label)
    expected = dict(sorted(expected_inventory.items()))
    if actual_inventory != expected:
        raise GenerationInputError(f"{label} inventory differs from verified provenance")

    snapshot = Path(
        tempfile.mkdtemp(
            prefix=f".{source.name}.tickettune-input-",
            dir=source.parent,
        )
    )
    snapshot_stat = os.lstat(snapshot)
    body_failed = False
    try:
        for relative, digest in expected.items():
            source_artifact = source / relative
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            _link_regular_snapshot(
                source_artifact,
                destination,
                label=f"{label} artifact {relative}",
            )
            try:
                snapshot_digest = _sha256_regular_snapshot(
                    destination,
                    label=f"private {label} snapshot artifact",
                )
            except AdapterCompatibilityError as exc:
                raise GenerationInputError(str(exc)) from exc
            if snapshot_digest != digest:
                raise GenerationInputError(
                    f"private {label} snapshot differs from verified artifact {relative}"
                )
        if _regular_tree_inventory(snapshot, label=f"private {label} snapshot") != expected:
            raise GenerationInputError(f"private {label} snapshot inventory is incomplete")
        try:
            yield snapshot
        except BaseException:
            body_failed = True
            raise
        if _regular_tree_inventory(snapshot, label=f"private {label} snapshot") != expected:
            raise GenerationInputError(f"private {label} snapshot changed during generation")
    finally:
        current: os.stat_result | None
        try:
            current = os.lstat(snapshot)
        except OSError:
            current = None
        if current is not None and (
            stat.S_ISDIR(current.st_mode)
            and (current.st_dev, current.st_ino) == (snapshot_stat.st_dev, snapshot_stat.st_ino)
        ):
            shutil.rmtree(snapshot)
        elif current is not None and not body_failed:
            raise GenerationInputError(f"private {label} snapshot path changed before cleanup")


def _adapter_snapshot_inventory(provenance: AdapterProvenance) -> dict[str, str]:
    inventory = _regular_tree_inventory(provenance.adapter_path, label="PEFT adapter")
    required = {
        "adapter_config.json": provenance.config_sha256,
        **provenance.weight_sha256,
    }
    if any(inventory.get(name) != digest for name, digest in required.items()):
        raise GenerationInputError("PEFT adapter inventory changed after validation")
    return inventory


def _merged_snapshot_inventory(provenance: VerifiedMergedModel) -> dict[str, str]:
    merged = Path(provenance.merged_model)
    provenance_path = Path(provenance.provenance_path)
    expected = dict(provenance.artifact_sha256)
    expected[provenance_path.relative_to(merged).as_posix()] = provenance.provenance_sha256
    return dict(sorted(expected.items()))


def _device(torch_module: Any, *, force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch_module.cuda.is_available():
        return "cuda"
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _seed(torch_module: Any, seed: int) -> None:
    manual_seed = getattr(torch_module, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
    if torch_module.cuda.is_available():
        manual_seed_all = getattr(torch_module.cuda, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)


def _dtype(config: FineTuneConfig, libraries: GenerationLibraries) -> Any:
    configured = config.model.torch_dtype.casefold()
    if configured == "auto":
        return "auto"
    if configured in {"bfloat16", "bf16"}:
        return libraries.torch.bfloat16
    if configured in {"float16", "fp16", "half"}:
        return libraries.torch.float16
    return libraries.torch.float32


def _model_load_kwargs(
    config: FineTuneConfig,
    libraries: GenerationLibraries,
    *,
    allow_download: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dtype": _dtype(config, libraries),
        "device_map": None,
        "local_files_only": not allow_download,
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.revision:
        kwargs["revision"] = config.model.revision
    return kwargs


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render the same conversational prompt shape used by TRL training."""

    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def _completion_to_expected(completion: object) -> dict[str, Any]:
    value: object = completion
    if isinstance(completion, list) and completion:
        final = completion[-1]
        if isinstance(final, dict):
            value = final.get("content", final)
    if isinstance(value, str):
        try:
            decoded = loads_strict(value)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            raise ValueError("test completion does not contain valid expected JSON") from exc
        if isinstance(decoded, dict):
            return decoded
    if isinstance(value, dict):
        return value
    raise ValueError("test record must contain an expected object or JSON completion")


def _read_generation_records(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    try:
        payload = _read_regular_snapshot(path, label="generation dataset")
    except AdapterCompatibilityError as exc:
        raise ValueError(str(exc)) from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            "generation dataset bytes changed after manifest verification: "
            f"expected {expected_sha256}, observed {actual_sha256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: generation dataset is not UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = loads_strict(line)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise ValueError(f"{path}:{line_number}: invalid JSON: {detail}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def _load_generation_cohort(
    config: FineTuneConfig,
    *,
    dataset_path: Path | None = None,
) -> GenerationCohort:
    """Load the exact deterministic test rows plus all release split identities."""

    verification = verify_prepared_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
        required_splits=("train", "validation", "test"),
    )
    canonical_source = verification.split_paths["test"]
    source = dataset_path.expanduser().resolve() if dataset_path else canonical_source
    if source != canonical_source:
        raise DatasetIntegrityError(
            verification.manifest_path,
            "generation dataset must be the canonical manifest test split: "
            f"expected {canonical_source}, received {source}",
        )
    records = tuple(
        _read_generation_records(
            source,
            expected_sha256=verification.split_sha256["test"],
        )
    )
    ordered_ids: list[str] = []
    expected_json: list[bytes] = []
    for index, record in enumerate(records, 1):
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise DatasetIntegrityError(
                verification.manifest_path,
                f"test record {index} must contain a non-empty string ID",
            )
        expected_value = record.get("expected")
        expected = (
            expected_value
            if isinstance(expected_value, dict)
            else _completion_to_expected(record.get("completion"))
        )
        ordered_ids.append(identifier)
        expected_json.append(canonical_json_bytes(expected))
    expected_ids = verification.split_ids["test"]
    if tuple(ordered_ids) != expected_ids:
        raise DatasetIntegrityError(
            verification.manifest_path,
            "generation records do not match the verified ordered test IDs",
        )
    return GenerationCohort(
        verification=verification,
        records=records,
        ordered_ids=tuple(ordered_ids),
        expected_json=tuple(expected_json),
    )


def _validate_predictions_against_cohort(
    predictions: tuple[GeneratedPrediction, ...],
    cohort: GenerationCohort,
    *,
    label: str,
) -> None:
    """Bind generated in-memory IDs and expected objects to verified test bytes."""

    actual_ids = tuple(prediction.id for prediction in predictions)
    if actual_ids != cohort.ordered_ids:
        raise ValueError(f"{label} ordered IDs differ from the verified test records")
    actual_expected = tuple(canonical_json_bytes(prediction.expected) for prediction in predictions)
    if actual_expected != cohort.expected_json:
        raise ValueError(f"{label} expected objects differ from the verified test records")
    verification = cohort.verification
    for index, prediction in enumerate(predictions, 1):
        if (
            prediction.dataset_manifest_sha256 != verification.manifest_sha256
            or prediction.dataset_split_sha256 != verification.split_sha256["test"]
        ):
            raise ValueError(
                f"{label} prediction {index} differs from the verified test artifact identity"
            )
        if prediction.training_dataset_sha256 and (
            prediction.training_dataset_sha256.get("test") != verification.split_sha256["test"]
        ):
            raise ValueError(
                f"{label} prediction {index} training test hash differs from evaluation"
            )


def _generation_config_sha256(
    config: FineTuneConfig,
    verification: DatasetVerification,
) -> str:
    """Bind inference controls to the exact verified held-out split."""

    payload = {
        "schema_version": 1,
        "seed": config.seed,
        "model": config.model.model_dump(mode="json"),
        "generation": config.generation.model_dump(mode="json"),
        "dataset_manifest_sha256": verification.manifest_sha256,
        "dataset_split": "test",
        "dataset_split_sha256": verification.split_sha256["test"],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[-1])
    try:
        return len(value)
    except TypeError:
        return 0


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _open_output_parent(path: Path, *, label: str) -> tuple[Path, int]:
    """Open/create every parent component without following symbolic links."""

    destination = _lexical_absolute(path)
    if not destination.name:
        raise GenerationOutputError(f"{label} must name a file: {destination}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    anchor = Path(destination.anchor or os.sep)
    try:
        parent_descriptor = os.open(anchor, directory_flags)
    except OSError as exc:  # pragma: no cover - operating-system root failure
        raise GenerationOutputError(f"cannot open {label} filesystem root: {anchor}") from exc
    current = anchor
    try:
        relative_parts = destination.parent.relative_to(anchor).parts
        for part in relative_parts:
            current /= part
            try:
                child_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=parent_descriptor)
                try:
                    child_descriptor = os.open(
                        part,
                        directory_flags,
                        dir_fd=parent_descriptor,
                    )
                except OSError as exc:
                    raise GenerationOutputError(
                        f"{label} parent cannot be a symbolic link: {current}"
                    ) from exc
            except OSError as exc:
                raise GenerationOutputError(
                    f"{label} parent cannot be a symbolic link: {current}"
                ) from exc
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor

        try:
            existing = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise GenerationOutputError(f"{label} cannot be a symbolic link: {destination}")
            if not stat.S_ISREG(existing.st_mode):
                raise GenerationOutputError(f"{label} must be a regular file: {destination}")
        return destination, parent_descriptor
    except Exception:
        os.close(parent_descriptor)
        raise


def _validate_immutable_output_path(path: Path, *, label: str) -> Path:
    """Preflight an immutable output before expensive work begins."""

    destination, parent_descriptor = _open_output_parent(path, label=label)
    os.close(parent_descriptor)
    return destination


def _read_output_payload(parent_descriptor: int, name: str, *, label: str) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise GenerationOutputError(f"{label} cannot be a symbolic link: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GenerationOutputError(f"{label} must be a regular file: {name}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise GenerationOutputError(f"{label} changed while it was being read: {name}")
    return payload


def _write_immutable_payload(
    path: Path,
    payload: bytes,
    *,
    label: str,
    equivalent: Callable[[bytes, bytes], bool] | None = None,
    require_absent: bool = False,
) -> Path:
    """Publish bytes once through one no-follow parent directory descriptor."""

    destination, parent_descriptor = _open_output_parent(path, label=label)

    def acceptable(existing: bytes) -> bool:
        return existing == payload or bool(equivalent and equivalent(existing, payload))

    temporary_name = f".{destination.name}.{secrets.token_hex(16)}"
    descriptor: int | None = None
    try:
        existing = _read_output_payload(parent_descriptor, destination.name, label=label)
        if existing is not None:
            if require_absent:
                raise GenerationOutputError(f"refusing to reuse existing {label}: {destination}")
            if acceptable(existing):
                return destination
            raise GenerationOutputError(f"refusing to overwrite different {label}: {destination}")

        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            temporary_name,
            create_flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.fsync(parent_descriptor)
        except FileExistsError as exc:
            existing = _read_output_payload(parent_descriptor, destination.name, label=label)
            if existing is not None and not require_absent and acceptable(existing):
                return destination
            raise GenerationOutputError(
                f"refusing to overwrite different {label}: {destination}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)
    return destination


def _prediction_identity(payload: bytes) -> bytes | None:
    """Return semantic prediction bytes with non-deterministic latency omitted."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            decoded = loads_strict(line)
            row = GeneratedPrediction.model_validate(decoded)
        except (json.JSONDecodeError, StrictJSONError, ValueError):
            return None
        rows.append(row.model_dump(mode="json", exclude={"latency_ms"}))
    if not rows:
        return None
    return canonical_json_bytes(rows)


def _prediction_payloads_equivalent(existing: bytes, candidate: bytes) -> bool:
    return _prediction_identity(existing) == _prediction_identity(candidate)


def _write_predictions(path: Path, predictions: tuple[GeneratedPrediction, ...]) -> Path:
    payload = b"".join(
        canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in predictions
    )
    return _write_immutable_payload(
        path,
        payload,
        label="predictions",
        equivalent=_prediction_payloads_equivalent,
    )


def generate_predictions(
    config: FineTuneConfig,
    *,
    dataset_path: Path | None = None,
    adapter_path: Path | None = None,
    merged_model_path: Path | None = None,
    output_path: Path | None = None,
    allow_download: bool = False,
    require_training_manifest: bool = False,
    _libraries: GenerationLibraries | None = None,
) -> tuple[GeneratedPrediction, ...]:
    """Generate from a base, PEFT adapter, or verified local safe merge."""

    cohort = _load_generation_cohort(config, dataset_path=dataset_path)
    verification = cohort.verification
    if adapter_path is not None and merged_model_path is not None:
        raise AdapterCompatibilityError(
            "generation accepts either adapter_path or merged_model_path, not both"
        )
    if adapter_path is not None and adapter_path.expanduser().is_symlink():
        raise AdapterCompatibilityError("adapter directory cannot be a symbolic link")
    validated_output_path = (
        _validate_immutable_output_path(output_path, label="predictions output")
        if output_path is not None
        else None
    )
    adapter_provenance = (
        validate_adapter_compatibility(
            adapter_path,
            model_name_or_path=config.model.name_or_path,
            model_revision=config.model.revision,
            require_training_manifest=require_training_manifest,
        )
        if adapter_path is not None
        else None
    )
    merged_provenance: VerifiedMergedModel | None = (
        verify_merged_model(
            merged_model_path,
            expected_base_model=config.model.name_or_path,
            expected_model_revision=config.model.revision,
        )
        if merged_model_path is not None
        else None
    )
    lineage_manifest_sha256 = (
        adapter_provenance.training_manifest_sha256
        if adapter_provenance is not None
        else (merged_provenance.training_manifest_sha256 if merged_provenance is not None else None)
    )
    lineage_config_sha256 = (
        adapter_provenance.training_config_sha256
        if adapter_provenance is not None
        else (merged_provenance.training_config_sha256 if merged_provenance is not None else None)
    )
    lineage_dataset_sha256 = (
        adapter_provenance.training_dataset_sha256
        if adapter_provenance is not None
        else (
            dict(merged_provenance.training_dataset_sha256) if merged_provenance is not None else {}
        )
    )
    lineage_qualification_sha256 = (
        adapter_provenance.qualification_sha256
        if adapter_provenance is not None
        else (dict(merged_provenance.qualification_sha256) if merged_provenance is not None else {})
    )
    if require_training_manifest and (
        adapter_provenance is not None or merged_provenance is not None
    ):
        configured_sha256 = hashlib.sha256(
            canonical_json_bytes(config.model_dump(mode="json"))
        ).hexdigest()
        if lineage_manifest_sha256 is None or lineage_config_sha256 != configured_sha256:
            raise AdapterCompatibilityError(
                "release generation requires a matching completed training manifest and config"
            )
        verified_dataset_sha256: dict[str, str] = {
            "source": verification.source_sha256,
            "manifest": verification.manifest_sha256,
        }
        for split_name, digest in verification.split_sha256.items():
            verified_dataset_sha256[split_name] = digest
        if any(
            lineage_dataset_sha256.get(name) != digest
            for name, digest in verified_dataset_sha256.items()
        ):
            raise AdapterCompatibilityError(
                "training manifest source, prepared-manifest, or split hashes do not match "
                "generation"
            )
        qualification = config.data.qualification
        if qualification is not None and qualification.required:
            required_qualification = {
                "qualification_review_manifest",
                "qualification_report",
            }
            if not required_qualification.issubset(lineage_qualification_sha256):
                raise AdapterCompatibilityError(
                    "qualified release generation requires training-manifest qualification hashes"
                )
    records = cohort.records
    generation_config_sha256 = _generation_config_sha256(config, verification)
    libraries = _libraries or _load_generation_libraries()
    _seed(libraries.torch, config.seed)
    with ExitStack() as input_snapshots:
        model_source = config.model.name_or_path
        adapter_load_source: str | None = None
        if merged_provenance is not None:
            merged_source = Path(merged_provenance.merged_model)
            merged_snapshot = input_snapshots.enter_context(
                _verified_input_snapshot(
                    merged_source,
                    expected_inventory=_merged_snapshot_inventory(merged_provenance),
                    label="merged model",
                )
            )
            model_source = str(merged_snapshot)
        if adapter_provenance is not None:
            adapter_snapshot = input_snapshots.enter_context(
                _verified_input_snapshot(
                    adapter_provenance.adapter_path,
                    expected_inventory=_adapter_snapshot_inventory(adapter_provenance),
                    label="PEFT adapter",
                )
            )
            adapter_load_source = str(adapter_snapshot)

        tokenizer_kwargs = _model_load_kwargs(config, libraries, allow_download=allow_download)
        tokenizer_kwargs.pop("dtype")
        if merged_provenance is not None:
            tokenizer_kwargs.pop("revision", None)
            tokenizer_kwargs["local_files_only"] = True
            tokenizer_kwargs["trust_remote_code"] = False
        tokenizer = libraries.AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
        if "qwen" in config.model.name_or_path.casefold():
            tokenizer.eos_token = QWEN_EOS_TOKEN
        elif config.model.eos_token:
            tokenizer.eos_token = config.model.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = _model_load_kwargs(config, libraries, allow_download=allow_download)
        if merged_provenance is not None:
            model_kwargs.pop("revision", None)
            model_kwargs["local_files_only"] = True
            model_kwargs["trust_remote_code"] = False
        model = libraries.AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        if adapter_provenance is not None:
            if adapter_load_source is None:  # pragma: no cover - defensive invariant.
                raise GenerationInputError("PEFT adapter snapshot was not created")
            model = libraries.PeftModel.from_pretrained(
                model,
                adapter_load_source,
                is_trainable=False,
                local_files_only=not allow_download,
            )
        device = _device(libraries.torch, force_cpu=config.training.use_cpu)
        model.to(device)
        model.eval()

    results: list[GeneratedPrediction] = []
    for index, record in enumerate(records):
        prompt = record.get("prompt")
        if not isinstance(prompt, list) or not all(isinstance(item, dict) for item in prompt):
            raise ValueError(f"record {index + 1} has no conversational prompt list")
        messages = [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in prompt
        ]
        expected_value = record.get("expected")
        expected = (
            expected_value
            if isinstance(expected_value, dict)
            else _completion_to_expected(record.get("completion"))
        )
        prompt_text = render_prompt(tokenizer, messages)
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        input_length = _token_count(encoded["input_ids"])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
            "repetition_penalty": config.generation.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if config.generation.do_sample:
            generation_kwargs.update(
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
            )
        started = time.perf_counter()
        with libraries.torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        generated_ids = generated[0][input_length:]
        text = str(tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
        results.append(
            GeneratedPrediction(
                id=cohort.ordered_ids[index],
                expected=expected,
                prediction=text,
                latency_ms=latency_ms,
                prompt_tokens=input_length,
                generated_tokens=_token_count(generated_ids),
                model_name_or_path=config.model.name_or_path,
                model_revision=config.model.revision,
                dataset_manifest_sha256=verification.manifest_sha256,
                dataset_split_sha256=verification.split_sha256["test"],
                generation_config_sha256=generation_config_sha256,
                training_manifest_sha256=lineage_manifest_sha256,
                training_config_sha256=lineage_config_sha256,
                training_dataset_sha256=lineage_dataset_sha256,
                qualification_sha256=lineage_qualification_sha256,
                adapter_path=(str(adapter_provenance.adapter_path) if adapter_provenance else None),
                adapter_config_sha256=(
                    adapter_provenance.config_sha256 if adapter_provenance else None
                ),
                adapter_weight_sha256=(
                    adapter_provenance.weight_sha256 if adapter_provenance else {}
                ),
                merged_model_path=(
                    merged_provenance.merged_model if merged_provenance is not None else None
                ),
                merge_provenance_sha256=(
                    merged_provenance.provenance_sha256 if merged_provenance is not None else None
                ),
                merged_artifact_sha256=(
                    dict(merged_provenance.artifact_sha256) if merged_provenance is not None else {}
                ),
                merged_adapter_config_sha256=(
                    merged_provenance.adapter_config_sha256
                    if merged_provenance is not None
                    else None
                ),
                merged_adapter_weight_files=(
                    merged_provenance.adapter_weight_files if merged_provenance is not None else ()
                ),
                merged_adapter_weight_sha256=(
                    merged_provenance.adapter_weight_sha256 if merged_provenance is not None else ()
                ),
            )
        )
    if adapter_provenance is not None:
        adapter_after_generation = validate_adapter_compatibility(
            adapter_provenance.adapter_path,
            model_name_or_path=config.model.name_or_path,
            model_revision=config.model.revision,
            require_training_manifest=require_training_manifest,
        )
        if adapter_after_generation != adapter_provenance:
            raise AdapterCompatibilityError("adapter bytes or inventory changed during generation")
    if merged_provenance is not None:
        try:
            merged_after_generation = verify_merged_model(
                Path(merged_provenance.merged_model),
                expected_base_model=config.model.name_or_path,
                expected_model_revision=config.model.revision,
            )
        except ExportValidationError as exc:
            raise GenerationOutputError(
                "merged-model bytes or inventory changed during generation"
            ) from exc
        if merged_after_generation != merged_provenance:
            raise GenerationOutputError("merged-model bytes or inventory changed during generation")
    predictions = tuple(results)
    _validate_predictions_against_cohort(
        predictions,
        cohort,
        label="generated",
    )
    if validated_output_path is not None:
        _write_predictions(validated_output_path, predictions)
    return predictions
