from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from tickettune.deployment_proof import APPROVED_PRODUCTION_COMPOSE_SHA256

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "deploy" / "vllm" / "production"
COMPOSE_PATH = PRODUCTION / "compose.yaml"
NGINX_PATH = PRODUCTION / "nginx.conf"
ENTRYPOINT_PATH = PRODUCTION / "vllm-entrypoint.py"
PROMETHEUS_PATH = PRODUCTION / "prometheus.yml"
ALERTS_PATH = PRODUCTION / "alerts.yml"
ALERTMANAGER_PATH = PRODUCTION / "alertmanager.yml"
RELEASE_MANIFEST_PATH = PRODUCTION / "release-manifest.example.json"

VLLM_IMAGE = (
    "vllm/vllm-openai:v0.24.0@"
    "sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8"
)
NGINX_IMAGE = (
    "nginx:1.30.3-alpine@sha256:61f816887864af764259e28edb5331584ae2e42de57b3cb1e4f134d757559951"
)
PROMETHEUS_IMAGE = (
    "prom/prometheus:v3.13.0@"
    "sha256:0e698e35e50d1ddc2d11a4a55b089fe62eb71358a5c204dfafd21bdf8ffe04b8"
)
ALERTMANAGER_IMAGE = (
    "prom/alertmanager:v0.33.1@"
    "sha256:a89f8d4520954079275441eecdb71444328bd90633dd4eddfc33b9ed657f349b"
)


def test_approved_production_profile_compose_digest_is_current() -> None:
    assert hashlib.sha256(COMPOSE_PATH.read_bytes()).hexdigest() == (
        APPROVED_PRODUCTION_COMPOSE_SHA256
    )


def _compose() -> dict[str, object]:
    value = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _entrypoint() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tickettune_vllm_entrypoint", ENTRYPOINT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _volume(service: dict[str, object], target: str) -> dict[str, object]:
    volumes = service["volumes"]
    assert isinstance(volumes, list)
    return next(volume for volume in volumes if volume["target"] == target)


def _adapter_digest(root: Path) -> str:
    inventory: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    canonical = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _secret(path: Path, value: bytes) -> Path:
    path.write_bytes(value)
    path.chmod(0o440)
    assert stat.S_IMODE(path.stat().st_mode) == 0o440
    return path


def _entrypoint_environment(tmp_path: Path) -> tuple[dict[str, str], Path, str]:
    adapter = tmp_path / "adapter"
    (adapter / "nested").mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "nested" / "adapter_model.safetensors").write_bytes(b"weights")
    expected_digest = _adapter_digest(adapter)
    runtime_root = tmp_path / "runtime-cache"
    runtime_root.mkdir()
    secret = _secret(tmp_path / "api-key", b"A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6")
    environment = {
        "VLLM_API_KEY_FILE": str(secret),
        "SECRET_GROUP_ID": str(os.getgid()),
        "RELEASE_ID": "tickettune-quality-20260718-a",
        "RELEASE_GIT_REVISION": "a" * 40,
        "EXPECTED_ADAPTER_SHA256": expected_digest,
        "TICKETTUNE_ADAPTER_PATH": str(adapter),
        "TICKETTUNE_RUNTIME_CACHE_ROOT": str(runtime_root),
        "HOME": str(runtime_root / "home"),
        "XDG_CACHE_HOME": str(runtime_root / "xdg"),
        "VLLM_CACHE_ROOT": str(runtime_root / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime_root / "torchinductor"),
        "TRITON_CACHE_DIR": str(runtime_root / "triton"),
        "CUDA_CACHE_PATH": str(runtime_root / "cuda"),
    }
    return environment, adapter, expected_digest


def test_production_compose_pins_images_and_isolates_networks() -> None:
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)

    assert services["vllm"]["image"] == VLLM_IMAGE
    assert services["gateway"]["image"] == NGINX_IMAGE
    assert services["prometheus"]["image"] == PROMETHEUS_IMAGE
    assert services["alertmanager"]["image"] == ALERTMANAGER_IMAGE
    assert {service["platform"] for service in services.values()} == {"linux/amd64"}

    assert "ports" not in services["vllm"]
    assert "ports" not in services["prometheus"]
    assert "ports" not in services["alertmanager"]
    assert services["gateway"]["ports"] == ["${BIND_ADDRESS:-127.0.0.1}:${TLS_PORT:-8443}:8443"]

    networks = compose["networks"]
    assert networks["edge"].get("internal") is not True
    assert (
        networks["edge"]["driver_opts"]["com.docker.network.bridge.enable_ip_masquerade"] == "false"
    )
    assert networks["model_internal"]["internal"] is True
    assert networks["observability_internal"]["internal"] is True
    assert networks["notification_egress"].get("internal") is not True
    assert services["gateway"]["networks"] == [
        "edge",
        "model_internal",
        "observability_internal",
    ]
    assert services["vllm"]["networks"] == ["model_internal"]
    assert services["prometheus"]["networks"] == ["observability_internal"]
    assert services["alertmanager"]["networks"] == [
        "observability_internal",
        "notification_egress",
    ]


def test_file_secrets_use_required_supplementary_group() -> None:
    compose = _compose()
    services = compose["services"]
    secrets = compose["secrets"]
    assert isinstance(services, dict)
    assert isinstance(secrets, dict)

    assert set(secrets) == {
        "vllm_api_key",
        "tls_certificate",
        "tls_private_key",
        "alertmanager_webhook_url",
    }
    for secret in secrets.values():
        assert set(secret) == {"file"}
        assert secret["file"].startswith("${")

    group_expression = "${SECRET_GROUP_ID:?Set SECRET_GROUP_ID to the numeric host secret group}"
    assert services["vllm"]["group_add"] == [group_expression]
    assert services["gateway"]["group_add"] == [group_expression]
    assert services["alertmanager"]["group_add"] == [group_expression]
    assert "group_add" not in services["prometheus"]
    assert services["vllm"]["secrets"] == ["vllm_api_key"]
    assert set(services["gateway"]["secrets"]) == {"tls_certificate", "tls_private_key"}
    assert services["alertmanager"]["secrets"] == ["alertmanager_webhook_url"]


def test_services_are_non_root_read_only_and_have_effective_limits() -> None:
    services = _compose()["services"]
    assert services["gateway"]["user"] == "101:101"
    assert services["vllm"]["user"] == "2000:0"
    assert services["prometheus"]["user"] == "65534:65534"
    assert services["alertmanager"]["user"] == "65534:65534"

    for service in services.values():
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
        assert service["read_only"] is True
        assert service["pids_limit"] == service["deploy"]["resources"]["limits"]["pids"]
        assert service["cpus"] == service["deploy"]["resources"]["limits"]["cpus"]
        assert service["mem_limit"] == service["deploy"]["resources"]["limits"]["memory"]
        assert service["logging"]["driver"] == "local"
        assert service["logging"]["options"]["max-size"]
        assert service["logging"]["options"]["max-file"]


def test_vllm_binds_and_cache_environment_fail_closed() -> None:
    vllm = _compose()["services"]["vllm"]
    environment = vllm["environment"]

    assert vllm["entrypoint"] == ["python3", "/opt/tickettune/vllm-entrypoint.py"]
    assert "vllm-entrypoint.sh" not in json.dumps(vllm)
    assert "VLLM_API_KEY" not in json.dumps(environment)
    assert "--api-key" not in json.dumps(vllm["command"])
    assert environment["TICKETTUNE_ADAPTER_PATH"] == "/models/adapter"
    assert environment["TICKETTUNE_RUNTIME_CACHE_ROOT"] == "/runtime-cache"
    assert environment["HOME"] == "/runtime-cache/home"
    assert environment["XDG_CACHE_HOME"] == "/runtime-cache/xdg"
    assert environment["VLLM_CACHE_ROOT"] == "/runtime-cache/vllm"
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == "/runtime-cache/torchinductor"
    assert environment["TRITON_CACHE_DIR"] == "/runtime-cache/triton"
    assert environment["CUDA_CACHE_PATH"] == "/runtime-cache/cuda"

    adapter = _volume(vllm, "/models/adapter")
    hf_cache = _volume(vllm, "/home/vllm/.cache/huggingface")
    runtime_cache = _volume(vllm, "/runtime-cache")
    entrypoint = _volume(vllm, "/opt/tickettune/vllm-entrypoint.py")
    assert adapter["read_only"] is True
    assert hf_cache["read_only"] is True
    assert runtime_cache.get("read_only") is not True
    assert entrypoint["read_only"] is True
    for mount in (adapter, hf_cache, runtime_cache, entrypoint):
        assert mount["bind"]["create_host_path"] is False

    assert vllm["labels"]["com.tickettune.adapter-sha256"].startswith("${EXPECTED_ADAPTER_SHA256:")


def test_vllm_has_explicit_context_scheduler_and_log_limits() -> None:
    command = _compose()["services"]["vllm"]["command"]
    assert "--max-model-len" in command
    assert "--max-num-seqs" in command
    assert "--max-num-batched-tokens" in command
    assert "--disable-log-requests" in command
    assert "--disable-uvicorn-access-log" in command
    assert "--enable-lora" in command
    assert "--lora-modules" in command
    assert "--enable-lora-runtime-updating" not in command


def test_runtime_healthchecks_probe_listeners_not_config_files() -> None:
    services = _compose()["services"]
    assert "/health" in json.dumps(services["vllm"]["healthcheck"]["test"])
    assert "127.0.0.1:9091/metrics" in json.dumps(services["gateway"]["healthcheck"]["test"])
    assert "127.0.0.1:9090/-/ready" in json.dumps(services["prometheus"]["healthcheck"]["test"])
    assert "127.0.0.1:9093/-/ready" in json.dumps(services["alertmanager"]["healthcheck"]["test"])
    assert "promtool" not in json.dumps(services["prometheus"]["healthcheck"])
    assert "nginx -t" not in json.dumps(services["gateway"]["healthcheck"])


def test_nginx_tls_and_metrics_listeners_are_strictly_allowlisted() -> None:
    config = NGINX_PATH.read_text(encoding="utf-8")

    assert "listen 8443 ssl;" in config
    assert "ssl_protocols TLSv1.2 TLSv1.3;" in config
    assert "server_tokens off;" in config
    assert "limit_req_zone" in config
    assert config.count("limit_req zone=tickettune_api") == 2
    assert "client_max_body_size 64k;" in config
    assert "add_header X-Request-ID $request_id always;" in config
    assert "proxy_set_header Authorization $http_authorization;" in config
    assert "location = /v1/models" in config
    assert "location = /v1/chat/completions" in config

    assert "listen 9091;" in config
    assert config.count("location = /metrics") == 1
    assert "proxy_pass http://tickettune_vllm;" in config
    assert config.count("location /") == 2
    assert config.count("return 404;") == 2

    log_format = next(
        line for line in config.splitlines() if "log_format tickettune_redacted" in line
    )
    for sensitive in ("$request_body", "$http_authorization", "$args", "$request_uri"):
        assert sensitive not in log_format


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"a" * 31,
        b"a" * 31 + b" ",
        b"a" * 31 + b"\n",
        b"a" * 31 + b"\x00",
        b"a" * 31 + bytes([255]),
        b"a" * 4097,
    ],
)
def test_python_entrypoint_rejects_unsafe_api_keys(tmp_path: Path, payload: bytes) -> None:
    entrypoint = _entrypoint()
    secret = _secret(tmp_path / "api-key", payload)

    with pytest.raises(ValueError, match="API key"):
        entrypoint.load_api_key(secret, expected_group_id=os.getgid())


def test_python_entrypoint_enforces_group_mode_and_release_identity(tmp_path: Path) -> None:
    entrypoint = _entrypoint()
    secret = _secret(tmp_path / "api-key", b"A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6")

    assert entrypoint.load_api_key(secret, expected_group_id=os.getgid()).startswith("A1b2")
    secret.chmod(0o400)
    with pytest.raises(ValueError, match="0440"):
        entrypoint.load_api_key(secret, expected_group_id=os.getgid())
    secret.chmod(0o440)
    with pytest.raises(ValueError, match="group"):
        entrypoint.load_api_key(secret, expected_group_id=os.getgid() + 1)

    for release_id, revision, digest in (
        ("replace-with-release", "a" * 40, "b" * 64),
        ("valid-release", "0" * 40, "b" * 64),
        ("valid-release", "a" * 40, "0" * 64),
        ("valid-release", "not-a-sha", "b" * 64),
    ):
        with pytest.raises(ValueError):
            entrypoint.validate_release_identity(release_id, revision, digest)


def test_adapter_inventory_digest_is_canonical_and_rejects_unsafe_entries(
    tmp_path: Path,
) -> None:
    entrypoint = _entrypoint()
    adapter = tmp_path / "adapter"
    (adapter / "z").mkdir(parents=True)
    (adapter / "z" / "weights.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    assert entrypoint.adapter_inventory_sha256(adapter) == _adapter_digest(adapter)

    target = tmp_path / "outside"
    target.write_text("outside", encoding="utf-8")
    (adapter / "unsafe-link").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        entrypoint.adapter_inventory_sha256(adapter)
    (adapter / "unsafe-link").unlink()

    fifo = adapter / "unsafe-fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        entrypoint.adapter_inventory_sha256(adapter)


def test_entrypoint_preflights_caches_and_execs_without_secret_in_argv(
    tmp_path: Path,
) -> None:
    entrypoint = _entrypoint()
    environment, _adapter, expected_digest = _entrypoint_environment(tmp_path)
    captured: dict[str, object] = {}

    def fake_exec(program: str, argv: list[str], child_environment: dict[str, str]) -> None:
        captured.update(program=program, argv=argv, environment=child_environment)

    entrypoint.execute(["vllm", "serve", "model"], environment, exec_fn=fake_exec)

    assert captured["program"] == "vllm"
    assert captured["argv"] == ["vllm", "serve", "model"]
    child_environment = captured["environment"]
    assert isinstance(child_environment, dict)
    assert child_environment["VLLM_API_KEY"].startswith("A1b2")
    assert child_environment["EXPECTED_ADAPTER_SHA256"] == expected_digest
    assert child_environment["VLLM_API_KEY"] not in captured["argv"]
    for name in (
        "HOME",
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        assert Path(environment[name]).is_dir()


def test_prometheus_and_alertmanager_contracts() -> None:
    prometheus = yaml.safe_load(PROMETHEUS_PATH.read_text(encoding="utf-8"))
    rules = yaml.safe_load(ALERTS_PATH.read_text(encoding="utf-8"))
    alertmanager = yaml.safe_load(ALERTMANAGER_PATH.read_text(encoding="utf-8"))
    services = _compose()["services"]

    assert prometheus["rule_files"] == ["/etc/prometheus/alerts.yml"]
    assert prometheus["alerting"]["alertmanagers"][0]["static_configs"] == [
        {"targets": ["alertmanager:9093"]}
    ]
    jobs = {job["job_name"]: job for job in prometheus["scrape_configs"]}
    assert jobs["tickettune-vllm"]["static_configs"] == [{"targets": ["gateway:9091"]}]
    assert jobs["tickettune-vllm"]["metrics_path"] == "/metrics"
    assert jobs["tickettune-prometheus"]["static_configs"] == [{"targets": ["127.0.0.1:9090"]}]
    assert jobs["tickettune-alertmanager"]["static_configs"] == [{"targets": ["alertmanager:9093"]}]

    alerts = rules["groups"][0]["rules"]
    names = {alert["alert"] for alert in alerts}
    assert {
        "TicketTuneVllmDown",
        "TicketTuneRequestsWaiting",
        "TicketTuneAlertmanagerDown",
        "TicketTunePrometheusStorageCapacity",
    } <= names
    assert all(alert["for"] for alert in alerts)

    webhook = alertmanager["receivers"][0]["webhook_configs"][0]
    assert webhook["url_file"] == "/run/secrets/alertmanager_webhook_url"
    assert "url" not in webhook
    assert (
        "--storage.tsdb.retention.size=${PROMETHEUS_RETENTION_SIZE:-5GB}"
        in services["prometheus"]["command"]
    )


def test_examples_are_quality_profile_specific_and_fail_closed() -> None:
    values = {
        key: value
        for line in (PRODUCTION / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }

    assert values["BASE_MODEL"] == "Qwen/Qwen2.5-7B-Instruct"
    assert values["SERVED_MODEL_NAME"] == "tickettune-qwen-7b-quality"
    assert "artifacts/qwen-7b-quality/runs/replace-with-qualified-run-id" in values["ADAPTER_PATH"]
    assert values["RELEASE_ID"] == "replace-with-immutable-release-id"
    assert values["RELEASE_GIT_REVISION"] == "0" * 40
    assert values["EXPECTED_ADAPTER_SHA256"] == "0" * 64
    assert values["SECRET_GROUP_ID"].isdigit()


def test_release_manifest_example_and_readme_define_proof_boundaries() -> None:
    manifest = json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    readme = (PRODUCTION / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.lower().split())

    assert {
        "schema_version",
        "release_id",
        "project_name",
        "compose_file",
        "compose_sha256",
        "env_file",
        "env_sha256",
        "model",
        "adapter_path",
        "adapter_sha256",
    } <= set(manifest)
    assert manifest["schema_version"] == "2.0"
    assert manifest["compose_sha256"] == "0" * 64
    assert manifest["env_sha256"] == "0" * 64
    assert manifest["adapter_sha256"] == "0" * 64
    for evidence in (
        "dataset_manifest",
        "test_split",
        "training_manifest",
        "qualification_report",
        "evaluation_manifest",
        "evaluation_report",
        "parity_report",
    ):
        assert "replace-with" in manifest[f"{evidence}_file"]
        assert manifest[f"{evidence}_sha256"] == "0" * 64
    assert "configuration proof" in normalized_readme
    assert "does not prove" in normalized_readme
    assert "completed training manifest" in normalized_readme
    assert "prepared dataset manifest" in normalized_readme
    assert "exact prepared test split" in normalized_readme
    assert "passing dataset qualification" in normalized_readme
    assert "passing evaluation manifest" in normalized_readme
    assert "passing evaluation report" in normalized_readme
    assert "passing parity report" in normalized_readme
    assert "chgrp" in readme and "0440" in readme
    assert "create_host_path" in readme
    assert "sort_keys=True" in readme
    assert "separators=(',', ':')" in readme
    assert "ensure_ascii=False" in readme
    assert "target-host firewall" in readme.lower()
    assert "egress" in readme.lower()
    assert "promtool check config" in readme
    assert "amtool check-config" in readme
    assert ALERTMANAGER_IMAGE in readme


def test_entrypoint_help_and_syntax_are_runnable() -> None:
    result = subprocess.run(  # noqa: S603
        [os.sys.executable, str(ENTRYPOINT_PATH), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "vLLM" in result.stdout
