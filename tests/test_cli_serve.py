"""CLI tests for ``serve`` and the first-run flow (all external seams are monkeypatched)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ollama2cpp import config, constants, gguf, hardware, llamacpp, ollama, selector, server
from ollama2cpp.__main__ import app
from ollama2cpp.hardware import Device, HostInfo
from tests.gguf_builder import write_gguf

runner = CliRunner()
GB = 1024**3
GPU = Device("CUDA0", "Test GPU", 8192, 6000)
HOST = HostInfo(devices=(GPU,), ram_total_mib=32768, ram_free_mib=16000, physical_cores=8, logical_cores=16)
BUILD = "b1234"


def _chat(name: str, size: int, tag: str = "latest", **options: Any) -> tuple[str, str, int, dict[str, Any]]:
    """Spec for a model whose manifest declares ``size`` bytes and whose blob is a tiny valid GGUF."""
    return (name, tag, size, {"blob_writer": lambda path: write_gguf(path, n_layer=2), **options})


@dataclass
class Rig:
    """Recorded calls and switchable state for the monkeypatched seams."""

    root: Path
    binary: Path
    bench: Path
    host: HostInfo = HOST
    binary_present: bool = True
    running: tuple[str, str | None] = ("none", None)
    install_error: Exception | None = None
    installs: list[bool] = field(default_factory=list)
    starts: list[dict[str, Any]] = field(default_factory=list)
    stops: list[int] = field(default_factory=list)
    describes: list[tuple[str, int]] = field(default_factory=list)

    def invoke(self, *args: str) -> Any:
        """Run the CLI."""
        return runner.invoke(app, list(args))

    def save_config(self, **overrides: Any) -> config.Config:
        """Write a ready-to-use config (as if a first run happened)."""
        values: dict[str, Any] = {
            "ollama_path": str(self.root),
            "default_model": "big:latest",
            "llama_server": str(self.binary),
            "llama_bench": str(self.bench),
        }
        cfg = config.Config(**{**values, **overrides})
        config.save(cfg)
        return cfg

    def model(self, name: str) -> ollama.OllamaModel:
        """Return an installed model by name."""
        return ollama.resolve_model(self.root, name)


@pytest.fixture
def rig(fake_ollama: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """Build a fake Ollama store plus seams for llama.cpp, hardware and the server."""
    root = fake_ollama(
        [
            _chat("big", 6 * GB, projector=True),
            _chat("gemma4", 3 * GB, "12b"),
            _chat("tiny", 1 * GB),
            _chat("huge", 20 * GB),
        ]
    )
    binary = tmp_path / "bin" / "llama-server.exe"
    bench = tmp_path / "bin" / "llama-bench.exe"
    state = Rig(root=root, binary=binary, bench=bench)
    monkeypatch.setenv(constants.ENV_OLLAMA_MODELS, str(root))

    def find_binary(name: str, configured: str | None = None) -> Path | None:
        del configured
        if not state.binary_present:
            return None
        return state.binary if name == "llama-server" else state.bench

    def install(*, assume_yes: bool = False) -> Path:
        state.installs.append(assume_yes)
        if state.install_error is not None:
            raise state.install_error
        state.binary_present = True
        return state.binary

    def describe(host: str, port: int, base: Path | None = None) -> tuple[str, str | None]:
        del base
        state.describes.append((host, port))
        return state.running

    def start(argv: Any, **kwargs: Any) -> int:
        state.starts.append({"argv": list(argv), **kwargs})
        return 0

    def stop(base: Path | None = None, **_kwargs: Any) -> bool:
        del base
        state.stops.append(1)
        state.running = ("none", None)
        return True

    monkeypatch.setattr(llamacpp, "find_binary", find_binary)
    monkeypatch.setattr(llamacpp, "install", install)
    monkeypatch.setattr(llamacpp, "version", lambda _b: BUILD)
    monkeypatch.setattr(hardware, "host_info", lambda _b: state.host)
    monkeypatch.setattr(server, "describe_running", describe)
    monkeypatch.setattr(server, "start", start)
    monkeypatch.setattr(server, "stop", stop)
    monkeypatch.setattr(server, "port_in_use", lambda _h, _p: False)
    return state


def _no_ollama(*_args: object, **_kwargs: object) -> None:
    return None


def _argv(result_output: str) -> str:
    return " ".join(result_output.split())


# --- first run ---------------------------------------------------------------------------------------------------


def test_first_run_writes_config_and_picks_model(rig: Rig) -> None:
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == "big:latest"  # largest chat model that fits 8 GiB; huge does not
    assert cfg.ollama_path == str(rig.root)
    assert cfg.llama_server == str(rig.binary)
    assert cfg.llama_bench == str(rig.bench)
    assert "big:latest" in result.output
    assert "Traceback" not in result.output


def test_first_run_respects_model_flag(rig: Rig) -> None:
    result = rig.invoke("serve", "--dry-run", "--yes", "--model", "gemma4:12b")
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == "gemma4:12b"


def test_first_run_installs_when_binary_missing(rig: Rig) -> None:
    rig.binary_present = False
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 0, result.output
    assert rig.installs == [True]
    cfg = config.load()
    assert cfg is not None
    assert cfg.llama_server == str(rig.binary)


def test_first_run_install_without_yes_asks(rig: Rig) -> None:
    rig.binary_present = False
    assert rig.invoke("serve", "--dry-run").exit_code == 0
    assert rig.installs == [False]


def test_first_run_install_declined_exits_1_cleanly(rig: Rig) -> None:
    rig.binary_present = False
    rig.install_error = llamacpp.LlamaCppError("Installation declined. Install llama.cpp manually")
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 1
    assert "declined" in result.output
    assert "Traceback" not in result.output
    assert config.load() is None


def test_first_run_no_ollama_found(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "find_ollama_path", _no_ollama)
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 1
    assert "Ollama" in result.output
    assert "Traceback" not in result.output
    assert config.load() is None


def test_first_run_invalid_ollama_path_warns_and_falls_back(rig: Rig, tmp_path: Path) -> None:
    bad = tmp_path / "nowhere"
    result = rig.invoke("serve", "--dry-run", "--yes", "--ollama-path", str(bad))
    assert result.exit_code == 0, result.output
    assert "nowhere" in result.output
    assert "ignoring" in result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.ollama_path == str(rig.root)


def _second_store(rig: Rig) -> Path:
    """Build a second fake Ollama store that contains only ``alt:latest``."""
    other = rig.root.parent / "other-ollama"
    digest = "sha256:" + hashlib.sha256(b"alt").hexdigest()
    manifest = other / "manifests" / "registry.ollama.ai" / "library" / "alt" / "latest"
    manifest.parent.mkdir(parents=True)
    layer = {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": 2 * GB}
    manifest.write_text(json.dumps({"schemaVersion": 2, "layers": [layer]}), encoding="utf-8")
    (other / "blobs").mkdir()
    write_gguf(other / "blobs" / digest.replace(":", "-"), n_layer=2)
    return other


def test_first_run_explicit_ollama_path_used(rig: Rig) -> None:
    other = _second_store(rig)
    result = rig.invoke("serve", "--dry-run", "--yes", "--ollama-path", str(other))
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.ollama_path == str(other)  # a first run saves the flag
    assert cfg.default_model == "alt:latest"


def test_regular_run_valid_flag_is_per_run(rig: Rig) -> None:
    rig.save_config()
    other = _second_store(rig)
    result = rig.invoke("serve", "--dry-run", "--ollama-path", str(other))
    assert result.exit_code == 0, result.output
    assert "--alias alt:latest" in _argv(result.output)  # big:latest is absent there: chosen for this run only
    assert str(other / "blobs") in result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.ollama_path == str(rig.root)
    assert cfg.default_model == "big:latest"


def test_regular_run_invalid_flag_uses_configured_path(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.save_config()
    monkeypatch.setattr(ollama, "find_ollama_path", _no_ollama)  # discovery must not be needed
    result = rig.invoke("serve", "--dry-run", "--ollama-path", str(tmp_path / "bogus"))
    assert result.exit_code == 0, result.output
    assert "bogus" in result.output
    assert "ignoring" in result.output
    assert str(rig.root / "blobs") in result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.ollama_path == str(rig.root)


def test_regular_run_invalid_flag_and_configured_gone_rediscovers(rig: Rig, tmp_path: Path) -> None:
    rig.save_config(ollama_path=str(tmp_path / "moved-away"))
    result = rig.invoke("serve", "--dry-run", "--ollama-path", str(tmp_path / "bogus"))
    assert result.exit_code == 0, result.output
    assert "ignoring" in result.output
    assert "moved-away" in result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.ollama_path == str(rig.root)


def test_first_run_no_chat_model(fake_ollama: Callable[..., Path], rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = rig.root.parent / "empty-ollama"
    (empty / "manifests").mkdir(parents=True)
    monkeypatch.setenv(constants.ENV_OLLAMA_MODELS, str(empty))
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 1
    assert "ollama pull" in result.output
    assert "Traceback" not in result.output
    assert config.load() is None


def test_first_run_nothing_fits_warns_and_picks_smallest(rig: Rig) -> None:
    rig.host = HostInfo((Device("CUDA0", "Tiny GPU", 100, 100),), 32768, 16000, 8, 16)
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == "tiny:latest"
    assert "fits" in result.output


def test_first_run_cpu_only_uses_ram(rig: Rig) -> None:
    rig.host = HostInfo((), 32768, 16000, 8, 16)
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == "huge:latest"  # 20 GB fits 32 GiB RAM
    assert "-ngl 0" in _argv(result.output)


def test_first_run_model_not_found(rig: Rig) -> None:
    result = rig.invoke("serve", "--dry-run", "--yes", "--model", "nope")
    assert result.exit_code == 1
    assert "not found" in result.output
    assert "Traceback" not in result.output
    assert config.load() is None


# --- regular run -------------------------------------------------------------------------------------------------


def test_regular_run_builds_expected_argv(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    blob = rig.model("big").blob_path
    assert blob is not None
    out = _argv(result.output)
    assert "-m" in out
    assert str(blob) in out
    assert "--jinja" in out
    assert "--alias big:latest" in out
    assert "--host 127.0.0.1 --port 8080" in out
    assert "-c 8192" in out
    assert rig.starts == []
    assert rig.describes == []  # a dry run never inspects or stops a running server


def test_regular_run_starts_server_foreground(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("serve")
    assert result.exit_code == 0, result.output
    (call,) = rig.starts
    assert call["detach"] is False
    assert call["model"] == "big:latest"
    assert (call["host"], call["port"]) == ("127.0.0.1", 8080)
    assert "--jinja" in call["argv"]


def test_detach_prints_url(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("serve", "--detach", "--port", "9099", "--host", "0.0.0.0")
    assert result.exit_code == 0, result.output
    (call,) = rig.starts
    assert call["detach"] is True
    assert call["port"] == 9099
    assert "http://0.0.0.0:9099" in result.output


def test_uses_saved_profile_args(rig: Rig) -> None:
    model = rig.model("big")
    cfg = rig.save_config()
    assert model.digest is not None
    cfg.profiles[config.profile_key(model.digest, 8192)] = config.Profile(
        args=["-ngl", "7", "-t", "3"],
        ctx=6144,
        tg_tps=42.5,
        measured=True,
        fingerprint=hardware.fingerprint(HOST, GPU, BUILD),
    )
    config.save(cfg)
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    out = _argv(result.output)
    assert "-ngl 7 -t 3" in out
    assert "-c 6144" in out
    assert "optimized profile" in out
    assert "42.5" in out
    assert "stale" not in out


def test_unmeasured_profile_is_labelled(rig: Rig) -> None:
    model = rig.model("big")
    cfg = rig.save_config()
    assert model.digest is not None
    cfg.profiles[config.profile_key(model.digest, 8192)] = config.Profile(
        args=["-ngl", "5"], ctx=8192, fingerprint=hardware.fingerprint(HOST, GPU, BUILD)
    )
    config.save(cfg)
    result = rig.invoke("serve", "--dry-run")
    assert "unmeasured profile" in result.output
    assert "-ngl 5" in _argv(result.output)


def test_stale_fingerprint_warns_but_uses(rig: Rig) -> None:
    model = rig.model("big")
    cfg = rig.save_config()
    assert model.digest is not None
    cfg.profiles[config.profile_key(model.digest, 8192)] = config.Profile(
        args=["-ngl", "7"],
        ctx=8192,
        measured=True,
        tg_tps=10.0,
        fingerprint=hardware.fingerprint(HOST, GPU, "b1"),
    )
    config.save(cfg)
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "stale" in result.output
    assert "optimize" in result.output
    assert "-ngl 7" in _argv(result.output)


def test_no_profile_uses_seed_and_hints_optimize(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("serve", "--dry-run")
    assert "optimize" in result.output
    out = _argv(result.output)
    assert "-ngl 3 " in out  # 2 blocks + output layer, full offload on the tiny test GGUF
    assert "-fa on" in out


def test_ctx_without_profile_is_refit_and_reported(rig: Rig) -> None:
    rig.save_config()
    rig.host = HostInfo((Device("CUDA0", "Small GPU", 700, 700),), 32768, 16000, 8, 16)
    result = rig.invoke("serve", "--dry-run", "--ctx", "1000000")
    assert result.exit_code == 0, result.output
    assert "1000000" in result.output  # the notice names the requested context
    assert "-c 1000000" not in _argv(result.output)


def test_extra_args_after_double_dash_appended_last(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("serve", "--dry-run", "--", "--temp", "0.5", "--no-webui")
    assert result.exit_code == 0, result.output
    assert _argv(result.output).endswith("--temp 0.5 --no-webui")


def test_multi_gpu_pins_device(rig: Rig) -> None:
    rig.save_config()
    rig.host = HostInfo((GPU, Device("CUDA1", "Other GPU", 4096, 4096)), 32768, 16000, 8, 16)
    result = rig.invoke("serve", "--dry-run")
    assert "--device CUDA0" in _argv(result.output)


def test_configured_device_is_pinned(rig: Rig) -> None:
    rig.save_config(device="CUDA0")
    result = rig.invoke("serve", "--dry-run")
    assert "--device CUDA0" in _argv(result.output)


def test_single_gpu_not_pinned(rig: Rig) -> None:
    rig.save_config()
    assert "--device" not in rig.invoke("serve", "--dry-run").output


def test_low_free_vram_warns(rig: Rig) -> None:
    rig.save_config()
    rig.host = HostInfo((Device("CUDA0", "Busy GPU", 8192, 100),), 32768, 16000, 8, 16)
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "free" in result.output


def test_model_flag_used_for_this_run_only(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("serve", "--dry-run", "--model", "tiny")
    assert "--alias tiny:latest" in _argv(result.output)
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == "big:latest"


def test_default_model_no_longer_installed_is_reselected(rig: Rig) -> None:
    rig.save_config(default_model="removed:latest")
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "removed:latest" in result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == "big:latest"


def test_model_not_found_is_clean_error(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("serve", "--dry-run", "--model", "nope")
    assert result.exit_code == 1
    assert "not found" in result.output
    assert "Traceback" not in result.output


def test_vision_adds_projector(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("serve", "--dry-run", "--vision")
    projector = next(layer for layer in rig.model("big").layers if layer.media_type.endswith("projector"))
    expected = rig.root / "blobs" / projector.digest.replace(":", "-")
    assert f"--mmproj {expected}" in _argv(result.output)


def test_vision_without_projector_warns(rig: Rig) -> None:
    rig.save_config(default_model="tiny:latest")
    result = rig.invoke("serve", "--dry-run", "--vision")
    assert result.exit_code == 0, result.output
    assert "--mmproj" not in result.output
    assert "projector" in result.output


def test_vision_projector_blob_missing_warns(rig: Rig) -> None:
    rig.save_config()
    projector = next(layer for layer in rig.model("big").layers if layer.media_type.endswith("projector"))
    (rig.root / "blobs" / projector.digest.replace(":", "-")).unlink()
    result = rig.invoke("serve", "--dry-run", "--vision")
    assert result.exit_code == 0, result.output
    assert "--mmproj" not in result.output


# --- already running ---------------------------------------------------------------------------------------------


def test_already_running_same_model_exits_0(rig: Rig) -> None:
    rig.save_config()
    rig.running = ("ours", "big:latest")
    result = rig.invoke("serve")
    assert result.exit_code == 0, result.output
    assert "already running" in result.output
    assert rig.starts == []
    assert rig.stops == []


def test_already_running_same_model_hints_that_settings_may_differ(rig: Rig) -> None:
    rig.save_config()
    rig.running = ("ours", "big:latest")
    result = rig.invoke("serve", "--ctx", "4096")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "already running" in text
    assert "--restart" in text
    assert "differ" in text
    assert (rig.starts, rig.stops) == ([], [])


def test_restart_same_model_stops_and_starts_again_with_the_new_settings(rig: Rig) -> None:
    rig.save_config()
    rig.running = ("ours", "big:latest")
    result = rig.invoke("serve", "--restart", "--ctx", "4096")
    assert result.exit_code == 0, result.output
    assert rig.stops == [1]
    (call,) = rig.starts
    argv = call["argv"]
    assert argv[argv.index("-c") + 1] == "4096"  # the new settings, not the old server's
    assert call["model"] == "big:latest"


def test_restart_same_model_picks_up_a_new_profile(rig: Rig) -> None:
    cfg = rig.save_config()
    model = rig.model("big")
    assert model.digest is not None
    cfg.profiles[config.profile_key(model.digest, 8192)] = config.Profile(
        args=["-ngl", "7", "-t", "3"], ctx=8192, fingerprint=hardware.fingerprint(HOST, GPU, BUILD)
    )
    config.save(cfg)
    rig.running = ("ours", "big:latest")
    assert rig.invoke("serve", "--restart").exit_code == 0
    argv = rig.starts[0]["argv"]
    assert argv[argv.index("-ngl") + 1] == "7"


def test_restart_same_model_waits_for_the_port(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()
    rig.running = ("ours", "big:latest")
    monkeypatch.setattr(server, "port_in_use", lambda _h, _p: True)
    monkeypatch.setattr("ollama2cpp.workflow.PORT_FREE_TIMEOUT", 0.05)
    result = rig.invoke("serve", "--restart")
    assert result.exit_code == 1
    assert "still in use" in result.output
    assert rig.starts == []


def test_restart_same_model_with_a_foreign_listener_still_refuses(rig: Rig) -> None:
    rig.save_config()
    rig.running = ("foreign", "big:latest")
    result = rig.invoke("serve", "--restart")
    assert result.exit_code == 1
    assert (rig.stops, rig.starts) == ([], [])


def test_running_other_model_exits_1_without_restart(rig: Rig) -> None:
    rig.save_config()
    rig.running = ("ours", "tiny:latest")
    result = rig.invoke("serve")
    assert result.exit_code == 1
    assert "tiny:latest" in result.output
    assert "--restart" in result.output
    assert rig.starts == []
    assert rig.stops == []


def test_restart_replaces_our_server(rig: Rig) -> None:
    rig.save_config()
    rig.running = ("ours", "tiny:latest")
    result = rig.invoke("serve", "--restart")
    assert result.exit_code == 0, result.output
    assert rig.stops == [1]
    assert len(rig.starts) == 1


def test_restart_waits_for_port_and_gives_up(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()
    rig.running = ("ours", "tiny:latest")
    monkeypatch.setattr(server, "port_in_use", lambda _h, _p: True)
    monkeypatch.setattr("ollama2cpp.workflow.PORT_FREE_TIMEOUT", 0.05)
    result = rig.invoke("serve", "--restart")
    assert result.exit_code == 1
    assert "still in use" in result.output
    assert rig.starts == []


def test_restart_refuses_foreign_process(rig: Rig) -> None:
    rig.save_config()
    rig.running = ("foreign", None)
    result = rig.invoke("serve", "--restart")
    assert result.exit_code == 1
    assert "8080" in result.output
    assert "--port" in result.output
    assert rig.stops == []
    assert rig.starts == []
    assert "Traceback" not in result.output


def test_foreign_process_without_restart_exits_1(rig: Rig) -> None:
    rig.save_config()
    rig.running = ("foreign", "something")
    result = rig.invoke("serve")
    assert result.exit_code == 1
    assert rig.stops == []


# --- error discipline --------------------------------------------------------------------------------------------


def test_dangling_manifest_error_message(fake_ollama: Callable[..., Path], rig: Rig) -> None:
    fake_ollama([_chat("gone", 2 * GB)], missing_blob={"gone:latest"})
    rig.save_config()
    result = rig.invoke("serve", "--dry-run", "--model", "gone")
    assert result.exit_code == 1
    assert "blob missing" in result.output
    assert "ollama pull gone:latest" in result.output
    assert "Traceback" not in result.output


def test_corrupt_config_treated_as_first_run(rig: Rig) -> None:
    home = constants.home()
    home.mkdir(parents=True)
    (home / "config.json").write_text("{not json", encoding="utf-8")
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert (home / "config.json.bak").exists()
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == "big:latest"


def test_ollama_path_gone_rediscovered(rig: Rig, tmp_path: Path) -> None:
    rig.save_config(ollama_path=str(tmp_path / "moved-away"))
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "moved-away" in result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.ollama_path == str(rig.root)


def test_ollama_path_gone_and_nothing_found(rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config(ollama_path=str(tmp_path / "moved-away"))
    monkeypatch.setattr(ollama, "find_ollama_path", _no_ollama)
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_binary_missing_on_regular_run_installs(rig: Rig) -> None:
    rig.save_config()
    rig.binary_present = False
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 0, result.output
    assert rig.installs == [True]


def test_unexpected_exception_is_one_line_without_traceback(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()

    def boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(server, "start", boom)
    result = rig.invoke("serve")
    assert result.exit_code == 1
    assert "kaboom" in result.output
    assert "Traceback" not in result.output


def test_unexpected_exception_traceback_with_verbose(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()

    def boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(server, "start", boom)
    result = runner.invoke(app, ["-v", "serve"])
    assert result.exit_code == 1
    assert "kaboom" in result.output
    assert "Traceback" in result.output


def test_llamacpp_error_shows_its_message(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()

    def fail(*_a: object, **_k: object) -> int:
        raise llamacpp.LlamaCppError("port 8080 on 127.0.0.1 is held by another process")

    monkeypatch.setattr(server, "start", fail)
    result = rig.invoke("serve")
    assert result.exit_code == 1
    assert "held by another process" in result.output


def test_keyboard_interrupt_exits_130(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()

    def interrupt(*_a: object, **_k: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "start", interrupt)
    result = rig.invoke("serve")
    assert result.exit_code == 130
    assert "Traceback" not in result.output


# --- bare invocation ---------------------------------------------------------------------------------------------


def test_bare_invocation_runs_serve(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke()
    assert result.exit_code == 0, result.output
    assert len(rig.starts) == 1


def test_group_model_option_applies_to_serve(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("--model", "tiny")
    assert result.exit_code == 0, result.output
    assert rig.starts[0]["model"] == "tiny:latest"


def test_serve_model_option_wins_over_group(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("--model", "tiny", "serve", "--dry-run", "--model", "gemma4:12b")
    assert "--alias gemma4:12b" in _argv(result.output)


# --- OLLAMA_MODEL naming a model that is not installed ---------------------------------------------------------------

MISSING_ENV_MODEL = "qwen3.5:0.8b"


def test_first_run_with_a_missing_env_model_falls_back_to_auto_select(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, MISSING_ENV_MODEL)
    result = rig.invoke("serve", "--dry-run", "--yes")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert f"OLLAMA_MODEL={MISSING_ENV_MODEL} is not installed" in text
    assert "auto-selected" in text
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == "big:latest"
    assert "Traceback" not in result.output


def test_bare_first_run_with_a_missing_env_model_works(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, MISSING_ENV_MODEL)
    result = rig.invoke()
    assert result.exit_code == 0, result.output
    assert rig.starts[0]["model"] == "big:latest"


def test_regular_run_with_a_missing_env_model_uses_the_saved_default(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config(default_model="tiny:latest")
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, MISSING_ENV_MODEL)
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert f"OLLAMA_MODEL={MISSING_ENV_MODEL} is not installed" in text
    assert "saved default" in text
    assert "--alias tiny:latest" in _argv(result.output)


def test_bare_regular_run_with_a_missing_env_model_uses_the_saved_default(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.save_config(default_model="tiny:latest")
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, MISSING_ENV_MODEL)
    result = rig.invoke()
    assert result.exit_code == 0, result.output
    assert rig.starts[0]["model"] == "tiny:latest"


def test_regular_run_with_a_missing_env_model_and_no_default_auto_selects(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.save_config(default_model=None)
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, MISSING_ENV_MODEL)
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "auto-selected" in " ".join(result.output.split())
    assert "--alias big:latest" in _argv(result.output)


def test_env_model_that_is_installed_wins_over_the_saved_default(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, "tiny")
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "--alias tiny:latest" in _argv(result.output)
    assert "not installed" not in result.output


def test_model_flag_naming_a_missing_model_still_fails_even_with_the_env_set(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.save_config()
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, "tiny")
    for args in (("serve", "--dry-run", "--model", "nope"), ("--model", "nope", "serve", "--dry-run")):
        result = rig.invoke(*args)
        assert result.exit_code == 1, args
        assert "not found" in result.output
        assert "Available" in result.output


def test_model_flag_naming_a_missing_model_fails_on_a_first_run_too(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, "tiny")
    result = rig.invoke("serve", "--dry-run", "--yes", "--model", "nope")
    assert result.exit_code == 1
    assert "not found" in result.output
    assert config.load() is None


# --- profile context and validation notices ---------------------------------------------------------------------


def _save_profile(rig: Rig, *, ctx: int, keys: tuple[int, ...], **fields: Any) -> None:
    cfg = rig.save_config()
    model = rig.model("big")
    assert model.digest is not None
    profile = config.Profile(args=["-ngl", "7"], ctx=ctx, fingerprint=hardware.fingerprint(HOST, GPU, BUILD), **fields)
    for key in keys:
        cfg.profiles[config.profile_key(model.digest, key)] = profile
    config.save(cfg)


def test_profile_with_a_reduced_context_warns_about_the_difference(rig: Rig) -> None:
    _save_profile(rig, ctx=2048, keys=(8192,))
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "Using profile context 2048 instead of requested 8192" in text
    assert "optimize --ctx 8192" in text
    assert "--ctx 2048" in text
    assert "-c 2048" in _argv(result.output)


def test_profile_found_by_the_requested_key_warns_when_its_context_differs(rig: Rig) -> None:
    _save_profile(rig, ctx=2048, keys=(2048, 4096))
    result = rig.invoke("serve", "--dry-run", "--ctx", "4096")
    assert "Using profile context 2048 instead of requested 4096" in " ".join(result.output.split())
    assert "-c 2048" in _argv(result.output)


def test_profile_found_by_the_actual_key_is_silent(rig: Rig) -> None:
    _save_profile(rig, ctx=2048, keys=(2048, 4096))
    result = rig.invoke("serve", "--dry-run", "--ctx", "2048")
    assert result.exit_code == 0, result.output
    assert "Using profile context" not in result.output
    assert "-c 2048" in _argv(result.output)


def test_profile_with_the_requested_context_is_silent(rig: Rig) -> None:
    _save_profile(rig, ctx=8192, keys=(8192,))
    assert "Using profile context" not in rig.invoke("serve", "--dry-run").output


def test_unvalidated_profile_is_labelled(rig: Rig) -> None:
    _save_profile(rig, ctx=8192, keys=(8192,), measured=True, tg_tps=33.0, validated=False)
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "not validated (optimization was interrupted)" in text
    assert "optimized profile" in text  # still used like a measured one
    assert "-ngl 7" in _argv(result.output)


def test_validated_profile_has_no_validation_notice(rig: Rig) -> None:
    _save_profile(rig, ctx=8192, keys=(8192,), measured=True, tg_tps=33.0)
    assert "not validated" not in rig.invoke("serve", "--dry-run").output


# --- the low-VRAM warning compares against the need, not the file size ---------------------------------------


def test_low_free_vram_warning_compares_with_the_need_at_the_chosen_ctx(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.save_config()
    rig.host = HostInfo((Device("CUDA0", "Test GPU", 8192, 7123),), 32768, 16000, 8, 16)
    # the 6 GiB manifest is smaller than the 7123 MiB free, but the real need (weights + KV + overhead) is not
    assert rig.model("big").size // (1024 * 1024) < 7123
    monkeypatch.setattr(selector, "estimate_need_mib", lambda *_a, **_k: 7544)
    text = " ".join(rig.invoke("serve", "--dry-run").output.split())
    assert "Only 7123 MiB of VRAM is free" in text
    assert "7544 MiB" in text


def test_no_low_free_vram_warning_when_the_need_fits(rig: Rig) -> None:
    rig.save_config(default_model="tiny:latest")
    rig.host = HostInfo((Device("CUDA0", "Test GPU", 8192, 7123),), 32768, 16000, 8, 16)
    assert "of VRAM is free" not in rig.invoke("serve", "--dry-run").output


def test_no_low_free_vram_warning_when_the_model_is_larger_than_the_card_but_the_card_is_free(rig: Rig) -> None:
    rig.save_config(default_model="huge:latest")
    rig.host = HostInfo((Device("CUDA0", "Test GPU", 8192, 8192),), 32768, 16000, 8, 16)
    assert "of VRAM is free" not in rig.invoke("serve", "--dry-run").output


# --- an unreadable (not corrupt) config -------------------------------------------------------------------------


def test_unreadable_config_is_a_clean_error_and_the_file_stays(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()
    path = config.config_file()
    before = path.read_text(encoding="utf-8")
    real_read_text = Path.read_text

    def denied(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == constants.CONFIG_FILENAME:
            msg = "denied"
            raise PermissionError(msg)
        return real_read_text(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", denied)
        result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 1
    assert "Cannot read the configuration" in " ".join(result.output.split())
    assert "Traceback" not in result.output
    assert path.read_text(encoding="utf-8") == before
    assert not path.with_name("config.json.bak").exists()
    assert rig.installs == []  # no first run was started over the unreadable file


def test_low_free_vram_warning_is_skipped_when_the_gguf_header_cannot_be_read(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_profile(rig, ctx=8192, keys=(8192,))  # a saved profile means serve itself never needs the header
    rig.host = HostInfo((Device("CUDA0", "Busy GPU", 8192, 100),), 32768, 16000, 8, 16)

    def unreadable(_path: Path) -> None:
        msg = "bad header"
        raise gguf.GgufError(msg)

    monkeypatch.setattr(gguf, "read_gguf", unreadable)
    result = rig.invoke("serve", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "of VRAM is free" not in result.output
