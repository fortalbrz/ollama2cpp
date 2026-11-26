"""Tests for llama.cpp binary management and argv construction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ollama2cpp import llamacpp, ui
from ollama2cpp.llamacpp import LlamaCppError, ServeSettings, build_serve_argv, find_binary, install, update, version
from ollama2cpp.proc import ProcResult

WINGET_ARGV = [
    "winget",
    "install",
    "--id",
    "ggml.llamacpp",
    "-e",
    "--accept-package-agreements",
    "--accept-source-agreements",
]
RELEASES = "https://github.com/ggml-org/llama.cpp/releases"


def _fake_winget_tree(local: Path, name: str = "llama-server") -> Path:
    pkg = local / "Microsoft" / "WinGet" / "Packages" / "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
    pkg.mkdir(parents=True, exist_ok=True)
    exe = pkg / f"{name}.exe"
    exe.write_bytes(b"MZ")
    return exe


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp, "_is_windows", lambda: True)


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr(llamacpp.shutil, "which", lambda *_a, **_k: None)
    return local


def test_find_binary_prefers_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "custom" / "llama-server.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(llamacpp.shutil, "which", lambda *_a, **_k: str(tmp_path / "other"))
    assert find_binary("llama-server", str(exe)) == exe


def test_find_binary_configured_missing_falls_back_to_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    on_path = tmp_path / "bin" / "llama-server"
    monkeypatch.setattr(
        llamacpp.shutil, "which", lambda name, *_a, **_k: str(on_path) if name == "llama-server" else None
    )
    assert find_binary("llama-server", str(tmp_path / "nope")) == on_path


def test_find_binary_tries_exe_suffix_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: None
) -> None:
    on_path = tmp_path / "llama-server.exe"
    monkeypatch.setattr(
        llamacpp.shutil, "which", lambda name, *_a, **_k: str(on_path) if name.endswith(".exe") else None
    )
    assert find_binary("llama-server") == on_path


def test_find_binary_winget_glob(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = _isolate(monkeypatch, tmp_path)
    exe = _fake_winget_tree(local)
    assert find_binary("llama-server") == exe


def test_find_binary_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    assert find_binary("llama-server") is None


def test_find_binary_no_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("LOCALAPPDATA")
    assert find_binary("llama-server") is None


def test_install_declined_raises(monkeypatch: pytest.MonkeyPatch, windows: None) -> None:
    monkeypatch.setattr(ui, "confirm", lambda *_a, **_k: False)

    def boom(*_a: object, **_k: object) -> ProcResult:
        raise AssertionError("winget must not run")

    monkeypatch.setattr(llamacpp.proc, "run", boom)
    with pytest.raises(LlamaCppError, match="declined"):
        install()


def test_install_runs_winget_with_yes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, windows: None) -> None:
    local = _isolate(monkeypatch, tmp_path)
    calls: list[list[str]] = []
    asked: list[bool] = []

    def fake_confirm(question: str, *, assume_yes: bool = False) -> bool:
        asked.append(assume_yes)
        return True

    def fake_run(argv: Sequence[str], *, timeout: float | None = None) -> ProcResult:
        calls.append(list(argv))
        _fake_winget_tree(local)
        return ProcResult(0, "", "")

    monkeypatch.setattr(ui, "confirm", fake_confirm)
    monkeypatch.setattr(llamacpp.proc, "run", fake_run)
    found = install(assume_yes=True)
    assert calls == [WINGET_ARGV]
    assert asked == [True]
    assert found.name == "llama-server.exe"


def test_install_winget_missing(monkeypatch: pytest.MonkeyPatch, windows: None) -> None:
    monkeypatch.setattr(ui, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(llamacpp.proc, "run", lambda *_a, **_k: ProcResult(127, "", "not found"))
    with pytest.raises(LlamaCppError) as info:
        install(assume_yes=True)
    message = str(info.value)
    assert "winget install --id ggml.llamacpp -e" in message
    assert RELEASES in message


def test_install_winget_fails(monkeypatch: pytest.MonkeyPatch, windows: None) -> None:
    monkeypatch.setattr(ui, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(llamacpp.proc, "run", lambda *_a, **_k: ProcResult(1, "", "boom"))
    with pytest.raises(LlamaCppError, match="exit code 1"):
        install(assume_yes=True)


def test_install_binary_still_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, windows: None) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(ui, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(llamacpp.proc, "run", lambda *_a, **_k: ProcResult(0, "", ""))
    with pytest.raises(LlamaCppError, match="not found"):
        install(assume_yes=True)


def test_install_non_windows_never_calls_winget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp, "_is_windows", lambda: False)

    def boom(*_a: object, **_k: object) -> ProcResult:
        raise AssertionError("winget must not run")

    monkeypatch.setattr(llamacpp.proc, "run", boom)
    with pytest.raises(LlamaCppError) as info:
        install(assume_yes=True)
    assert "brew install llama.cpp" in str(info.value)
    assert RELEASES in str(info.value)


def test_update_runs_winget_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: Sequence[str], *, timeout: float | None = None) -> ProcResult:
        calls.append(list(argv))
        return ProcResult(0, "ok", "")

    monkeypatch.setattr(llamacpp.proc, "run", fake_run)
    assert update().returncode == 0
    assert calls == [["winget", "upgrade", "--id", "ggml.llamacpp", "-e"]]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            ProcResult(
                0,
                "",
                "load_backend: loaded RPC backend from x\nversion: 1.0.0-dev (build 12721, commit 79c69f4b0)\n"
                "built with Clang 22.1.8 for Windows x86_64\n",
            ),
            "b12721",
        ),
        (ProcResult(0, "version: 1.0.0-dev (build 4242, commit abc)\n", ""), "b4242"),
        (ProcResult(0, "", "load_backend: nothing useful\n"), "unknown"),
        (ProcResult(127, "", "missing"), "unknown"),
        (ProcResult(1, "version: 1.0.0-dev (build 1, commit a)", ""), "unknown"),
    ],
)
def test_version(monkeypatch: pytest.MonkeyPatch, result: ProcResult, expected: str) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: Sequence[str], *, timeout: float | None = None) -> ProcResult:
        seen.append(list(argv))
        return result

    monkeypatch.setattr(llamacpp.proc, "run", fake_run)
    assert version(Path("llama-server")) == expected
    assert seen == [[str(Path("llama-server")), "--version"]]


def _settings(model: Path, **overrides: object) -> ServeSettings:
    base: dict[str, object] = {"model_path": model, "alias": "m", "host": "127.0.0.1", "port": 8080, "ctx": 8192}
    base.update(overrides)
    return ServeSettings(**base)  # type: ignore[arg-type]


def test_build_argv_basic(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "m.gguf"
    argv = build_serve_argv(binary, _settings(model))
    assert argv[:3] == [str(binary), "-m", str(model)]
    assert "--jinja" in argv
    assert argv[3:] == ["--alias", "m", "--host", "127.0.0.1", "--port", "8080", "-c", "8192", "--jinja"]
    assert all(isinstance(a, str) for a in argv)


def test_build_argv_optional_flags(tmp_path: Path) -> None:
    mmproj = tmp_path / "proj.gguf"
    argv = build_serve_argv(tmp_path / "ls", _settings(tmp_path / "m.gguf", device="Vulkan1", mmproj=mmproj))
    assert argv[argv.index("--device") + 1] == "Vulkan1"
    assert argv[argv.index("--mmproj") + 1] == str(mmproj)
    assert argv.index("--jinja") < argv.index("--device") < argv.index("--mmproj")


def test_build_argv_path_with_spaces(tmp_path: Path) -> None:
    folder = tmp_path / "Users" / "Jörge Á" / "ollama models"
    model = folder / "sha256-abc def.gguf"
    mmproj = folder / "proj ü.gguf"
    binary = tmp_path / "Program Files" / "llama server.exe"
    argv = build_serve_argv(binary, _settings(model, mmproj=mmproj))
    assert argv.count(str(model)) == 1
    assert argv.count(str(mmproj)) == 1
    assert argv[0] == str(binary)
    assert argv[argv.index("-m") + 1] == str(model)
    assert len(argv) == 14


def test_extra_args_override_profile(tmp_path: Path) -> None:
    s = _settings(tmp_path / "m.gguf", profile_args=["-ngl", "20"], extra_args=["-ngl", "5"])
    argv = build_serve_argv(tmp_path / "ls", s)
    assert argv[-4:] == ["-ngl", "20", "-ngl", "5"]


def test_serve_settings_lists_not_shared(tmp_path: Path) -> None:
    a = _settings(tmp_path / "a")
    b = _settings(tmp_path / "b")
    a.profile_args.append("x")
    assert b.profile_args == []
    assert b.extra_args == []
