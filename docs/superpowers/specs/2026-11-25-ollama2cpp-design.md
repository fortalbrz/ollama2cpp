# ollama2cpp — design

Date: 2026-11-25 · Status: awaiting approval

## Intent and why

Serve models already pulled with Ollama through `llama-server` (llama.cpp), without
downloading or converting anything. **The goal is maximum generation speed
(tokens/s)** on the user's own hardware; Ollama is only the model store.
Console app, Typer + Rich, Windows first (the user's host: Windows 11, RTX 5060 Ti 8 GB
+ AMD iGPU, 7 Ollama models, `llama-server` 1.0.0-dev built with the **Vulkan** backend).

## Goals

1. First run with no config: ensure llama.cpp is installed, locate the Ollama models
   directory, choose the best model that fits VRAM (unless `--model` given), write the
   config, then continue as a normal run.
2. Normal run: if a server is already healthy, say so and exit; otherwise resolve the
   model's GGUF blob from the Ollama manifest and start
   `llama-server -m <models>/blobs/sha256-<digest> …`.
3. `optimize`: measure and save the fastest working `llama-server` settings per model.
4. `list`: show available Ollama models.
5. A few extra commands (below).

## Non-goals / anti-goals

- No model downloading, conversion, or writing into the Ollama directory (read-only).
- No chat UI, no proxying of the Ollama API.
- Not a failure-prone "clever" optimizer: it must never save a config that crashes or
  OOMs; heuristic-only results are labelled as unmeasured.
- v1 does not use the manifest's `draft` (speculative decoding) or `projector` (vision)
  layers; `info` shows that they exist, and `serve --vision` may add `--mmproj`
  (off by default because it costs VRAM). Speculative decoding is a later candidate.
- Linux/macOS: code stays portable (pathlib, no Windows-only calls except behind
  `sys.platform`), but auto-install is Windows/winget only; elsewhere print install
  instructions.

## Approaches considered (optimizer)

| | How | Trade-off |
|-|-|-|
| A. Heuristic only | Rules from GGUF header + VRAM/RAM/cores | Instant, no benchmark; settings are guesses |
| B. Benchmark only | `llama-bench` sweep | Real numbers; slow, can OOM on bad bounds |
| **C. Hybrid (chosen by user)** | Heuristic gives safe bounds and the seed; short `llama-bench` sweep around it; save the winner | Most code, but measures tokens/s on this host and cannot try OOM configs |

## Design

### Layout (all absolute imports, per `ruff.toml`)

`src/ollama2cpp/` (move the stray `src/__main__.py` and `src/__init__.py` in here):

| Module | Responsibility |
|-|-|
| `__main__.py` | Typer app, commands, global options (`--model/-m`, `--verbose`, `--version`). No-subcommand = `serve`. |
| `constants.py` | Defaults (already exists; fix its docstring). |
| `config.py` | Load/save `DEFAULT_PATH/config.json` (atomic write). Holds `ollama_path`, `default_model`, `llama_server` path, host/port, per-model profiles. |
| `ollama.py` | Locate Ollama dir; enumerate manifests (`manifests/<registry>/<namespace>/<name>/<tag>`); parse layers; resolve `application/vnd.ollama.image.model` digest to `blobs/sha256-<hex>` (blob name uses `-` for `:`). |
| `gguf.py` | Minimal GGUF header reader (no deps): architecture, block count, context length, KV heads, head dim, embedding/pooling flag, file type. |
| `hardware.py` | Devices, VRAM total/free, RAM, physical/logical cores, free-space. |
| `llamacpp.py` | Find/install/update `llama-server`/`llama-bench`; build argv; health check; spawn/stop; pid file. |
| `selector.py` | Best-model-that-fits logic. |
| `optimizer.py` | Heuristic bounds + `llama-bench` sweep + profile persistence. |
| `ui.py` | Rich console, tables, panels, spinners/progress, notices. Plain strings (drop the copied `t()` i18n scaffolding). |

### Ollama path discovery (in order)
`--ollama-path` option → `$OLLAMA_MODELS` → `~/.ollama/models` → `%LOCALAPPDATA%\Ollama\models`
→ `/usr/share/ollama/.ollama/models`. First one containing `manifests/` wins. The
stored config path is re-validated each run (if gone, rediscover).

### Hardware and model selection
- VRAM source: `llama-server --list-devices` (works for Vulkan/CUDA/Metal, reports per
  device total/free), falling back to `nvidia-smi --query-gpu=…`. On multi-device hosts
  (here NVIDIA + AMD iGPU) pin the device with the most VRAM via `--device`
  and never split onto the iGPU unless the user asks (`--device` override stored in config).
- Selection uses **total** VRAM, with a warning when free VRAM is much lower
  (Ollama or another process is holding it; `status` shows this).
- Candidate = manifest with a model layer, GGUF not embedding/pooling, excluding
  names/archs like `bert`, `nomic-bert`, `*-embed*`. Estimated need =
  weights + KV cache at the default context (`n_layer·2·n_ctx·n_kv_heads·head_dim·2 B`)
  + ~10 % / 512 MiB compute overhead. Pick the **largest candidate that fits fully
  in VRAM** (full offload is the biggest speed factor); if none fit, pick the smallest
  chat model and warn it will be partially offloaded/CPU-bound. `--model` always wins.
  Model names accept `name[:tag]`; missing tag = `latest`.

### Config file
`DEFAULT_PATH` (`~/.ollama2cpp`) `/config.json`, JSON (no extra dependency; `tomllib`
can't write). Contains the fields above plus
`profiles[<digest>] = {args, tg_tps, pp_tps, measured: bool, host_fingerprint, llama_build, timestamp}`.
Unknown keys are preserved; corrupt file → back up as `.bak`, rediscover, never crash.
First run is detected by absence of this file.

### Commands

| Command | Behavior |
|-|-|
| *(none)* / `serve` | The flow in Goals. Options: `--model`, `--host`, `--port` (default 127.0.0.1:8080), `--ctx`, `--detach`, `--restart`, `--dry-run` (print command), `--vision`, extra llama args after `--`. Saved profile args are applied unless overridden. Runs foreground, streaming server output, Ctrl-C stops cleanly. |
| `list` | Rich table: name:tag, size, quant, arch, ctx, kind (chat/embedding), "fits VRAM?", optimized? (`--json` too). |
| `optimize [MODEL]` | Hybrid optimizer (below). `--quick` = heuristic only, `--all` = every chat model, `--time-limit`, `--ctx`. |
| `status` | Server running? (HTTP `/health` + pid file), config, GPU/VRAM free vs total, saved profile. |
| `stop` | Stop the server started by this tool (pid file). |
| `info MODEL` | GGUF details, layers (projector/draft/template), VRAM estimate, saved profile. |
| `doctor` | Checks llama.cpp (version, backend), Ollama dir, GPU, config; suggests fixes. |
| `update` | `winget upgrade ggml.llamacpp`. |
| `config` | Show config and its path; `--set key=value`; `--reset`. |

Install step uses `winget install --id ggml.llamacpp -e --accept-package-agreements
--accept-source-agreements` when `llama-server` is absent (PATH, then the winget
package dir); `upgrade` only via `update`. Newly installed binaries may not be on the
current PATH: search the winget packages dir and store the absolute path in config.

### Optimizer (C)
1. **Heuristic seed** from GGUF + hardware: `-ngl` (all layers if the VRAM estimate fits,
   else the largest count that does), `-c` (user/default 8192, reduced to fit),
   `--flash-attn on` when the backend supports it, `-t` = physical cores,
   `-b/-ub` 2048/512, KV cache `f16`.
2. **Sweep** with `llama-bench -o json` (`-p 512 -n 128`, few repetitions) using
   coordinate descent around the seed, one parameter at a time, not a full grid:
   `-fa {0,1}`, `-ub {256,512,1024,2048}`, `-t` around physical cores,
   `-ngl` (try seed and seed±small steps when partially offloaded), optional KV cache
   quantisation `q8_0` only with `--allow-kv-quant` (affects quality, so opt-in).
3. **Objective:** maximize generation speed (tg tokens/s); prompt speed is the tie-breaker.
4. A candidate that fails/OOMs/times out is logged and skipped (never aborts the run);
   if every measurement fails, save the heuristic seed with `measured: false`.
5. Save the winner as the model's profile (with the benchmark table kept in the config);
   `serve` uses it automatically. A fingerprint mismatch (GPU/VRAM/RAM/llama.cpp build
   changed) shows a "profile is stale, re-run optimize" notice but still uses it.
6. Rich progress bar per candidate and a final before/after table. Ctrl-C keeps the best
   result so far.

### Cross-cutting
- Subprocess calls through one thin wrapper (list-form argv, timeouts, no `shell=True`),
  so tests mock a single seam.
- Logging via `%`-style `logging`; user-facing messages via `ui`. Failure of one model
  during `list`/`optimize --all` is logged and shown with a sentinel row, never aborts.
- Server ports/hosts validated; default bind 127.0.0.1.

### Testing and tooling
- pytest with fixtures that build a fake Ollama tree and tiny synthetic GGUF files;
  `subprocess` wrapper and HTTP health check are monkeypatched. Typer `CliRunner`
  for commands. Gate: coverage ≥ 90 % on `src/ollama2cpp`.
- Housekeeping fixes: `pyproject.toml` coverage target `src/llmopenfinance` →
  `src/ollama2cpp`; `ruff.toml` `known-first-party` → `ollama2cpp`; keywords; add
  `psutil` dependency (RAM/cores/process control); add a README.
- Verification order per CLAUDE.md: `ruff format`, `ruff check --fix`, `pyrefly check`,
  `pytest -x`.
- A live smoke test (marked `live`, deselected by default) against the user's real
  Ollama directory and `llama-server`.

### Behaviors settled after design review

- **Templates/params/compatibility.** `serve` always passes `--jinja` and relies on the
  chat template embedded in the GGUF; Ollama's `template`, `params`, `system` and
  `license` layers are ignored in v1 (shown by `info`). A blob whose GGUF header can't
  be read or whose architecture the installed `llama-server` rejects is marked
  "unsupported" in `list`, skipped by the selector, and a failed startup reports the
  server's last log lines plus a hint to run `update`. No automatic load-and-retry fallback.
- **Optimizer validation.** `llama-bench` does not exercise the real server path, so after
  the sweep the winner is launched as a real `llama-server` at the serving context,
  waits for `/health`, and runs one short completion. On failure it steps down once
  (smaller `-c`, then lower `-ngl`) and re-validates; if that fails too it saves the
  heuristic seed as `measured: false`. Profiles are keyed by `(digest, ctx)`;
  `serve --ctx` with no matching profile recomputes a safe heuristic fit and says so.
- **Already-running server.** Identified via `/props` (model path/alias) plus the pid
  file. Same model: report and exit 0. Different model: warn and exit non-zero unless
  `--restart`. `--restart`/`stop` only stop a process recorded in our pid file; a foreign
  process on the port is never killed (report who holds the port instead).
- **VRAM held by other processes (e.g. Ollama).** Never stops anything by default. `optimize`
  has a pre-flight check: if free VRAM can't hold the model plus overhead it shows the
  shortfall and refuses; `--unload-ollama` (explicit opt-in) asks Ollama to unload models
  (`keep_alive: 0` via its local API) and waits for VRAM to free. Selection (`serve` first
  run) uses total VRAM; `serve` itself warns if free VRAM is too low.
- **Installing llama.cpp.** Ask once for confirmation (`--yes` or non-interactive skips
  the question and proceeds); if `winget` is missing or fails, print manual install
  instructions and exit with a clear error.

## Calls I made (for you to check)

- Ollama's template/params layers ignored in v1; `--jinja` always on.
- Optimizer ends with a real-server validation, and profiles are per `(model, ctx)`.
- Never kills a process this tool didn't start; Ollama is only unloaded with `--unload-ollama`.
- Install of llama.cpp asks for confirmation first (`--yes` to skip).
- JSON config instead of TOML; plain strings instead of i18n; `psutil` dependency.
- Default context 8192; default bind `127.0.0.1:8080`.
- `winget install` (not `upgrade`) when absent, separate `update` command.
- Embeddings excluded from "best model"; vision and draft layers unused in v1.
- KV-cache quantisation opt-in only.
- `AGENTS.md` (referenced by CLAUDE.md) does not exist; not creating it as part of this work.

## Left to the builder
Exact JSON schema field names, GGUF parsing details, the exact sweep step values and
time budgets, Rich layout/styling, error-message wording, and how `--list-devices`
output is parsed on each backend.
