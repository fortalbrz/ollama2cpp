# ollama2cpp

Start `llama-server` (llama.cpp) on the GGUF files of the models you already pulled with Ollama.

Ollama is convenient, but llama.cpp usually generates tokens faster once its settings fit your GPU
(layers offloaded, flash attention, threads, batch sizes). `ollama2cpp` reuses Ollama's model store, so
there is nothing to download or convert, picks sensible llama-server settings, and can benchmark your
machine to find the fastest ones. You get an OpenAI-compatible server at `http://127.0.0.1:8080`.

The Ollama directory is only ever read. Nothing is copied, moved or modified there.

## Requirements

- Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/).
- Ollama installed, with at least one model pulled (`ollama pull qwen3.5:9b`).
- llama.cpp (`llama-server`, `llama-bench`). On Windows the first run offers to install it with
  `winget install ggml.llamacpp`; elsewhere install it yourself and put it on `PATH`.

## Install and run

```
uv sync
uv run ollama2cpp --help
```

To get an `ollama2cpp` command on your `PATH`:

```
uv tool install .
ollama2cpp --help
```

## Quick start

```
uv run ollama2cpp            # same as `ollama2cpp serve`
```

The first run, in order:

1. Installs llama.cpp with winget if `llama-server` is not found (asks first; `--yes` skips the question).
2. Finds the Ollama models directory (`--ollama-path`, `$OLLAMA_MODELS`, `~/.ollama/models`,
   `%LOCALAPPDATA%\Ollama\models`, the Linux service path).
3. Picks the largest chat model that fits your VRAM (or the one you named with `--model`).
4. Writes `~/.ollama2cpp/config.json` and prints a summary.
5. Starts `llama-server` in the foreground. Ctrl-C stops it.

Later runs reuse the saved configuration. If the model has a saved profile (see `optimize`) its settings are
used, otherwise a heuristic seed is. Use `serve --dry-run` to see the exact command without running it.

## Commands

Global options (before the command): `-v/--verbose`, `-m/--model NAME` (also read from `$OLLAMA_MODEL`),
`--version`.

| Command | What it does | Options |
| --- | --- | --- |
| `serve` (default) | Start llama-server on a model. Extra llama-server arguments go after `--`. | `-m/--model`, `--ollama-path`, `--host`, `--port`, `--ctx`, `--detach`, `--restart`, `--dry-run`, `--vision`, `-y/--yes` |
| `list` | Installed models: kind, size, quantisation, whether it fits VRAM, optimized or not. | `--json` |
| `info MODEL` | GGUF details, layers, VRAM need and saved profiles of a model. | |
| `status` | Server state, config path, GPU memory, default model's profile. | |
| `stop` | Stop the server that ollama2cpp started. | |
| `doctor` | Check llama.cpp, Ollama, GPU, free VRAM and config, with fix hints. | |
| `update` | Upgrade llama.cpp with winget. | |
| `config` | Show the configuration and its path. | `--set KEY=VALUE` (repeatable), `--reset`, `-y/--yes` |
| `optimize [MODEL]` | Find the fastest settings for a model and save them. | `--quick`, `--all`, `--time-limit SECONDS`, `--ctx`, `--allow-kv-quant`, `--unload-ollama`, `--stop-server`, `-y/--yes` |
| `ver` | Print the version. | |

Examples:

```
uv run ollama2cpp serve --model gemma4:12b --ctx 16384 --port 9000
uv run ollama2cpp serve --detach              # return once the server is ready
uv run ollama2cpp serve -- --threads 8        # extra llama-server arguments
uv run ollama2cpp stop
```

Settable config keys: `host`, `port`, `device`, `ollama_path`, `default_model`, `llama_server`.

Sample `list` output (trimmed):

```
Ollama models in C:\Users\you\.ollama\models
┌───────────────────┬──────────┬────────┬────────┬────────┬─────────────┬───────────┬───────────┐
│ Name              │ Size GiB │ Quant  │ Arch   │ Ctx    │ Kind        │ Fits VRAM │ Optimized │
├───────────────────┼──────────┼────────┼────────┼────────┼─────────────┼───────────┼───────────┤
│ gemma4:12b        │ 6.9      │ Q4_K_M │ gemma4 │ 262144 │ chat        │ no        │ no        │
│ lfm3-audio:2b     │ 1.0      │ Q4_K_M │ lfm3   │ 128000 │ unsupported │ -         │ -         │
│ qwen3-embedding:4b│ 2.3      │ Q4_K_M │ qwen3  │ 40960  │ embedding   │ -         │ -         │
│ qwen3.5:9b-q4_k_m │ 5.2      │ Q4_K_M │ qwen35 │ 262144 │ chat        │ yes       │ no        │
└───────────────────┴──────────┴────────┴────────┴────────┴─────────────┴───────────┴───────────┘
```

(The real table also has a `Note` column. `list --json` prints only a JSON array on stdout.)

## How `optimize` works

```
uv run ollama2cpp optimize qwen3.5:9b-q4_k_m
```

1. Heuristic seed: from the GGUF metadata and your hardware it computes layers to offload, flash attention,
   thread count and batch sizes.
2. `llama-bench` sweep: it benchmarks variations around the seed, within `--time-limit` seconds per model
   (default 300), and keeps the one with the best generation speed (tokens/s).
3. Real-server validation: the winner is started in a real `llama-server` and must come up and answer
   before it is accepted (a failure steps down once, then the unmeasured seed is saved).
4. The result is saved per model and context size in the config file; `serve` uses it automatically.

Options:

- `--quick`: skip the benchmark and save the heuristic seed (unmeasured). Seconds, not minutes.
- `--all`: optimize every chat model, one after another (a failure on one does not stop the rest).
- `--ctx N`: context size to optimize for (default 8192). Profiles are per model and context size.
- `--allow-kv-quant`: also try a `q8_0` KV cache. Faster or smaller, but it can affect output quality, so it is off by default.
- `--unload-ollama`: if Ollama holds the VRAM the model needs, ask it to unload its models first.
- `--stop-server`: stop the ollama2cpp server that is running (it holds VRAM) before benchmarking.

The sweep optimizes for the VRAM that is free right now. A model that does not fit completely is benchmarked
with a partial offload (the number of GPU layers is part of the sweep), and a model larger than the card is
always a partial offload. `optimize` refuses only when other processes hold a lot of VRAM (more than 1 GiB) and
the model needs memory that is not free; it then says how much is in use and how to free it (see
Troubleshooting). Ctrl-C keeps the best result measured so far: it is saved without the final validation and
`serve` says so.

## Configuration

Stored in `~/.ollama2cpp/config.json` (set `OLLAMA2CPP_HOME` to use another directory). It holds the
paths of `llama-server`/`llama-bench` and the Ollama directory, the default model, host, port, an optional
pinned `device`, and the saved `profiles`. Writes are atomic and a broken file is moved to `config.json.bak` and treated as missing, never a crash.
`ollama2cpp config` shows it; `config --set port=9000` edits it; `config --reset` deletes it.

Environment variables:

| Variable | Effect |
| --- | --- |
| `OLLAMA_MODELS` | Ollama models directory (same variable Ollama uses). |
| `OLLAMA_MODEL` | Default model name, as if `--model` were given. |
| `OLLAMA2CPP_HOME` | Where ollama2cpp keeps its config and server pid file. |

## Safety

- The Ollama directory is read-only for ollama2cpp.
- `stop`, `--restart` and `--stop-server` only affect a server that ollama2cpp started itself (tracked in
  its own pid file). Other llama-server processes, and Ollama, are never killed.
- `--unload-ollama` only asks Ollama to unload models through its API; it does not stop Ollama.

## Troubleshooting

- Run `uv run ollama2cpp doctor` first; it names the failing check and a fix.
- `optimize` says VRAM is short: Ollama keeps models loaded for a while after use. Run `ollama stop MODEL`
  or pass `--unload-ollama`. Also close other GPU users, or stop our own server (`stop` / `--stop-server`).
- Several GPUs (for example an NVIDIA card plus an integrated AMD GPU under Vulkan): `doctor` and `status`
  list them. ollama2cpp picks the best one; to pin another, use `config --set device=Vulkan1`
  (ids as printed by `llama-server --list-devices`). `config --set device=none` goes back to automatic.
- `Model 'x' not found`: a model you name with `--model` (or as the `optimize` argument) must be installed; the
  error lists the installed models. An `OLLAMA_MODEL` that names a model you do not have is not an error:
  ollama2cpp warns and falls back to the saved default, then to the automatic choice. Unset it to silence
  the warning.
- `llama-bench was not found` (`optimize` without `--quick`): run `ollama2cpp doctor` to see what is missing, then
  `ollama2cpp update` (or install the full llama.cpp release that ships `llama-bench`).
- `serve` says the server is already running: its settings may differ from the current profile or options (for
  example after `optimize`); run `serve --restart` to stop it and start it again with the current ones.
- `llama-server was not found` and winget is missing (or this is not Windows): install llama.cpp from
  <https://github.com/ggml-org/llama.cpp/releases>, put it on `PATH`, or `config --set llama_server=PATH`.
- Port in use: `serve --port 9000`, or `stop` if it is our own server.

## Development

```
uv run ruff format .
uv run ruff check . --fix
uv run pyrefly check
uv run pytest -x
```

`uv run pytest` skips the `live` tests and enforces 90% coverage. The live smoke tests use your real Ollama
directory and real `llama-server` (they never start a server or run a benchmark) and skip when either is
missing, or when your models live in a custom `OLLAMA_MODELS` directory (the tests clear that variable so they
only look at the default locations). Run them with `--no-cov`: they exercise a small part of the code, so the
90% coverage gate would fail a live-only run.

```
uv run pytest -m live --no-cov
```
