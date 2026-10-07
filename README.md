# İbnü'n-Nedîm Gömme (`ibnun-nedim`)

<p align="center"><img src="assets/ikon.png" alt="İbnü'n-Nedîm Gömme icon" width="256"></p>

<p align="center"><sub>Icon: made by Ercan Er with Google Gemini (2026-10); Arabic name «ابن النديم» set in Amiri (SIL Open Font License).</sub></p>

[Türkçe](README.tr.md)

A single-file, pure-Rust (candle) embedding server for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), with an OpenAI-compatible `/v1/embeddings` API and a small desktop window. Runs fully locally, on CPU. No Python, no Docker, no external runtime.

## Name: why Ibn al-Nadim?

> Bismillahirrahmanirrahim. In the name of God, the Most Gracious, the Most Merciful.

This server is named after Abu al-Faraj Muhammad ibn Ishaq al-Nadim, a *warraq* (copyist and bookseller) of tenth-century Baghdad. According to the TDV Encyclopedia of Islam entry [İbnü'n-Nedîm](https://islamansiklopedisi.org.tr/ibnun-nedim), he was a warraq like his father and wrote *al-Fihrist* in 377 AH (987 CE). The work, also known as *Fihrist al-kutub*, *Fihrist al-ulum* and *Fihrist al-ulama*, is the first of its kind in the bibliographic literature of the Islamic world.

The name lands here because of a division of labour. [el-Fihrist](https://github.com/Ercaner1988/el-Fihrist) is the catalogue: it indexes skills, code and memory. To search a catalogue by meaning, every text first has to be read and its meaning turned into numbers, much as the warraq took up each book and decided its place in the Fihrist. İbnü'n-Nedîm Gömme ("embedding") does that reading: it turns text into bge-m3 vectors, and the catalogue searches with them. The catalogue tool in el-Fihrist is also called `ibnunnedim` (the CLI) and takes its embeddings from this server; to keep them apart, one is the **CLI** and the other the **Gömme**.

Source: Nasuhi Ünal Karaarslan, "İbnü'n-Nedîm", *TDV İslâm Ansiklopedisi*, vol. ⚠ TO BE VERIFIED, pp. ⚠ TO BE VERIFIED, https://islamansiklopedisi.org.tr/ibnun-nedim. Volume and pages are left open because the entry could not be opened from this session. See also the [el-Fihrist](https://islamansiklopedisi.org.tr/el-fihrist) entry.

### Sister repositories

- [el-Fihrist](https://github.com/Ercaner1988/el-Fihrist): skill, code and memory catalogue for AI agents. Takes its embeddings from this server; the nearest-neighbour index for bge-m3 vectors (`bge-dizin`) lives in this repository.
- [Nazar](https://github.com/Ercaner1988/Nazar): search inside local file contents, named after *nazar*, reasoning from the known to the unknown in kalam. Its embedding backend is this server.
- [agent-reach-rs](https://github.com/Ercaner1988/agent-reach-rs) (ARR): web reading engine for agents; reads TDV Encyclopedia of Islam citations from the entry itself.
- [kilim-tema](https://github.com/Ercaner1988/kilim-tema): shared theme (Kilim palette, Samarkand paper).

## Download

Get the latest build from the [Releases page](https://github.com/Ercaner1988/ibnun-nedim/releases). `release.yml` builds these six assets per tag:

| File | Platform |
|---|---|
| `ibnun-nedim-x86_64-pc-windows-msvc.exe` | Windows, x86-64 |
| `ibnun-nedim-aarch64-pc-windows-msvc.exe` | Windows, ARM64 |
| `ibnun-nedim-x86_64-unknown-linux-gnu` | Linux, x86-64 |
| `ibnun-nedim-aarch64-unknown-linux-gnu` | Linux, ARM64 |
| `ibnun-nedim-x86_64-apple-darwin` | macOS, Intel |
| `ibnun-nedim-aarch64-apple-darwin` | macOS, Apple Silicon |

The x86-64 builds are compiled for `x86-64-v3` (AVX2 + FMA + BMI — Intel Haswell 2013+, AMD Excavator/Zen 2015+). On an older CPU the app refuses to start and says so instead of crashing.

CI starts each build and embeds a test sentence before publishing it, except the macOS Intel build: it is cross-compiled on Apple Silicon, where Rosetta has no AVX2, so it is built but not run in CI.

## First run

Double-click the binary. A window opens and downloads the model (BAAI/bge-m3, ~2.2 GB) once from the Hugging Face hub into the standard HF cache, with a progress bar. Later runs reuse the cached files and skip the download.

For servers, run headless instead: `--headless` or `BGE_HEADLESS=1` skips the window and behaves like a plain CLI server.

## Unsigned binaries

These builds aren't code-signed, so your OS will warn you the first time:

- **Windows**: SmartScreen — click "More info" then "Run anyway".
- **macOS**: Gatekeeper — right-click the binary and choose "Open", or run `xattr -d com.apple.quarantine <file>`.
- **Linux**: mark it executable first: `chmod +x <file>`.

## API

`POST /v1/embeddings` — body is `{"model": "bge-m3", "input": "text"}` or `{"model": "bge-m3", "input": ["text1", "text2"]}`. `model` is accepted but ignored (bge-m3 is the only model this server runs). Returns 1024-dim, L2-normalized CLS-pooled vectors.

`GET /health` — 200 OK once the server is up.

```bash
curl http://127.0.0.1:11435/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":"hello world"}'
```

## Configuration

Environment variables read at startup:

| Variable | Default | Notes |
|---|---|---|
| `BGE_HOST` | `127.0.0.1` | `0.0.0.0` exposes the server to your LAN/Docker — there is no authentication, only do this on a trusted network. |
| `BGE_PORT` | `11435` | |
| `BGE_PARALLEL` | `4` | Max batches embedded concurrently (each batch is one forward pass spread over all cores). |
| `BGE_PARTI_TOKEN` | `2048` | Max padded tokens (batch size x longest input) per batch. |
| `BGE_PROFIL` | off | `1` prints one timestamped (UTC) line per request: tokens, batches, tokenizer ms, forward ms. |
| `BGE_HEADLESS` | unset | `1` skips the GUI window. |

## Connecting tools

The Connections screen detects and wires up local RAG tools:

- **Open Notebook, Open WebUI, AnythingLLM** — probed over HTTP and connected automatically when found.
- **LibreChat, Dify** — no HTTP probe; the screen offers a "Copy config" button with a ready-to-paste config snippet instead.

A "Runs in Docker" toggle switches the URL given to a tool from `127.0.0.1` to `host.docker.internal` when the tool itself runs inside a container.

Connecting never deletes or edits your existing models or credentials, and does nothing if the tool already points at this server. Switching a tool's embedding model still has consequences:

- **Open Notebook**: existing sources may need re-embedding for consistent search. The app shows a note after it changed the default model.
- **Open WebUI**: knowledge bases need to be re-indexed. The app shows a note after the change.
- **AnythingLLM**: changing the embedder **deletes every workspace's embedded documents** (AnythingLLM resets its vector database). The app writes nothing until you confirm with "Delete and connect".

## Performance

Measured on a Ryzen 5 7430U laptop (6 cores / 12 threads), ~1,500-character chunks, `BGE_PARALLEL=4`: roughly 1.9–2.5 seconds per chunk per request batch.

The portable `x86-64-v3` release build performs the same as a `target-cpu=native` build on that machine: median 2.49 s/chunk vs. 2.78 s/chunk, identical vectors (cosine 1.000000).

Vectors match a llama.cpp bge-m3 reference server at cosine similarity ≥ 0.9999 (CLS pooling, L2-normalized).

To reproduce, run two servers on different ports and, with [Bun](https://bun.sh) installed (the recommended runtime for the benchmark scripts):

```bash
BENCH_SOURCE=book.txt bun bench.ts native=11434 v3=11435
```

The scripts use only portable APIs, so Node.js 22.18 or newer also runs them directly (`node bench.ts ...`, `node lane-bench.ts ...`); Bun is simply the recommended choice.

### Concurrency: short queries vs. batch jobs

A single request already keeps `BGE_PARALLEL` threads busy. When several clients call the
server at once (e.g. a chat tool doing a bulk re-index while another tool sends a live search
query), each concurrent multi-input request adds its own `BGE_PARALLEL` threads, and a short
single-input query gets starved: measured median latency went from ~0.65 s (idle) to 7.4 s
under one concurrent batch job and 47 s under three.

`BGE_TOPLU_IZIN` (default 1) gates multi-input requests through a semaphore; a single-input
request under ~512 characters (a typical search query) always skips the gate. This trades some
batch throughput for short-query latency: gate permit 1 brought the median back to ~1.8 s at a
cost of about 27% batch throughput, measured with 3 concurrent clients over 3 rotated rounds.

To reproduce:

```bash
BENCH_SOURCE=book.txt bun lane-bench.ts gated=./target/release/ibnun-nedim.exe|1 ungated=./target/release/ibnun-nedim.exe|999
```

## Install (Windows, mcp-tools)

`kur.ps1` builds the release binary and copies it to `%USERPROFILE%\Desktop\mcp-tools\ibnun-nedim`. An older copy there is kept as `ibnun-nedim.<time>.eski.exe`, and the copy is checked against the build with SHA-256. If the copy is running it stops; `-Durdur` stops it first, `-DerlemeYok` installs an existing build, `-Hedef` picks another folder.

```powershell
./kur.ps1
```

## Build from source

```bash
cargo build --release
```

`.cargo/config.toml` sets `target-cpu=x86-64-v3` for `x86_64` builds automatically: candle's element-wise operations only use the SIMD instructions the compile target allows, and an SSE2-baseline build was about 10x slower.

Optional: `cargo build --release --features fihrist` adds [el-Fihrist](https://github.com/Ercaner1988/el-Fihrist)'s theme catalogue to the Theme menu. The catalogue is only read while that menu is open; the chosen pack is saved next to your other settings and the built-in theme stays the fallback.

## Credits & licenses

Code is [MIT licensed](LICENSE).

The server downloads and runs [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3); check its license on that page before using it.

The bundled Inter font is licensed under the [SIL Open Font License 1.1](crates/bge-theme/assets/fonts/LICENSE.txt).

Built with [candle](https://github.com/huggingface/candle), [egui/eframe](https://github.com/emilk/egui), [axum](https://github.com/tokio-rs/axum), and [hf-hub](https://github.com/huggingface/hf-hub).
