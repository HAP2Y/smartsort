# SmartSort

**A local-first AI inference orchestration platform**, demonstrated through privacy-preserving file classification on macOS and Linux.

SmartSort started as a single-process file sorter that uses a local LLM (via Ollama) to organise messy directories by content. It has since been extended into a distributed inference platform: a job router, a pluggable queue backend (in-memory or Redis Streams), worker pools per model, and an orchestrator that submits work and reassembles results — all running locally, with a Docker Compose topology that scales the same shape across processes and hosts.

The original single-process classifier remains the reference workload; the distributed runtime treats it as one of several pluggable inference units.

---

## What this project demonstrates

- **Distributed system design without the framework** — producer/consumer topology with explicit `Job` / `JobResult` envelopes, route-keyed queues, consumer groups, ack semantics, and orchestrator-side timeout handling. No Celery, no Airflow — just the protocol and two backends behind it.
- **Pluggable infrastructure** — one `QueueBackend` protocol, two implementations (`InMemoryQueueBackend` for tests / single-process runs, `RedisStreamBackend` for multi-process / multi-host). Application code is identical on both.
- **Intelligent workload routing** — small documents go to a fast model queue, large documents to a large-context model queue, images to an OCR queue, everything else to a cheap rules engine. The router is a data-driven rule list, not branching code.
- **Fault isolation across the queue boundary** — a crashing worker reports the exception as a `JobResult.error` instead of poisoning the queue; the orchestrator fills timeouts with `Unknown_Unsorted` so the end-user always gets a complete plan.
- **Container-ready scale path** — Dockerfile + Compose stack with Redis, per-route worker pools, and an on-demand dispatcher. `docker compose up --scale ai-small-worker=4` is the entire horizontal scale story; the same image is the building block for a Kubernetes Deployment.
- **127 tests, fully offline** — the network is mocked end-to-end (Ollama HTTP + Redis-style ack/dequeue paths via the in-memory backend). CI runs the same suite plus an out-of-process CLI smoke test.

---

## Architecture

```
                    ┌─────────────┐
files in a dir ───▶ │   Router    │── picks queue by file traits ──┐
                    └─────────────┘                                 │
                                                                    ▼
                    ┌─────────────────────────────────────────────────────┐
                    │           QueueBackend (memory | Redis Streams)     │
                    │                                                     │
                    │   smartsort:jobs:rules     smartsort:jobs:ai-small  │
                    │   smartsort:jobs:ai-large  smartsort:jobs:ocr       │
                    └─────────────────────────────────────────────────────┘
                          │            │             │            │
                          ▼            ▼             ▼            ▼
                      ┌───────┐    ┌───────┐     ┌───────┐    ┌───────┐
                      │Worker │    │Worker │ ... │Worker │    │Worker │
                      │ rules │    │ai-sm. │     │ai-lg. │    │  ocr  │
                      └───┬───┘    └───┬───┘     └───┬───┘    └───┬───┘
                          │            │             │            │
                          └────────────┴──────┬──────┴────────────┘
                                              ▼
                                ┌────────────────────────────┐
                                │  smartsort:results stream  │
                                └────────────┬───────────────┘
                                             ▼
                                      ┌──────────────┐
                                      │ Orchestrator │── classification plan
                                      └──────────────┘
```

Components live in `inference/`:

| Module | Responsibility |
| --- | --- |
| `inference/types.py` | `Job` and `JobResult` dataclasses with dict serialisation. The wire format is intentionally small and stable. |
| `inference/queue.py` | `QueueBackend` protocol, `InMemoryQueueBackend` (thread-safe stdlib queues), `RedisStreamBackend` (Streams + consumer groups, XACK + XDEL on completion). |
| `inference/router.py` | First-match `RouteRule` list. Defaults: images → `ocr`, large extractable docs → `ai-large`, small docs → `ai-small`, fallback → `rules`. |
| `inference/worker.py` | Polling worker that wraps any `Classifier`. Catches exceptions and surfaces them as `JobResult.error` so a bad model never poisons the queue. |
| `inference/orchestrator.py` | Submits jobs, drains the result stream, fills timeouts with `Classification.unknown()` so callers always get a complete plan. |

The classifier protocol from the original single-process pipeline (`classifier/pipeline.py`) is reused unchanged — each worker hosts a `ClassificationPipeline` internally, which means the same classification logic ships in both runtimes.

---

## Distributed runtime

### Run a worker

```bash
# Local, single-process (no Redis required) — useful for development and tests:
smartsort serve-worker --route rules --backend memory

# Production-style: connect to Redis, pick a route, optionally override the model:
smartsort serve-worker --route ai-small  --backend redis --redis-url redis://localhost:6379/0
smartsort serve-worker --route ai-large  --backend redis --model qwen2.5:32b
```

A worker subscribes to one route, polls the queue with a configurable timeout, runs each dequeued file through its `Classifier`, publishes a `JobResult`, and acks. Throughput scales by running more workers on the same route — same binary, different `--route`.

### Submit work

```bash
# Spin in-process workers and dispatch (good for demos):
smartsort dispatch ~/Downloads --backend memory --inline-workers 2

# Dispatch onto an external Redis-backed worker fleet:
smartsort dispatch ~/Downloads --backend redis --redis-url redis://localhost:6379/0 --apply
```

The dispatcher walks the directory, asks the router which queue each file belongs on, enqueues a `Job`, then drains the result stream until every submitted job has reported back (or the timeout expires). The resulting `{file_path: Classification}` plan is identical in shape to the single-process `smartsort run` output, so the file mover (`movers/organizer.py`) consumes it unchanged.

### Docker Compose

```bash
docker compose up --build              # redis + one worker per route
docker compose up --scale ai-small-worker=4   # horizontal scale, one command
docker compose run --rm dispatcher dispatch /work --backend redis --redis-url redis://redis:6379/0
```

The Compose file (`docker-compose.yml`) defines `redis`, `rules-worker`, `ai-small-worker`, `ai-large-worker`, and a profiled on-demand `dispatcher`. Workers mount a shared `./workdir` so the same files are visible to every container — this is what a Kubernetes `PersistentVolumeClaim` becomes in the next step.

---

## Classification (the reference workload)

The classifier each worker runs is the same one shipped in the single-process CLI. Files flow through a `ClassificationPipeline` whose first confident hit wins:

1. **High-confidence filename overrides** (`Rules (HC)`) — IMM forms, T4 slips, IRCC, IELTS, WES, NOC, LMIA, PCC, ECA, ITA, "employment verification", "police clearance", `PR_` prefix, etc. Confidence 100, short-circuits later steps.
2. **Local AI** (`Local AI`) — Ollama reads the first few pages of each file (PDF / DOCX / CSV / text). Sensitive content is redacted (emails, phone numbers, URLs, JWTs, AWS keys) before the model sees anything. Only accepted if confidence ≥ `confidence_threshold`.
3. **Keyword + extension fallback** (`Rules`) — system / hidden file detection, then multi-word phrase matches (`"air india"`, `"reality flip"`) before single-word keywords so the more specific signal wins. Generic archives (`.zip`, `.dmg`, `.pkg`) fall through to `Archives_and_Apps`.

Adding a new classification source (OCR, hash dedupe, an ML model) is a single new class implementing the `Classifier` protocol. The same protocol is what workers consume on the distributed side.

The category set lives in `config/categories.yaml`: `Canadian_PR_Docs`, `AstroQuant_Sidereal`, `Guidewire_PSE_Work`, `Resumes_Career_Tech`, `Financial_Taxes`, `Medical_Health`, `Travel_Transit`, `Franchise_Business_Research`, `Media_Images`, `Archives_and_Apps`, `Metadata_System`, `Unknown_Unsorted`.

---

## Prerequisites

1. Python 3.10+
2. (Optional, for AI routes) [Ollama](https://ollama.com/download) with at least one local model:
   ```bash
   ollama pull qwen2.5:14b   # small / fast — default for ai-small route
   ollama pull qwen2.5:32b   # large context — used for ai-large route
   ```
3. (Optional, for distributed mode) Redis 7+, or just `docker compose up redis`.
4. Install:
   ```bash
   pip install -e ".[dev]"   # editable install — gives you the `smartsort` command
   pip install redis         # only needed if you use --backend redis
   ```

---

## Single-process usage (unchanged from v0.2)

```bash
smartsort run ~/Downloads                  # dry-run (prints the plan, moves nothing)
smartsort run ~/Downloads --apply          # actually move files
smartsort run ~/Downloads --no-ai          # rules-only (no Ollama)
smartsort run ~/Downloads -r               # recurse into subdirectories
smartsort run ~/Downloads -vv              # debug logging (one -v = info)
smartsort undo ~/Downloads                 # revert the last sort
smartsort check-rules                      # validate categories.yaml + show summary
```

Each `--apply` writes a `.smartsort_undo.json` log into the target directory. `undo` restores files to their original paths and removes the empty category folders left behind. Files already nested inside a SmartSort category folder are skipped on subsequent runs, so re-running is safe and idempotent.

`classifier/ai_local.py` contains a `PROMPT_TEMPLATE` with disambiguation rules ("employment-verification letters → PR docs, not Career"). When you change the category set, edit those rules so the LLM understands your taxonomy. Each rule should explain *why* a category is what it is, plus what it explicitly **isn't**, so the model has tie-breakers.

`classifier/redactor.py` strips emails, phone numbers, URLs, JWTs, and AWS keys from extracted text before it reaches the LLM. Add patterns there to redact additional entities before any text leaves the machine.

---

## End-to-end walkthrough on `~/Downloads`

A staged path that exercises the single-process pipeline first, then the in-memory distributed runtime, then the full Redis-backed worker fleet. Stop at whichever level of testing you need.

### 0. Preflight (one-time)

```bash
git fetch origin
git checkout claude/distributed-inference-platform-jgYrc
git pull

pip install -e ".[dev]"
pip install redis                       # only needed for the Redis backend

ollama pull qwen2.5:14b                 # default model for the ai-small route
ollama serve &                          # if it isn't already running

smartsort check-rules                   # sanity check categories.yaml
python -m pytest tests/ -q              # expect: 127 passed
```

### 1. Always start on a copy

The tool is idempotent and ships with `undo`, but the first run on `~/Downloads` is safer against a sandbox copy:

```bash
mkdir -p ~/smartsort-sandbox
cp -R ~/Downloads/* ~/smartsort-sandbox/
```

Use `~/smartsort-sandbox` as the target until you trust the plan, then point the same commands at `~/Downloads`.

### 2. Single-process baseline

Confirms config + Ollama are wired correctly before involving the queue layer.

```bash
smartsort run ~/smartsort-sandbox --no-ai          # rules-only dry-run
smartsort run ~/smartsort-sandbox -vv              # full pipeline, dry-run, debug logs
```

Expect a "Classification Plan" table. Nothing has moved yet.

### 3. Distributed runtime, no Redis (in-process workers)

Cheapest exercise of the new code path: orchestrator and workers share one process via the in-memory backend.

```bash
smartsort dispatch ~/smartsort-sandbox \
    --backend memory \
    --inline-workers 2 \
    --timeout 120 \
    -vv
```

Look for:

- A "Routing plan" table listing the four routes (`ocr`, `ai-large`, `ai-small`, `rules`).
- A "Submitted N jobs" line broken down per route — the router fanning files out.
- A "Classification Plan" table identical in shape to the single-process one.
- `submitted=N completed=N failed=0` — the orchestrator confirming every job round-tripped through the queue.

If `failed > 0`, re-run with `-vv` to see which worker errored on which file.

### 4. Distributed runtime, real Redis (separate processes)

Three terminals (or one terminal + `tmux`).

#### 4a. Start Redis

```bash
docker run --rm -p 6379:6379 --name smartsort-redis redis:7-alpine
# or, locally: redis-server
```

#### 4b. One worker per route

```bash
# terminal 1
smartsort serve-worker --route rules    --backend redis --redis-url redis://localhost:6379/0 -v

# terminal 2
smartsort serve-worker --route ai-small --backend redis --redis-url redis://localhost:6379/0 -v

# terminal 3 (optional, if you have a larger model pulled)
smartsort serve-worker --route ai-large --backend redis --redis-url redis://localhost:6379/0 \
    --model qwen2.5:32b -v
```

Each prints `Worker <name> listening on route '<r>' via redis.` and then sits idle.

#### 4c. Dispatch from a fourth terminal

```bash
smartsort dispatch ~/smartsort-sandbox \
    --backend redis --redis-url redis://localhost:6379/0 \
    --timeout 300 -vv
```

Jobs land on the worker terminals (`-v` makes each worker log its `processed` / `failed` counters). Workers print final stats on `Ctrl-C` shutdown.

#### 4d. Scale test — proves throughput actually scales

Kill the single `ai-small` worker and start two named instances:

```bash
smartsort serve-worker --route ai-small --backend redis --name ai-small-1
smartsort serve-worker --route ai-small --backend redis --name ai-small-2
```

Re-run the dispatch. Total wall time drops, and each worker's `processed` count is roughly half of what one worker reported alone.

### 5. Same flow under Docker Compose (one command)

```bash
mkdir -p workdir
cp -R ~/smartsort-sandbox/* workdir/

docker compose up --build              # redis + 1 worker per route
# in another terminal:
docker compose run --rm dispatcher dispatch /work \
    --backend redis --redis-url redis://redis:6379/0 \
    --timeout 300 -vv

# horizontal scale demo:
docker compose up --scale ai-small-worker=4 -d
```

The AI workers in compose target `host.docker.internal:11434`, which works on Docker Desktop (Mac/Windows) by default. On Linux, either run Ollama in a sibling container or add `--add-host=host.docker.internal:host-gateway` to the worker services.

### 6. Apply for real, then undo

Once the dry-run plan looks right:

```bash
smartsort dispatch ~/smartsort-sandbox \
    --backend redis --redis-url redis://localhost:6379/0 \
    --apply
```

Files move into category subfolders and a `.smartsort_undo.json` log is written into the target directory. To revert:

```bash
smartsort undo ~/smartsort-sandbox
```

Once you trust the output, repeat steps 3–6 against `~/Downloads` directly.

### Useful debugging flags

| Flag | Effect |
| --- | --- |
| `-v` / `-vv` | INFO / DEBUG logging on either command |
| `--timeout 600` | extend the dispatcher's wait window for big directories |
| `--no-ai` | skip Ollama entirely (only valid on `smartsort run`, not `dispatch`) |
| `--inline-workers 0` | dispatch with zero workers — proves the orchestrator times out cleanly and fills with `Unknown_Unsorted` instead of hanging |

Fastest single-command diagnostic for a misbehaving directory: `smartsort dispatch <dir> --backend memory --inline-workers 1 -vv` — single thread, full debug logs, no network.

---

## Settings reference

| Setting | Purpose |
| --- | --- |
| `confidence_threshold` | Minimum AI confidence (0–100) before AI's answer is accepted. Below this, the file falls through to the keyword rules. |
| `max_extract_chars` | Upper bound on characters extracted per file before sending to the LLM. Lower = faster, less context. |
| `default_local_model` | Ollama model tag for the `ai-small` route (e.g. `qwen2.5:14b`). Must be pulled locally. |
| `large_model` *(optional)* | Ollama model tag for the `ai-large` route (e.g. `qwen2.5:32b`). Defaults to `default_local_model` if absent. |

---

## Project layout

```
inference/                  # distributed inference runtime
  types.py                  # Job / JobResult wire format
  queue.py                  # QueueBackend protocol, InMemory + Redis Streams impls
  router.py                 # rule-based queue selection
  worker.py                 # polling worker around the Classifier protocol
  orchestrator.py           # producer + result drainer

classifier/                 # the reference workload (file classification)
  types.py                  # FileItem + Classification dataclasses
  pipeline.py               # Classifier Protocol + ClassificationPipeline runner
  classifiers.py            # concrete pipeline classifiers (HC / AI / Rules)
  rules.py                  # filename tokeniser + regex + keyword engine
  extractor.py              # PDF / DOCX / CSV / text extraction (multi-page)
  ai_local.py               # OllamaClient + build_prompt + parse_response
  redactor.py               # PII / secret redaction before text leaves the machine

movers/
  organizer.py              # idempotent moves, undo log, category-folder awareness

config/
  categories.yaml

tests/                      # 127 tests, all offline
  test_rules.py             # filename-classification regression suite (~55 cases)
  test_pipeline.py          # pipeline ordering / fallback / exception handling
  test_ai_local.py          # mocked Ollama: health, prompt, parse, transport errors
  test_extractor.py         # PDF / DOCX / CSV / text + redaction integration
  test_redactor.py          # PII / secret pattern coverage
  test_organizer.py         # move + undo round-trips, collision suffixing
  test_dryrun_smoke.py      # end-to-end CLI dry-run on a fixture directory
  test_inference_queue.py        # queue backend round-trip + multi-route polling
  test_inference_router.py       # routing decisions by extension + size
  test_inference_worker.py       # exception capture, ack behaviour
  test_inference_orchestrator.py # end-to-end submit + collect with stub workers

Dockerfile
docker-compose.yml          # redis + per-route worker pools + dispatcher
.github/workflows/ci.yml
main.py                     # run, undo, check-rules, serve-worker, dispatch
pyproject.toml
```

---

## Tests

```bash
python -m pytest tests/ -q
```

127 tests covering the rules engine, the typed `Classification` pipeline, the Ollama client (with the network fully mocked), text extraction, the PII redactor, organizer move + undo round-trips, an end-to-end CLI dry-run, plus the new distributed-inference layer: queue backend round-trips, router decisions, worker exception capture, and an end-to-end orchestrator run that uses stub classifiers to verify the producer/worker contract offline.

CI runs the same suite plus an out-of-process CLI dry-run; see `.github/workflows/ci.yml`.

---

## Privacy

- All classification runs locally. The AI step talks only to `http://localhost:11434` (Ollama).
- `classifier/redactor.py` strips emails, phone numbers, URLs, JWTs, and AWS access keys from extracted text before it reaches the model.
- The undo log only stores filesystem paths and timestamps — no file contents.
- Redis, when used, runs locally too — `docker compose up redis` binds to localhost.

---

## Roadmap

- Pull route definitions and per-route worker concurrency out of code and into `config/categories.yaml`.
- Per-worker Prometheus `/metrics` endpoint (currently exposed as in-memory `WorkerStats`).
- Swap `--inline-workers` from threads to `multiprocessing.Process` so the GIL stops capping CPU-bound classifiers.
- Kubernetes manifests (Deployment per route + HorizontalPodAutoscaler keyed on queue depth).
- OCR worker (Tesseract / PaddleOCR) on the `ocr` route — the queue is already wired; only the classifier is missing.
