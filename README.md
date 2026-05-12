# SmartSort

> A local-first AI inference orchestration platform, demonstrated through privacy-preserving file classification.

SmartSort began as a single-process CLI that organises a directory using a local LLM (via Ollama). It now ships with a distributed inference runtime — a job router, a pluggable queue backend (in-memory or Redis Streams), worker pools per model, and an orchestrator — that scales the same classification workload across multiple processes and machines. The dispatcher can stand a Docker Compose fleet up sized for the workload at hand, run the job, and tear it back down in a single command.

The platform is workload-agnostic. File classification is the reference implementation; any class that implements the `Classifier` protocol can run as a worker.

---

## What this project demonstrates

- **Distributed system design from primitives** — explicit `Job` / `JobResult` envelopes, route-keyed queues, consumer groups, ack semantics, orchestrator-side timeout handling. No Celery, no Airflow.
- **Pluggable infrastructure** — a single `QueueBackend` protocol with two implementations: in-memory (single-process, tests) and Redis Streams (multi-process, multi-host). Application code is identical on either.
- **Intelligent workload routing** — small documents go to a fast model queue, large documents to a large-context model queue, images to OCR, everything else to a cheap rules engine. The router is a data-driven rule list, not branching code.
- **Workload-aware autoscaling** — the dispatcher pre-routes the file set locally, sizes the Compose fleet to match (more `ai-small` replicas when there are lots of small PDFs, more `ai-large` when there are big ones), and brings it up in one command.
- **Fault isolation across the queue boundary** — a crashing worker reports the exception as a `JobResult.error` instead of poisoning the queue; the orchestrator fills timeouts with `Unknown_Unsorted` so the end-user always receives a complete plan.
- **Container-ready scale path** — Dockerfile + Compose stack with Redis, per-route worker pools, and an on-demand dispatcher. `docker-compose up --scale ai-small-worker=4` is the entire horizontal scale story; the same image is the building block for a Kubernetes Deployment.
- **175 tests, fully offline** — the network is mocked end-to-end. CI runs the same suite plus an out-of-process CLI smoke test.

---

## Architecture

```
┌─────────┐   Router decides    ┌──────────────────────────────────────┐
│ files   │ ──────────────────▶ │ Queue backend (memory | Redis)       │
└─────────┘   route by ext+size │   smartsort:jobs:rules               │
                                │   smartsort:jobs:ai-small            │
                                │   smartsort:jobs:ai-large            │
                                │   smartsort:jobs:ocr                 │
                                └──────────────┬───────────────────────┘
                                               │  workers pull, classify, publish
                                               ▼
                                ┌──────────────────────────────────────┐
                                │ smartsort:results stream             │
                                └──────────────┬───────────────────────┘
                                               ▼
                                      ┌─────────────────┐
                                      │  Orchestrator   │── classification plan
                                      └─────────────────┘
```

| Module | Responsibility |
| --- | --- |
| `inference/types.py` | `Job` and `JobResult` dataclasses with dict serialisation — the wire format. |
| `inference/queue.py` | `QueueBackend` protocol, `InMemoryQueueBackend`, `RedisStreamBackend` (XADD / XREADGROUP / XACK + XDEL). |
| `inference/router.py` | First-match `RouteRule` list deciding which queue each file lands on. |
| `inference/worker.py` | Polling worker around any `Classifier`; catches exceptions and surfaces them as `JobResult.error`. |
| `inference/orchestrator.py` | Submits jobs, drains the result stream, fills timeouts with `Classification.unknown()`. |

| Route | Sent here when | Worker classifier |
| --- | --- | --- |
| `ai-small` | small extractable docs (.pdf / .docx / .txt / ...) under 2 MB | **Local AI → HC rules → fallback rules** |
| `ai-large` | large extractable docs ≥ 2 MB | **Local AI → HC rules → fallback rules** (larger model) |
| `ocr` | images (.png / .jpg / ...) | not implemented — dispatcher classifies as Unknown locally |
| `unroutable` | exotic types the router can't place | dispatcher classifies as Unknown locally |

**Pipeline order matters.** AI runs first on every file the worker
sees, so the LLM gets to weigh in on every document. Rules only catch
the cases where AI declines (no extractable text, output not in the
allowed category set, confidence below `confidence_threshold`). The
previous order put HC rules first, which short-circuited the LLM for
any filename with a familiar keyword — meaning the AI fleet was
expensive plumbing that almost never ran.

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo> && cd smartsort

# 2. Create and activate a virtualenv (recommended)
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 3. Install the package and dev extras
pip install -e ".[dev]"
pip install redis                   # only needed for --backend redis

# 4. Pull a model for the AI routes
ollama pull qwen2.5:14b             # default for the ai-small route
ollama serve &                      # if it isn't already running
```

Optional for the distributed runtime: Docker / Docker Compose (Desktop on macOS / Windows, or `docker-ce` + `docker-compose` on Linux).

---

## Quickstart

```bash
# Local single-process (the original v0.2 path):
smartsort run ~/Downloads                          # dry-run
smartsort run ~/Downloads --apply                  # actually move files
smartsort undo ~/Downloads                         # revert the last sort

# Distributed in-process (no Redis, threads inside one Python process):
smartsort run ~/Downloads --distributed --workers 2

# Distributed via Redis + Docker, one command:
smartsort run ~/Downloads --distributed --backend redis --up

# Same, but tear the fleet down afterwards:
smartsort run ~/Downloads --distributed --backend redis --up --down --apply
```

---

## Three ways to run

| Mode | Command | When to use |
| --- | --- | --- |
| **Local** | `smartsort run <dir>` | Default. One process, inline classifier pipeline. Best for a single directory on one machine. |
| **In-process distributed** | `smartsort run <dir> --distributed` | Same process, routed through queues and worker threads. Useful for exercising the distributed code path and for parallelising CPU-bound classifiers on one box. |
| **Redis-backed distributed** | `smartsort run <dir> --distributed --backend redis [--up]` | External worker fleet over Redis Streams. Scales horizontally — locally, in Docker, or in Kubernetes. |

All other flags (`--apply`, `--recursive`, `--no-ai`, `-v`) work in every mode.

---

## Redis mode in detail

### Recommended: one-command lifecycle

```bash
smartsort run ~/Downloads --distributed --backend redis --up
```

This:

1. Pre-routes the file set locally to count jobs per queue.
2. Runs `docker-compose up -d --build --scale ai-small-worker=N ...` with replica counts sized to the workload.
3. Submits jobs to Redis.
4. Drains the result stream into a classification plan.
5. Leaves the fleet running for the next run (or tears it down if `--down` is passed).

### Autoscaling targets

| Service | Files per worker | Max replicas |
| --- | --- | --- |
| `ai-small-worker` | 25 | 2 |
| `ai-large-worker` | 10 | 1 |

Only AI workers are Compose services in the AI-first architecture —
rules run as a fallback inside each AI worker, so a separate
rules-worker container would be redundant. OCR-routed files (images)
are flagged Unknown on the dispatcher until a real OCR classifier
ships.

The table lives in `main.py:COMPOSE_SCALE` — change the numbers to match your hardware.

> **Why the AI caps are low.** A single Ollama instance serialises LLM calls (one model in memory, one inference at a time). Past 2–3 workers per AI route you only add memory pressure, never throughput. If you point workers at a multi-instance Ollama setup (e.g. a GPU pod pool in Kubernetes), raise the caps to match.

### Memory footprint

The full Compose fleet is intentionally lean:

| Component | RSS (steady state) | Notes |
| --- | --- | --- |
| `redis` | ~30 MB | capped at 128 MB |
| `rules-worker` | ~80 MB | slim image — no PyMuPDF, no python-docx, no pandas (`Dockerfile`) |
| `ai-small-worker` ×2 | ~150 MB each | text-extraction image (`Dockerfile.ai`), capped at 768 MB |
| `ai-large-worker` ×1 | ~150 MB | same image as ai-small |
| **Inside Docker** | **~610 MB** | |
| Ollama on host (qwen2.5:14b) | ~9 GB | model lives in Ollama, not in any worker container |
| Ollama on host (+ qwen2.5:32b) | +~20 GB | only loaded if you actually use the `ai-large` route with the 32B model |

Tactics used to keep this small:

- **Two-image build.** `Dockerfile` is the slim base used by the rules worker (typer + rich + pyyaml + requests + redis — pure-Python wheels). `Dockerfile.ai` adds PyMuPDF + python-docx only for the workers that actually call them. ~150 MB lighter rules image.
- **No `pandas` / `numpy`.** CSV preview uses stdlib `csv` — pandas was 125 MB of dependency for "show me the columns and the first two rows".
- **No `litellm`.** Was in `requirements.txt` from an earlier scaffold; never used in code.
- **Lazy backend imports.** PyMuPDF and python-docx are imported only when the file actually needs them, so the rules-worker can run on the slim image without them installed.
- **Per-service `mem_limit`.** `docker-compose.yml` declares hard ceilings, so a misbehaving worker can't eat the host.
- **Conservative AI replica caps.** See the table above — replicas past 2 on a single Ollama instance burn memory for zero throughput gain.

If you're memory-constrained, prefer a smaller model:

```bash
ollama pull qwen2.5:7b      # ~5 GB
ollama pull qwen2.5:3b      # ~2 GB
```

…and set `models.ai-small: qwen2.5:7b` in `config/categories.yaml`.

---

## Diagnostics

The dispatcher does three things to keep distributed runs debuggable:

**Pre-flight.** Before submitting, `--distributed --backend redis` checks that Redis is reachable, that each route with queued work has at least one consumer subscribed (via `XINFO CONSUMERS`), and that Ollama answers on the configured host. The check panel is printed up front:

```
── Pre-flight ─────────────────────────────────────────
✓ redis:           reachable at redis://localhost:6379/0
✓ workers/rules:   1 consumer(s): rules-1
✓ workers/ai-small: 2 consumer(s): ai-small-1, ai-small-2
✓ ollama:          reachable at http://localhost:11434
```

If Redis is down, the dispatcher aborts immediately rather than enqueuing jobs nobody can drain.

**Live progress.** While the orchestrator drains the result stream, a one-line summary prints every 5 seconds:

```
[ 30s] 47/134 done | rules 24/26 | ocr 0/10 | ai-small 21/92 | ai-large 2/6
[ 60s] 78/134 done | rules 26/26 | ocr 4/10 | ai-small 44/92 | ai-large 4/6
```

**Per-job worker logs.** With `-v` (the default for compose workers), every worker prints one INFO line per job:

```
ai-small-1: Resume.pdf -> Resumes_Career_Tech (Local AI, 92%) in 8240ms
ai-small-1: report.zip -> ERROR (Timeout: read timed out) in 60010ms
```

Tail any worker with `docker-compose logs -f ai-small-worker` to watch jobs flow in real time.

**Post-mortem on timeout.** If anything failed when the timeout hit, the dispatcher prints which queues still have entries in flight and dumps the last 5 log lines from each compose worker so you don't have to context-switch:

```
── Post-mortem ────────────────────────────────────────
queue ai-small: 14 entries still in flight
── ai-small-worker (last 5 log lines) ──
... worker ai-small-1 starting on routes=['ai-small']
... ai-small-1: report.pdf -> Resumes_Career_Tech (Local AI, 87%) in 12450ms
```

Plus a hint block suggesting the typical next step (smaller model, longer timeout, more replicas).

### Manual lifecycle

```bash
docker-compose up -d --build                                   # start once
smartsort run ~/Downloads --distributed --backend redis        # dispatch N times
docker-compose up -d --scale ai-small-worker=4                 # scale by hand
docker-compose down                                            # stop when done
```

### Running workers without Docker

```bash
smartsort serve-worker --routes rules,ocr  --backend redis -v
smartsort serve-worker --routes ai-small   --backend redis -v
smartsort serve-worker --routes ai-large   --backend redis --model qwen2.5:32b -v
```

A single worker can subscribe to multiple routes — above, one cheap process drains `rules + ocr` while a dedicated worker handles each AI model.

---

## Why Redis (and how it scales)

The Redis backend is the production-style story. It's deliberately built on **Redis Streams + consumer groups** rather than plain lists or pub/sub, because that combination gives every property the platform actually needs:

| Property | How it's realised | Why it matters |
| --- | --- | --- |
| **At-least-once delivery** | `XREADGROUP` reserves an entry in the consumer's Pending Entries List until it's acked with `XACK`. | A crashing worker doesn't silently drop the job — the entry is still claimable from PEL. |
| **Exactly-once-per-group** | One consumer group per worker pool means each job is delivered to *one* consumer in the group, regardless of how many workers join. | Scale out by adding more workers to the same group; no coordination layer required. |
| **Horizontal scaling** | `docker-compose up --scale ai-small-worker=N` (or `kubectl scale deploy ai-small-worker --replicas=N`). | Throughput on the AI route grows linearly with worker count until you saturate Ollama or the GPU. |
| **Workload-aware autoscaling** | `smartsort run --up` pre-routes files locally and sizes the fleet via `--scale svc=N` per route before submitting. | Cold-start cost is paid once; the next dispatch is instant. |
| **Stateless dispatcher** | `smartsort run` writes only to Redis and reads only from the result stream. No local state survives the process. | Fire it from your laptop, from CI, from a cron pod — wherever can reach Redis. |
| **Backpressure observability** | `redis-cli XLEN smartsort:jobs:<route>` returns the queue depth at any instant. | Hook the autoscaler / a dashboard up to queue depth instead of CPU. |
| **Pluggable backend** | Workers, router, and orchestrator depend only on the `QueueBackend` protocol; the Redis implementation is one of two ships today. | Swap to Kafka / NATS / SQS / RabbitMQ by writing one class. Application code is untouched. |
| **Cross-host portability** | Workers don't bind to localhost; they speak only to `REDIS_URL`. The Compose image is the same artefact you'd push to a Kubernetes Deployment. | The path from "one laptop" to "GPU node pool in a cluster" is configuration, not refactoring. |

### Throughput model

For a workload of *N* files split across queues:

```
wall_time ≈ max(
    queue_depth(rules)     / (workers_rules    × throughput_rules),
    queue_depth(ai-small)  / (workers_ai-small × throughput_ai-small),
    queue_depth(ai-large)  / (workers_ai-large × throughput_ai-large),
    queue_depth(ocr)       / (workers_ocr      × throughput_ocr),
)
```

The router rebalances the numerator across queues; horizontal scaling rebalances the denominator. The system is throughput-bound by Ollama (per-token latency on AI routes) and disk (filename rules), not by Redis itself — a single Redis instance comfortably handles tens of thousands of stream entries per second, well above any plausible Ollama saturation point.

### What this enables next

- Drop a GPU-only worker pool on a separate host pointing at the same Redis URL — the AI routes seamlessly fan out to it.
- Add a different workload (audio transcription, embedding generation, OCR for real) as a new route + worker without touching anything that already works.
- A Kubernetes HorizontalPodAutoscaler keyed off `XLEN` per route gives true reactive autoscaling — the structure is already there, the manifests are the only missing piece.

---

## How classification works

Each worker hosts a `ClassificationPipeline`. The first confident result wins:

1. **High-confidence filename overrides** — IMM forms, T4 slips, IRCC, IELTS, WES, NOC, LMIA, PCC, ECA, ITA, "employment verification", "police clearance", `PR_` prefix. Confidence 100, short-circuits later steps.
2. **Local AI** — Ollama reads the first few pages of each file. PII and secrets (emails, phone numbers, URLs, JWTs, AWS keys) are redacted before the model sees anything. Accepted only if confidence ≥ `confidence_threshold`.
3. **Keyword + extension fallback** — system / hidden file detection, multi-word phrase matches (`"air india"`, `"reality flip"`) before single-word keywords. Generic archives (`.zip`, `.dmg`) fall through to `Archives_and_Apps`.

Categories live in `config/categories.yaml`. Adding a new classification source (OCR, ML model, hash dedupe) is one class implementing the `Classifier` protocol — the same protocol workers consume on the distributed side.

---

## Settings

`config/categories.yaml` is the single source of truth for everything operational.

| Setting | Purpose |
| --- | --- |
| `confidence_threshold` | Minimum AI confidence (0–100) before AI's answer wins; below this the file falls through to the rules safety net. **60** is a good fit for 7B models (which self-report 60–75 even when right); raise to 80+ for 14B / 32B models which are more confident. |
| `max_extract_chars` | Upper bound on characters extracted per file before the LLM sees it. |
| `models.<route>` | **Per-route model** — what each AI worker actually loads (e.g. `models.ai-small: qwen2.5:7b`). |
| `default_local_model` | Legacy fallback used by the inline `smartsort run` path and as the default for `ai-small` if `models.ai-small` is unset. |
| `large_model` | Legacy fallback for `ai-large` if `models.ai-large` is unset. |

Resolution order for AI workers: `--model` CLI flag → `settings.models.<route>` → legacy `large_model`/`default_local_model`. The worker prints which model it ended up using at startup.

### Picking a model

Memory rule of thumb: **parameter count × ~0.7 GB**. Pick per-route based on traffic shape:

| Model | RAM in Ollama | Calls/min on M-series CPU | Good for |
| --- | --- | --- | --- |
| `qwen2.5:3b` | ~2 GB | 30–60 | high-throughput `ai-small` on a small machine |
| `qwen2.5:7b` | ~5 GB | 15–30 | recommended default for `ai-small` |
| `qwen2.5:14b` | ~9 GB | 5–10 | `ai-small` on a beefy box, `ai-large` default |
| `qwen2.5:32b` | ~20 GB | 1–3 | `ai-large` only — too slow for the fan-out queue |

Pull whatever you reference before running:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
```

---

## Project layout

```
inference/                 # distributed runtime
  types.py                 # Job / JobResult wire format
  queue.py                 # QueueBackend protocol + memory & Redis Streams impls
  router.py                # rule-based queue selection
  worker.py                # polling worker over the Classifier protocol
  orchestrator.py          # producer + result drainer

classifier/                # reference workload (file classification)
  pipeline.py              # Classifier Protocol + ClassificationPipeline
  classifiers.py           # HC / AI / Rules pipeline classifiers
  rules.py                 # filename tokeniser + regex + keyword engine
  extractor.py             # PDF / DOCX / CSV / text extraction
  ai_local.py              # Ollama client + prompt + parser
  redactor.py              # PII / secret redaction before text leaves the box

movers/organizer.py        # idempotent moves + undo log
config/categories.yaml
Dockerfile                 # slim base image used by the rules-worker
Dockerfile.ai              # adds PyMuPDF + python-docx for AI workers
docker-compose.yml         # redis + per-route worker pools (with mem_limit)
main.py                    # run, undo, check-rules, serve-worker
tests/                     # 175 tests, all offline
```

---

## Tests

```bash
python -m pytest tests/ -q
```

175 tests covering the rules engine, the `Classification` pipeline, the Ollama client (network mocked), text extraction, PII redaction, organizer move + undo round-trips, an end-to-end CLI dry-run, plus the distributed-inference layer: queue round-trips, router decisions, worker exception capture, and an end-to-end orchestrator run with stub workers.

CI runs the same suite plus an out-of-process CLI smoke test (`.github/workflows/ci.yml`).

---

## Privacy

- All classification runs locally. The AI step talks only to `http://localhost:11434` (Ollama).
- `classifier/redactor.py` strips PII and secrets from extracted text before it reaches the model.
- The undo log stores filesystem paths and timestamps — no file contents.
- Redis, when used, runs locally; `docker-compose up redis` binds to localhost only.

---

## Roadmap

- Pull route definitions and per-route worker concurrency into `config/categories.yaml`.
- Per-worker Prometheus `/metrics` endpoint (currently exposed as `WorkerStats`).
- Swap inline `--workers` from threads to `multiprocessing.Process` so the GIL stops capping CPU-bound classifiers.
- Kubernetes manifests (Deployment per route + HPA keyed on queue depth).
- Real OCR worker (Tesseract / PaddleOCR) on the `ocr` route.

---

## License

MIT — see [`LICENSE`](LICENSE).
