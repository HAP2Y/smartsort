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
- **157 tests, fully offline** — the network is mocked end-to-end. CI runs the same suite plus an out-of-process CLI smoke test.

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
| `rules` | fallback when nothing else matches | filename rules engine |
| `ai-small` | small extractable docs (.pdf / .docx / .txt < 2 MB) | small Ollama model + rules safety net |
| `ai-large` | large extractable docs ≥ 2 MB | larger Ollama model + rules safety net |
| `ocr` | images (.png / .jpg / .tif / .bmp / ...) | rules fallback until a real OCR worker lands |

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
| `rules-worker` (also drains `ocr`) | 100 | 2 |
| `ai-small-worker` | 50 | 2 |
| `ai-large-worker` | 20 | 1 |

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

…and set `default_local_model: qwen2.5:7b` in `config/categories.yaml`.

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

| Setting | Purpose |
| --- | --- |
| `confidence_threshold` | Minimum AI confidence (0–100) before AI's answer is accepted. |
| `max_extract_chars` | Upper bound on characters extracted per file before the LLM sees it. |
| `default_local_model` | Ollama tag for `ai-small` (e.g. `qwen2.5:14b`). Must be pulled locally. |
| `large_model` *(optional)* | Ollama tag for `ai-large`. Defaults to `default_local_model`. |

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
tests/                     # 157 tests, all offline
```

---

## Tests

```bash
python -m pytest tests/ -q
```

157 tests covering the rules engine, the `Classification` pipeline, the Ollama client (network mocked), text extraction, PII redaction, organizer move + undo round-trips, an end-to-end CLI dry-run, plus the distributed-inference layer: queue round-trips, router decisions, worker exception capture, and an end-to-end orchestrator run with stub workers.

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
