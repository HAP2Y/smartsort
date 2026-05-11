# SmartSort

**A local-first AI inference orchestration platform**, demonstrated through privacy-preserving file classification.

Started as a single-process tool that organises a messy directory using a local LLM (via Ollama). Extended into a distributed runtime: a job router, a pluggable queue backend (in-memory or Redis Streams), worker pools per model, and an orchestrator — all running locally, scalable via Docker Compose and Kubernetes.

The same CLI runs in both modes. Add `--distributed` to scale out.

---

## Quickstart

```bash
git clone <repo> && cd smartsort
pip install -e ".[dev]"
ollama pull qwen2.5:14b && ollama serve &

smartsort run ~/Downloads                  # dry-run plan
smartsort run ~/Downloads --apply          # actually move files
smartsort undo ~/Downloads                 # revert
```

That's the original single-process flow. To scale across a worker fleet, see [Distributed mode](#distributed-mode).

---

## Two ways to run

| Mode | Command | When |
| --- | --- | --- |
| **Local** | `smartsort run <dir>` | Default. One process, inline classifier pipeline. Best for a single directory on one machine. |
| **Distributed (in-process)** | `smartsort run <dir> --distributed` | Same machine, same process, but routed through queues and worker threads. Useful for testing the distributed path and for parallelising classification on one box. |
| **Distributed (Redis)** | `smartsort run <dir> --distributed --backend redis` | External worker fleet over Redis Streams. Scale by running more workers — locally, in Docker, or in Kubernetes. |

Every other flag (`--apply`, `--recursive`, `--no-ai`, `-v`) works in all modes.

---

## Distributed mode

The local pipeline becomes a router → per-route queues → worker pools → result stream → orchestrator topology.

```
┌─────────┐   Router decides    ┌──────────────────────────────────────┐
│ files   │ ──────────────────▶ │ Queue backend (memory | redis)       │
└─────────┘   queue by ext+size │   smartsort:jobs:rules               │
                                │   smartsort:jobs:ai-small            │
                                │   smartsort:jobs:ai-large            │
                                │   smartsort:jobs:ocr                 │
                                └──────────────┬───────────────────────┘
                                               │ workers pull, classify, publish
                                               ▼
                                ┌──────────────────────────────────────┐
                                │ smartsort:results stream             │
                                └──────────────┬───────────────────────┘
                                               ▼
                                      ┌─────────────────┐
                                      │  Orchestrator   │── classification plan
                                      └─────────────────┘
```

| Route | Sent here when | Worker classifier |
| --- | --- | --- |
| `rules` | fallback when nothing else matches | filename rules engine |
| `ai-small` | small extractable docs (.pdf / .docx / .txt < 2 MB) | small Ollama model + rules safety net |
| `ai-large` | large extractable docs ≥ 2 MB | larger Ollama model + rules safety net |
| `ocr` | images (.png / .jpg / ...) | falls back to rules until a real OCR classifier ships |

### Option A — in-process distributed (no Redis)

The fastest way to try it. Workers run as threads inside the dispatcher process.

```bash
smartsort run ~/Downloads --distributed --workers 2 -vv
```

### Option B — Redis-backed workers (the production-style path)

Two steps: start the worker fleet, then submit work. Workers stay running across runs.

```bash
# 1. Start Redis + one worker per route, in the background.
docker compose up --build -d

# 2. From your host, submit a directory. Workers in containers pick the jobs up.
smartsort run ~/Downloads --distributed --backend redis \
    --redis-url redis://localhost:6379/0
```

That's it. To scale a single route horizontally:

```bash
docker compose up --scale ai-small-worker=4 -d
```

The dispatcher (your `smartsort run`) is stateless. You can fire it from anywhere that can reach Redis — host shell, another container, CI.

### Running workers without Docker

```bash
# Terminal 1
smartsort serve-worker --routes rules,ocr   --backend redis -v

# Terminal 2
smartsort serve-worker --routes ai-small    --backend redis -v

# Terminal 3 (optional)
smartsort serve-worker --routes ai-large    --backend redis --model qwen2.5:32b -v
```

A single worker can subscribe to multiple routes — the example above runs one cheap worker for `rules + ocr` and a separate worker per AI model.

---

## How classification works (the workload the platform is built around)

Each worker hosts a `ClassificationPipeline`. The first confident result wins:

1. **High-confidence filename overrides** — IMM forms, T4 slips, IRCC, IELTS, WES, NOC, LMIA, PCC, ECA, ITA, "employment verification", "police clearance", `PR_` prefix. Confidence 100, short-circuits later steps.
2. **Local AI** — Ollama reads the first few pages (PDF / DOCX / CSV / text). Sensitive content (emails, phone numbers, URLs, JWTs, AWS keys) is redacted before the model sees anything. Accepted only if confidence ≥ `confidence_threshold`.
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
Dockerfile
docker-compose.yml         # redis + per-route worker pools + on-demand cli
main.py                    # run, undo, check-rules, serve-worker
tests/                     # 127 tests, all offline
```

---

## Tests

```bash
python -m pytest tests/ -q
```

127 tests covering the rules engine, the Classification pipeline, the Ollama client (network mocked), text extraction, PII redaction, organizer move + undo round-trips, an end-to-end CLI dry-run, plus the distributed-inference layer: queue round-trips, router decisions, worker exception capture, and an end-to-end orchestrator run with stub workers.

CI runs the same suite plus an out-of-process CLI smoke test (`.github/workflows/ci.yml`).

---

## Privacy

- All classification runs locally. The AI step talks only to `http://localhost:11434` (Ollama).
- `classifier/redactor.py` strips PII and secrets from extracted text before it reaches the model.
- The undo log stores filesystem paths and timestamps — no file contents.
- Redis, when used, runs locally; `docker compose up redis` binds to localhost only.

---

## Roadmap

- Pull route definitions and per-route worker concurrency into `config/categories.yaml`.
- Per-worker Prometheus `/metrics` endpoint (currently exposed as `WorkerStats`).
- Swap inline `--workers` from threads to `multiprocessing.Process` so the GIL stops capping CPU-bound classifiers.
- Kubernetes manifests (Deployment per route + HPA keyed on queue depth).
- Real OCR worker (Tesseract / PaddleOCR) on the `ocr` route.
