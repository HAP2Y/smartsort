# SmartSort

Intelligent, local-first, privacy-preserving file organisation for macOS and Linux.

SmartSort scans a messy directory and organises it by **content and context**, not just file extensions. It runs a fast filename rules engine first, falls back to a local LLM (via Ollama) for ambiguous files, and never sends your documents to a cloud service.

## How it classifies

Files flow through a `ClassificationPipeline` of pluggable classifiers. The first one that returns a confident, known-category result wins:

1. **High-confidence filename overrides** (`Rules (HC)`) — IMM forms, T4 slips, IRCC, IELTS, WES, NOC, LMIA, PCC, ECA, ITA, "employment verification", "police clearance", `PR_` prefix, etc. Confidence 100, short-circuits all later steps.
2. **Local AI** (`Local AI`) — Ollama reads the first few pages of each file (PDF / DOCX / CSV / text). Sensitive content is redacted (emails, phone numbers, URLs, JWTs, AWS keys) before being passed to the model. Only accepted if confidence ≥ `confidence_threshold`.
3. **Keyword + extension fallback** (`Rules`) — system / hidden file detection, then multi-word phrase matches (`"air india"`, `"reality flip"`) before single-word keywords so the more specific signal wins. Generic archives (`.zip`, `.dmg`, `.pkg`) fall through to `Archives_and_Apps` at confidence 75.

Adding a new classification source (OCR, ML model, hash dedupe) is a single new class implementing the `Classifier` protocol — see `classifier/pipeline.py` and `classifier/classifiers.py`.

The category set lives in `config/categories.yaml` and includes: `Canadian_PR_Docs`, `AstroQuant_Sidereal`, `Guidewire_PSE_Work`, `Resumes_Career_Tech`, `Financial_Taxes`, `Medical_Health`, `Travel_Transit`, `Franchise_Business_Research`, `Media_Images`, `Archives_and_Apps`, `Metadata_System`, `Unknown_Unsorted`.

## Prerequisites

1. Python 3.10+
2. (Optional, for AI step) [Ollama](https://ollama.com/download) with a local model:
   ```bash
   ollama pull qwen2.5:32b
   ```
   The model name lives in `config/categories.yaml` under `settings.default_local_model`.
3. Install:
   ```bash
   # editable install (recommended) — gives you the `smartsort` command
   pip install -e ".[dev]"
   # …or just runtime deps
   pip install -r requirements.txt
   ```

## Usage

```bash
smartsort run ~/Downloads                  # dry-run (prints the plan, moves nothing)
smartsort run ~/Downloads --apply          # actually move files
smartsort run ~/Downloads --no-ai          # rules-only (no Ollama)
smartsort run ~/Downloads -r               # recurse into subdirectories
smartsort run ~/Downloads -vv              # debug logging (one -v = info)
smartsort undo ~/Downloads                 # revert the last sort
smartsort check-rules                      # validate categories.yaml + show summary
```

Without an editable install you can still invoke the script directly: `python main.py run ~/Downloads ...`.

Each `--apply` writes a `.smartsort_undo.json` log into the target directory; `undo` reads that log, restores files to their original paths, and removes the empty category folders left behind. Files already nested inside a SmartSort category folder are skipped on subsequent runs, so re-running is safe and idempotent.

## Customise it for yourself

The default categories are tuned for the original author's life (Canadian PR docs, Guidewire work, an astro-quant project). Forking and rewiring it for your own setup is a 5–10 minute job, almost entirely in `config/categories.yaml`.

### 1. Fork and clone

```bash
git clone https://github.com/<your-fork>/smartsort.git
cd smartsort
pip install -e ".[dev]"
```

### 2. Reshape `config/categories.yaml`

The whole engine is driven by this one file. Each top-level entry under `categories:` is a folder name, a list of file extensions the category accepts, and a list of keywords / phrases. The file ends with a `settings:` block that controls thresholds and the AI model.

```yaml
categories:
  Work_Acme:                            # folder name (created on --apply)
    extensions: [.pdf, .docx, .pptx]    # only files with these extensions can match
    keywords:
      - acme                            # single token, exact match (case-insensitive)
      - "client onboarding"             # multi-word phrase: matched as adjacent tokens
      - jira

  Family_Photos:
    extensions: [.jpg, .jpeg, .png, .heic, .mp4]
    keywords: [family, kids, vacation, "school play"]

  # ...and so on. Add as many categories as you like.

  Unknown_Unsorted:                     # MUST exist — this is the catch-all
    extensions: []
    keywords: []

settings:
  confidence_threshold: 80              # AI answers below this fall through to keyword rules
  max_extract_chars: 1000               # upper bound on extracted text per file
  default_local_model: qwen2.5:32b      # any model you've pulled with `ollama pull`
```

**How matching actually works** (worth knowing before editing):
- The filename is first split on underscores, dashes, dots, parentheses, whitespace, and camelCase boundaries — so `MyFile_Acme-2026.pdf` becomes the tokens `[my, file, acme, 2026, pdf]`.
- **Single-word keywords** match a token exactly. Keywords ≥ 5 characters also match prefixes (`doctor` → `doctors`).
- **Multi-word keywords** (anything with a space) match as an adjacent phrase against the joined tokens (`"client onboarding"` matches `Client_Onboarding_Acme.pdf`).
- **Multi-word phrases beat single-word matches**, so `"air india"` will win over a generic `ticket` keyword, regardless of order in the YAML.
- Extensions are filters, not matches — a file is only considered for a category if its extension is in that category's `extensions` list.

### 3. Add unambiguous filename markers (high-confidence rules)

If you have filenames where one substring is a dead giveaway — invoice numbers, project codes, employer prefixes — add a regex to `HIGH_CONFIDENCE_PATTERNS` in `classifier/rules.py`. These match against the tokenised filename, return confidence 100, and short-circuit both AI and keyword rules.

```python
HIGH_CONFIDENCE_PATTERNS = [
    # ...existing patterns...
    (r'\binv\d{4,}\b',     'Financial_Taxes', "Invoice number"),
    (r'^acme\b',           'Work_Acme',       "Filename prefix 'ACME_'"),
    (r'\bproject zenith\b','Work_Acme',       "Project Zenith doc"),
]
```

Use `\b` word boundaries; patterns are anchored to the tokenised, lowercased filename (spaces between tokens). Run `pytest tests/test_rules.py -k <your_keyword>` after adding patterns to lock them in with a regression test.

### 4. Tune the AI prompt (optional)

`classifier/ai_local.py` contains a `PROMPT_TEMPLATE` with disambiguation rules ("EVL letters → PR docs, not Career"). When you change the category set, edit those rules so the LLM understands your taxonomy. Each rule should explain *why* a category is what it is, plus what it explicitly **isn't**, so the model has tie-breakers.

### 5. Adjust redaction (optional)

`classifier/redactor.py` strips emails, phone numbers, URLs, JWTs, and AWS keys from extracted text before it reaches the LLM. Add patterns there if you want to redact extra entities (medical record numbers, internal employee IDs, etc.) before any text leaves the machine.

### 6. Verify

```bash
smartsort check-rules                   # validates YAML, lists categories + counts
smartsort run ~/Downloads --no-ai       # rules-only dry-run
smartsort run ~/Downloads -vv           # debug logs (which classifier picked what)
python -m pytest tests/ -q              # full test suite
```

Then write a couple of regression tests in `tests/test_rules.py` for the filenames you care most about — five minutes of test-writing now will save you debugging when you tweak rules later.

## Settings reference

| Setting | Purpose |
| --- | --- |
| `confidence_threshold` | Minimum AI confidence (0–100) before AI's answer is accepted. Below this, the file falls through to the keyword rules. |
| `max_extract_chars` | Upper bound on characters extracted per file before sending to the LLM. Lower = faster, less context for the model. |
| `default_local_model` | Ollama model tag (e.g. `qwen2.5:32b`, `llama3.1:8b`). Must be pulled locally. |

## Project layout

```
classifier/
  types.py        # FileItem + Classification dataclasses (the shared vocabulary)
  pipeline.py     # Classifier Protocol + ClassificationPipeline runner
  classifiers.py  # concrete pipeline classifiers (HighConfidence / AI / Rules)
  rules.py        # filename tokeniser + high-confidence regex + keyword engine
  extractor.py    # PDF / DOCX / CSV / text extraction (multi-page) + redaction
  ai_local.py     # split into OllamaClient + build_prompt + parse_response
  redactor.py     # PII / secret redaction before any text leaves the machine
movers/
  organizer.py    # idempotent moves, undo log, category-folder awareness
config/
  categories.yaml
tests/
  test_rules.py        # filename-classification regression suite (~55 cases)
  test_pipeline.py     # pipeline ordering / fallback / exception handling
  test_ai_local.py     # mocked Ollama: health, prompt, parse, transport errors
  test_extractor.py    # PDF / DOCX / CSV / text + redaction integration
  test_redactor.py     # PII / secret pattern coverage
  test_organizer.py    # move + undo round-trips, collision suffixing
  test_dryrun_smoke.py # end-to-end CLI dry-run on a fixture directory
.github/workflows/ci.yml
main.py                # `run`, `undo`, `check-rules` Typer commands
pyproject.toml
```

## Tests

```bash
python -m pytest tests/ -q
```

114+ tests covering the rules engine, the typed Classification pipeline, the Ollama client (with the network fully mocked), text extraction, the PII redactor, organizer move + undo round-trips, and an end-to-end CLI dry-run smoke test that asserts every fixture is routed to the expected category. CI runs the same suite plus an out-of-process CLI dry-run; see `.github/workflows/ci.yml`.

## Privacy

- All classification runs locally. The AI step talks only to `http://localhost:11434` (Ollama).
- `classifier/redactor.py` strips emails, phone numbers, URLs, JWTs, and AWS access keys from extracted text before it reaches the model.
- The undo log only stores filesystem paths and timestamps — no file contents.
