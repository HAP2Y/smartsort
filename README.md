# SmartSort

Intelligent, local-first, privacy-preserving file organization for macOS (and Linux).

SmartSort scans a messy directory and organizes it by **content and context**, not just file extensions. It runs a fast filename rules engine first, falls back to a local LLM (via Ollama) for ambiguous files, and never sends your documents to a cloud service.

## How it classifies

Files are classified through three ordered passes:

1. **High-confidence filename overrides** — IMM forms, T4 slips, IRCC, IELTS, WES, NOC, LMIA, PCC, ECA, ITA, "employment verification", "police clearance", `PR_` prefix, etc. These short-circuit any further analysis.
2. **Local AI** — Ollama reads the first few pages of each file (PDFs, DOCX, CSVs, text). Sensitive content is redacted (emails, phone numbers, JWTs, AWS keys) before being passed to the model.
3. **Keyword + extension fallback** — multi-word phrases (e.g. `"air india"`, `"reality flip"`) are checked before single-word keywords so the more specific signal wins. Generic archives (`.zip`, `.dmg`, `.pkg`) fall through to `Archives_and_Apps`.

The category set is defined in `config/categories.yaml` and includes: `Canadian_PR_Docs`, `AstroQuant_Sidereal`, `Guidewire_PSE_Work`, `Resumes_Career_Tech`, `Financial_Taxes`, `Medical_Health`, `Travel_Transit`, `Franchise_Business_Research`, `Media_Images`, `Archives_and_Apps`, `Metadata_System`, `Unknown_Unsorted`.

## Prerequisites

1. Python 3.10+
2. (Optional, for AI step) [Ollama](https://ollama.com/download) with a local model:
   ```bash
   ollama pull qwen2.5:32b
   ```
   The model name lives in `config/categories.yaml` under `settings.default_local_model`.
3. Install Python deps:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Dry-run (default — shows the plan, moves nothing):

```bash
python main.py run ~/Downloads
```

Apply the plan:

```bash
python main.py run ~/Downloads --apply
```

Skip the AI step entirely (rules only):

```bash
python main.py run ~/Downloads --no-ai
```

Undo the last sort in a directory:

```bash
python main.py undo ~/Downloads
```

Each `--apply` writes a `.smartsort_undo.json` log into the target directory; `undo` reads that log, restores files to their original paths, and removes empty category folders.

Files already nested inside a SmartSort category folder are skipped on subsequent runs, so re-running over the same directory is safe and idempotent.

## Configuration

`config/categories.yaml` controls the category list, allowed extensions per category, keyword lists, and engine settings:

| Setting | Purpose |
| --- | --- |
| `confidence_threshold` | Minimum AI confidence (0–100) before AI's answer is accepted. Below this, falls through to keyword rules. |
| `max_extract_chars` | Upper bound on characters extracted per file before sending to the LLM. |
| `default_local_model` | Ollama model tag (e.g. `qwen2.5:32b`). |

Add a new category by appending to `categories:` with `extensions:` and `keywords:` lists. Multi-word keywords like `"employment verification"` are matched as adjacent-token phrases.

For unambiguous filename markers, add a regex to `HIGH_CONFIDENCE_PATTERNS` in `classifier/rules.py` — these win over both AI and keyword matches.

## Project layout

```
classifier/
  rules.py        # filename tokenizer + high-confidence regex + keyword engine
  extractor.py    # PDF / DOCX / CSV / text extraction with multi-page support
  ai_local.py     # Ollama client + classification prompt
  redactor.py     # PII / secret redaction before any text leaves the machine
movers/
  organizer.py    # idempotent moves, undo log, category-folder awareness
config/
  categories.yaml
tests/
  test_rules.py   # filename-classification regression suite
main.py           # `run` and `undo` CLI commands
```

## Tests

```bash
python -m pytest tests/ -q
```

The test suite locks in classification for ~55 representative filenames covering each category, including the underscore / camelCase tokenization fixes (`Happy_imm5476e_Signed.pdf`, `previewFormPCCDetail.pdf`, `MSUBARODA_WESEducationalCredentialsForwarding.pdf`) and the EVL → `Canadian_PR_Docs` routing.

## Privacy

- All classification runs locally. The AI step talks only to `http://localhost:11434` (Ollama).
- `classifier/redactor.py` strips emails, phone numbers, URLs, JWTs, and AWS access keys from extracted text before it reaches the model.
- The undo log only stores filesystem paths and timestamps — no file contents.
