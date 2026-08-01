"""SmartSort CLI entry point."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

import typer
import yaml
from rich.console import Console
from rich.table import Table

from classifier.ai_local import LocalAIClassifier
from classifier.classifiers import (
    HighConfidenceRulesClassifier,
    LocalAIPipelineClassifier,
    RulesClassifier,
)
from classifier.extractor import FileExtractor
from classifier.pipeline import ClassificationPipeline
from classifier.rules import RulesEngine
from classifier.types import Classification, FileItem
from inference import Orchestrator, Router, Worker, build_backend
from inference.router import (
    ROUTE_AI_LARGE,
    ROUTE_AI_SMALL,
    ROUTE_OCR,
    ROUTE_RULES,
)
from movers.offboarding import Disposition, PolicyError, RetentionPolicy
from movers.organizer import Organizer

CONFIG_PATH = Path(__file__).parent / "config" / "categories.yaml"
OFFBOARD_POLICY_PATH = Path(__file__).parent / "config" / "offboarding.yaml"

app = typer.Typer(help="SmartSort - Local-first file classification & sorting", no_args_is_help=True)
console = Console()
log = logging.getLogger("smartsort")


# ---------------------------------------------------------------------- helpers


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _gather_files(target: Path, recursive: bool, category_names: set[str]) -> list[Path]:
    """Collect files to classify, skipping system files and already-organised ones."""
    candidates: Iterable[Path] = target.rglob("*") if recursive else target.iterdir()
    out: list[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        if p.name.startswith(".smartsort"):
            continue
        try:
            rel_parts = p.relative_to(target).parts
        except ValueError:
            continue
        if rel_parts and rel_parts[0] in category_names:
            continue
        out.append(p)
    return out


def _build_pipeline(config: dict, no_ai: bool) -> tuple[ClassificationPipeline, str]:
    """Wire up the classifier pipeline. Returns (pipeline, ai_status_message)."""
    rules = RulesEngine(str(CONFIG_PATH))
    threshold = config["settings"]["confidence_threshold"]
    categories = list(config["categories"].keys())

    classifiers = [HighConfidenceRulesClassifier(rules)]
    ai_status = "AI skipped (--no-ai)."

    if not no_ai:
        ai = LocalAIClassifier(model=config["settings"]["default_local_model"])
        ok, msg = ai.is_running()
        ai_status = msg
        if ok:
            extractor = FileExtractor(max_chars=config["settings"]["max_extract_chars"])
            classifiers.append(
                LocalAIPipelineClassifier(ai, extractor, categories, threshold, enabled=True)
            )
        else:
            log.warning("AI disabled: %s", msg)

    classifiers.append(RulesClassifier(rules))
    return ClassificationPipeline(classifiers), ai_status


# ------------------------------------------------------------------- commands


@app.command()
def run(
    target_dir: str = typer.Argument(..., help="Directory to sort"),
    apply: bool = typer.Option(False, "--apply", help="Apply changes. Defaults to dry-run."),
    no_ai: bool = typer.Option(False, "--no-ai", help="Skip AI classification entirely (rules only)."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
    distributed: bool = typer.Option(
        False, "--distributed",
        help="Run via the queue + worker runtime instead of the inline pipeline.",
    ),
    backend: str = typer.Option(
        "memory", "--backend",
        help="Queue backend when --distributed: 'memory' (in-process) or 'redis'.",
    ),
    redis_url: str = typer.Option(
        "redis://localhost:6379/0", "--redis-url",
        help="Redis URL when --backend=redis.",
    ),
    workers: int = typer.Option(
        2, "--workers",
        help="Inline workers per route (memory backend only). Use 0 with redis to defer to external workers.",
    ),
    timeout: float = typer.Option(
        900.0, "--timeout",
        help="Max seconds to wait for distributed results before filling Unknown_Unsorted. "
             "Budget ~3-4s per file per worker: 200 files over 4 workers needs ~200s, "
             "but a cold Ollama model load can add a minute on the first call.",
    ),
    up: bool = typer.Option(
        False, "--up",
        help="With --backend=redis, run `docker compose up -d --build` first to start the worker fleet, "
             "then dispatch. Workers are left running afterward for reuse.",
    ),
    down: bool = typer.Option(
        False, "--down",
        help="After dispatching, run `docker compose down` to tear the fleet back down.",
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (-v, -vv)."),
):
    """Sort files in TARGET_DIR.

    Two modes:

      smartsort run <dir>                       # local: one process, inline pipeline
      smartsort run <dir> --distributed         # distributed: router + queues + workers
      smartsort run <dir> --distributed --backend redis --redis-url redis://...
    """
    _configure_logging(verbose)

    target = Path(target_dir).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        console.print(f"[red]Error: Directory {target_dir} not found.[/red]")
        raise typer.Exit(1)

    config = _load_config()
    categories = list(config["categories"].keys())
    files = _gather_files(target, recursive=recursive, category_names=set(categories))
    scope = "recursively" if recursive else "at top level"
    console.print(f"[cyan]Found {len(files)} files {scope} in {target}.[/cyan]")
    if not files:
        console.print("[yellow]Nothing to do.[/yellow]")
        return

    if (up or down) and not (distributed and backend == "redis"):
        console.print(
            "[yellow]--up/--down only apply with --distributed --backend redis; ignoring.[/yellow]"
        )
        up = False
        down = False

    brought_up = False
    if up:
        # Pre-route locally to size the fleet — every file goes through
        # an AI worker, so the autoscaler counts cover the full workload.
        counts = _route_counts(files, Router.default())
        targets = _scale_targets(counts)
        console.print(
            "[dim]Route counts: "
            + (", ".join(f"{r}={n}" for r, n in sorted(counts.items())) or "none")
            + "[/dim]"
        )
        brought_up = _compose_up(scale=targets)
        if not brought_up:
            raise typer.Exit(1)

    try:
        if distributed:
            plan = _run_distributed(
                files=files,
                config=config,
                backend=backend,
                redis_url=redis_url,
                workers=workers,
                timeout=timeout,
            )
        else:
            plan = _run_local(files=files, config=config, no_ai=no_ai)

        _print_plan(plan, apply)

        if apply:
            organizer = Organizer(str(target), category_names=categories)
            organizer.move_files({fp: c.to_dict() for fp, c in plan.items()}, apply=True)
            console.print(f"[bold green]\nFiles sorted. Undo log: {organizer.undo_log}[/bold green]")
            console.print("[dim]Run `smartsort undo <dir>` to revert.[/dim]")
        else:
            console.print("\n[yellow]Dry-run complete. Run with --apply to move files.[/yellow]")
    finally:
        if down and brought_up:
            _compose_down()
        elif brought_up and not down:
            console.print(
                "[dim]Worker fleet left running. "
                "Stop it with `docker compose down` or re-run with --down.[/dim]"
            )


def _run_local(*, files: list[Path], config: dict, no_ai: bool) -> dict[str, Classification]:
    """Inline single-process pipeline (the original v0.2 path)."""
    with console.status("[bold yellow]Building pipeline (Ollama health check)..."):
        pipeline, ai_status = _build_pipeline(config, no_ai=no_ai)
    console.log(ai_status)

    plan: dict[str, Classification] = {}
    with console.status("[bold green]Classifying files...") as status:
        for path in files:
            status.update(f"[bold green]Classifying: {path.name}")
            plan[str(path)] = pipeline.classify(FileItem(path=path))
    return plan


# --------------------------------------------------- compose lifecycle


# Per-route scale tuning.
#
# Only AI workers exist as Compose services — rules are run inline on the
# dispatcher (the queue is for expensive work, not for filename matching),
# and OCR has no implementation yet so OCR-routed files are classified as
# Unknown locally rather than enqueued.
#
# `files_per_worker` is the soft saturation point. `max_workers` caps the
# autoscaler so a directory of 10k files doesn't spawn 400 containers.
# Caps stay low because a single host Ollama serialises LLM calls — replicas
# past 2-3 add memory pressure without throughput. Point at a multi-instance
# Ollama setup and raise the caps to scale further.
COMPOSE_SCALE = {
    "ai-small-worker": {"files_per_worker": 25, "max_workers": 2},
    "ai-large-worker": {"files_per_worker": 10, "max_workers": 1},
}


def _compose_cmd() -> list[str] | None:
    """Return the docker-compose invocation that works on this host, or None."""
    import shutil
    if shutil.which("docker"):
        # docker compose (v2 plugin) is the modern form; docker-compose (v1)
        # is the legacy binary. We'll prefer v2 and fall back to v1.
        import subprocess
        probe = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None


def _route_counts(files: list[Path], router: Router) -> dict[str, int]:
    """Pure pre-pass: ask the router where each file would go, without enqueuing."""
    counts: dict[str, int] = {}
    for path in files:
        route = router.route(FileItem(path=path))
        counts[route] = counts.get(route, 0) + 1
    return counts


def _scale_targets(counts: dict[str, int]) -> dict[str, int]:
    """Map compose service -> desired replicas, sized for the workload.

    Only AI services scale here — rules and OCR are not Compose services
    in the new architecture (rules runs inline; OCR is unimplemented and
    classified Unknown on the dispatcher).
    """
    import math
    ai_small = counts.get(ROUTE_AI_SMALL, 0)
    ai_large = counts.get(ROUTE_AI_LARGE, 0)

    def _scale(jobs: int, cfg: dict) -> int:
        if jobs <= 0:
            return 1  # keep one warm worker in case stray jobs land
        return min(cfg["max_workers"], max(1, math.ceil(jobs / cfg["files_per_worker"])))

    return {
        "ai-small-worker": _scale(ai_small, COMPOSE_SCALE["ai-small-worker"]),
        "ai-large-worker": _scale(ai_large, COMPOSE_SCALE["ai-large-worker"]),
    }


def _compose_up(scale: dict[str, int] | None = None) -> bool:
    """Start the worker fleet via `docker compose up -d --build`.

    Returns True on success. Honours `scale` by passing `--scale svc=N` for
    each service so the fleet is sized to the workload from the start.
    """
    import subprocess
    base = _compose_cmd()
    if base is None:
        console.print("[red]--up requires docker (compose v2) or docker-compose on PATH.[/red]")
        return False
    if not Path("docker-compose.yml").exists():
        console.print("[red]--up needs docker-compose.yml in the current directory.[/red]")
        return False

    cmd = base + ["up", "-d", "--build"]
    if scale:
        for service, n in scale.items():
            cmd += ["--scale", f"{service}={n}"]
        scale_summary = ", ".join(f"{s}={n}" for s, n in scale.items())
        console.print(f"[bold cyan]Bringing fleet up: {scale_summary}[/bold cyan]")
    else:
        console.print("[bold cyan]Bringing fleet up...[/bold cyan]")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print("[red]docker compose up failed; see output above.[/red]")
        return False

    # Workers subscribe to Redis on startup; give them a beat so the first
    # XADD doesn't land before any consumer has joined the group.
    import time
    time.sleep(2)
    return True


def _compose_down() -> None:
    import subprocess
    base = _compose_cmd()
    if base is None:
        return
    console.print("[bold cyan]Tearing fleet down...[/bold cyan]")
    subprocess.run(base + ["down"])


# --------------------------------------------------- distributed runner


def _run_distributed(
    *,
    files: list[Path],
    config: dict,
    backend: str,
    redis_url: str,
    workers: int,
    timeout: float,
) -> dict[str, Classification]:
    """Producer/worker path via the inference package.

    Architecture: every file goes onto an AI worker queue. Workers run
    AI-first (Local AI → HC rules → fallback rules), so the LLM gets to
    weigh in on every file and rules only catch the cases where AI
    declines. The router still picks ai-small / ai-large by size +
    extension; images route to UNROUTABLE (no OCR worker today) and are
    classified Unknown_Unsorted on the dispatcher with a clear reason.
    """
    from inference.diagnostics import (
        ProgressReporter,
        preflight_ollama,
        preflight_redis,
        preflight_workers,
        queue_depths,
        tail_compose_logs,
    )
    from inference.router import ROUTE_UNROUTABLE
    from inference.prefilter import Prefilter
    from classifier.ai_local import DEFAULT_OLLAMA_URL

    backend_kwargs = {"url": redis_url} if backend == "redis" else {}
    qb = build_backend(backend, **backend_kwargs)
    router = Router.default()

    # -------------------- Pre-route locally. Files no worker can read
    # (archives, installers, spreadsheets, extensionless files, and images
    # while there is no OCR worker) are classified here by filename rules
    # rather than being stamped Unknown or enqueued to a queue nobody
    # drains. AI-first is preserved for everything a worker *can* read.
    prefilter = Prefilter(RulesEngine(str(CONFIG_PATH)))
    plan: dict[str, Classification] = {}
    enqueueable: list[Path] = []
    prefiltered_hits = 0
    for path in files:
        item = FileItem(path=path)
        route = router.route(item)
        if route == ROUTE_UNROUTABLE:
            verdict = prefilter.classify(item)
            plan[str(path)] = verdict
            if verdict.is_known:
                prefiltered_hits += 1
        else:
            enqueueable.append(path)

    if plan:
        console.print(
            f"[cyan]Prefilter handled {len(plan)} non-worker files "
            f"({prefiltered_hits} classified, {len(plan) - prefiltered_hits} unknown).[/cyan]"
        )

    if not enqueueable:
        console.print("[cyan]No files require worker processing.[/cyan]")
        qb.close()
        return plan

    # -------------------- Inline workers for the memory backend
    inline_workers: list[Worker] = []
    threads: list = []
    if workers > 0 and backend == "memory":
        for route in (ROUTE_AI_SMALL, ROUTE_AI_LARGE):
            for i in range(workers):
                classifier = _build_worker_classifier(route, config, None)
                w = Worker(name=f"{route}-{i}", routes=[route], classifier=classifier, backend=qb)
                inline_workers.append(w)
                threads.append(w.run_in_thread())
        console.print(f"[dim]Spawned {len(inline_workers)} inline AI workers.[/dim]")
    elif workers > 0 and backend == "redis":
        console.print(
            "[yellow]--workers ignored with --backend=redis; "
            "start external workers via `smartsort serve-worker` or docker-compose.[/yellow]"
        )

    # Per-route counts so preflight knows which routes need consumers
    # and progress reporting can show denominators.
    expected_by_route: dict[str, int] = {}
    for path in enqueueable:
        r = router.route(FileItem(path=path))
        expected_by_route[r] = expected_by_route.get(r, 0) + 1

    # -------------------- Pre-flight
    if backend == "redis":
        console.rule("[bold cyan]Pre-flight")
        checks = [preflight_redis(redis_url)]
        active_routes = [r for r, n in expected_by_route.items() if n > 0]
        checks.extend(preflight_workers(redis_url, active_routes))
        if any(r in active_routes for r in (ROUTE_AI_SMALL, ROUTE_AI_LARGE)):
            checks.append(preflight_ollama(DEFAULT_OLLAMA_URL))
        for c in checks:
            colour = "green" if c.ok else "red"
            console.print(f"[{colour}]{c.status} {c.name}[/{colour}]: {c.detail}")
        blockers = [c for c in checks if not c.ok and c.name == "redis"]
        if blockers:
            console.print(
                "[red]Aborting: cannot reach Redis. Bring the fleet up with --up "
                "or start it manually with `docker-compose up -d`.[/red]"
            )
            qb.close()
            raise typer.Exit(1)
        console.rule()

    # -------------------- Submit + drain
    orchestrator = Orchestrator(backend=qb, router=router)
    pending = orchestrator.submit(enqueueable)
    console.print(f"[cyan]Submitted {len(pending)} jobs via {backend} backend.[/cyan]")
    for route, count in orchestrator.stats.by_route.items():
        console.print(f"  [magenta]{route}[/magenta]: {count}")

    progress = ProgressReporter(
        expected_total=len(pending),
        expected_by_route=dict(orchestrator.stats.by_route),
        tick_seconds=5.0,
        _printer=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )
    worker_plan = orchestrator.collect(pending, timeout=timeout, on_result=progress.on_result)
    progress.final()
    plan.update(worker_plan)

    for w in inline_workers:
        w.stop()
    for t in threads:
        t.join(timeout=2.0)

    # -------------------- Post-mortem (only on failures)
    if backend == "redis" and orchestrator.stats.failed > 0:
        console.rule("[bold red]Post-mortem")
        depths = queue_depths(qb, list(expected_by_route))
        for route, depth in depths.items():
            if depth > 0:
                console.print(f"[red]queue {route}[/red]: {depth} entries still in flight")
        services = []
        if depths.get(ROUTE_AI_SMALL, 0) > 0:
            services.append("ai-small-worker")
        if depths.get(ROUTE_AI_LARGE, 0) > 0:
            services.append("ai-large-worker")
        if services:
            logs = tail_compose_logs(services, lines=10)
            for svc, body in logs.items():
                console.print(f"[bold]── {svc} (last 10 log lines) ──[/bold]")
                console.print(body or "[dim](empty)[/dim]")
        console.print(
            "\n[yellow]Hints:[/yellow]\n"
            "  • Smaller / faster model: set `models.ai-small: qwen2.5:3b` in config/categories.yaml.\n"
            "  • Longer timeout: --timeout 1200.\n"
            "  • More replicas: `docker-compose up -d --scale ai-small-worker=N`.\n"
            "  • Lower confidence_threshold so the LLM's lower-confidence answers count.\n"
        )
        console.rule()

    qb.close()

    console.print(
        f"[dim]submitted={orchestrator.stats.submitted} "
        f"completed={orchestrator.stats.completed} "
        f"failed={orchestrator.stats.failed}[/dim]"
    )
    return plan


@app.command()
def undo(
    target_dir: str = typer.Argument(..., help="Directory whose last sort to revert"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
):
    """Revert the most recent sort using the .smartsort_undo.json log."""
    _configure_logging(verbose)

    target = Path(target_dir).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        console.print(f"[red]Error: Directory {target_dir} not found.[/red]")
        raise typer.Exit(1)

    config = _load_config()
    organizer = Organizer(str(target), category_names=list(config["categories"].keys()))
    restored, missing, errors = organizer.undo()

    console.print(f"[green]Restored:[/green] {restored}")
    if missing:
        console.print(f"[yellow]Missing (already moved/deleted):[/yellow] {missing}")
    for err in errors:
        console.print(f"[red]Error:[/red] {err}")


@app.command()
def offboard(
    target_dir: str = typer.Argument(
        None, help="Directory to scan (e.g. ~/Downloads). Not needed with --explain.",
    ),
    export_to: str = typer.Option(
        "./Offboarding_Bundle", "--export-to",
        help="Where the KEEP bundle is written, organised by category.",
    ),
    apply: bool = typer.Option(False, "--apply", help="Actually write the bundle. Defaults to dry-run."),
    move: bool = typer.Option(
        False, "--move",
        help="Move KEEP files instead of copying. Default is copy, so originals "
             "stay put until you have verified the bundle.",
    ),
    no_ai: bool = typer.Option(False, "--no-ai", help="Rules only, skip the LLM."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
    explain: bool = typer.Option(False, "--explain", help="Print the retention policy and exit."),
    manifest: str = typer.Option(
        None, "--manifest",
        help="Write a CSV audit trail of every file and its disposition. "
             "Defaults to <export-to>/manifest.csv when --apply is used.",
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
):
    """Separate your own records from the company's work product.

    Classifies every file, then applies the retention policy in
    config/offboarding.yaml to split them three ways:

      KEEP    your records — immigration, payslips, tax forms, offer and
              relieving letters, certifications, resumes, personal projects
      LEAVE   company work product and customer data
      REVIEW  credentials and anything the classifier wasn't sure about

    Only KEEP files are exported, and files are copied (not moved) unless
    you pass --move. Credentials are never exported under any flag.
    """
    _configure_logging(verbose)

    config = _load_config()
    categories = list(config["categories"].keys())
    policy = RetentionPolicy.load(OFFBOARD_POLICY_PATH)

    try:
        policy.validate(categories)
    except PolicyError as e:
        console.print(f"[red]Retention policy is out of sync with categories.yaml:[/red]\n  {e}")
        raise typer.Exit(1)

    if explain:
        _print_policy(policy)
        return

    if not target_dir:
        console.print("[red]Error: TARGET_DIR is required (omit it only with --explain).[/red]")
        raise typer.Exit(1)

    target = Path(target_dir).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        console.print(f"[red]Error: Directory {target_dir} not found.[/red]")
        raise typer.Exit(1)

    files = _gather_files(target, recursive=recursive, category_names=set(categories))
    scope = "recursively" if recursive else "at top level"
    console.print(f"[cyan]Found {len(files)} files {scope} in {target}.[/cyan]")
    if not files:
        console.print("[yellow]Nothing to do.[/yellow]")
        return

    plan = _run_local(files=files, config=config, no_ai=no_ai)
    buckets = policy.partition(plan)

    console.rule("[bold blue]Offboarding Split")
    _print_disposition(buckets, policy)

    destination = Path(export_to).expanduser().resolve()
    result = policy.export(plan, destination, apply=apply, move=move)

    console.rule()
    verb = "Moved" if move else "Copied"
    if apply:
        console.print(f"[bold green]{verb} {result.count} files into {destination}[/bold green]")
        for err in result.errors:
            console.print(f"[red]Error:[/red] {err}")
    else:
        console.print(
            f"[yellow]Dry-run: {result.count} files would be "
            f"{'moved' if move else 'copied'} into {destination}.[/yellow]"
        )
        console.print("[dim]Re-run with --apply to write the bundle.[/dim]")

    manifest_path = _write_manifest(plan, policy, destination, manifest, apply)
    if manifest_path:
        console.print(f"[dim]Manifest: {manifest_path}[/dim]")

    review = buckets[Disposition.REVIEW]
    if review:
        console.print(
            f"\n[bold yellow]⚠ {len(review)} file(s) need your decision — "
            "nothing was done with them. They remain in the source folder.[/bold yellow]"
        )
        # Counts alone are not actionable: you cannot decide about a file you
        # cannot name. List every REVIEW file, secrets first.
        secrets = [p for p, c in review if c in policy.never_export]
        others = [p for p, c in review if c not in policy.never_export]

        if secrets:
            console.print(
                "\n[red]Credential-bearing — never exported. Rotate or destroy "
                "rather than taking copies:[/red]"
            )
            for p in sorted(secrets):
                console.print(f"    [red]{Path(p).name}[/red]")

        if others:
            console.print(
                f"\n[yellow]Unclassified ({len(others)}) — the classifier had no "
                "confident answer. Check these by hand:[/yellow]"
            )
            for p in sorted(others):
                console.print(f"    {Path(p).name}")


def _write_manifest(
    plan: dict[str, Classification],
    policy: RetentionPolicy,
    destination: Path,
    manifest: str | None,
    apply: bool,
) -> Path | None:
    """Write a full audit trail of every file and what happened to it.

    Offboarding is exactly the situation where you may later need to show
    what you took and what you left, so the manifest covers all three
    dispositions rather than only the exported set.
    """
    if manifest:
        path = Path(manifest).expanduser().resolve()
    elif apply:
        path = destination / "manifest.csv"
    else:
        return None

    rows = []
    for filepath, c in plan.items():
        disp = policy.disposition(c.category)
        exported = disp is Disposition.KEEP and policy.is_exportable(c.category)
        rows.append({
            "disposition": disp.value,
            "exported": "yes" if (exported and apply) else "no",
            "category": c.category,
            "confidence": c.confidence,
            "method": c.method,
            "filename": Path(filepath).name,
            "source_path": filepath,
            "reason": c.reason,
        })
    rows.sort(key=lambda r: (r["disposition"], r["category"], r["filename"]))

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["disposition"])
            writer.writeheader()
            writer.writerows(rows)
    except OSError as e:
        console.print(f"[red]Could not write manifest:[/red] {e}")
        return None
    return path


def _print_policy(policy: RetentionPolicy) -> None:
    table = Table(title="Offboarding retention policy")
    table.add_column("Disposition", style="bold")
    table.add_column("Categories")
    styles = {Disposition.KEEP: "green", Disposition.LEAVE: "red", Disposition.REVIEW: "yellow"}
    for disp in Disposition:
        cats = sorted(c for c, d in policy.by_category.items() if d is disp)
        table.add_row(f"[{styles[disp]}]{disp.value.upper()}[/{styles[disp]}]", "\n".join(cats))
    console.print(table)
    if policy.never_export:
        console.print(
            "[red]Never exported under any flag:[/red] " + ", ".join(sorted(policy.never_export))
        )


def _print_disposition(buckets: dict, policy: RetentionPolicy) -> None:
    styles = {Disposition.KEEP: "green", Disposition.LEAVE: "red", Disposition.REVIEW: "yellow"}
    blurb = {
        Disposition.KEEP: "your records — exported",
        Disposition.LEAVE: "company / customer property — left behind",
        Disposition.REVIEW: "needs your decision — untouched",
    }
    for disp in Disposition:
        entries = buckets[disp]
        colour = styles[disp]
        console.print(
            f"\n[bold {colour}]{disp.value.upper()}[/bold {colour}] "
            f"({len(entries)} files — {blurb[disp]})"
        )
        by_cat: dict[str, int] = {}
        for _, cat in entries:
            by_cat[cat] = by_cat.get(cat, 0) + 1
        for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
            flag = "  [red](never exported)[/red]" if cat in policy.never_export else ""
            console.print(f"    {cat}: {n}{flag}")


@app.command(name="check-rules")
def check_rules():
    """Validate categories.yaml and print a summary of registered rules."""
    config = _load_config()
    cats = config["categories"]
    table = Table(title="Categories")
    table.add_column("Category", style="magenta")
    table.add_column("Extensions", style="cyan")
    table.add_column("Keywords", justify="right", style="green")
    for name, data in cats.items():
        table.add_row(
            name,
            ", ".join(data.get("extensions", [])) or "(any)",
            str(len(data.get("keywords", []))),
        )
    console.print(table)
    settings = config.get("settings", {})
    console.print(
        f"\n[dim]threshold={settings.get('confidence_threshold')} "
        f"max_extract_chars={settings.get('max_extract_chars')} "
        f"default_local_model={settings.get('default_local_model')!r}[/dim]"
    )


# -------------------------------------------------------------------- output


def _print_plan(plan: dict[str, Classification], apply: bool) -> None:
    console.rule("[bold blue]Classification Plan")
    table = Table(title="Execution Plan" if apply else "Plan (dry-run)", show_lines=True)
    table.add_column("Filename", style="cyan", max_width=40)
    table.add_column("Category", style="magenta")
    table.add_column("Conf %", justify="right", style="green")
    table.add_column("Method", style="yellow")
    table.add_column("Reason", style="white", max_width=50)

    for fp, c in plan.items():
        table.add_row(Path(fp).name, c.category, str(c.confidence), c.method, c.reason)
    console.print(table)

    summary: dict[str, int] = {}
    for c in plan.values():
        summary[c.category] = summary.get(c.category, 0) + 1
    summary_table = Table(title="Summary")
    summary_table.add_column("Category", style="magenta")
    summary_table.add_column("Files", justify="right", style="green")
    for cat, count in sorted(summary.items(), key=lambda x: -x[1]):
        summary_table.add_row(cat, str(count))
    console.print(summary_table)


# --------------------------------------------------- distributed inference


def _model_for_route(config: dict, route: str, override: str | None = None) -> str:
    """Resolve which Ollama model a given route should use.

    Resolution order:
      1. explicit --model override
      2. settings.models.<route>             (preferred, per-route)
      3. settings.large_model                (legacy, for ai-large only)
      4. settings.default_local_model        (legacy fallback)
    """
    if override:
        return override
    settings = config["settings"]
    per_route = (settings.get("models") or {}).get(route)
    if per_route:
        return per_route
    if route == ROUTE_AI_LARGE:
        return settings.get("large_model", settings["default_local_model"])
    return settings["default_local_model"]


def _build_worker_classifier(route: str, config: dict, model_override: str | None):
    """Pick the right classifier for a given queue route.

    Order is **HC-rules → AI → keyword-rules**, which splits the difference
    between the two previous designs.

    Going fully rules-first starved the LLM: any file with a familiar
    keyword short-circuited it, so the AI fleet was expensive plumbing that
    rarely ran. But going fully AI-first put the LLM ahead of markers that
    are true *by definition*, and it lost to them. Observed in one run: the
    sector-ETF exports split across two categories because the model reads
    ``Date,Open,High,Low,Close,Volume`` as generic finance — xlc/xle landed
    in AstroQuant while xlb/xlf/xlp/xlu landed in Financial_Taxes. Same for
    ``PATEL_CAN_PAY_SLIP_*``, where the model contradicted the project's own
    pay-slip rule.

    ``HighConfidenceRulesClassifier`` is a deliberately small, curated regex
    list (IMM form numbers, T4, PCC, ticker filenames) — cases where a
    filename is definitional and a 7B model second-guessing it is strictly
    worse. It matches a small minority of files, so the LLM still sees the
    bulk of the corpus. ``RulesClassifier`` (fuzzy keyword matching) stays
    *behind* the AI, which is what the AI-first change was really protecting.
    """
    rules = RulesEngine(str(CONFIG_PATH))
    categories = list(config["categories"].keys())
    threshold = config["settings"]["confidence_threshold"]

    if route in (ROUTE_AI_SMALL, ROUTE_AI_LARGE):
        model = _model_for_route(config, route, model_override)
        ai = LocalAIClassifier(model=model)
        extractor = FileExtractor(max_chars=config["settings"]["max_extract_chars"])
        return ClassificationPipeline([
            HighConfidenceRulesClassifier(rules),  # definitional filename markers win outright
            LocalAIPipelineClassifier(ai, extractor, categories, threshold, enabled=True),
            RulesClassifier(rules),                # last resort: keyword + extension fallback
        ])
    if route in (ROUTE_RULES, ROUTE_OCR):
        log.warning(
            "route %r has no dedicated worker in the new architecture; "
            "serving as rules-pipeline for backward compatibility", route,
        )
        return ClassificationPipeline([
            HighConfidenceRulesClassifier(rules),
            RulesClassifier(rules),
        ])
    raise typer.BadParameter(f"unknown route: {route!r}")


@app.command(name="serve-worker")
def serve_worker(
    routes: str = typer.Option(
        ...,
        "--routes", "-r",
        help=f"Comma-separated routes to subscribe to (any of: {ROUTE_RULES}, {ROUTE_AI_SMALL}, {ROUTE_AI_LARGE}, {ROUTE_OCR}). "
             f"One worker can cover multiple routes — handy for absorbing the OCR queue with a rules fallback.",
    ),
    backend: str = typer.Option("redis", "--backend", help="Queue backend: redis | memory."),
    redis_url: str = typer.Option("redis://localhost:6379/0", "--redis-url"),
    model: str = typer.Option(None, "--model", help="Override the Ollama model for AI routes."),
    name: str = typer.Option(None, "--name", help="Worker name (defaults to first-route + pid)."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
):
    """Run a long-lived worker that consumes jobs from one or more route queues.

    The worker uses the classifier configured for its primary route. Pass
    multiple routes to drain several queues from one process — for example
    `--routes rules,ocr` runs a single cheap worker that handles both the
    rules fallback and the OCR queue while a separate AI worker handles the
    LLM-bound routes.
    """
    _configure_logging(verbose)
    config = _load_config()

    route_list = [r.strip() for r in routes.split(",") if r.strip()]
    if not route_list:
        raise typer.BadParameter("at least one route required")

    import os
    worker_name = name or f"{route_list[0]}-{os.getpid()}"

    backend_kwargs = {"url": redis_url, "consumer_name": worker_name} if backend == "redis" else {}
    qb = build_backend(backend, **backend_kwargs)
    # Classifier is keyed off the primary route — if you mix AI and non-AI
    # routes on one worker, name the AI route first so the LLM is wired up.
    classifier = _build_worker_classifier(route_list[0], config, model)
    worker = Worker(name=worker_name, routes=route_list, classifier=classifier, backend=qb)

    # Loud, structured startup banner so the user can see at a glance which
    # model each worker is actually using and whether Ollama is reachable.
    primary = route_list[0]
    console.rule(f"[bold cyan]Worker {worker_name}")
    console.print(f"[bold]routes  [/bold] {route_list}")
    console.print(f"[bold]backend [/bold] {backend}")
    if primary in (ROUTE_AI_SMALL, ROUTE_AI_LARGE):
        from classifier.ai_local import DEFAULT_OLLAMA_URL, LocalAIClassifier
        resolved_model = _model_for_route(config, primary, model)
        console.print(f"[bold]model   [/bold] {resolved_model}")
        console.print(f"[bold]ollama  [/bold] {DEFAULT_OLLAMA_URL}")
        ok, msg = LocalAIClassifier(model=resolved_model).is_running()
        colour = "green" if ok else "red"
        console.print(f"[{colour}]health  → {msg}[/{colour}]")
        if not ok:
            console.print(
                "[yellow]Worker will run with rules-fallback only until Ollama is reachable.[/yellow]"
            )
    else:
        console.print(f"[bold]model   [/bold] (rules-only worker, no LLM)")
    console.rule()
    console.print("[dim]Ctrl-C to stop.[/dim]")
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
        console.print("\n[yellow]Worker stopping...[/yellow]")
    finally:
        qb.close()
        console.print(
            f"[cyan]Processed {worker.stats.processed}, "
            f"failed {worker.stats.failed}.[/cyan]"
        )


if __name__ == "__main__":
    app()
