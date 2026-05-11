"""SmartSort CLI entry point."""
from __future__ import annotations

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
from movers.organizer import Organizer

CONFIG_PATH = Path(__file__).parent / "config" / "categories.yaml"

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
        300.0, "--timeout",
        help="Max seconds to wait for distributed results before filling Unknown_Unsorted.",
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
        # Pre-route locally to size the fleet for the actual workload.
        # Router.route() is pure (filename + size only), so doing it here and
        # again inside the orchestrator is cheap and avoids leaking router
        # internals into _compose_up.
        counts = _route_counts(files, Router.default())
        targets = _scale_targets(counts)
        console.print(
            "[dim]Route counts: "
            + ", ".join(f"{r}={n}" for r, n in sorted(counts.items()))
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
# `files_per_worker` is the soft saturation point — above it we add another
# worker. `max_workers` caps the autoscaler so a directory of 10k files
# doesn't try to spawn 400 containers.
#
# AI route caps are deliberately conservative. The throughput bottleneck on
# AI routes is the single host Ollama instance, not the Python worker — once
# Ollama is saturated, extra workers just queue on it while burning memory
# (each container is ~150 MB, and the Ollama-side serialisation means
# replicas past 2-3 give zero throughput gain on a single GPU / CPU). If you
# point workers at a multi-instance Ollama setup, raise these caps.
#
# rules-worker subscribes to both `rules` and `ocr`, so its scaling target
# combines the two route counts.
COMPOSE_SCALE = {
    "rules-worker":    {"files_per_worker": 100, "max_workers": 2},
    "ai-small-worker": {"files_per_worker": 50,  "max_workers": 2},
    "ai-large-worker": {"files_per_worker": 20,  "max_workers": 1},
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
    """Map compose service -> desired replicas, sized for the workload."""
    import math
    # rules-worker handles both rules and ocr queues (see docker-compose.yml).
    rules_total = counts.get(ROUTE_RULES, 0) + counts.get(ROUTE_OCR, 0)
    ai_small    = counts.get(ROUTE_AI_SMALL, 0)
    ai_large    = counts.get(ROUTE_AI_LARGE, 0)

    def _scale(jobs: int, cfg: dict) -> int:
        if jobs <= 0:
            return 1  # keep one warm worker in case stray jobs land
        return min(cfg["max_workers"], max(1, math.ceil(jobs / cfg["files_per_worker"])))

    return {
        "rules-worker":    _scale(rules_total, COMPOSE_SCALE["rules-worker"]),
        "ai-small-worker": _scale(ai_small,    COMPOSE_SCALE["ai-small-worker"]),
        "ai-large-worker": _scale(ai_large,    COMPOSE_SCALE["ai-large-worker"]),
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
    """Producer/worker path via the inference package."""
    from inference.diagnostics import (
        ProgressReporter,
        preflight_ollama,
        preflight_redis,
        preflight_workers,
        queue_depths,
        tail_compose_logs,
    )
    from classifier.ai_local import DEFAULT_OLLAMA_URL

    backend_kwargs = {"url": redis_url} if backend == "redis" else {}
    qb = build_backend(backend, **backend_kwargs)
    router = Router.default()

    inline_workers: list[Worker] = []
    threads: list = []
    if workers > 0 and backend == "memory":
        for route in (ROUTE_RULES, ROUTE_AI_SMALL, ROUTE_AI_LARGE, ROUTE_OCR):
            for i in range(workers):
                classifier = _build_worker_classifier(route, config, None)
                w = Worker(name=f"{route}-{i}", routes=[route], classifier=classifier, backend=qb)
                inline_workers.append(w)
                threads.append(w.run_in_thread())
        console.print(f"[dim]Spawned {len(inline_workers)} inline workers across 4 routes.[/dim]")
    elif workers > 0 and backend == "redis":
        console.print(
            "[yellow]--workers ignored with --backend=redis; "
            "start external workers via `smartsort serve-worker` or docker compose.[/yellow]"
        )

    # Compute the per-route counts up-front so preflight knows which
    # routes need consumers and progress can show denominators.
    expected_by_route = _route_counts(files, router)

    # ------------------------------------------------- preflight checks
    if backend == "redis":
        console.rule("[bold cyan]Pre-flight")
        checks = [preflight_redis(redis_url)]
        # Only require workers on routes that actually have jobs queued.
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
        # Worker / Ollama failures are warnings only — the orchestrator's
        # timeout will still produce a usable plan with fallbacks.
        console.rule()

    orchestrator = Orchestrator(backend=qb, router=router)
    pending = orchestrator.submit(files)
    console.print(f"[cyan]Submitted {len(pending)} jobs via {backend} backend.[/cyan]")
    for route, count in orchestrator.stats.by_route.items():
        console.print(f"  [magenta]{route}[/magenta]: {count}")

    # ------------------------------------------------- live progress
    progress = ProgressReporter(
        expected_total=len(pending),
        expected_by_route=dict(orchestrator.stats.by_route),
        tick_seconds=5.0,
        _printer=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )
    plan = orchestrator.collect(pending, timeout=timeout, on_result=progress.on_result)
    progress.final()

    for w in inline_workers:
        w.stop()
    for t in threads:
        t.join(timeout=2.0)

    # ------------------------------------------------- post-mortem
    if backend == "redis" and orchestrator.stats.failed > 0:
        console.rule("[bold red]Post-mortem")
        depths = queue_depths(qb, list(expected_by_route))
        for route, depth in depths.items():
            if depth > 0:
                console.print(f"[red]queue {route}[/red]: {depth} entries still in flight")
        # Best-effort: dump the last few log lines from each compose
        # service so the user doesn't have to context-switch to find
        # them. Skipped silently if docker-compose isn't on PATH.
        ai_busy = (depths.get(ROUTE_AI_SMALL, 0) > 0
                   or depths.get(ROUTE_AI_LARGE, 0) > 0)
        services = ["rules-worker"]
        if ai_busy:
            services += ["ai-small-worker", "ai-large-worker"]
        logs = tail_compose_logs(services, lines=5)
        for svc, body in logs.items():
            console.print(f"[bold]── {svc} (last 5 log lines) ──[/bold]")
            console.print(body or "[dim](empty)[/dim]")
        console.print(
            "\n[yellow]Hints:[/yellow]\n"
            "  • If AI workers are slow, switch to a smaller model in config/categories.yaml "
            "(`models.ai-small: qwen2.5:3b`).\n"
            "  • Increase --timeout if the queue is just deep.\n"
            "  • Scale a route by hand: `docker-compose up -d --scale ai-small-worker=N`.\n"
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
    """Pick the right classifier for a given queue route."""
    rules = RulesEngine(str(CONFIG_PATH))
    categories = list(config["categories"].keys())
    threshold = config["settings"]["confidence_threshold"]

    if route == ROUTE_RULES:
        # Combine HC + fallback rules behind a tiny pipeline so the worker
        # presents one Classifier surface to the queue runner.
        return ClassificationPipeline([
            HighConfidenceRulesClassifier(rules),
            RulesClassifier(rules),
        ])
    if route in (ROUTE_AI_SMALL, ROUTE_AI_LARGE):
        model = _model_for_route(config, route, model_override)
        ai = LocalAIClassifier(model=model)
        extractor = FileExtractor(max_chars=config["settings"]["max_extract_chars"])
        return ClassificationPipeline([
            HighConfidenceRulesClassifier(rules),  # cheap gate before LLM
            LocalAIPipelineClassifier(ai, extractor, categories, threshold, enabled=True),
            RulesClassifier(rules),  # safety net if AI declines
        ])
    if route == ROUTE_OCR:
        # OCR is not implemented yet; fall back to rules so the queue still drains.
        log.warning("OCR route not implemented; serving rules-only worker for %s", route)
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
