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

    orchestrator = Orchestrator(backend=qb, router=router)
    pending = orchestrator.submit(files)
    console.print(f"[cyan]Submitted {len(pending)} jobs via {backend} backend.[/cyan]")
    for route, count in orchestrator.stats.by_route.items():
        console.print(f"  [magenta]{route}[/magenta]: {count}")

    with console.status("[bold green]Awaiting worker results..."):
        plan = orchestrator.collect(pending, timeout=timeout)

    for w in inline_workers:
        w.stop()
    for t in threads:
        t.join(timeout=2.0)
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
        model = model_override or (
            config["settings"].get("large_model", config["settings"]["default_local_model"])
            if route == ROUTE_AI_LARGE
            else config["settings"]["default_local_model"]
        )
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

    console.print(
        f"[green]Worker {worker_name} listening on {route_list} via {backend}.[/green]"
    )
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
