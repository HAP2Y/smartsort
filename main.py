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
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (-v, -vv)."),
):
    """Sort files in TARGET_DIR using rules + local AI."""
    _configure_logging(verbose)

    target = Path(target_dir).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        console.print(f"[red]Error: Directory {target_dir} not found.[/red]")
        raise typer.Exit(1)

    console.rule("[bold blue]SmartSort Initialisation")

    config = _load_config()
    categories = list(config["categories"].keys())
    console.log("[green]✓[/green] Configuration loaded.")

    with console.status("[bold yellow]Building pipeline (Ollama health check)..."):
        pipeline, ai_status = _build_pipeline(config, no_ai=no_ai)
    console.log(ai_status)

    organizer = Organizer(str(target), category_names=categories)

    console.rule("[bold blue]Scanning Files")

    files = _gather_files(target, recursive=recursive, category_names=set(categories))
    skipped = sum(
        1
        for sub in target.iterdir()
        if sub.is_dir() and sub.name in categories
        for inner in sub.iterdir()
        if inner.is_file()
    )
    scope = "recursively" if recursive else "at top level"
    console.print(
        f"[cyan]Found {len(files)} files {scope} in {target}[/cyan] "
        f"(skipping {skipped} already inside category folders).\n"
    )

    plan: dict[str, Classification] = {}
    with console.status("[bold green]Classifying files...") as status:
        for path in files:
            status.update(f"[bold green]Classifying: {path.name}")
            plan[str(path)] = pipeline.classify(FileItem(path=path))

    _print_plan(plan, apply)

    if apply:
        organizer.move_files({fp: c.to_dict() for fp, c in plan.items()}, apply=True)
        console.print(f"[bold green]\nFiles sorted. Undo log: {organizer.undo_log}[/bold green]")
        console.print("[dim]Run `smartsort undo <dir>` (or `python main.py undo <dir>`) to revert.[/dim]")
    else:
        console.print("\n[yellow]Dry-run complete. Run with --apply to move files.[/yellow]")


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


if __name__ == "__main__":
    app()
