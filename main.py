import typer
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table

from classifier.rules import RulesEngine
from classifier.extractor import FileExtractor
from classifier.ai_local import LocalAIClassifier
from movers.organizer import Organizer

app = typer.Typer(help="SmartSort - Intelligent Local-First File Organization", no_args_is_help=True)
console = Console()

CONFIG_PATH = Path(__file__).parent / "config" / "categories.yaml"


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


@app.command()
def run(
    target_dir: str = typer.Argument(..., help="Directory to sort"),
    apply: bool = typer.Option(False, "--apply", help="Apply changes. Defaults to dry-run."),
    local_only: bool = typer.Option(True, "--local-only", help="Disable cloud AI fallback entirely."),
    no_ai: bool = typer.Option(False, "--no-ai", help="Skip AI classification entirely (rules only)."),
):
    """Sort files in TARGET_DIR using rules + local AI."""
    target = Path(target_dir).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        console.print(f"[red]Error: Directory {target_dir} not found.[/red]")
        raise typer.Exit(1)

    console.rule("[bold blue]SmartSort Initialization")

    config = load_config()
    categories = list(config['categories'].keys())
    threshold = config['settings']['confidence_threshold']

    console.log("[green]✓[/green] Configuration loaded.")

    rules = RulesEngine(str(CONFIG_PATH))
    extractor = FileExtractor(max_chars=config['settings']['max_extract_chars'])
    ai_local = LocalAIClassifier(model=config['settings']['default_local_model'])
    organizer = Organizer(str(target), category_names=categories)

    # Pre-flight Ollama check
    ai_enabled = False
    if not no_ai:
        with console.status("[bold yellow]Checking Ollama status..."):
            is_running, msg = ai_local.is_running()
            if is_running:
                ai_enabled = True
                console.log(f"[green]✓[/green] {msg}")
            else:
                console.log(f"[red]✗ AI Disabled:[/red] {msg}")
                console.log("[yellow]⚠ Proceeding with Rules-Based classification ONLY.[/yellow]")
    else:
        console.log("[yellow]AI skipped (--no-ai).[/yellow]")

    console.rule("[bold blue]Scanning Files")

    plan: dict[str, dict] = {}
    skipped_organized = 0
    files_to_process = []
    for f in target.iterdir():
        if not f.is_file():
            continue
        if f.name.startswith('.smartsort'):
            continue
        files_to_process.append(f)

    # Also scan one level into category directories so we can show counts but skip them.
    for sub in target.iterdir():
        if sub.is_dir() and sub.name in categories:
            for inner in sub.iterdir():
                if inner.is_file():
                    skipped_organized += 1

    console.print(f"[cyan]Found {len(files_to_process)} files in {target}[/cyan]"
                  f" (skipping {skipped_organized} already inside category folders).\n")

    with console.status("[bold green]Classifying files...") as status:
        for file_path in files_to_process:
            filepath_str = str(file_path)
            cat, conf, method, reason = "Unknown_Unsorted", 0, "None", "Initialized"

            # Step 1: HIGH-CONFIDENCE filename overrides (IMM/T4/EVL/PR_/...)
            hc = rules.high_confidence_match(filepath_str)
            if hc:
                cat, conf, reason = hc
                method = "Rules (HC)"

            # Step 2: AI on extracted text content
            if cat == "Unknown_Unsorted" and ai_enabled:
                status.update(f"[bold green]AI Analyzing: {file_path.name}...")
                snippet = extractor.extract_text(filepath_str)

                if snippet and snippet.strip():
                    ai_cat, ai_conf, ai_reason = ai_local.classify(file_path.name, snippet, categories)
                    if ai_conf >= threshold and ai_cat != "Unknown_Unsorted":
                        cat, conf, reason, method = ai_cat, ai_conf, ai_reason, "Local AI"
                    else:
                        reason = f"AI unsure ({ai_conf}%)"
                else:
                    reason = "AI skipped (no extractable text)"
            elif cat == "Unknown_Unsorted":
                reason = "AI disabled"

            # Step 3: Keyword rules + archive/system extension fallback
            if cat == "Unknown_Unsorted":
                rule_cat, rule_conf, rule_reason = rules.classify(filepath_str)
                if rule_cat != "Unknown_Unsorted":
                    cat, conf, reason, method = rule_cat, rule_conf, rule_reason, "Rules"
                else:
                    reason = f"{reason} -> Rules also failed"

            plan[filepath_str] = {"category": cat, "confidence": conf, "method": method, "reason": reason}

    console.rule("[bold blue]Classification Plan")

    table = Table(title="Category Plan (--dry-run)" if not apply else "Execution Plan", show_lines=True)
    table.add_column("Filename", style="cyan", max_width=40)
    table.add_column("Category", style="magenta")
    table.add_column("Conf %", justify="right", style="green")
    table.add_column("Method", style="yellow")
    table.add_column("Reason", style="white", max_width=50)

    for fp, data in plan.items():
        table.add_row(
            Path(fp).name,
            data['category'],
            str(data['confidence']),
            data['method'],
            str(data['reason']),
        )

    console.print(table)

    # Per-category summary
    summary: dict[str, int] = {}
    for data in plan.values():
        summary[data['category']] = summary.get(data['category'], 0) + 1
    summary_table = Table(title="Summary", show_lines=False)
    summary_table.add_column("Category", style="magenta")
    summary_table.add_column("Files", justify="right", style="green")
    for cat, count in sorted(summary.items(), key=lambda x: -x[1]):
        summary_table.add_row(cat, str(count))
    console.print(summary_table)

    if apply:
        organizer.move_files(plan, apply=True)
        console.print(f"[bold green]\nFiles sorted. Undo log: {organizer.undo_log}[/bold green]")
        console.print("[dim]Run `python main.py undo <dir>` to revert.[/dim]")
    else:
        console.print("\n[yellow]Dry-run complete. Run with --apply to move files.[/yellow]")


@app.command()
def undo(
    target_dir: str = typer.Argument(..., help="Directory whose last sort to revert"),
):
    """Revert the most recent sort using the .smartsort_undo.json log."""
    target = Path(target_dir).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        console.print(f"[red]Error: Directory {target_dir} not found.[/red]")
        raise typer.Exit(1)

    config = load_config()
    organizer = Organizer(str(target), category_names=list(config['categories'].keys()))
    restored, missing, errors = organizer.undo()

    console.print(f"[green]Restored:[/green] {restored}")
    if missing:
        console.print(f"[yellow]Missing (already moved/deleted):[/yellow] {missing}")
    for err in errors:
        console.print(f"[red]Error:[/red] {err}")


if __name__ == "__main__":
    app()
