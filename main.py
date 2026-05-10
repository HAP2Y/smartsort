import typer
import os
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from classifier.rules import RulesEngine
from classifier.extractor import FileExtractor
from classifier.ai_local import LocalAIClassifier
from movers.organizer import Organizer

app = typer.Typer(help="SmartSort - Intelligent Local-First File Organization")
console = Console()

def load_config():
    config_path = Path(__file__).parent / "config" / "categories.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@app.command()
def run(
    target_dir: str = typer.Argument(..., help="Directory to sort"),
    apply: bool = typer.Option(False, "--apply", help="Apply changes. Defaults to dry-run."),
    local_only: bool = typer.Option(True, "--local-only", help="Disable cloud AI fallback entirely.")
):
    target = Path(target_dir).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        console.print(f"[red]Error: Directory {target_dir} not found.[/red]")
        raise typer.Exit(1)

    console.rule("[bold blue]SmartSort Initialization")
    
    config = load_config()
    categories = list(config['categories'].keys())
    threshold = config['settings']['confidence_threshold']
    
    console.log("[green]✓[/green] Configuration loaded.")

    rules = RulesEngine(str(Path(__file__).parent / "config" / "categories.yaml"))
    extractor = FileExtractor(max_chars=config['settings']['max_extract_chars'])
    ai_local = LocalAIClassifier(model=config['settings']['default_local_model'])
    organizer = Organizer(str(target))

    # PRE-FLIGHT CHECK (This defines ai_enabled!)
    ai_enabled = False
    with console.status("[bold yellow]Checking Ollama status...") as status:
        is_running, msg = ai_local.is_running()
        if is_running:
            ai_enabled = True
            console.log(f"[green]✓[/green] {msg}")
        else:
            console.log(f"[red]✗ AI Disabled:[/red] {msg}")
            console.log("[yellow]⚠ Proceeding with Rules-Based classification ONLY.[/yellow]")

    console.rule("[bold blue]Scanning Files")

    plan = {}
    files_to_process = [f for f in target.iterdir() if f.is_file() and not f.name.startswith('.smartsort')]
    
    console.print(f"[cyan]Found {len(files_to_process)} files in {target}...[/cyan]\n")

    with console.status("[bold green]Classifying files...") as status:
        for file_path in files_to_process:
            filepath_str = str(file_path)
            cat, conf, method, reason = "Unknown_Unsorted", 0, "None", "Initialized"
            
            # Step 1: AI FIRST (If enabled and text can be extracted)
            if ai_enabled:
                status.update(f"[bold green]AI Analyzing: {file_path.name}...")
                snippet = extractor.extract_text(filepath_str)
                
                if snippet and snippet.strip():
                    ai_cat, ai_conf, ai_reason = ai_local.classify(file_path.name, snippet, categories)
                    if ai_conf >= threshold:
                        cat, conf, reason, method = ai_cat, ai_conf, ai_reason, "Local AI"
                else:
                    reason = "AI skipped (No extractable text)"
            else:
                reason = "AI Disabled"

            # Step 2: FALLBACK TO RULES (If AI failed, was skipped, or has low confidence)
            if conf < threshold and cat != "Metadata_System":
                rule_cat, rule_conf, rule_reason = rules.classify(filepath_str)
                
                if rule_conf >= threshold:
                    cat, conf, reason, method = rule_cat, rule_conf, rule_reason, "Rules"
                else:
                    # If both fail, format a clean reason message
                    if method == "None" or "skipped" in reason or "Disabled" in reason:
                        reason = f"{reason} -> Rules also failed"
                    else:
                        reason = f"AI unsure ({conf}%) -> Rules failed"
                    cat = "Unknown_Unsorted"

            plan[filepath_str] = {"category": cat, "confidence": conf, "method": method, "reason": reason}

    console.rule("[bold blue]Classification Plan")

    # Display Plan
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
            str(data['reason'])
        )
    
    console.print(table)

    if apply:
        organizer.move_files(plan, apply=True)
        console.print("[bold green]\nFiles sorted successfully! Undo log saved.[/bold green]")
    else:
        console.print("\n[yellow]Dry-run complete. Run with --apply to move files.[/yellow]")

if __name__ == "__main__":
    app()