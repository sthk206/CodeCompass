from pathlib import Path
import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from codecompass.indexing.store import CodeStore
from codecompass.retrieval.search import baseline_search, hyde_search, query_expansion_search
import asyncio


app = typer.Typer(
    name="codecompass",
    help="AI-powered repository onboarding assistant",
)
console = Console()

@app.command()
def hello():
    """Test command."""
    print("CodeCompass is working!")

@app.command()
def index(
    repo_path: Path = typer.Argument(
        ".",
        help="Path to repository",
        exists=True
    )   
):
    """Index repository into vectordb"""
    from codecompass.indexing.store import index_repository
    repo_path = repo_path.resolve()
    console.print(f"[bold]Indexing:[/bold] {repo_path}")

    try:
        index_repository(repo_path)
        console.print(f"[green]Indexing complete.[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
        
@app.command()
def status(
    repo_path: Path = typer.Argument(
        ".",
        help="Path to repository",
        exists=True
    )    
):
    from codecompass.indexing.store import CodeStore
    repo_path = repo_path.resolve()
    store = CodeStore(repo_path)

    stats = store.get_stats()
    
    if stats["status"] == "not_indexed":
        console.print(f"[yellow]Repository not indexed.[/yellow]")
        console.print(f"Run: [bold]codecompass index {repo_path}[/bold]")
    else:
        console.print(f"[green]✓ Repository indexed[/green]")
        console.print(f"  Path: [magenta]{stats['repo_path']}[/magenta]")
        console.print(f"  Chunks: [cyan]{stats['chunk_count']}[/cyan]")
        console.print(f"  Indexed at: [cyan][bold]{stats['indexed_at']}[/bold][/cyan]")

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    search_type: int = typer.Option(
        0,
        "--stype", "-s",
        help="Type of search query - 0(default), 1(hyde search), 2(query expansion)"

    ),
    repo_path: Path = typer.Option(
        ".",
        "--repo", "-r",
        help="Path to the repository",
        exists=True,
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
):
    """Search for code in an indexed repository"""
    from codecompass.retrieval.search import search_code
    
    
    repo_path = repo_path.resolve()
    strategies = {0: hyde_search, 1: baseline_search, 2: query_expansion_search}
    
    if search_type not in strategies:
        console.print("[red]Invalid search type[/red]")
        raise typer.Exit(1)
    
    try:
        results = strategies[search_type](repo_path, query, limit)
        
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        for r in results:
            console.print(f"\n[bold blue]{r.name}[/bold blue] ({r.chunk_type})")
            console.print(f"[dim]{r.file_path}:{r.start_line}-{r.end_line}[/dim]")
            if r.docstring:
                console.print(f"[italic]{r.docstring}[/italic]")
            console.print(f"Score: [plain]{r.score:.4f}[/plain]", highlight=False)
            
            console.print("─" * 50)
            
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    
@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the codebase"),
    repo_path: Path = typer.Option(
        ".",
        "--repo", "-r",
        help="Path to the repository",
        exists=True,
    ),
    search_type: int = typer.Option(
        0,
        "--stype", "-s",
        help="Search strategy: 0=HyDE (default), 1=baseline, 2=query expansion"
    ),
):
    """Ask a question about the codebase."""
    from codecompass.retrieval.rag import answer_question
    
    repo_path = repo_path.resolve()
    
    console.print(f"[dim]Searching codebase...[/dim]")
    
    try:
        answer = answer_question(repo_path, question, search_type=search_type)
        console.print("\n")
        console.print(Markdown(answer))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

def get_repo_path(path: str | None) -> Path:
    """Resolve repository path."""
    return Path(path).resolve() if path else Path.cwd()

@app.command()
def chat(
    path: str = typer.Argument(None, help="Repository path"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable conversation memory"),
    stream: bool = typer.Option(False, "--stream", help="Show agent's work in progress"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enables debugging mode (print full system prompt + token usage)"),
):
    """Start an interactive chat session."""
    repo_path = get_repo_path(path)
    
    # Ensure indexed
    store = CodeStore(repo_path)
    if store.get_stats().get("status") == "not_indexed":
        console.print("[yellow]Repository not indexed. Indexing now...[/yellow]")
        with console.status("[bold green]Indexing..."):
            store.index()
        console.print("[green]✓ Done[/green]\n")
    
    # Create agent - pass memory flag
    from codecompass.agent.graph import CodeCompassAgent
    agent = CodeCompassAgent(repo_path, use_memory=not no_memory, debug=debug)
    
    console.print(Panel(
        Align.center(Text("Welcome to CodeCompass! 🧭", style="bold green")),
        border_style="green",
        box=box.DOUBLE,
    ))
    console.print(f"\n[dim]Repository:[/dim] {repo_path}")
    console.print(f"[dim]Memory:[/dim] [cyan]{'enabled' if not no_memory else 'disabled'}[/cyan]")
    console.print(f"[dim]Commands: '/clear', '/save', '/exit', '/help'.[/dim]\n")
    console.print(Rule(characters="=", style="dim"))



    state_path = repo_path / "state.json"
    
    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith("/"):
                if user_input in ("/exit", "/quit", "/q"):
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif user_input == "/clear":
                    agent.reset()
                    console.print("[dim]Conversation cleared.[/dim]")
                    continue
                elif user_input == "/save":
                    try:
                        agent.save_state(state_path)
                        console.print(f"[dim]State saved to {state_path}[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Failed to save state: {e}[/yellow]")
                    continue
                elif user_input == "/help":
                    console.print(Panel(
                        "/clear - Clear conversation\n/save - Save Conversation\n/exit - Exit chat\n/help - Show help",
                        title="Commands"
                    ))
                    continue
                else:
                    console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                    continue
            
            console.print()
            
            if stream:
                console.print("[bold green]CodeCompass:\n[/bold green]")
                
                async def run_stream():
                    async for event in agent.chat_stream_async(user_input):
                        if event["type"] == "thinking":
                            console.print("  [dim]💭 Thinking...\n[/dim]")
                        elif event["type"] == "token":
                            console.print(event["content"], end="")  # Token by token
                        elif event["type"] == "tool_call":
                            console.print(f"  [dim]🔧 Calling: {event['tool']}[/dim]")
                        elif event["type"] == "tool_result":
                            console.print(f"  [dim]📄 Got result[/dim]")
                    console.print()  # Final newline
                
                asyncio.run(run_stream())
            else:
                with console.status("[bold green]Thinking..."):
                    response = agent.chat(user_input)
                console.print("[bold green]CodeCompass:[/bold green]")
                console.print(Markdown(response))
            
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def diagram(
    repo_path: Path = typer.Argument(
        ".",
        help="Path to the repository",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    diagram_type: str = typer.Option(
        "architecture",
        "--type", "-t",
        help="Type of diagram: architecture, dependency, flow, class",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (default: {type}_diagram.md)",
    ),
    format: str = typer.Option(
        "md",
        "--format", "-f",
        help="Output format: md (markdown) or mermaid (raw)",
    ),
    max_modules: int = typer.Option(
        30,
        "--max-modules", "-m",
        help="Maximum modules to include",
    ),
    show_functions: bool = typer.Option(
        False,
        "--show-functions",
        help="Show function names in architecture diagram",
    ),
):
    """Generate a diagram of the codebase structure.
    
    Examples:
    
        codecompass diagram . --type architecture
        
        codecompass diagram ./myproject --type flow -o entrypoints.md
        
        codecompass diagram . --type class --max-modules 50
    """
    from codecompass.diagrams.generator import (
        generate_architecture_diagram,
        generate_dependency_diagram,
        generate_flow_diagram,
        generate_class_diagram,
        save_diagram,
    )
    
    repo_path = repo_path.resolve()
    
    # Default output filename
    if output is None:
        output = Path(f"{diagram_type}_diagram.md")
    
    console.print(f"[bold]Generating {diagram_type} diagram...[/bold]")
    console.print(f"[dim]Repository: {repo_path}[/dim]")
    
    try:
        # Generate appropriate diagram
        if diagram_type == "architecture":
            result = generate_architecture_diagram(
                repo_path, 
                max_modules=max_modules,
                show_functions=show_functions,
            )
        elif diagram_type == "dependency":
            result = generate_dependency_diagram(repo_path)
        elif diagram_type == "flow":
            result = generate_flow_diagram(repo_path)
        elif diagram_type == "class":
            result = generate_class_diagram(repo_path, max_classes=max_modules)
        else:
            console.print(f"[red]Unknown diagram type: {diagram_type}[/red]")
            console.print("Available types: architecture, dependency, flow, class")
            raise typer.Exit(1)
        
        # Save the diagram
        saved_path = save_diagram(result, output, format=format)
        
        console.print(f"\n[green]✓ Diagram saved to: {saved_path}[/green]")
        console.print(f"[dim]  Title: {result.title}[/dim]")
        console.print(f"[dim]  Modules analyzed: {result.modules_analyzed}[/dim]")
        
        # Preview first few lines
        console.print(f"\n[bold]Preview:[/bold]")
        lines = result.mermaid_code.split("\n")[:10]
        for line in lines:
            console.print(f"  [cyan]{line}[/cyan]")
        if len(result.mermaid_code.split("\n")) > 10:
            console.print(f"  [dim]... ({len(result.mermaid_code.split(chr(10)))} total lines)[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error generating diagram: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()