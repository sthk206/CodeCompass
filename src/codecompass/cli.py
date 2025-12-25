from pathlib import Path
import typer
from rich.console import Console
from rich.markdown import Markdown
from codecompass.retrieval.search import baseline_search, hyde_search, query_expansion_search

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
    

@app.command()
def chat(
    repo_path: Path = typer.Argument(
        ".",
        help="Path to the repository",
        exists=True,
    ),
    search_type: int = typer.Option(
        0,
        "--stype", "-s",
        help="Search strategy: 0=HyDE (default), 1=baseline, 2=query expansion"
    ),
):
    """Start an interactive chat about the codebase."""
    from codecompass.retrieval.rag import answer_question
    from codecompass.indexing.store import CodeStore
    
    repo_path = repo_path.resolve()
    store = CodeStore(repo_path)
    
    # Check if indexed
    if not store.is_indexed():
        console.print(f"[yellow]Repository not indexed. Indexing now...[/yellow]")
        from codecompass.indexing.store import index_repository
        index_repository(repo_path)
    
    stats = store.get_stats()
    
    search_names = {0: "HyDE", 1: "Baseline", 2: "Query Expansion"}
    
    console.print("\n[bold green]╔═══════════════════════════════════════════════════════╗[/bold green]")
    console.print("[bold green]║           Welcome to CodeCompass! 🧭                   ║[/bold green]")
    console.print("[bold green]╚═══════════════════════════════════════════════════════╝[/bold green]")
    console.print(f"\n[dim]Repository:[/dim] {repo_path}")
    console.print(f"[dim]Indexed chunks:[/dim] {stats.get('chunk_count', 'unknown')}")
    console.print(f"[dim]Search strategy:[/dim] {search_names.get(search_type, 'HyDE')}")
    console.print(f"\n[dim]Type 'exit' to quit, 'help' for commands.[/dim]\n")
    
    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()
            
            if not question:
                continue
            
            if question.lower() == "exit":
                console.print("[dim]Goodbye![/dim]")
                break
            
            if question.lower() == "help":
                console.print("""
[bold]Commands:[/bold]
  exit     - Quit the chat
  help     - Show this help message

[bold]Tips:[/bold]
  • Ask about specific functions: "What does the login function do?"
  • Ask about architecture: "How is authentication implemented?"
  • Ask for explanations: "Explain the UserService class"
""")
                continue
            
            console.print("[dim]Thinking...[/dim]")
            answer = answer_question(repo_path, question, search_type=search_type)
            console.print(f"\n[bold green]CodeCompass:[/bold green]")
            console.print(Markdown(answer))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break

if __name__ == "__main__":
    app()