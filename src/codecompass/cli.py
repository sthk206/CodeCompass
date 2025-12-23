from pathlib import Path
import typer
from rich.console import Console
from rich.markdown import Markdown

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
    
    try:
        if search_type == 0:
            results = search_code(repo_path, query, limit=limit)
        elif search_type == 1:
            from codecompass.llm.ollama import generate
            prompt = f"""Write a 3-5 line Python function that would match this search query.
Output ONLY the code, no explanation.

Query: {query}
````python
"""
            hypothetical = "def" + generate(prompt)
            print(hypothetical, "\n!!!")
            results = search_code(repo_path, hypothetical, limit=limit)
        elif search_type == 2:
            from codecompass.llm.ollama import generate
            prompt = f"""Add 5-10 Python library or decorator names related to this code search query.
Only output the expanded query, nothing else.

Query: {query}
Expanded query:"""
            expanded_query = generate(prompt)
            print(expanded_query, "\n!!!")
            results = search_code(repo_path, expanded_query, limit=limit)

        else:
            raise ValueError("Search Type not valid")
        
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

if __name__ == "__main__":
    app()