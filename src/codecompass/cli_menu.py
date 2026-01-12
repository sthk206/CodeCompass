"""Interactive menu for CodeCompass using questionary"""

from pathlib import Path
from typing import Callable
import questionary
from questionary import Style, Choice
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

console = Console()

# Custom style for high visibility
MENU_STYLE = Style([
    ("qmark", "fg:green bold"),
    ("question", "fg:white bold"),
    ("answer", "fg:green bold"),
    ("pointer", "fg:green bold"),
    ("highlighted", "fg:green bold"),
    ("selected", "fg:green bold"),
    ("separator", "fg:gray"),
    ("instruction", "fg:gray italic"),
    ("text", "fg:white"),
    ("disabled", "fg:gray italic"),
])

COMMAND_GROUPS = {
    "Repository": [
        "index",
        "status",
    ],
    "LLM / RAG": [
        "chat",
        "ask",
        "search",
    ],
    "Utilities": [
        "diagram",
    ],
}

COMMANDS = {
    "index": {
        "icon": "📂",
        "title": "Index Repository",
        "description": "Index a repository into the vector database",
        "requires_index": False,
    },
    "status": {
        "icon": "📈",
        "title": "Check Status",
        "description": "Check the indexing status of a repository",
        "requires_index": False,
    },
    "search": {
        "icon": "🔍",
        "title": "Search Code",
        "description": "Hybrid (semantic + lexical) retrieval over indexed code",
        "requires_index": True,
    },
    "ask": {
        "icon": "❓",
        "title": "Ask Question",
        "description": "Ask a single question about the codebase",
        "requires_index": True,
    },
    "chat": {
        "icon": "💬",
        "title": "Interactive Chat",
        "description": "Start an interactive chat session",
        "requires_index": True,
    },
    "diagram": {
        "icon": "📊",
        "title": "Generate Diagram",
        "description": "Generate architecture/dependency diagrams",
        "requires_index": False,
    },
}


def check_indexed(repo_path: Path) -> bool:
    """Check if repository is indexed"""
    try:
        from codecompass.indexing.store import CodeStore
        store = CodeStore(repo_path)
        stats = store.get_stats()
        return stats.get("status") != "not_indexed"
    except Exception:
        return False


def show_header():
    """Display the CodeCompass header"""
    console.clear()
    console.print()
    console.print(Panel(
        Align.center(Text(" Welcome to CodeCompass! 🧭 ", style="bold green")),
        # subtitle="AI-powered repository assistant",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2),

    ))
    console.print()


def get_repo_path_input(default: str = ".") -> Path | None:
    """Prompt for repository path"""
    path = questionary.path(
        "Repository path:",
        default=default,
        only_directories=True,
        style=MENU_STYLE,
    ).ask()
    
    if path is None:
        return None
    
    resolved = Path(path).resolve()
    if not resolved.exists():
        console.print(f"[red]Path does not exist: {resolved}[/red]")
        return None
    return resolved


def select_command(repo_path: Path, is_indexed: bool) -> str | None:
    """Show command selection menu with sections"""
    choices = []

    choices.append(questionary.Separator(" "))
    for section_title, commands in COMMAND_GROUPS.items():
        # Section header
        choices.append(questionary.Separator(f"{section_title}"))

        for cmd in commands:
            info = COMMANDS[cmd]
            disabled = info["requires_index"] and not is_indexed

            label = f"  {info['icon']} {info['title']:<20} {info['description']}"

            if disabled:
                label = f"  {info['icon']} {info['title']:<20} | REQUIRES INDEXING"

                choices.append(
                    Choice(
                        title=label,
                        value=cmd,
                        disabled=True
                    )
                )
            else:
                choices.append(Choice(title=label, value=cmd))

    choices.append(questionary.Separator("────────────────────────"))


    choices.append(Choice(title="❌  Exit", value="exit"))

    # Show repo + index status
    status = "[green]✓ Indexed[/green]" if is_indexed else "[yellow]○ Not indexed[/yellow]"
    console.print(f"  Repository: [cyan]{repo_path}[/cyan]")
    console.print(f"  Status: {status}\n")

    return questionary.select(
        "Select a command:",
        choices=choices,
        style=MENU_STYLE,
        instruction="(↑↓ to move, Enter to select)",
    ).ask()



def prompt_index_params(repo_path: Path) -> dict | None:
    """Gather parameters for index command"""
    console.print(Panel("[bold]Index Repository[/bold]", border_style="blue"))
    
    path = questionary.path(
        "Repository path:",
        default=".",
        only_directories=True,
        style=MENU_STYLE,
    ).ask()
    
    if path is None:
        return None
    
    if not questionary.confirm(
        f"Index {Path(path).resolve()}?",
        default=True,
        style=MENU_STYLE,
    ).ask():
        return None
    
    return {"repo_path": Path(path)}


def prompt_status_params(repo_path: Path) -> dict | None:
    """Gather parameters for status command"""
    console.print(Panel("[bold]Check Status[/bold]", border_style="blue"))
    
    path = questionary.path(
        "Repository path:",
        default=".",
        only_directories=True,
        style=MENU_STYLE,
    ).ask()
    
    if path is None:
        return None
    
    return {"repo_path": Path(path)}


def prompt_search_params(repo_path: Path) -> dict | None:
    """Gather parameters for search command"""
    console.print(Panel("[bold]Search Code[/bold]", border_style="blue"))
    
    query = questionary.text(
        "Search query:",
        style=MENU_STYLE,
        validate=lambda x: len(x.strip()) > 0 or "Query cannot be empty",
    ).ask()
    
    if query is None:
        return None
    
    search_type = questionary.select(
        "Search strategy:",
        choices=[
            Choice("HyDE Search (default)", value=0),
            Choice("Baseline Search", value=1),
            Choice("Query Expansion", value=2),
        ],
        style=MENU_STYLE,
    ).ask()
    
    if search_type is None:
        return None
    
    limit = questionary.text(
        "Number of results:",
        default="5",
        style=MENU_STYLE,
        validate=lambda x: x.isdigit() and int(x) > 0 or "Must be a positive number",
    ).ask()
    
    if limit is None:
        return None
    
    return {
        "query": query,
        "search_type": search_type,
        "repo_path": repo_path,
        "limit": int(limit),
    }


def prompt_ask_params(repo_path: Path) -> dict | None:
    """Gather parameters for ask command"""
    console.print(Panel("[bold]Ask Question[/bold]", border_style="blue"))
    
    question = questionary.text(
        "Your question:",
        style=MENU_STYLE,
        validate=lambda x: len(x.strip()) > 0 or "Question cannot be empty",
    ).ask()
    
    if question is None:
        return None
    
    no_stream = questionary.confirm(
        "Disable streaming output?",
        default=False,
        style=MENU_STYLE,
    ).ask()
    
    if no_stream is None:
        return None
    
    debug = questionary.confirm(
        "Enable debug mode?",
        default=False,
        style=MENU_STYLE,
    ).ask()
    
    if debug is None:
        return None
    
    return {
        "question": question,
        "repo_path": repo_path,
        "no_stream": no_stream,
        "debug": debug,
    }


def prompt_chat_params(repo_path: Path) -> dict | None:
    """Gather parameters for chat command"""
    console.print(Panel("[bold]Interactive Chat[/bold]", border_style="blue"))
    
    console.print("[dim]Configure chat session options:[/dim]\n")
    
    answers = questionary.form(
        no_memory=questionary.confirm(
            "Disable conversation memory?",
            default=False,
            style=MENU_STYLE,
        ),
        no_stream=questionary.confirm(
            "Disable streaming output?",
            default=False,
            style=MENU_STYLE,
        ),
        debug=questionary.confirm(
            "Enable debug mode?",
            default=False,
            style=MENU_STYLE,
        ),
    ).ask()
    
    if answers is None:
        return None
    
    return {
        "path": str(repo_path),
        "no_memory": answers["no_memory"],
        "no_stream": answers["no_stream"],
        "debug": answers["debug"],
    }


def prompt_diagram_params(repo_path: Path) -> dict | None:
    """Gather parameters for diagram command"""
    console.print(Panel("[bold]Generate Diagram[/bold]", border_style="blue"))
    
    diagram_type = questionary.select(
        "Diagram type:",
        choices=[
            Choice("Architecture - Module structure overview", value="architecture"),
            Choice("Dependency - Import relationships", value="dependency"),
            Choice("Flow - Entry points and call flow", value="flow"),
            Choice("Class - Class hierarchy", value="class"),
        ],
        style=MENU_STYLE,
    ).ask()
    
    if diagram_type is None:
        return None
    
    output_format = questionary.select(
        "Output format:",
        choices=[
            Choice("Markdown (.md)", value="md"),
            Choice("Raw Mermaid (.mermaid)", value="mermaid"),
        ],
        style=MENU_STYLE,
    ).ask()
    
    if output_format is None:
        return None
    
    output_path = questionary.text(
        "Output path:",
        default=f".codecompass/diagrams/{diagram_type}_diagram.md",
        style=MENU_STYLE,
    ).ask()
    
    if output_path is None:
        return None
    
    max_modules = questionary.text(
        "Maximum modules to include:",
        default="30",
        style=MENU_STYLE,
        validate=lambda x: x.isdigit() and int(x) > 0 or "Must be a positive number",
    ).ask()
    
    if max_modules is None:
        return None
    
    show_functions = False
    if diagram_type == "architecture":
        show_functions = questionary.confirm(
            "Show function names?",
            default=False,
            style=MENU_STYLE,
        ).ask()
        
        if show_functions is None:
            return None
    
    return {
        "repo_path": repo_path,
        "diagram_type": diagram_type,
        "output": Path(output_path),
        "format": output_format,
        "max_modules": int(max_modules),
        "show_functions": show_functions,
    }


# Map commands to their parameter prompts
PARAM_PROMPTS = {
    "index": prompt_index_params,
    "status": prompt_status_params,
    "search": prompt_search_params,
    "ask": prompt_ask_params,
    "chat": prompt_chat_params,
    "diagram": prompt_diagram_params,
}


def run_interactive_menu():
    """Main entry point for interactive menu"""
    show_header()
    
    # First, get repository path
    console.print("[bold]Please specify your project path, or press Enter to use the current directory:[/bold]\n")
    
    repo_path = get_repo_path_input()
    if repo_path is None:
        return
    
    # Main loop
    while True:
        show_header()
        
        is_indexed = check_indexed(repo_path)
        command = select_command(repo_path, is_indexed)
        
        if command is None or command == "exit":
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break
        
        # Clear and show parameter form
        show_header()
        
        # Get parameters for the selected command
        prompt_fn = PARAM_PROMPTS.get(command)
        if prompt_fn is None:
            console.print(f"[red]Unknown command: {command}[/red]")
            continue
        
        params = prompt_fn(repo_path)
        
        if params is None:
            # User cancelled, go back to menu
            continue
        
        # Execute the command
        console.print()
        execute_command(command, params)
        
        # Wait for user before returning to menu
        console.print()
        questionary.press_any_key_to_continue(
            "Press any key to return to menu...",
            style=MENU_STYLE,
        ).ask()


def execute_command(command: str, params: dict):
    """Execute a CLI command with the given parameters"""
    # Import here to avoid circular imports
    from codecompass.cli import index, status, search, ask, chat, diagram
    
    console.print(f"[bold green]Running {command}...[/bold green]\n")
    
    try:
        if command == "index":
            index(repo_path=params["repo_path"])
        elif command == "status":
            status(repo_path=params["repo_path"])
        elif command == "search":
            search(
                query=params["query"],
                search_type=params["search_type"],
                repo_path=params["repo_path"],
                limit=params["limit"],
            )
        elif command == "ask":
            ask(
                question=params["question"],
                repo_path=params["repo_path"],
                no_stream=params["no_stream"],
                debug=params["debug"],
            )
        elif command == "chat":
            chat(
                path=params["path"],
                no_memory=params["no_memory"],
                no_stream=params["no_stream"],
                debug=params["debug"],
            )
        elif command == "diagram":
            diagram(
                repo_path=params["repo_path"],
                diagram_type=params["diagram_type"],
                output=params["output"],
                format=params["format"],
                max_modules=params["max_modules"],
                show_functions=params["show_functions"],
            )
    except SystemExit:
        # Typer raises SystemExit on errors, catch it to stay in menu
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")