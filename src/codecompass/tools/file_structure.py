from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from codecompass.tools.base import ToolResult, execute_with_metrics


class FileStructureInput(BaseModel):
    """Input schema for get_file_structure tool"""
    directory: str = Field(default=".", description="Directory path relative to repo root")
    max_depth: int = Field(default=3, description="Maximum depth to show")


class FileStructureTool(BaseTool):
    """Get the directory tree structure"""
    
    name: str = "get_file_structure"
    description: str = (
        "Get the directory tree structure of the repository. "
        "Use when asking about project organization or what files exist."
    )
    args_schema: type[BaseModel] = FileStructureInput
    
    repo_path: Path
    show_icons: bool = False

    
    def __init__(self, repo_path: Path = Path(".")):
        super().__init__(repo_path=repo_path)

    
    def _run(self, directory: str = ".", max_depth: int = 3) -> str:
        
        return execute_with_metrics(
            self.name, self._execute, 
            directory=directory, max_depth=max_depth
        )
    
    def _execute(self, directory: str = ".", max_depth: int = 3) -> ToolResult:
        """Get directory structure"""
        try:
            target = self.repo_path / directory
            
            if not target.exists():
                return ToolResult(success=False, data= None, error=f"Error: Directory not found: {directory}")
            
            lines = [f"📁 {directory}/"]
            self._build_tree(target, "", 0, max_depth, lines)
            
            return ToolResult(success=True, data="\n".join(lines))
            
        except Exception as e:
            return ToolResult(success=False, data= None, error=f"Error getting structure: {e}")
    
    def _build_tree(self, path: Path, prefix: str, depth: int, max_depth: int, lines: list):
        """Recursively build directory tree"""
        if depth >= max_depth:
            return
        
        skip = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
        
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return "Error: Permission Error"

        entries = [e for e in entries if not e.name.startswith(".") and e.name not in skip]
        
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{"📁 " if self.show_icons else ""}{entry.name}/")
                extension = "    " if is_last else "│   "
                self._build_tree(entry, prefix + extension, depth + 1, max_depth, lines)
            else:
                # icon = self._get_icon(entry.suffix)
                # lines.append(f"{prefix}{connector}{icon} {entry.name}")
                icon = self._get_icon(entry.suffix) if self.show_icons else ""
                lines.append(f"{prefix}{connector}{icon} {entry.name}".strip())
    
    def _get_icon(self, suffix: str) -> str:
        """Get icon for file type"""
        return {
            ".py": "🐍", ".js": "📜", ".ts": "📘", ".json": "📋",
            ".yaml": "⚙️", ".yml": "⚙️", ".md": "📝", ".txt": "📄",
        }.get(suffix.lower(), "📄")