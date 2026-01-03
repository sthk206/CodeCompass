from pathlib import Path
from langchain_core.tools import BaseTool

from codecompass.tools.get_dependencies import GetDependenciesTool

from .search_code import SearchCodeTool
from .read_file import ReadFileTool
from .find_references import FindReferencesTool
from .file_structure import FileStructureTool
from .git_history import GitHistoryTool


def create_tools(repo_path: Path) -> list[BaseTool]:
    """Factory function to create all tools for a repository"""
    return [
        SearchCodeTool(repo_path),
        ReadFileTool(repo_path),
        FindReferencesTool(repo_path),
        FileStructureTool(repo_path),
        GitHistoryTool(repo_path),
        GetDependenciesTool(repo_path),
    ]


def get_tools_description(tools: list[BaseTool]) -> str:
    """Format tool descriptions for system prompt"""
    lines = []
    for tool in tools:
        lines.append(f"### {tool.name}")
        lines.append(tool.description)
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "create_tools",
    "get_tools_description",
    "SearchCodeTool",
    "ReadFileTool",
    "FindReferencesTool",
    "FileStructureTool",
    "GitHistoryTool",
    "GetDependenciesTool",
]