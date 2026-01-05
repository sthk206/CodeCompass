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
    """Format tool descriptions for system prompt, including parameters."""
    lines = []
    for tool in tools:
        lines.append(f"### {tool.name}")
        lines.append(tool.description)
        lines.append("")
        lines.append("**Parameters:**")
        
        # Get schema from the Pydantic args_schema
        schema = tool.args_schema.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            default = param_info.get("default")
            
            req_marker = "(required)" if param_name in required else f"(default: {default})"
            lines.append(f"- `{param_name}` ({param_type}) {req_marker}: {param_desc}")
        
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