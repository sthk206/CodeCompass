"""Dependency analysis tool"""

import ast
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from codecompass.tools.base import ToolResult, execute_with_metrics


# Common standard library modules
STDLIB = {
    "abc", "argparse", "ast", "asyncio", "base64", "collections",
    "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal",
    "enum", "functools", "glob", "hashlib", "html", "http", "importlib",
    "inspect", "io", "itertools", "json", "logging", "math", "os",
    "pathlib", "pickle", "pprint", "queue", "random", "re", "shutil",
    "signal", "socket", "sqlite3", "string", "struct", "subprocess",
    "sys", "tempfile", "threading", "time", "traceback", "types",
    "typing", "unittest", "urllib", "uuid", "warnings", "weakref",
}

class GetDependenciesInput(BaseModel):
    """Input schema for get_dependencies tool"""
    file_path: str = Field(description="Path to the file to analyze")


class GetDependenciesTool(BaseTool):
    """Analyze imports and dependencies of a file"""
    
    name: str = "get_dependencies"
    description: str = (
        "Get the imports and dependencies for a Python file. "
        "Use when asking what a file depends on or what modules it uses."
    )
    args_schema: type[BaseModel] = GetDependenciesInput
    
    repo_path: Path
    
    
    def __init__(self, repo_path: Path = Path(".")):
        super().__init__(repo_path=repo_path)

    def _run(self, file_path: str) -> str:
        
        return execute_with_metrics(
            self.name, self._execute, 
            file_path=file_path
        )
    
    def _execute(self, file_path: str) -> ToolResult:
        """Analyze file dependencies"""
        try:
            full_path = self.repo_path / file_path
            
            if not full_path.exists():
                return ToolResult(success=False, data= None, error=f"Error: File not found: {file_path}")
            
            if full_path.suffix != ".py":
                return ToolResult(success=False, data= None, error="Error: Only Python files supported")
            
            content = full_path.read_text()
            imports = self._extract_imports(content)
            
            if not imports:
                return ToolResult(success=True, data=f"No imports found in `{file_path}`")
            
            # Categorize
            stdlib, third_party, local = [], [], []
            for module, names, is_relative in imports:
                formatted = f"from {module} import {', '.join(names)}" if names else f"import {module}"
                
                if is_relative or module.startswith("."):
                    local.append(formatted)
                elif module.split(".")[0] in STDLIB:
                    stdlib.append(formatted)
                else:
                    third_party.append(formatted)
            
            # Format output
            lines = [f"**Dependencies for `{file_path}`:**\n"]
            
            if stdlib:
                lines.append("**Standard Library:**\n" + "\n".join(f"- `{i}`" for i in stdlib))
            if third_party:
                lines.append("\n**Third-Party:**\n" + "\n".join(f"- `{i}`" for i in third_party))
            if local:
                lines.append("\n**Local/Relative:**\n" + "\n".join(f"- `{i}`" for i in local))
            
            return ToolResult(success=True, data="\n".join(lines))
            
        except SyntaxError as e:
            return ToolResult(success=False, data= None, error=f"Error: Syntax error in file: {e}")
        except Exception as e:
            return ToolResult(success=False, data= None, error=f"Error analyzing dependencies: {e}")
    
    def _extract_imports(self, content: str) -> list[tuple]:
        """Extract imports using AST"""
        imports = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return imports
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, [], False))
            elif isinstance(node, ast.ImportFrom):
                module = ("." * node.level) + (node.module or "")
                names = [alias.name for alias in node.names]
                imports.append((module, names, node.level > 0))
        
        return imports