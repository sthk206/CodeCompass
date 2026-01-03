# src/codecompass/tools/read_file.py
"""Read file contents tool"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from codecompass.tools.base import ToolResult, execute_with_metrics


class ReadFileInput(BaseModel):
    """Input schema for read_file tool"""
    file_path: str = Field(description="Path to the file relative to repository root")
    start_line: Optional[int] = Field(default=None, description="Starting line number (1-indexed)")
    end_line: Optional[int] = Field(default=None, description="Ending line number (1-indexed)")


class ReadFileTool(BaseTool):
    """Read the contents of a specific file"""
    
    name: str = "read_file"
    description: str = (
        "Read the contents of a specific file. Use when you need to see "
        "the full content of a file or a specific section."
    )
    args_schema: type[BaseModel] = ReadFileInput
    
    repo_path: Path
    
    def __init__(self, repo_path: Path = Path(".")):
        super().__init__(repo_path=repo_path)

    def _run(self, 
        file_path: str, 
        start_line: Optional[int] = None, 
        end_line: Optional[int] = None
    ) -> str:
        
        return execute_with_metrics(
            self.name, self._execute, 
            file_path=file_path, start_line=start_line, end_line=end_line
        )
    
    def _execute(
        self, 
        file_path: str, 
        start_line: Optional[int] = None, 
        end_line: Optional[int] = None
    ) -> ToolResult:
        """Read file contents"""
        try:
            full_path = self.repo_path / file_path
            
            # Security check
            if not full_path.resolve().is_relative_to(self.repo_path.resolve()):
                return ToolResult(success=False, data= None, error="Error: Access denied - path outside repository")
            
            if not full_path.exists():
                return ToolResult(success=False, data= None, error=f"Error: File not found: {file_path}")
            
            if not full_path.is_file():
                return ToolResult(success=False, data= None, error=f"Error: Not a file: {file_path}")
            
            content = full_path.read_text()
            lines = content.splitlines()
            
            # Apply line range
            if start_line is not None or end_line is not None:
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                lines = lines[start_idx:end_idx]
                
                # Add line numbers
                numbered = [f"{i + start_idx + 1:4d} | {line}" for i, line in enumerate(lines)]
                content = "\n".join(numbered)
            
            # Get language for syntax highlighting
            lang = self._get_language(full_path.suffix)
            
            return ToolResult(success=True, data=f"**{file_path}**\n```{lang}\n{content}\n```")
            
        except UnicodeDecodeError:
            return ToolResult(success=False, data= None, error=f"Error: Cannot read binary file: {file_path}")
        except Exception as e:
            return ToolResult(success=False, data= None, error=f"Error reading file: {e}")
    
    def _get_language(self, suffix: str) -> str:
        """Map file extension to language"""
        return {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".go": "go", ".rs": "rust",
            ".md": "markdown", ".json": "json", ".yaml": "yaml",
        }.get(suffix.lower(), "")