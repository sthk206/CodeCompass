"""Find symbol references tool"""

import re
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from codecompass.tools.base import ToolResult, execute_with_metrics

class FindReferencesInput(BaseModel):
    """Input schema for find_references tool"""
    symbol: str = Field(description="Exact name of the function, class, or variable to find")


class FindReferencesTool(BaseTool):
    """Find all references to a symbol in the codebase"""
    
    name: str = "find_references"
    description: str = (
        "Find all places in the codebase where a specific function, class, or variable is used. "
        "Use this when asking questions like 'where is X used?' or 'what calls X?'"
    )
    args_schema: type[BaseModel] = FindReferencesInput
    
    repo_path: Path
    
    def __init__(self, repo_path: Path = Path(".")):
        super().__init__(repo_path=repo_path)

    def _run(self, symbol: str) -> str:
        
        return execute_with_metrics(
            self.name, self._execute, 
            symbol=symbol
        )
    
    def _execute(self, symbol: str) -> ToolResult:
        """Find all references to the symbol"""
        try:
            references = {"definition": [], "import": [], "usage": []}
            skip_dirs = {"venv", ".venv", "node_modules", "__pycache__", ".git"}
            
            for py_file in self.repo_path.rglob("*.py"):
                if any(part in py_file.parts for part in skip_dirs):
                    continue
                
                try:
                    content = py_file.read_text()
                except (UnicodeDecodeError, PermissionError):
                    continue
                
                rel_path = str(py_file.relative_to(self.repo_path))
                pattern = rf'\b{re.escape(symbol)}\b'
                
                for i, line in enumerate(content.splitlines(), 1):
                    if re.search(pattern, line):
                        context = self._classify_reference(line, symbol)
                        references[context].append(f"`{rel_path}:{i}` - `{line.strip()}`")
            
            if not any(references.values()):
                return ToolResult(success=True, data=f"No references found for `{symbol}`")

            
            # Format output
            parts = []
            if references["definition"]:
                parts.append("**Definitions:**\n" + "\n".join(f"- {r}" for r in references["definition"][:5]))
            if references["import"]:
                parts.append("**Imports:**\n" + "\n".join(f"- {r}" for r in references["import"][:10]))
            if references["usage"]:
                usage_list = references["usage"][:15]
                parts.append(f"**Usages ({len(references['usage'])} found):**\n" + "\n".join(f"- {r}" for r in usage_list))
                if len(references["usage"]) > 15:
                    parts.append(f"  ... and {len(references['usage']) - 15} more")
            
            return ToolResult(success=True, data="\n\n".join(parts))

            
        except Exception as e:
            return ToolResult(success=False, data= None, error=f"Error finding references: {e}")
    
    def _classify_reference(self, line: str, symbol: str) -> str:
        """Classify the type of reference"""
        stripped = line.strip()
        if re.match(rf'^(def|class|async def)\s+{re.escape(symbol)}\b', stripped):
            return "definition"
        if re.match(rf'^{re.escape(symbol)}\s*=', stripped):
            return "definition"
        if stripped.startswith(("import ", "from ")):
            return "import"
        return "usage"