"""Semantic code search tool"""
from pathlib import Path
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from codecompass.retrieval.search import hyde_search

from .base import ToolResult, execute_with_metrics


class SearchCodeInput(BaseModel):
    query: str = Field(description="Natural language description of what to find")
    top_k: int = Field(default=5, description="Number of results")


class SearchCodeTool(BaseTool):
    """Search the codebase semantically"""
    
    name: str = "search_code"
    description: str = (
        "Search the codebase semantically for relevant code snippets."
        "Use this when looking for implementations, functionality, or specific code patterns."
    )
    args_schema: type[BaseModel] = SearchCodeInput
    
    repo_path: Path
    
    def __init__(self, repo_path: Path = Path(".")):
        super().__init__(repo_path=repo_path)
        from codecompass.indexing.store import CodeStore
        self._store = CodeStore(repo_path)
    
    def _run(self, query: str, top_k: int = 5) -> str:
        return execute_with_metrics(self.name, self._execute, query=query, top_k=top_k)
    
    def _execute(self, query: str, top_k: int = 5) -> ToolResult:
        results = hyde_search(self.repo_path, query, limit=top_k)
        
        if not results:
            return ToolResult(success=True, data=f"No matching code found for query `{query}`")
        
        formatted = []
        for r in results:
            formatted.append(
                f"### {r.file_path}:{r.start_line}-{r.end_line}\n"
                f"**{r.chunk_type}**: `{r.name}`\n"
                f"```python\n{r.code}\n```"
            )
        
        return ToolResult(success=True, data="\n\n".join(formatted))