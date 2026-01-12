from pathlib import Path
from datetime import datetime
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from codecompass.tools.base import ToolResult, execute_with_metrics

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


class GitHistoryInput(BaseModel):
    """Input schema for get_git_history tool"""
    file_path: Optional[str] = Field(default=None, description="Optional file to get history for")
    limit: int = Field(default=10, description="Number of commits to return")


class GitHistoryTool(BaseTool):
    """Get git commit history"""
    
    name: str = "get_git_history"
    description: str = (
        "Get git commit history. Use for understanding what changed, when, why, and by whom."
    )
    args_schema: type[BaseModel] = GitHistoryInput
    
    repo_path: Path
    _git_repo: Optional[object] = None
    
    def __init__(self, repo_path: Path = Path(".")):
        super().__init__(repo_path=repo_path)
    
    @property
    def git_repo(self):
        """Lazy-load git repo"""
        if self._git_repo is None and GIT_AVAILABLE:
            try:
                self._git_repo = git.Repo(self.repo_path)
            except git.InvalidGitRepositoryError:
                pass
        return self._git_repo
    
    def _run(self, 
        file_path: Optional[str] = None,
        limit: int = 10
    ) -> str:
        
        return execute_with_metrics(
            self.name, self._execute, 
            file_path=file_path, limit=limit
        )
    
    def _execute(self, file_path: Optional[str] = None, limit: int = 10) -> ToolResult:
        """Get git history"""
        max_commit_length = 300
        
        if not GIT_AVAILABLE:
            return ToolResult(success=False, data=None, error="GitPython not installed. Run: pip install gitpython")
        
        if self.git_repo is None:
            return ToolResult(success=False, data=None, error="Not a git repository")
        
        try:
            if file_path:
                commits = list(self.git_repo.iter_commits(paths=file_path, max_count=limit))
                header = f"**Git history for `{file_path}`:**\n"
            else:
                commits = list(self.git_repo.iter_commits(max_count=limit))
                header = "**Recent git history:**\n"
            
            if not commits:
                return ToolResult(success=True, data="No commits found.")
            
            lines = [header]
            for commit in commits:
                date = datetime.fromtimestamp(commit.committed_date).strftime("%Y-%m-%d %H:%M")
                message = commit.message.split("\n")[0]
                if len(message) > max_commit_length:
                    message = message[:(max_commit_length - 3)] + "..."
                lines.append(f"- `{commit.hexsha[:7]}` ({date}) **{commit.author.name}**: {message}")
            
            return ToolResult(success=True, data="\n".join(lines))

            
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Error getting history: {e}")