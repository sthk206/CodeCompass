# codecompass/retrieval/search.py
"""Search functionality for CodeCompass."""

from pathlib import Path
from dataclasses import dataclass

from codecompass.indexing.store import CodeStore


@dataclass
class SearchResult:
    """A search result with relevance score."""
    id: str
    file_path: str
    name: str
    chunk_type: str
    code: str
    start_line: int
    end_line: int
    docstring: str
    score: float  # Similarity score (lower = more similar for L2 distance)
    
    def format(self) -> str:
        """Format the result for display."""
        location = f"{self.file_path}:{self.start_line}-{self.end_line}"
        header = f"📄 {self.name} ({self.chunk_type}) - {location}"
        
        code_preview = self.code[:500] + "..." if len(self.code) > 500 else self.code
        
        return f"{header}\n{code_preview}"


def search_code(repo_path: Path, query: str, limit: int = 5) -> list[SearchResult]:
    """Search for code in an indexed repository."""
    import pprint
    store = CodeStore(repo_path)
    
    if not store.is_indexed():
        raise ValueError(f"Repository not indexed. Run: codecompass index {repo_path}")
    
    raw_results = store.search(query, limit=limit)
    
    results = []
    for r in raw_results:
        # summary = {k: r[k] for k in r if k != "code"}  
        # pprint.pprint(summary)
        results.append(SearchResult(
            id=r["id"],
            file_path=r["file_path"],
            name=r["name"],
            chunk_type=r["chunk_type"],
            code=r["code"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            docstring=r.get("docstring", ""),
            score=r.get("_relevance_score", 0.0),
        ))
    
    return results