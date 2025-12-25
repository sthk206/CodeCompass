# codecompass/retrieval/search.py
"""Search functionality for CodeCompass."""

from pathlib import Path
from dataclasses import dataclass

from codecompass.indexing.store import CodeStore
from codecompass.llm.ollama import generate

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

# Baseline Search
def baseline_search(repo_path: Path, query: str, limit: int = 5):
    results = search_code(repo_path, query, limit=limit)
    return results

# HYDE
def hyde_search(repo_path: Path, query: str, limit: int = 5):
    prompt = f"""Write a 3-5 line Python function that would match this search query.
Output ONLY the code, no explanation.

Query: {query}
````python
"""
    hypothetical = generate(prompt)
    hypothetical = hypothetical.strip()
    if hypothetical.startswith("```"):
        hypothetical = hypothetical.split("```")[1]
        hypothetical = hypothetical.replace("python", "", 1).strip()
    results = search_code(repo_path, hypothetical, limit=limit)
    return results

# EXPANDED QUERY
def query_expansion_search(repo_path: Path, query: str, limit: int = 5):

    prompt = f"""Add 5-10 related technical keywords to this code search query.
Only output the expanded query, nothing else.

Query: {query}
Expanded query:"""
    expanded_query = generate(prompt)
    results = search_code(repo_path, expanded_query, limit=limit)
    return results

# EXPANDED QUERY WITH CONTEXT
def query_expansion_search_context(repo_path: Path, query: str, limit: int = 5):
        
    store = CodeStore(repo_path)
    stats = store.get_stats()
    imports = stats.get("imports", [])[:100]
    imports_str = ", ".join(imports) if imports else "standard Python libraries"
    
    prompt = f"""Add 5-10 keywords to this code search query.
This repo uses these libraries: {imports_str}

ONLY add keywords that are directly relevant to the query.
Do NOT list unrelated libraries.

Query: {query}
Expanded query:"""
    expanded_query = generate(prompt)
    results = search_code(repo_path, expanded_query, limit=limit)
    return results

# -----------------------------------------------------------------------------
# Additional evaluations query expansion w/ context with different prompts
# -----------------------------------------------------------------------------

# Variation 1: Fewer imports (15) + always prepend original query
def query_expansion_context_v1(repo_path: Path, query: str, limit: int = 5):
    """Fewer imports, always include original query."""
    from codecompass.indexing.store import CodeStore
    
    store = CodeStore(repo_path)
    stats = store.get_stats()
    imports = stats.get("imports", [])[:15]  # Reduced from 100
    imports_str = ", ".join(imports)
    
    prompt = f"""Add 3-5 relevant keywords to this search query.
Available libraries: {imports_str}

Query: {query}
Keywords:"""
    keywords = generate(prompt).strip()
    expanded = f"{query} {keywords}"  # Always include original
    
    results = search_code(repo_path, expanded, limit=limit)
    return results


# Variation 2: Pick from list (more constrained)
def query_expansion_context_v2(repo_path: Path, query: str, limit: int = 5):
    """Ask LLM to pick relevant imports from the list."""
    from codecompass.indexing.store import CodeStore
    
    store = CodeStore(repo_path)
    stats = store.get_stats()
    imports = stats.get("imports", [])[:20]
    imports_str = ", ".join(imports)
    
    prompt = f"""Which of these libraries are relevant to the query? 
Pick 1-3 that are most relevant. Output only library names separated by spaces.

Libraries: {imports_str}
Query: {query}
Relevant:"""
    relevant = generate(prompt).strip()
    expanded = f"{query} {relevant}"
    
    results = search_code(repo_path, expanded, limit=limit)
    return results


# Variation 3: Minimal prompt
def query_expansion_context_v3(repo_path: Path, query: str, limit: int = 5):
    """Minimal prompt"""
    from codecompass.indexing.store import CodeStore
    
    store = CodeStore(repo_path)
    stats = store.get_stats()
    imports = stats.get("imports", [])[:100]
    imports_str = ", ".join(imports)
    
    prompt = f"""Query: {query}
Repo uses: {imports_str}
Only output the terms, nothing else.
Add 2-3 related terms:"""
    terms = generate(prompt).strip()
    expanded = f"{query} {terms}"
    
    results = search_code(repo_path, expanded, limit=limit)
    return results


# Variation 4: No LLM - rule-based matching
def query_expansion_context_v4(repo_path: Path, query: str, limit: int = 5):
    """No LLM, just match query words to imports."""
    from codecompass.indexing.store import CodeStore
    
    store = CodeStore(repo_path)
    stats = store.get_stats()
    imports = stats.get("imports", [])
    
    query_lower = query.lower()
    matched = []
    
    # Simple associations
    associations = {
        "embed": ["ollama", "sentence_transformers", "nomic"],
        "chunk": ["tree_sitter", "ast"],
        "vector": ["lancedb", "chromadb", "faiss"],
        "search": ["lancedb", "vector"],
        "cli": ["typer", "click", "argparse"],
        "command": ["typer", "click"],
        "parse": ["tree_sitter", "ast"],
        "llm": ["ollama", "openai", "anthropic"],
        "generate": ["ollama", "openai"],
        "store": ["lancedb", "database"],
        "index": ["lancedb", "vector"],
        "rag": ["lancedb", "ollama", "retrieval"],
    }
    
    for keyword, related in associations.items():
        if keyword in query_lower:
            for term in related:
                if term in imports and term not in matched:
                    matched.append(term)
    
    expanded = f"{query} {' '.join(matched[:5])}"
    
    results = search_code(repo_path, expanded, limit=limit)
    return results