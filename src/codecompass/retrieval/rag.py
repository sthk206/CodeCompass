from pathlib import Path

from codecompass.indexing.store import CodeStore
from codecompass.llm.ollama import generate
from codecompass.retrieval.search import (
    baseline_search, hyde_search, query_expansion_search
)

SYSTEM_PROMPT = """You are CodeCompass, an AI assistant that helps developers understand codebases.

You have been given relevant code snippets from the repository to help answer the user's question.
Base your answer on the provided code context. If the context doesn't contain enough information 
to fully answer the question, say so.

Be concise and specific. Reference file names and function names when relevant."""

SEARCH_STRATEGIES = {
    0: hyde_search,           # Default - best performing
    1: baseline_search,
    2: query_expansion_search,
}

def answer_question(
    repo_path: Path, 
    question: str, 
    num_chunks: int = 5,
    search_type: int = 0,
) -> str:
    """Answer a question about a repository using RAG."""

    store = CodeStore(repo_path)
    if not store.is_indexed():
        return f"Repository not indexed. Please run: codecompass index {repo_path}"

    # Get search strategy
    search_fn = SEARCH_STRATEGIES.get(search_type, hyde_search)
    
    # Search for relevant chunks
    results = search_fn(repo_path, question, num_chunks)
    
    if not results:
        return "No relevant code found for your question."
    
    # Build context from results
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"""
### Result {i}: {r.name} ({r.chunk_type})
**File:** {r.file_path} (lines {r.start_line}-{r.end_line})
```python
{r.code}
```
""")
    context = "\n".join(context_parts)

    prompt = f"""## Relevant Code Context
{context}

## User Question
{question}

## Your Answer
"""
    answer = generate(prompt, system=SYSTEM_PROMPT)
    return answer


def format_sources(results: list) -> str:
    """Format search results as sources."""
    if not results:
        return ""
    
    lines = ["\n📚 Sources:"]
    for r in results:
        lines.append(f"  • {r.file_path}:{r.start_line} - {r.name}")
    
    return "\n".join(lines)
