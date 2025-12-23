from pathlib import Path

from codecompass.indexing.store import CodeStore
from codecompass.llm.ollama import generate

SYSTEM_PROMPT = """You are CodeCompass, an AI assistant that helps developers understand codebases.

You have been given relevant code snippets from the repository to help answer the user's question.
Base your answer on the provided code context. If the context doesn't contain enough information 
to fully answer the question, say so.

Be concise and specific. Reference file names and function names when relevant."""

def answer_question(repo_path: Path, question: str, num_chunks: int = 5) -> str:
    """Answer a question about a repostory using RAG"""

    store = CodeStore(repo_path)
    if not store.is_indexed():
        return f"Repository not indexed. Please run: codecompass index {repo_path}"

    # search num_chunks items in store with question
    results = store.search(question, limit=num_chunks)
    if not results:
        return "No relevant code found for your question."
    
    # form prompt
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"""
### Result {i}: {r['name']} ({r['chunk_type']})
**File:** {r['file_path']} (lines {r['start_line']}-{r['end_line']})
```python
{r['code']}
```
""")
    context = "\n".join(context_parts)

    prompt = f"""
## Relevant Code Context
{context}

## User Question
{question}

## Your Answer
"""
    # call generate
    answer = generate(prompt, system=SYSTEM_PROMPT)
    return answer

def format_sources(results: list[dict]) -> str:
    """Format search results as sources"""
    if not results:
        return ""
    
    lines = ["\n📚 Sources:"]
    for r in results:
        lines.append(f"  • {r['file_path']}:{r['start_line']} - {r['name']}")
    
    return "\n".join(lines)