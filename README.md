# CodeCompass 🧭

**AI-Powered Repository Understanding Assistant**

CodeCompass helps developers quickly understand and navigate unfamiliar codebases using natural language. Built as a local-first RAG system with an autonomous LangGraph agent, it manages the full pipeline for understanding complex codebases, from AST-based chunking and hybrid retrieval to systematic evaluation of retrieval strategies and fine-tuning approaches.

> Ask questions like *"How does authentication work?"* or *"What would break if I change this function?"* and get accurate, context-aware answers grounded in your actual codebase.

---
https://github.com/user-attachments/assets/c7d2b1c6-d93b-4563-a5d5-0bdac8d30999

## Highlights

- **Hybrid RAG Pipeline**: AST-aware code chunking → LanceDB vector store → semantic + BM25 hybrid search with HyDE query transformation
- **Autonomous Agent**: Autonomous ReAct LangGraph-powered agent with 6 specialized code-naviation tools
- **Rigorous Evaluation**: Systematic benchmarks comparing retrieval strategies (HyDE improves recall by 13.9%) and fine-tuning approaches
- **Local-First**: Runs entirely on your machine with Ollama—no API keys or cloud dependencies, well-suited for privacy-sensitive use cases

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [Technical Deep-Dive](#technical-deep-dive)
  - [RAG Pipeline](#rag-pipeline)
  - [Retrieval Strategy Evaluation](#retrieval-strategy-evaluation)
  - [LangGraph Agent](#langgraph-agent)
  - [Fine-Tuning Evaluation](#fine-tuning-evaluation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---


## Features

| Feature | Description |
|---------|-------------|
| **Interactive Terminal** | Arrow-navigable menu interface built with Questionary for easy command selection |
| **Conversation Memory** | Multi-turn chat with context retention across questions |
| **Git-Aware** | Understands commit history and file changes |
| **Diagram Generation** | Auto-generate architecture, dependency, flow, and class diagrams as Mermaid |
| **Streaming Responses** | Real-time token streaming with tool call visibility |

---

## Quick Start

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) (GPU/Metal recommended for fast inference)
- ~5GB disk space for models

### Installation

#### Option 1: PyPI Package (Recommended)

```bash
# 1. Navigate to the repository you want to explore
cd /path/to/your/repo

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install CodeCompass
pip install codecompass-ai

# 4. Start Ollama (in a separate terminal)
ollama serve
# Or use the Ollama Desktop App (recommended for macOS - enables Metal GPU acceleration)

# 5. Pull required models (~5GB total)
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 6. Start exploring
codecompass
```

#### Option 2: Git Clone (Development)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/codecompass.git

# 2. Navigate to the project you want to explore
cd /path/to/your/repo

# 3. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install from local path
pip install "/path/to/codecompass"

# 5. Start Ollama and pull models (same as above)
ollama serve  # Or use Desktop App for Metal support
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 6. Start exploring
codecompass
```

> **Note for macOS users**: The [Ollama Desktop App](https://ollama.com/download) automatically utilizes Metal for GPU acceleration, providing significantly faster inference than CPU-only execution. Homebrew installs (`brew install ollama`) may not enable Metal by default.

### Verify Installation

```bash
codecompass --version      # Check installation
codecompass status         # Check Ollama connection and index status
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CodeCompass Architecture                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Query                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    LangGraph Agent                              │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │    │
│  │  │   Reason    │───▶│  Tool Call  │───▶│  Synthesize Answer  │  │    │
│  │  │  (Qwen 7B)  │◀───│   Router    │◀───│                     │  │    │
│  │  └─────────────┘    └──────┬──────┘    └─────────────────────┘  │    │
│  └────────────────────────────┼────────────────────────────────────┘    │
│                               │                                         │
│          ┌────────────────────┼────────────────────┐                    │
│          ▼                    ▼                    ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐              │
│  │ search_code  │    │  read_file   │    │find_references│              │
│  │   (HyDE +    │    │              │    │               │              │
│  │ Hybrid RAG)  │    └──────────────┘    └───────────────┘              │
│  └──────┬───────┘                                                       │
│         │            ┌──────────────┐    ┌───────────────┐              │
│         │            │get_structure │    │get_git_history│              │
│         │            └──────────────┘    └───────────────┘              │
│         │                                                               │
│         │            ┌──────────────┐                                   │
│         │            │  get_deps    │                                   │
│         ▼            └──────────────┘                                   │
│  ┌─────────────────────────────────────┐                                │
│  │           RAG Pipeline              │                                │
│  │  ┌─────────┐  ┌─────────┐  ┌─────┐  │                                │
│  │  │  AST    │─▶│ Embed   │─▶│Lance│  │                                │
│  │  │ Chunker │  │ (Nomic) │  │ DB  │  │                                │
│  │  └─────────┘  └─────────┘  └─────┘  │                                │
│  └─────────────────────────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## CLI Reference

CodeCompass provides both an interactive menu and direct commands.

### Interactive Mode

```bash
codecompass          # Launch interactive menu with arrow navigation
```

### Direct Commands

| Command | Description | Example |
|---------|-------------|---------|
| `index` | Index a repository into the vector database | `codecompass index .` |
| `status` | Check indexing status of a repository | `codecompass status` |
| `search` | Hybrid retrieval over indexed code (no LLM response) | `codecompass search "authentication flow"` |
| `ask` | Ask a one-shot question about the codebase | `codecompass ask "How does caching work?"` |
| `chat` | Start interactive chat session with memory | `codecompass chat` |
| `diagram` | Generate architecture/dependency diagrams | `codecompass diagram . --type architecture` |

### Command Details

**Search** — Use when you want to find relevant code without an LLM-synthesized answer:
```bash
codecompass search "database connection" --limit 10 --stype 0
# --stype: 0=HyDE (default), 1=baseline, 2=query expansion
```

**Ask** — One-shot question with full agent reasoning:
```bash
codecompass ask "What would break if I change the User model?" --debug
```

**Chat** — Interactive session with conversation memory:
```bash
codecompass chat --no-memory    # Disable memory for independent questions
codecompass chat --debug        # Show token usage and tool calls
```

**Diagram** — Generate Mermaid diagrams:
```bash
codecompass diagram . --type architecture   # Module structure
codecompass diagram . --type dependency     # Import relationships
codecompass diagram . --type flow           # Entry points and call flow
codecompass diagram . --type class          # Class hierarchy
```

---

## Technical Deep-Dive

### RAG Pipeline

#### AST-Based Chunking

Unlike naive line-based or token-based chunking, CodeCompass uses **tree-sitter** to parse Python files into their Abstract Syntax Tree, extracting semantically meaningful chunks:

```python
# Each chunk represents a complete semantic unit
@dataclass
class CodeChunk:
    id: str              # Unique identifier: "path/file.py::ClassName.method_name"
    file_path: str       # Relative path from repo root
    name: str            # Function/class/method name
    chunk_type: str      # "function" | "class" | "method"
    code: str            # Complete source code
    start_line: int      # 1-indexed line numbers
    end_line: int
    docstring: str       # Extracted docstring if present
    parent_class: str    # For methods, the containing class
    imports: list[str]   # File-level imports for context
```

**Why AST chunking?** Token-based chunking can split functions mid-logic, losing semantic coherence. AST chunking ensures each chunk is a complete, self-contained unit (function, class, or method) that preserves the full context needed for understanding.

The chunker handles decorated definitions (e.g., `@app.command()`) by including decorators with their associated functions, and recursively extracts methods from class bodies while maintaining parent-class relationships.

#### Embedding Strategy

CodeCompass uses **nomic-embed-text** (768 dimensions) for embeddings, chosen for its strong performance on code understanding tasks. Each chunk is embedded as a composite text that includes:

```
Name: {function_name}
Type: {function|class|method}
Class: {parent_class if method}
Description: {docstring}
File imports: {comma-separated imports}
Code:
{source code}
```

**Handling long code**: The embedding model has a 2048-token context limit. For chunks exceeding this, we apply **middle truncation** that preserves the head (85%) and tail (15%) of the code:

```python
def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_chars = int(max_chars * 0.85)
    tail_chars = max_chars - head_chars
    return text[:head_chars] + "\n\n... [truncated] ...\n\n" + text[-tail_chars:]
```

This preserves docstrings, function signatures, and return statements—the most informative parts for code search—while truncating implementation details from the middle.

#### Hybrid Search

The vector store (LanceDB) supports both semantic vector search and BM25 lexical search. CodeCompass uses **hybrid search** combining both:

```python
results = (
    table
    .search(query_type="hybrid", fts_columns="search_text")
    .vector(query_vector)
    .text(query)
    .limit(limit)
    .to_list()
)
```

**Why hybrid?** Vector search excels at semantic similarity ("find authentication code" matches `login`, `session`, `token`) while BM25 excels at exact matches (searching for `UserModel` finds exact occurrences). Combining both provides better recall than either alone.

---

### Retrieval Strategy Evaluation

I evaluated retrieval strategies on 19 test queries across 5 categories. **HyDE improved recall by 13.9%** over baseline vector search.

| Strategy | Recall@5 | Precision@5 | MRR |
|----------|----------|-------------|-----|
| **HyDE** | **0.733** | 0.295 | 0.671 |
| Query Expansion | 0.718 | 0.295 | 0.737 |
| Baseline (vector) | 0.644 | 0.263 | 0.754 |

HyDE (Hypothetical Document Embedding) generates a hypothetical code snippet matching the query, then searches for similar real code—bridging the semantic gap between natural language and code.

**Negative Result**: Context-aware expansion (providing repo imports to the LLM) performed worst (0.428 Recall@5). Testing 4 prompt variations confirmed that adding repository context introduces noise rather than signal.

📊 **[Full evaluation details →](evaluation/README.md#retrieval-strategy-evaluation)**

---

### LangGraph Agent

The agent is built with **LangGraph**, implementing a Reason–Act (ReAct) loop where the LLM decides which tools to call based on the query:

```python
def create_agent(repo_path: Path, debug: bool = False):
    tools = create_tools(repo_path)
    llm = ChatOllama(model="qwen2.5:7b", num_ctx=16384).bind_tools(tools)

    def reason(state: MessagesState, config: RunnableConfig):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm.invoke(messages, config=config)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("reason", reason)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "reason")

    return graph.compile()
```

#### Available Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `search_code` | Semantic code search via HyDE + hybrid retrieval | "How does X work?", "Where is Y implemented?" |
| `read_file` | Read file contents with optional line range | When exact file path is known |
| `find_references` | Find all usages of a symbol across the codebase | "What uses X?", "What would break if I change X?" |
| `get_file_structure` | Directory tree of the repository | Project overview, finding file locations |
| `get_git_history` | Recent commit history, optionally for a specific file | "Who modified X?", "What changed recently?" |
| `get_dependencies` | Analyze imports and dependencies of a file | "What does this file depend on?" |

#### Token Management

Tool calls for large codebases can return 3-5K tokens per call. The Ollama context window is set to 16,384 tokens (vs. default 4,096) to prevent the model from losing the system prompt and conversation history when processing large tool results.

I explored several token-management strategies, including tool-output compression and sliding-window truncation of older messages. In practice, these approaches added complexity and made token usage harder to reason about during development. Expanding Ollama’s context window to 16K tokens proved to be a simpler and more reliable solution, handling large codebases effectively without additional logic. The compression and windowing code remains in the codebase but is disabled by default. We rely on Ollama’s built-in truncation behavior, which retains the system prompt and the most recent assistant message when the context window is exceeded.

For debugging, the `DebugChatOllama` wrapper tracks token usage across calls:

```python
class DebugChatOllama(ChatOllama):
    def generate(self, messages, **kwargs):
        # ... logs full request on first call
        result = super().generate(messages, **kwargs)
        
        usage = getattr(result.generations[0][0].message, "usage_metadata", None)
        if usage:
            # Track new tokens, detect truncation, show cumulative spend
            console.print(f"[TOKENS] new={new_input} | out={output_tokens} | spent={cumulative}")
        return result
```

---

### Fine-Tuning Evaluation

I evaluated fine-tuning for tool calling and code explanation tasks.

#### Tool Calling: No Fine-Tuning Needed

Through prompt engineering, base Qwen 2.5 7B achieved **95-100% tool selection accuracy** across two benchmarks:

| Benchmark | Queries | Accuracy | Valid JSON |
|-----------|---------|----------|------------|
| A (intent-driven) | 21 | 95.2% | 100% |
| B (tool-driven) | 39 | 100% | 100% |

#### Code Explanation: Negative Result

I trained a LoRA adapter on Magicoder-OSS-Instruct-75K and built an LLM-as-Judge evaluation framework. **The fine-tuned model lost 10/10 comparisons**:

| Metric | Fine-tuned | Base Model |
|--------|------------|------------|
| Clarity | 3.30 | 4.50 |
| Accuracy | 4.10 | 5.00 |
| Insight | 1.60 | 3.50 |
| Avg words | 60 | 180 |

The fine-tuned model was more concise but sacrificed explanatory depth. The base model's pretraining already captured sufficient code understanding.

**Decision**: Single-model architecture using base Qwen 2.5 7B for all stages.

📊 **[Full evaluation with qualitative examples →](evaluation/README.md#fine-tuning-evaluation)**

---

## Configuration

Configuration via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODECOMPASS_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `CODECOMPASS_CHAT_MODEL` | `qwen2.5:7b` | LLM for agent reasoning |
| `CODECOMPASS_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CODECOMPASS_OLLAMA_CTX_WINDOW` | `16384` | Context window size |
| `CODECOMPASS_CHUNK_MAX_CHARS` | `5000` | Max characters per code chunk |

LanceDB data is stored in `~/.codecompass/` by default.

---

## Troubleshooting

### "Connection refused" error
Ensure Ollama is running:
```bash
ollama serve
# Or open the Ollama desktop app
```

### "Input length exceeds context length" during indexing
Decrease chunk size in config:
```bash
export CODECOMPASS_CHUNK_MAX_CHARS=3000
```

### Slow responses
- Ensure GPU/Metal is being used (Ollama auto-detects)
>On macOS with Apple Silicon (M-series), make sure you’re using the official Ollama app, since the Homebrew installation does not enable Metal GPU support by default.
- Try a smaller model: `export CODECOMPASS_CHAT_MODEL=qwen2.5:3b`

### Model forgets context in long conversations
Increase context window:
```bash
export CODECOMPASS_OLLAMA_CTX_WINDOW=32768
```

---

## Tips for Better Results

1. **Be specific**: "How does user authentication work?" beats "How does auth work?"
2. **Reference files**: "What does src/main.py do?" helps focus the search
3. **Ask follow-ups**: Memory is enabled by default, so "What calls that function?" works after discussing a function
4. **Use search for exploration**: `codecompass search` finds relevant code without LLM overhead when you just want to browse

---

## Project Structure

```
codecompass/
├── agent/           # LangGraph agent and conversation management
│   └── graph.py     # Agent definition, state management, debug wrapper
├── diagrams/        # Mermaid diagram generation
├── indexing/        # AST chunking and vector store
│   ├── chunker.py   # Tree-sitter based code extraction
│   └── store.py     # LanceDB vector store with hybrid search
├── llm/             # Ollama integration
├── retrieval/       # Search strategies (HyDE, query expansion)
├── tools/           # LangChain tools for agent
├── cli.py           # Typer CLI commands
└── cli_menu.py      # Interactive questionary menu
```

---

## License

MIT
