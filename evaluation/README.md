# CodeCompass Evaluation

This document details the systematic evaluation of retrieval strategies and fine-tuning approaches for CodeCompass.

## Table of Contents

- [Retrieval Strategy Evaluation](#retrieval-strategy-evaluation)
  - [Methodology](#methodology)
  - [Results](#results)
  - [Context-Aware Expansion: Negative Result](#context-aware-expansion-negative-result)
- [Fine-Tuning Evaluation](#fine-tuning-evaluation)
  - [Tool Calling Evaluation](#i-tool-calling-evaluation)
  - [Code Explanation Fine-Tuning](#ii-code-explanation-fine-tuning)
  - [Qualitative Results](#qualitative-results)

---

## Retrieval Strategy Evaluation

### Methodology

We evaluated four retrieval strategies on a benchmark of **19 test queries** across 5 categories:

| Category | Description | Example Query |
|----------|-------------|---------------|
| Feature Search | Finding specific functionality | "Find the login function" |
| Concept Search | Understanding implementations | "How does caching work?" |
| API Search | Finding interfaces | "What endpoints are available?" |
| Debug Search | Troubleshooting | "Where is the error handling?" |
| Natural Language | Conversational queries | "Show me how users are created" |

**Metrics:**
- **Recall@5**: Proportion of relevant results in top 5
- **Precision@5**: Proportion of top 5 results that are relevant
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant result

### Results

| Strategy | Recall@5 | Precision@5 | MRR |
|----------|----------|-------------|-----|
| **HyDE** | **0.733** | 0.295 | 0.671 |
| Query Expansion | 0.718 | 0.295 | 0.737 |
| Baseline (vector search) | 0.644 | 0.263 | 0.754 |
| Query Expansion + Context | 0.428 | 0.189 | 0.399 |

**Key Finding**: HyDE (Hypothetical Document Embedding) achieved the best recall, improving by **13.9%** over baseline vector search.

#### How HyDE Works

Instead of embedding the user's query directly, HyDE generates a hypothetical code snippet that would answer the query, then searches for real code similar to that hypothetical:

```python
def hyde_search(repo_path: Path, query: str, limit: int = 5):
    prompt = f"""Write a 3-5 line Python function that would match this search query.
Output ONLY the code, no explanation.

Query: {query}"""
    
    hypothetical = generate(prompt)  # LLM generates hypothetical code
    results = search_code(repo_path, hypothetical, limit=limit)
    return results
```

**Why it works**: The hypothetical code is in the same "language" as the indexed code chunks, leading to better embedding similarity than natural language queries.

#### Query Expansion

Adds related technical keywords to broaden the search:

```python
def query_expansion_search(repo_path: Path, query: str, limit: int = 5):
    prompt = f"""Add 5-10 related technical keywords to this code search query.
Only output the expanded query, nothing else.

Query: {query}
Expanded query:"""
    expanded_query = generate(prompt)
    results = search_code(repo_path, expanded_query, limit=limit)
    return results
```

#### Query Expansion + Context

I hypothesized that providing relevant repository context (imports/libraries used) would help the LLM generate more relevant query expansions.

```python
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
```

### Context-Aware Expansion: Negative Result

The context-aware strategy (providing repo imports to the LLM) performed surprisingly poorly. I hypothesized the prompt was suboptimal, so I tested 4 variations.

#### Experiment Design

Tested 5 variations of context-aware query expansion:

| Variant | Description | Implementation |
|---------|-------------|----------------|
| Original | 100 imports, verbose prompt | Provide all imports, ask for expansion |
| v1 | Fewer imports (15) | Reduced context, always prepend original query |
| v2 | Pick from list | LLM selects relevant imports from list |
| v3 | Minimal prompt | Simplified instructions |
| v4 | Rule-based | No LLM, keyword matching against imports |

#### Results

| Variant | Description | Recall@5 |
|---------|-------------|----------|
| v4: Rule-based | No LLM, keyword matching | 0.691 |
| v2: Pick from list | LLM selects relevant imports | 0.665 |
| v3: Minimal prompt | Simple prompt | 0.644 |
| v1: Fewer imports | 15 imports instead of 100 | 0.600 |
| Original | 100 imports, verbose prompt | 0.428 |


#### Implementation of Variants

**v2 (LLM selects relevant imports)**:
```python
# Variation 2: Pick from list (more constrained)
def query_expansion_context_v2(repo_path: Path, query: str, limit: int = 5):
    """Ask LLM to pick relevant imports from the list"""
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

```

#### Analysis
Even the best context-aware variant (rule-based, no LLM) underperformed simple query expansion (0.691 vs 0.718). **Simpler strategies (HyDE, basic query expansion) outperform complex context-aware approaches for code search.** Adding repository context introduces noise rather than signal—the LLM either hallucinates irrelevant connections or gets distracted by the additional information.

---

## Fine-Tuning Evaluation

### Background

I considered fine-tuning for two tasks:
1. **Tool calling**: Selecting the right tool for a query
2. **Code explanation**: Generating clear explanations of code

**Dataset consideration**: Evaluated xlam-60k (Salesforce), CodeSearchNet, code_x_glue_ct_code_to_text, and settled on **Magicoder-OSS-Instruct-75K** for its detailed yet concise explanations.

**Architecture decision**: Proposed a two-pass architecture with hot-swap adapters:

```
│  User Query                                                  │
│      ↓                                                       │
│  [Base Model] → Tool selection (95% accurate, no training)  │
│      ↓                                                       │
│  [Tools] → Code chunks                                       │
│      ↓                                                       │
│  [Fine-tuned Model] → Explanations only ← TRAIN THIS ONLY   │
│      ↓                                                       │
│  [Base Model] → Final synthesized answer                    │
```

Using LoRA to avoid users downloading two full models.

---

## I. Tool Calling Evaluation

### Starting Point

Initial evaluation showed ~70% correct tool selection with JSON format issues. Through prompt engineering alone, we achieved:
- **100% valid JSON**
- **95-100% correct tool selection**

### Benchmark A: Intent-Driven Queries

21 queries simulating real CodeCompass use cases across 7 categories.

#### Sample Queries

```json
{
    "id": "concept_3",
    "category": "understanding",
    "question": "How is authentication implemented? Walk me through the flow.",
    "expected_tool": "search_code",
    "why_valuable": "Understanding a system, not just finding a file",
    "notes": "Should search for auth, login, session, token patterns"
}
```

```json
{
    "id": "impact_1",
    "category": "impact",
    "question": "If I change the User model, what parts of the codebase might be affected?",
    "expected_tool": "find_references",
    "why_valuable": "Impact analysis before making changes",
    "notes": "Should find references to User class/model"
}
```

#### Results

```
======================================================================
BENCHMARK RESULTS: qwen2.5:7b
======================================================================

Overall Metrics:
  Valid JSON:      21/21 (100.0%)
  Correct Tool:    20/21 (95.2%)
  Has Args/Answer: 21/21 (100.0%)

By Category:
  debugging       ██████████ 3/3 (100%)
  impact          ██████░░░░ 2/3 (67%)
  no_tool         ██████████ 3/3 (100%)
  onboarding      ██████████ 3/3 (100%)
  planning        ██████████ 3/3 (100%)
  quality         ██████████ 2/2 (100%)
  understanding   ██████████ 4/4 (100%)

Failures (1):
  [impact_3] Expected: find_references, Got: search_code
```

**Analysis**: The single failure was an edge case where `search_code` and `find_references` were both reasonable choices. The model chose semantic search over reference tracing.

### Benchmark B: Tool-Driven Queries

39 simpler queries testing general tool invocation across all 6 tools.

#### Sample Queries

```json
{
    "id": "search_4",
    "category": "search_code",
    "question": "Find code related to user permissions and access control",
    "expected_tool": "search_code",
    "rationale": "Broad conceptual search"
}
```

```json
{
    "id": "refs_6",
    "category": "find_references",
    "question": "Show me all the places that call send_email",
    "expected_tool": "find_references",
    "rationale": "Explicit 'all places that call X' pattern"
}
```

#### Results

```
======================================================================
BENCHMARK RESULTS: qwen2.5:7b
======================================================================

Overall Metrics:
  Valid JSON:      39/39 (100.0%)
  Correct Tool:    39/39 (100.0%)
  Has Args/Answer: 36/39 (92.3%)

By Category:
  find_references      ██████████ 10/10 (100%)
  get_dependencies     ██████████ 5/5 (100%)
  get_file_structure   ██████████ 4/4 (100%)
  get_git_history      ██████████ 4/4 (100%)
  no_tool              ██████████ 5/5 (100%)
  read_file            ██████████ 5/5 (100%)
  search_code          ██████████ 6/6 (100%)
```

### Tool Calling Conclusion

**The base Qwen 2.5 7B model achieves 95-100% tool selection accuracy through prompt engineering alone.** Fine-tuning for tool calling is unnecessary.

---

## II. Code Explanation Fine-Tuning

Given that tool calling was solved, I focused fine-tuning efforts on code explanation.

### Methodology

Built an **LLM-as-Judge** evaluation framework before committing to fine-tuning:

| Component | Details |
|-----------|---------|
| Test Cases | 10 code patterns (async, decorators, design patterns, etc.) |
| Quality Dimensions | Clarity, Accuracy, Conciseness, Insight |
| Bias Mitigation | Position randomization (model A vs B) |
| Metrics | Win rate, dimension scores (1-5), response length |

### Experiment Setup

| Model | Details |
|-------|---------|
| Base | Qwen 2.5 7B (4-bit quantized) |
| Fine-tuned | LoRA adapter trained on Magicoder (1,400 examples) |
| Judge | Qwen 2.5 7B (separate inference) |

### Results

```
======================================================================
EVALUATION RESULTS
======================================================================

📊 Win Rate:
  codecompass:explain (fine-tuned): 0/10 (0.0%)
  qwen2.5:7b (base):                10/10 (100.0%)
  Ties:                             0/10 (0.0%)

🔍 Position Bias Check:
  Fine-tuned as A: 0/4 wins (0.0%)
  Fine-tuned as B: 0/6 wins (0.0%)

📈 Average Scores (1-5):
  Metric          Fine-tuned    Base Model    
  ---------------------------------------------
  clarity         3.30          4.50          ↓
  accuracy        4.10          5.00          ↓
  insight         1.60          3.50          ↓

📏 Average Response Length:
  Fine-tuned: 407 chars (59.6 words)
  Base:       1019 chars (180.2 words)

⏱️ Inference Time:
  Fine-tuned: 25.0s average
  Base:       11.4s average (+119.8% slower with adapter)
```

### Analysis

The fine-tuned model:
- Produced **shorter** responses (60 words vs 180 words)
- Scored **lower** on all quality dimensions
- Was **slower** due to additional LoRA adapter computations during inference

**Root cause**: The Magicoder dataset optimizes for concise, direct answers. While this works for code generation tasks, code *explanation* benefits from elaboration—discussing design patterns, edge cases, and usage examples.

---

## Qualitative Results

### Example 1: Retry with Exponential Backoff

**Code:**
```python
def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
```

**Base Model** (178 words):
> This code defines a `retry_with_backoff` function that attempts to execute another function (`func`) up to `max_retries` times. It introduces exponential backoff between retries, doubling the wait time after each failure.
>
> **Key Points:**
> - **Purpose**: To retry executing a function with increasing delays if it fails.
> - **Design Decisions**: Uses a loop to attempt execution multiple times. Implements exponential backoff to avoid overwhelming the system on repeated failures.
> - **Notable Patterns**: Exception handling retries until the last attempt, then rethrows. Exponential backoff: delays increase (e.g., 1s, 2s, 4s).
> - **Edge Cases**: If `func` succeeds on any attempt, returns immediately. Retries up to `max_retries`, then rethrows the last exception.
>
> **Example Usage:**
> ```python
> def fetch_data():
>     raise ConnectionError("Network issue")
> 
> try:
>     result = retry_with_backoff(fetch_data, max_retries=5)
> except Exception as e:
>     print(f"Failed after retries: {e}")
> ```
>
> This pattern is useful for handling transient errors in distributed systems.

**Fine-Tuned** (59 words):
> The `retry_with_backoff` function takes the original function to be retried, along with optional parameters for maximum retries and base delay. It attempts to execute the function within a loop, handling any exceptions that may occur. If an exception is caught, it waits for increasing delays before retrying until the maximum number of retries is reached or no exception occurs.

**Analysis**: The base model provides richer context—naming the pattern (exponential backoff), explaining *why* it's useful (avoid overwhelming systems), showing example usage, and noting edge cases. The fine-tuned model only describes *what* the code does.

---

### Example 2: Singleton Pattern

**Code:**
```python
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connection = None
        return cls._instance
    
    def connect(self, connection_string):
        if self._connection is None:
            self._connection = create_connection(connection_string)
        return self._connection
```

**Base Model** (135 words):
> This code defines a singleton class `DatabaseConnection` that ensures only one database connection exists at any time.
>
> - **Purpose**: Ensures a single instance of the database connection to avoid multiple connections.
> - **Key Design Decisions**: Uses the Singleton pattern with `__new__` method to control instantiation. Hides the actual connection object (`_connection`) as an attribute.
> - **Notable Patterns**:
>   - **Singleton Pattern**: Ensures only one instance of the class is created and provides a global point of access.
>   - **Lazy Initialization**: The `_connection` is created only when `connect` is called for the first time.
> - **Edge Cases**: Prevents multiple connections by checking if `_connection` exists before creating a new one.

**Fine-Tuned** (55 words):
> The `DatabaseConnection` class ensures that only one instance of the database connection exists throughout the application. The `_instance` attribute acts as a singleton pattern to maintain a single point of access to the database connection. The `connect` method establishes a connection if it hasn't been established yet and returns the connection object for further operations.

**Analysis**: The base model identifies two patterns (Singleton and Lazy Initialization), explains the design rationale, and notes implementation details (`__new__` method). The fine-tuned model provides a surface-level description without naming the patterns explicitly.

---

### Example 3: Memoization Decorator

**Code:**
```python
def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper
```

**Base Model** (139 words):
> This code defines a `memoize` function that acts as a decorator to memoize another function's results.
>
> - Uses a dictionary `cache` to store previously computed results.
> - The `wrapper` function checks if arguments have been seen before; if not, computes and stores.
> - Handles both positional and keyword arguments by converting to a hashable key.
> - Uses `functools.wraps` to preserve the wrapped function's metadata.
>
> **Notable:**
> - Sorts kwargs to handle equivalent kwarg orders (e.g., `f(a=1, b=2)` vs. `f(b=2, a=1)`).
> - Could raise issues with unhashable or non-pickleable arguments.
> - May consume significant memory for functions with many unique argument combinations.
>
> **Edge cases:**
> - Functions that always return the same value regardless of input won't benefit.
> - Recursive functions need careful handling to avoid infinite recursion.

**Fine-Tuned** (82 words):
> The `memoize` decorator is implemented using a dictionary to store the results of function calls. The key for each result is generated based on the input arguments and keyword arguments. This ensures that the memoization works correctly even when the order of keyword arguments changes. The `wrapper` function checks if the result for the given inputs is already cached, and returns it if available; otherwise, it computes the result using the original function and stores it in the cache before returning it.

**Analysis**: The base model goes beyond mechanics to discuss **practical concerns**: memory consumption, unhashable arguments, recursive function behavior. These insights are valuable for developers using the code. The fine-tuned model stays at the implementation level.

---

## Final Decision

**Single-model architecture using base Qwen 2.5 7B for all stages.**

The evaluation validated that:
1. **Tool calling**: Base model achieves 95-100% accuracy with prompt engineering
2. **Code explanation**: Fine-tuning regressed quality; base model's pretraining captured sufficient code understanding

This avoids the complexity of hot-swapping adapters and provides better user experience (faster inference, higher quality explanations).