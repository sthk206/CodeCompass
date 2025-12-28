# Fine-tuning Evaluations

## I. Tool Calling Evaluation

We started around 70% correct tool accuracy with JSON format issues. Through prompt engineering, we were able to get 100% valid json and up to 95.2% tool calling, with 100% tool calling for simpler queries.

### 1. Benchmark A (21 intent-driven queries simulating real use cases for CodeCompass)

#### a. Sample queries
```
    {
        "id": "concept_3",
        "category": "understanding",
        "question": "How is authentication implemented? Walk me through the flow.",
        "expected_tool": "search_code",
        "why_valuable": "Understanding a system, not just finding a file",
        "notes": "Should search for auth, login, session, token patterns"
    },
```
```
    {
        "id": "impact_1",
        "category": "impact",
        "question": "If I change the User model, what parts of the codebase might be affected?",
        "expected_tool": "find_references",
        "why_valuable": "Impact analysis before making changes",
        "notes": "Should find references to User class/model"
    }
```

#### b. Results
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

======================================================================
ANALYSIS & RECOMMENDATIONS:
======================================================================
  ✓ qwen2.5:7b achieves 95% accuracy.
  → Fine-tuning may NOT be necessary.
```

### 2. Benchmark B (39 tool-driven, simpler queries for testing general tool invocation)

#### a. Sample queries
```
    {
        "id": "search_4",
        "category": "search_code",
        "question": "Find code related to user permissions and access control",
        "expected_tool": "search_code",
        "rationale": "Broad conceptual search"
    }
```
```
    {
        "id": "refs_6",
        "category": "find_references",
        "question": "Show me all the places that call send_email",
        "expected_tool": "find_references",
        "rationale": "Explicit 'all places that call X' pattern"
    }
```

#### b. Results
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

======================================================================
ANALYSIS & RECOMMENDATIONS:
======================================================================

✓  JSON Compliance: 100% (good)
✓  Tool Selection: 100%
   Base model performs well. Fine-tuning may not be needed.
```

## II. Code Explanation Finetuning

The two-phase architecture separates tool selection from code explanation.
We hypothesized the explanation phase might benefit from domain-specific fine-tuning.

### 1. Methodology
Built an LLM-as-Judge evaluation framework before committing to fine-tuning:

- **10 code patterns** across categories (async, decorators, design patterns)
- **4 quality dimensions**: clarity, accuracy, conciseness, insight
- **Position randomization** to eliminate ordering bias
- **Automated win-rate calculation**

### 2. Experiment
- **Base model**: Qwen 2.5 7B (4-bit quantized)
- **Fine-tuned**: LoRA adapter trained on Magicoder dataset (1,400 examples)
- **Judge**: Qwen 2.5 7B (separate inference)

### 3. Results
```
======================================================================
EVALUATION: codecompass:explain vs qwen2.5:7b
Judge: qwen2.5:7b
Test cases: 10
======================================================================

[1/10] retry_backoff...
    ❌ Winner: baseline
[2/10] singleton...
    ❌ Winner: baseline
[3/10] memoize...
    ❌ Winner: baseline
[4/10] context_manager...
    ❌ Winner: baseline
[5/10] async_gather...
    ❌ Winner: baseline
[6/10] property_validator...
    ❌ Winner: baseline
[7/10] factory_method...
    ❌ Winner: baseline
[8/10] generator_pipeline...
    ❌ Winner: baseline
[9/10] descriptor...
    ❌ Winner: baseline
[10/10] rate_limiter...
    ❌ Winner: baseline

======================================================================
EVALUATION RESULTS
======================================================================

📊 Win Rate:
  codecompass:explain: 0/10 (0.0%)
  qwen2.5:7b: 10/10 (100.0%)
  Ties: 0/10 (0.0%)

🔍 Position Bias Check:
  Model as A: 0/4 wins (0.0%)
  Model as B: 0/6 wins (0.0%)

📈 Average Scores (1-5):
  Metric          codecompass:explain qwen2.5:7b     
  ---------------------------------------------
  clarity         3.30            4.50            ↓
  accuracy        4.10            5.00            ↓
  conciseness     2.50            4.00            ↓
  insight         1.60            3.50            ↓

📏 Average Response Length:
  codecompass:explain: 407 chars
  qwen2.5:7b: 1019 chars

======================================================================
SUMMARY STATISTICS
======================================================================

Metric                           Baseline      Fine-tuned            Diff
----------------------------------------------------------------------
Avg words/response                  180.2            59.6          -66.9%
Avg time/response (s)               11.37           25.00         +119.8%
Total words                          1802             596
Total time (s)                      113.7           250.0

======================================================================
VERDICT:
======================================================================
⚠️ qwen2.5:7b performed better (10000% win rate)
   Fine-tuning may have regressed quality. Check training.
```

### 4. QUALITATIVE RESULTS: SAMPLE RESPONSES

#### a. retry_backoff: Retry with exponential backoff

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

**BASELINE** (178 words, 15.0s):
````
This code defines a `retry_with_backoff` function that attempts to execute another function (`func`) up to `max_retries` times. It introduces exponential backoff between retries, doubling the wait time after each failure.

### Key Points:
- **Purpose**: To retry executing a function with increasing delays if it fails.
- **Design Decisions**:
  - Uses a loop to attempt execution multiple times.
  - Implements exponential backoff to avoid overwhelming the system on repeated failures.
- **Notable Patterns/Techniques**:
  - Exception handling: retries until the last attempt, then rethrows the exception.
  - Exponential backoff: delays increase exponentially (e.g., 1s, 2s, 4s).
- **Edge Cases Handled**:
  - If `func` succeeds on any attempt, it returns immediately without further retries.
  - Retries up to `max_retries`, then rethrows the last exception if all attempts fail.

### Example Usage:
```python
def fetch_data():
    # Simulate a network error
    raise ConnectionError("Network issue")

try:
    result = retry_with_backoff(fetch_data, max_retries=5)
    print(result)
except Exception as e:
    print(f"Failed after retries: {e}")
```

This pattern is useful for handling transient errors in distributed systems where occasional failures are expected and should be retried with increasing delays.
````

**FINE-TUNED** (59 words, 24.9s):
The `retry_with_backoff` function takes the original function to be retried, along with optional parameters for maximum retries and base delay. It attempts to execute the function within a loop, handling any exceptions that may occur. If an exception is caught, it waits for increasing delays before retrying until the maximum number of retries is reached or no exception occurs.

----------------------------------------------------------------------

#### b. singleton: Singleton design pattern

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

**BASELINE** (135 words, 9.9s):
````
This code defines a singleton class `DatabaseConnection` that ensures only one database connection exists at any time. Here's a concise breakdown:

- **Purpose**: Ensures a single instance of the database connection to avoid multiple connections.
- **Key Design Decisions**:
  - Uses the Singleton pattern with `__new__` method to control instantiation.
  - Hides the actual connection object (`_connection`) as an attribute.
- **Notable Patterns/Techniques**:
  - **Singleton Pattern**: Ensures only one instance of the class is created and provides a global point of access to it.
  - **Lazy Initialization**: The `_connection` is created only when `connect` is called for the first time.
- **Edge Cases Handled**:
  - Prevents multiple connections by checking if `_connection` exists before creating a new one.

In summary, this class manages database connections efficiently, ensuring that all instances share the same connection object.
````

**FINE-TUNED** (55 words, 24.9s):
The `DatabaseConnection` class ensures that only one instance of the database connection exists throughout the application. The `_instance` attribute acts as a singleton pattern to maintain a single point of access to the database connection. The `connect` method establishes a connection if it hasn't been established yet and returns the connection object for further operations.

----------------------------------------------------------------------

#### c. decorator_cache: Memoization decorator

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

**BASELINE** (139 words, 9.1s):
````
This code defines a `memoize` function that acts as a decorator to memoize another function's results. Key points:

- It uses a dictionary `cache` to store previously computed results.
- The inner `wrapper` function checks if the current arguments have been seen before; if not, it computes and stores the result.
- It handles both positional and keyword arguments by converting them into a hashable key.
- Uses `functools.wraps` to preserve the wrapped function's metadata.

Notable:
- Sorts kwargs to handle equivalent kwarg orders differently (e.g., `f(a=1, b=2)` vs. `f(b=2, a=1)`).
- Could raise issues with unhashable or non-pickleable arguments.
- May consume significant memory for functions with many unique argument combinations.

Edge cases:
- Functions that always return the same value regardless of input won't be memoized properly.
- Recursive functions need careful handling to avoid infinite recursion.
````

**FINE-TUNED** (82 words, 25.5s):
The `memoize` decorator is implemented using a dictionary to store the results of function calls. The key for each result is generated based on the input arguments and keyword arguments. This ensures that the memoization works correctly even when the order of keyword arguments changes. The `wrapper` function checks if the result for the given inputs is already cached, and returns it if available; otherwise, it computes the result using the original function and stores it in the cache before returning it.


### 5. Conclusion
The fine-tuned model had more concise explanations but at the cost of explanatory depth.
Base model's pretraining already captured sufficient code understanding.

**Decision**: Single-model architecture using base Qwen 2.5 7B for all stages.