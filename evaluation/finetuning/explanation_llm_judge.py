"""
CodeCompass Code Explanation Evaluation

Compares fine-tuned model vs baseline on code explanation quality.
Uses LLM-as-Judge for evaluation.

Usage:
    # Make sure both models are available in Ollama
    ollama pull qwen2.5:7b
    ollama create codecompass:explain -f Modelfile
    
    # Run evaluation
    python explanation_llm_judge.py --model codecompass:explain --baseline qwen2.5:7b
    
    # With custom judge (optional)
    python explanation_llm_judge.py --model codecompass:explain --baseline qwen2.5:7b --judge gpt-4

Metrics:
    - Win Rate: How often fine-tuned beats baseline
    - Quality Scores: Clarity, Accuracy, Conciseness, Insight (1-5)
    - Avg Response Length
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import ollama
except ImportError:
    print("Error: ollama package required. Run: pip install ollama")
    sys.exit(1)


# =============================================================================
# TEST CASES - Code snippets to explain
# =============================================================================

TEST_CASES = [
    {
        "id": "retry_backoff",
        "code": '''def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)''',
        "concepts": ["retry logic", "exponential backoff", "exception handling"],
    },
    {
        "id": "singleton",
        "code": '''class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance''',
        "concepts": ["singleton pattern", "__new__", "class-level state"],
    },
    {
        "id": "memoize",
        "code": '''def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper''',
        "concepts": ["decorator", "memoization", "caching", "closure"],
    },
    {
        "id": "context_manager",
        "code": '''class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.conn = None
    
    def __enter__(self):
        self.conn = connect(self.connection_string)
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        return False''',
        "concepts": ["context manager", "__enter__", "__exit__", "resource management"],
    },
    {
        "id": "async_gather",
        "code": '''async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]''',
        "concepts": ["async/await", "gather", "concurrent requests", "error filtering"],
    },
    {
        "id": "property_validator",
        "code": '''class User:
    def __init__(self, email):
        self._email = None
        self.email = email
    
    @property
    def email(self):
        return self._email
    
    @email.setter
    def email(self, value):
        if "@" not in value:
            raise ValueError("Invalid email")
        self._email = value.lower()''',
        "concepts": ["property decorator", "setter", "validation", "encapsulation"],
    },
    {
        "id": "factory_method",
        "code": '''class Serializer:
    @classmethod
    def from_format(cls, format_type):
        serializers = {
            "json": JsonSerializer,
            "xml": XmlSerializer,
            "csv": CsvSerializer,
        }
        if format_type not in serializers:
            raise ValueError(f"Unknown format: {format_type}")
        return serializers[format_type]()''',
        "concepts": ["factory method", "classmethod", "polymorphism"],
    },
    {
        "id": "generator_pipeline",
        "code": '''def read_large_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

def filter_lines(lines, pattern):
    for line in lines:
        if pattern in line:
            yield line

def process_file(path, pattern):
    lines = read_large_file(path)
    filtered = filter_lines(lines, pattern)
    return list(filtered)''',
        "concepts": ["generator", "lazy evaluation", "pipeline", "memory efficiency"],
    },
    {
        "id": "descriptor",
        "code": '''class Validated:
    def __init__(self, validator):
        self.validator = validator
        
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
        
    def __get__(self, obj, objtype=None):
        return getattr(obj, self.name, None)
        
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}")
        setattr(obj, self.name, value)''',
        "concepts": ["descriptor protocol", "__set_name__", "reusable validation"],
    },
    {
        "id": "rate_limiter",
        "code": '''class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def allow(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            return False
        self.calls.append(now)
        return True''',
        "concepts": ["rate limiting", "sliding window", "time-based filtering"],
    },
]


# =============================================================================
# LLM JUDGE PROMPT
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer evaluating the quality of code explanations.

You will be shown a code snippet and two explanations (A and B) in random order.
Your task is to evaluate which explanation is better.

Evaluation criteria:
1. **Clarity**: Is the explanation easy to understand?
2. **Accuracy**: Does it correctly describe what the code does?
3. **Conciseness**: Does it avoid unnecessary verbosity while covering key points?
4. **Insight**: Does it explain WHY the code is written this way, not just WHAT it does?
5. **Completeness**: Does it cover the important concepts without being exhaustive?

Respond with a JSON object:
{
    "winner": "A" or "B" or "tie",
    "scores": {
        "A": {"clarity": 1-5, "accuracy": 1-5, "conciseness": 1-5, "insight": 1-5},
        "B": {"clarity": 1-5, "accuracy": 1-5, "conciseness": 1-5, "insight": 1-5}
    },
    "reasoning": "Brief explanation of your decision"
}

Be fair and unbiased. Judge based on quality, not length."""


JUDGE_USER_TEMPLATE = """## Code to Explain

```python
{code}
```

## Explanation A

{explanation_a}

## Explanation B

{explanation_b}

---

Which explanation is better? Respond with JSON only."""


# =============================================================================
# EVALUATION LOGIC
# =============================================================================

@dataclass
class EvalResult:
    test_id: str
    model_explanation: str
    baseline_explanation: str
    winner: str  # "model", "baseline", "tie"
    model_scores: dict
    baseline_scores: dict
    reasoning: str
    model_was_a: bool  # Track position for bias detection


def generate_explanation(model: str, code: str) -> str:
    """Generate explanation from a model."""
    messages = [
        {
            "role": "system",
            "content": "Explain this code concisely. Focus on what it does, why it's designed this way, and any notable patterns."
        },
        {"role": "user", "content": code}
    ]
    
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.3}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[Error generating response: {e}]"


def judge_explanations(
    code: str,
    explanation_a: str,
    explanation_b: str,
    judge_model: str = "qwen2.5:7b"
) -> dict:
    """Use LLM to judge which explanation is better."""
    
    prompt = JUDGE_USER_TEMPLATE.format(
        code=code,
        explanation_a=explanation_a,
        explanation_b=explanation_b
    )
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = ollama.chat(
            model=judge_model,
            messages=messages,
            options={"temperature": 0}
        )
        
        result_text = response["message"]["content"].strip()
        
        # Parse JSON from response
        # Handle potential markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        return json.loads(result_text)
        
    except json.JSONDecodeError:
        return {
            "winner": "tie",
            "scores": {"A": {}, "B": {}},
            "reasoning": "Failed to parse judge response"
        }
    except Exception as e:
        return {
            "winner": "tie",
            "scores": {"A": {}, "B": {}},
            "reasoning": f"Judge error: {e}"
        }


def run_evaluation(
    model: str,
    baseline: str,
    judge_model: str,
    test_cases: list,
    verbose: bool = True
) -> list[EvalResult]:
    """Run full evaluation comparing model vs baseline."""
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"EVALUATION: {model} vs {baseline}")
    print(f"Judge: {judge_model}")
    print(f"Test cases: {len(test_cases)}")
    print(f"{'='*70}\n")
    
    for i, test in enumerate(test_cases):
        if verbose:
            print(f"[{i+1}/{len(test_cases)}] {test['id']}...")
        
        # Generate explanations
        model_explanation = generate_explanation(model, test["code"])
        baseline_explanation = generate_explanation(baseline, test["code"])
        
        # Randomize order to avoid position bias
        model_is_a = random.random() > 0.5
        
        if model_is_a:
            exp_a, exp_b = model_explanation, baseline_explanation
        else:
            exp_a, exp_b = baseline_explanation, model_explanation
        
        # Judge
        judgment = judge_explanations(
            test["code"],
            exp_a,
            exp_b,
            judge_model
        )
        
        # Map winner back to model/baseline
        raw_winner = judgment.get("winner", "tie")
        if raw_winner == "tie":
            winner = "tie"
        elif (raw_winner == "A" and model_is_a) or (raw_winner == "B" and not model_is_a):
            winner = "model"
        else:
            winner = "baseline"
        
        # Map scores
        scores = judgment.get("scores", {"A": {}, "B": {}})
        if model_is_a:
            model_scores = scores.get("A", {})
            baseline_scores = scores.get("B", {})
        else:
            model_scores = scores.get("B", {})
            baseline_scores = scores.get("A", {})
        
        result = EvalResult(
            test_id=test["id"],
            model_explanation=model_explanation,
            baseline_explanation=baseline_explanation,
            winner=winner,
            model_scores=model_scores,
            baseline_scores=baseline_scores,
            reasoning=judgment.get("reasoning", ""),
            model_was_a=model_is_a
        )
        
        results.append(result)
        
        if verbose:
            icon = "🏆" if winner == "model" else "❌" if winner == "baseline" else "🤝"
            print(f"    {icon} Winner: {winner}")
    
    return results


def print_summary(results: list[EvalResult], model: str, baseline: str):
    """Print evaluation summary."""
    
    total = len(results)
    model_wins = sum(1 for r in results if r.winner == "model")
    baseline_wins = sum(1 for r in results if r.winner == "baseline")
    ties = sum(1 for r in results if r.winner == "tie")
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    
    # Win rate
    print(f"\n📊 Win Rate:")
    print(f"  {model}: {model_wins}/{total} ({100*model_wins/total:.1f}%)")
    print(f"  {baseline}: {baseline_wins}/{total} ({100*baseline_wins/total:.1f}%)")
    print(f"  Ties: {ties}/{total} ({100*ties/total:.1f}%)")
    
    # Position bias check
    model_a_wins = sum(1 for r in results if r.winner == "model" and r.model_was_a)
    model_b_wins = sum(1 for r in results if r.winner == "model" and not r.model_was_a)
    model_a_total = sum(1 for r in results if r.model_was_a)
    model_b_total = total - model_a_total
    
    print(f"\n🔍 Position Bias Check:")
    if model_a_total > 0:
        print(f"  Model as A: {model_a_wins}/{model_a_total} wins ({100*model_a_wins/model_a_total:.1f}%)")
    if model_b_total > 0:
        print(f"  Model as B: {model_b_wins}/{model_b_total} wins ({100*model_b_wins/model_b_total:.1f}%)")
    
    # Average scores
    def avg_score(results, is_model, metric):
        scores = []
        for r in results:
            s = r.model_scores if is_model else r.baseline_scores
            if metric in s:
                scores.append(s[metric])
        return sum(scores) / len(scores) if scores else 0
    
    metrics = ["clarity", "accuracy", "conciseness", "insight"]
    
    print(f"\n📈 Average Scores (1-5):")
    print(f"  {'Metric':<15} {model:<15} {baseline:<15}")
    print(f"  {'-'*45}")
    for metric in metrics:
        model_avg = avg_score(results, True, metric)
        baseline_avg = avg_score(results, False, metric)
        diff = model_avg - baseline_avg
        indicator = "↑" if diff > 0.1 else "↓" if diff < -0.1 else "="
        print(f"  {metric:<15} {model_avg:<15.2f} {baseline_avg:<15.2f} {indicator}")
    
    # Response length comparison
    model_avg_len = sum(len(r.model_explanation) for r in results) / total
    baseline_avg_len = sum(len(r.baseline_explanation) for r in results) / total
    
    print(f"\n📏 Average Response Length:")
    print(f"  {model}: {model_avg_len:.0f} chars")
    print(f"  {baseline}: {baseline_avg_len:.0f} chars")
    
    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT:")
    print(f"{'='*70}")
    
    win_rate = model_wins / total
    if win_rate > 0.6:
        print(f"✅ {model} is significantly better ({win_rate:.0%} win rate)")
        print(f"   Fine-tuning improved code explanation quality.")
    elif win_rate > 0.45:
        print(f"🤷 Results are mixed ({win_rate:.0%} win rate)")
        print(f"   Fine-tuning shows marginal improvement.")
    else:
        print(f"⚠️ {baseline} performed better ({100-win_rate:.0%} win rate)")
        print(f"   Fine-tuning may have regressed quality. Check training.")
    
    print()


def save_results(results: list[EvalResult], model: str, baseline: str, output_path: Path):
    """Save detailed results to JSON."""
    
    output = {
        "model": model,
        "baseline": baseline,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "model_wins": sum(1 for r in results if r.winner == "model"),
            "baseline_wins": sum(1 for r in results if r.winner == "baseline"),
            "ties": sum(1 for r in results if r.winner == "tie"),
        },
        "results": [
            {
                "test_id": r.test_id,
                "winner": r.winner,
                "model_scores": r.model_scores,
                "baseline_scores": r.baseline_scores,
                "reasoning": r.reasoning,
                "model_explanation": r.model_explanation[:500],  # Truncate
                "baseline_explanation": r.baseline_explanation[:500],
            }
            for r in results
        ]
    }
    
    # Calculate win rate
    total = len(results)
    output["summary"]["model_win_rate"] = output["summary"]["model_wins"] / total
    
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate code explanation quality: fine-tuned vs baseline"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="codecompass",
        help="Fine-tuned model name in Ollama"
    )
    parser.add_argument(
        "--baseline", 
        type=str, 
        default="qwen2.5:7b",
        help="Baseline model to compare against"
    )
    parser.add_argument(
        "--judge", 
        type=str, 
        default=None,
        help="Model to use as judge (default: same as baseline)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file for results JSON"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    # Default judge to baseline (or use a separate judge model)
    judge_model = args.judge or args.baseline
    
    # Check models exist
    try:
        available = ollama.list()
        model_names = [m.model for m in available.models] if hasattr(available, 'models') else []
        model_names = [m.replace(':latest', '') for m in model_names]
        
        for check_model in [args.model, args.baseline, judge_model]:
            check_name = check_model.replace(':latest', '')
            if check_name not in model_names and check_model not in [m.model for m in available.models]:
                print(f"⚠️  Model '{check_model}' not found in Ollama.")
                print(f"   Available: {model_names}")
                print(f"   Run: ollama pull {check_model}")
                sys.exit(1)
                
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    # Run evaluation
    random.seed(42)  # For reproducibility
    results = run_evaluation(
        model=args.model,
        baseline=args.baseline,
        judge_model=judge_model,
        test_cases=TEST_CASES,
        verbose=not args.quiet
    )
    
    # Print summary
    print_summary(results, args.model, args.baseline)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"eval_judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        # output_path = Path(f"eval_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    save_results(results, args.model, args.baseline, output_path)


if __name__ == "__main__":
    main()