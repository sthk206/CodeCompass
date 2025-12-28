"""
CodeCompass Tools Benchmark A (Intent-driven complex queries, seperated by codecompass use cases)


Usage:
    # First, make sure Ollama is running with Qwen
    ollama pull qwen2.5:7b
    
    # Run benchmark
    python tools_benchmark_a.py --model qwen2.5:7b
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

try:
    import ollama
except ImportError:
    print("Error: ollama package required. Run: pip install ollama")
    sys.exit(1)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

CODECOMPASS_TOOLS = [
    {
        "name": "search_code",
        "description": "Semantic search across the codebase. Use for conceptual queries like 'how is X handled' or 'where is Y implemented' when you don't know the exact file or function name.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of what to find"},
                "file_type": {"type": "string", "description": "Optional file extension filter (e.g., '.py')"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_file",
        "description": "Read contents of a specific file. Use when you know the exact file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file relative to repo root"},
                "start_line": {"type": "integer", "description": "Optional starting line"},
                "end_line": {"type": "integer", "description": "Optional ending line"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "find_references",
        "description": "Find all usages of a specific symbol (function, class, variable). Use when you know the exact name and want to see where it's used.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Exact name of the function, class, or variable"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_file_structure",
        "description": "Get directory tree structure. Use for understanding project organization or finding where files are located.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path, use '.' for root", "default": "."},
                "max_depth": {"type": "integer", "description": "How deep to show", "default": 3}
            },
            "required": []
        }
    },
    {
        "name": "get_git_history",
        "description": "Get git commit history. Use for understanding what changed, when, why, and by whom.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Optional file to get history for"},
                "limit": {"type": "integer", "description": "Number of commits", "default": 10}
            },
            "required": []
        }
    },
    {
        "name": "get_dependencies",
        "description": "Get imports and dependencies of a file. Use to understand what a file depends on.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file to analyze"}
            },
            "required": ["file_path"]
        }
    }
]



SYSTEM_PROMPT = """You are CodeCompass, an AI assistant that helps developers understand and navigate codebases.

You have access to these tools:

{tools}

IMPORTANT RULES:
1. When you need information from the codebase, use a tool. Respond with JSON:
   {{"tool": "tool_name", "arguments": {{"arg": "value"}}}}

2. If you can answer directly from your knowledge (explaining concepts, giving advice), respond with:
   {{"tool": null, "answer": "your response"}}
CRITICAL: For the "answer" field, write everything on ONE LINE. Do not use newlines or line breaks.

3. Always respond with valid JSON only. No other text.

4. Choose the most specific tool for the task:
- **search_code**: Conceptual queries when no exact symbol or file is specified. Keep queries SHORT (2-4 keywords). Remove filler words. 
- **find_references**: When you KNOW the exact symbol name. Extract just the symbol. Use **find_references** whenever the user asks what would break, what is affected, or where a symbol is used, even if they phrase it as an impact or dependency question.
- **get_git_history**: Questions about changes, history, who modified, when changed.
- **read_file**: When you know the exact file path.
- **get_file_structure**: Project overview or finding file locations.
- **No tool**: General concepts that don't need codebase access.

Examples:

User: How is authentication implemented?
Response: {{"tool": "search_code", "arguments": {{"query": "authentication"}}}}

User: What would break if I change the UserService class?
Response: {{"tool": "find_references", "arguments": {{"symbol": "UserService"}}}}

User: What changed in auth recently?
Response: {{"tool": "get_git_history", "arguments": {{"file_path": "auth"}}}}
"""


# =============================================================================
# TEST CASES - Questions that actually benefit from AI assistance
# =============================================================================

TEST_CASES = [
    # =========================================================================
    # CATEGORY 1: Onboarding & Exploration
    # Why AI helps: New devs don't know where to start. IDE can't answer these.
    # =========================================================================
    {
        "id": "onboard_1",
        "category": "onboarding",
        "question": "I just joined this project. Can you give me an overview of how it's structured?",
        "expected_tool": "get_file_structure",
        "why_valuable": "New developer orientation - IDE can't explain project structure",
        "notes": "Should use get_file_structure to show layout, possibly with explanation"
    },
    {
        "id": "onboard_2", 
        "category": "onboarding",
        "question": "What are the main entry points to this application?",
        "expected_tool": "search_code",
        "why_valuable": "Finding entry points requires understanding, not just search",
        "notes": "Should search for main, app, entry, or similar patterns"
    },
    {
        "id": "onboard_3",
        "category": "onboarding",
        "question": "Where should I look to understand how the API works?",
        "expected_tool": "search_code",
        "why_valuable": "Guidance question - needs reasoning about where APIs are defined",
        "notes": "Should search for API routes, endpoints, handlers"
    },
    
    # =========================================================================
    # CATEGORY 2: Conceptual Understanding
    # Why AI helps: These require understanding concepts, not just text matching
    # =========================================================================
    {
        "id": "concept_1",
        "category": "understanding",
        "question": "How does this codebase handle errors? Is there a consistent pattern?",
        "expected_tool": "search_code",
        "why_valuable": "Pattern recognition across codebase - Ctrl+F can't do this",
        "notes": "Should search for error handling, exceptions, try/catch patterns"
    },
    {
        "id": "concept_2",
        "category": "understanding",
        "question": "What's the data flow from when a user submits a form to when it's saved in the database?",
        "expected_tool": "search_code",
        "why_valuable": "Tracing data flow requires multi-step reasoning",
        "notes": "May need multiple tool calls to trace the flow"
    },
    {
        "id": "concept_3",
        "category": "understanding",
        "question": "How is authentication implemented? Walk me through the flow.",
        "expected_tool": "search_code",
        "why_valuable": "Understanding a system, not just finding a file",
        "notes": "Should search for auth, login, session, token patterns"
    },
    {
        "id": "concept_4",
        "category": "understanding",
        "question": "Is there any caching in this codebase? How is it used?",
        "expected_tool": "search_code",
        "why_valuable": "Discovery question - you don't know IF caching exists",
        "notes": "Should search for cache, redis, memoize, etc."
    },
    
    # =========================================================================
    # CATEGORY 3: Impact Analysis & Dependencies
    # Why AI helps: Understanding ripple effects requires reasoning
    # =========================================================================
    {
        "id": "impact_1",
        "category": "impact",
        "question": "If I change the User model, what parts of the codebase might be affected?",
        "expected_tool": "find_references",
        "why_valuable": "Impact analysis before making changes",
        "notes": "Should find references to User class/model"
    },
    {
        "id": "impact_2",
        "category": "impact",
        "question": "What would break if we removed the legacy_api module?",
        "expected_tool": "find_references",
        "why_valuable": "Deprecation planning - need to know dependencies",
        "notes": "Should find what imports or uses legacy_api"
    },
    {
        "id": "impact_3",
        "category": "impact",
        "question": "Is the config.py file used everywhere or just in a few places?",
        "expected_tool": "find_references",
        "why_valuable": "Understanding coupling/dependency scope",
        "notes": "Could use find_references on config or search for config imports"
    },
    
    # =========================================================================
    # CATEGORY 4: Debugging & Investigation
    # Why AI helps: Debugging often requires exploring unknowns
    # =========================================================================
    {
        "id": "debug_1",
        "category": "debugging",
        "question": "Users are reporting slow login times. Where should I look to investigate?",
        "expected_tool": "search_code",
        "why_valuable": "Starting point for debugging - need to find relevant code",
        "notes": "Should search for login, auth, session creation"
    },
    {
        "id": "debug_2",
        "category": "debugging",
        "question": "There's a bug with payment processing. Show me the payment-related code.",
        "expected_tool": "search_code",
        "why_valuable": "Finding relevant code when you don't know the structure",
        "notes": "Should search for payment, charge, billing, transaction"
    },
    {
        "id": "debug_3",
        "category": "debugging",
        "question": "What changed in the authentication code recently? Could that explain the new bug?",
        "expected_tool": "get_git_history",
        "why_valuable": "Correlating bugs with recent changes",
        "notes": "Should get git history, possibly filtered to auth files"
    },
    
    # =========================================================================
    # CATEGORY 5: Planning & Architecture
    # Why AI helps: Requires understanding current state before making decisions
    # =========================================================================
    {
        "id": "plan_1",
        "category": "planning",
        "question": "I need to add rate limiting. Is there any existing rate limiting code I can extend?",
        "expected_tool": "search_code",
        "why_valuable": "Avoid reinventing - find existing patterns first",
        "notes": "Should search for rate limit, throttle, etc."
    },
    {
        "id": "plan_2",
        "category": "planning",
        "question": "We want to add a new API endpoint. Where are the existing endpoints defined?",
        "expected_tool": "search_code",
        "why_valuable": "Finding the right place to add new code",
        "notes": "Should search for routes, endpoints, @app.route, etc."
    },
    {
        "id": "plan_3",
        "category": "planning",
        "question": "What database ORM or query pattern does this project use?",
        "expected_tool": "search_code",
        "why_valuable": "Understanding tech stack decisions",
        "notes": "Should search for database, ORM, query, model patterns"
    },
    
    # =========================================================================
    # CATEGORY 6: Code Review & Quality
    # Why AI helps: Finding patterns/anti-patterns across codebase
    # =========================================================================
    {
        "id": "quality_1",
        "category": "quality",
        "question": "Are there any TODO or FIXME comments I should be aware of?",
        "expected_tool": "search_code",
        "why_valuable": "Finding technical debt across codebase",
        "notes": "Should search for TODO, FIXME, HACK, XXX"
    },
    {
        "id": "quality_2",
        "category": "quality",
        "question": "How are environment variables handled? Are there any hardcoded secrets?",
        "expected_tool": "search_code",
        "why_valuable": "Security review - finding potential issues",
        "notes": "Should search for env, config, secret, password, api_key"
    },
    
    # =========================================================================
    # CATEGORY 7: No Tool Needed
    # Why important: Model should know when NOT to use tools
    # =========================================================================
    {
        "id": "no_tool_1",
        "category": "no_tool",
        "question": "What's the difference between REST and GraphQL?",
        "expected_tool": None,
        "why_valuable": "General knowledge - shouldn't search codebase for this",
        "notes": "Should answer directly without tool"
    },
    {
        "id": "no_tool_2",
        "category": "no_tool",
        "question": "Can you explain what dependency injection is?",
        "expected_tool": None,
        "why_valuable": "Concept explanation - no tool needed",
        "notes": "Should answer directly without tool"
    },
    {
        "id": "no_tool_3",
        "category": "no_tool",
        "question": "What's a good way to structure Python tests?",
        "expected_tool": None,
        "why_valuable": "Best practice advice - general knowledge",
        "notes": "Should answer directly, maybe offer to show examples from codebase"
    },
]


# =============================================================================
# EVALUATION LOGIC
# =============================================================================

@dataclass
class TestResult:
    test_id: str
    category: str
    question: str
    expected_tool: Optional[str]
    actual_tool: Optional[str]
    raw_response: str
    
    valid_json: bool = False
    correct_tool: bool = False
    has_arguments: bool = False
    
    error: Optional[str] = None


def generate_response(model: str, question: str) -> str:
    """Generate response from model."""
    formatted_prompt = SYSTEM_PROMPT.format(tools=json.dumps(CODECOMPASS_TOOLS, indent=2))
    
    messages = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": question}
    ]
    
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0}
    )
    
    return response["message"]["content"].strip()


def evaluate_response(response: str, expected_tool: Optional[str]) -> dict:
    """Evaluate a model response."""
    result = {
        "valid_json": False,
        "correct_tool": False,
        "has_arguments": False,
        "actual_tool": None,
        "error": None
    }
    
    try:
        # Try to parse JSON
        parsed = json.loads(response)
        result["valid_json"] = True
        
        # Extract tool
        actual_tool = parsed.get("tool")
        result["actual_tool"] = actual_tool
        
        # Check tool correctness
        result["correct_tool"] = (actual_tool == expected_tool)
        
        # Check for arguments/answer
        if actual_tool:
            result["has_arguments"] = "arguments" in parsed and len(parsed.get("arguments", {})) > 0
        else:
            result["has_arguments"] = "answer" in parsed and len(str(parsed.get("answer", ""))) > 10
            
    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON: {str(e)[:50]}"
    
    return result


def run_benchmark(model: str, verbose: bool = True) -> List[TestResult]:
    """Run the full benchmark."""
    results = []
    
    print(f"\n{'='*70}")
    print(f"Running benchmark on: {model}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"{'='*70}\n")
    
    for i, test in enumerate(TEST_CASES):
        if verbose:
            print(f"[{i+1}/{len(TEST_CASES)}] {test['category']}: {test['question'][:50]}...")
        
        try:
            response = generate_response(model, test["question"])
            eval_result = evaluate_response(response, test["expected_tool"])
            
            result = TestResult(
                test_id=test["id"],
                category=test["category"],
                question=test["question"],
                expected_tool=test["expected_tool"],
                actual_tool=eval_result["actual_tool"],
                raw_response=response,
                valid_json=eval_result["valid_json"],
                correct_tool=eval_result["correct_tool"],
                has_arguments=eval_result["has_arguments"],
                error=eval_result["error"]
            )
            
            if verbose:
                status = "✓" if result.correct_tool and result.valid_json else "✗"
                expected = test["expected_tool"] or "no_tool"
                actual = eval_result["actual_tool"] or "no_tool"
                print(f"    {status} Expected: {expected:<20} Got: {actual}")
            
        except Exception as e:
            result = TestResult(
                test_id=test["id"],
                category=test["category"],
                question=test["question"],
                expected_tool=test["expected_tool"],
                actual_tool=None,
                raw_response="",
                error=str(e)
            )
            if verbose:
                print(f"    ✗ Error: {str(e)[:50]}")
        
        results.append(result)
    
    return results


def print_summary(results: List[TestResult], model: str):
    """Print benchmark summary."""
    total = len(results)
    valid_json = sum(1 for r in results if r.valid_json)
    correct_tool = sum(1 for r in results if r.correct_tool)
    has_args = sum(1 for r in results if r.has_arguments)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS: {model}")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  Valid JSON:      {valid_json}/{total} ({100*valid_json/total:.1f}%)")
    print(f"  Correct Tool:    {correct_tool}/{total} ({100*correct_tool/total:.1f}%)")
    print(f"  Has Args/Answer: {has_args}/{total} ({100*has_args/total:.1f}%)")
    
    # By category
    print(f"\nBy Category:")
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"total": 0, "correct": 0}
        categories[r.category]["total"] += 1
        if r.correct_tool:
            categories[r.category]["correct"] += 1
    
    for cat, stats in sorted(categories.items()):
        pct = 100 * stats["correct"] / stats["total"]
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {cat:<15} {bar} {stats['correct']}/{stats['total']} ({pct:.0f}%)")
    
    # Failures
    failures = [r for r in results if not r.correct_tool or not r.valid_json]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures[:10]:  # Show first 10
            expected = r.expected_tool or "no_tool"
            actual = r.actual_tool or "no_tool"
            print(f"  [{r.test_id}] Expected: {expected}, Got: {actual}")
            if r.error:
                print(f"           Error: {r.error}")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION:")
    print(f"{'='*70}")
    
    accuracy = correct_tool / total
    if accuracy >= 0.8:
        print(f"  ✓ {model} achieves {accuracy:.0%} accuracy.")
        print(f"  → Fine-tuning may NOT be necessary.")
        print(f"  → Focus on prompt engineering and RAG quality instead.")
    elif accuracy >= 0.6:
        print(f"  ~ {model} achieves {accuracy:.0%} accuracy.")
        print(f"  → Fine-tuning could help, but try prompt engineering first.")
        print(f"  → Consider fine-tuning on Magicoder for code understanding.")
    else:
        print(f"  ✗ {model} only achieves {accuracy:.0%} accuracy.")
        print(f"  → Fine-tuning is recommended.")
        print(f"  → Use SWE-agent trajectories or build custom dataset.")
    
    print()


def save_results(results: List[TestResult], model: str, output_path: Path):
    """Save detailed results to JSON."""
    output = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "valid_json": sum(1 for r in results if r.valid_json),
            "correct_tool": sum(1 for r in results if r.correct_tool),
            "accuracy": sum(1 for r in results if r.correct_tool) / len(results)
        },
        "results": [
            {
                "test_id": r.test_id,
                "category": r.category,
                "question": r.question,
                "expected_tool": r.expected_tool,
                "actual_tool": r.actual_tool,
                "valid_json": r.valid_json,
                "correct_tool": r.correct_tool,
                "raw_response": r.raw_response[:500],  # Truncate
                "error": r.error
            }
            for r in results
        ]
    }
    
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model on code navigation tool-calling")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama model to test")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    # Check Ollama
    try:
        models = [m.model for m in ollama.list()["models"]]
        if args.model not in models and f"{args.model}:latest" not in models:
            print(f"Model '{args.model}' not found. Available: {models}")
            print(f"\nTry: ollama pull {args.model}")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    # Run benchmark
    results = run_benchmark(args.model, verbose=not args.quiet)
    
    # Print summary
    print_summary(results, args.model)
    
    # Save results
    if args.output:
        save_results(results, args.model, Path(args.output))
    else:
        output_path = Path(f"benchmark_tools_A_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        # output_path = Path(f"benchmark_{args.model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_results(results, args.model, output_path)


if __name__ == "__main__":
    main()