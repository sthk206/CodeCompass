"""
CodeCompass Tools Benchmark B (Tool-driven simpler queries, seperated by distinct tool use cases)


Usage:
    # First, make sure Ollama is running with Qwen
    ollama pull qwen2.5:7b
    
    # Run benchmark
    python tools_benchmark_b.py --model qwen2.5:7b
"""


import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Optional, List
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
        "description": "Semantic search across the codebase for code matching a concept. Use when you DON'T know the exact name and need to discover/explore.",
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
        "description": "Read contents of a specific file. Use when you know the exact file path and want to see its contents.",
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
        "description": "Find all locations where a specific symbol (function, class, variable, module) is imported, called, or used. Use when you KNOW the exact name and want to find its usages/callers/dependents.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Exact name of the function, class, variable, or module"}
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
        "description": "Analyze imports and dependencies OF a specific file. Shows what external modules/packages a file imports. Use to understand what a file depends on.",
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
- **search_code**: Conceptual queries. Keep queries SHORT (2-4 keywords). Remove filler words.
- **find_references**: When you KNOW the exact symbol name. Extract just the symbol.
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
# TEST CASES
# =============================================================================

TEST_CASES = [
    # =========================================================================
    # SEARCH_CODE - Exploration/Discovery queries
    # =========================================================================
    {
        "id": "search_1",
        "category": "search_code",
        "question": "How is authentication implemented in this codebase?",
        "expected_tool": "search_code",
        "rationale": "Exploring a concept - don't know specific file/function names yet"
    },
    {
        "id": "search_2",
        "category": "search_code",
        "question": "Is there any caching logic? Show me where.",
        "expected_tool": "search_code",
        "rationale": "Discovery question - checking if something exists"
    },
    {
        "id": "search_3",
        "category": "search_code",
        "question": "Where is error handling done in this project?",
        "expected_tool": "search_code",
        "rationale": "Looking for a pattern across codebase"
    },
    {
        "id": "search_4",
        "category": "search_code",
        "question": "Find code related to user permissions and access control",
        "expected_tool": "search_code",
        "rationale": "Broad conceptual search"
    },
    {
        "id": "search_5",
        "category": "search_code",
        "question": "How does the app connect to the database?",
        "expected_tool": "search_code",
        "rationale": "Understanding implementation - exploring"
    },
    
    # =========================================================================
    # FIND_REFERENCES - Known symbol, find usages
    # These are the cases where IDE's F12/Shift+F12 would help, but the user
    # is asking via natural language, often for impact analysis
    # =========================================================================
    {
        "id": "refs_1",
        "category": "find_references",
        "question": "What calls the validate_user function?",
        "expected_tool": "find_references",
        "rationale": "Exact name given (validate_user), want callers"
    },
    {
        "id": "refs_2",
        "category": "find_references",
        "question": "Where is the UserService class used?",
        "expected_tool": "find_references",
        "rationale": "Exact name given (UserService), want usages"
    },
    {
        "id": "refs_3",
        "category": "find_references",
        "question": "If I modify the DatabaseConnection class, what might break?",
        "expected_tool": "find_references",
        "rationale": "Impact analysis - exact name given, find dependents"
    },
    {
        "id": "refs_4",
        "category": "find_references",
        "question": "What would break if we removed the legacy_api module?",
        "expected_tool": "find_references",
        "rationale": "Deprecation check - exact name given (legacy_api)"
    },
    {
        "id": "refs_5",
        "category": "find_references",
        "question": "Is the config module imported anywhere besides main.py?",
        "expected_tool": "find_references",
        "rationale": "Usage scope question - exact name given (config)"
    },
    {
        "id": "refs_6",
        "category": "find_references",
        "question": "Show me all the places that call send_email",
        "expected_tool": "find_references",
        "rationale": "Explicit 'all places that call X' pattern"
    },
    {
        "id": "refs_7",
        "category": "find_references",
        "question": "What depends on the utils.helpers module?",
        "expected_tool": "find_references",
        "rationale": "Dependency question with exact module name"
    },
    {
        "id": "refs_8",
        "category": "find_references",
        "question": "I want to rename the process_payment function. What files would I need to update?",
        "expected_tool": "find_references",
        "rationale": "Refactoring prep - need all usages of exact symbol"
    },
    
    # =========================================================================
    # READ_FILE - Known file path
    # =========================================================================
    {
        "id": "read_1",
        "category": "read_file",
        "question": "Show me the contents of src/main.py",
        "expected_tool": "read_file",
        "rationale": "Explicit file path given"
    },
    {
        "id": "read_2",
        "category": "read_file",
        "question": "What's in the README.md file?",
        "expected_tool": "read_file",
        "rationale": "Specific file requested"
    },
    {
        "id": "read_3",
        "category": "read_file",
        "question": "Open config/settings.py",
        "expected_tool": "read_file",
        "rationale": "Direct file open request"
    },
    {
        "id": "read_4",
        "category": "read_file",
        "question": "Can you display lines 50-100 of app/models.py?",
        "expected_tool": "read_file",
        "rationale": "File with specific line range"
    },
    {
        "id": "read_5",
        "category": "read_file",
        "question": "Let me see the pyproject.toml",
        "expected_tool": "read_file",
        "rationale": "Specific config file request"
    },
    
    # =========================================================================
    # GET_DEPENDENCIES - What does a file import
    # =========================================================================
    {
        "id": "deps_1",
        "category": "get_dependencies",
        "question": "What external packages does main.py use?",
        "expected_tool": "get_dependencies",
        "rationale": "Asking about imports OF a specific file"
    },
    {
        "id": "deps_2",
        "category": "get_dependencies",
        "question": "Show me the imports in src/services/auth.py",
        "expected_tool": "get_dependencies",
        "rationale": "Explicit imports request for specific file"
    },
    {
        "id": "deps_3",
        "category": "get_dependencies",
        "question": "What does the payment module depend on?",
        "expected_tool": "get_dependencies",
        "rationale": "Dependencies OF a module (not dependents)"
    },
    {
        "id": "deps_4",
        "category": "get_dependencies",
        "question": "Which libraries are imported in utils.py?",
        "expected_tool": "get_dependencies",
        "rationale": "Library/import analysis of specific file"
    },
    
    # =========================================================================
    # GET_FILE_STRUCTURE - Project layout
    # =========================================================================
    {
        "id": "struct_1",
        "category": "get_file_structure",
        "question": "What's the overall structure of this project?",
        "expected_tool": "get_file_structure",
        "rationale": "Project overview request"
    },
    {
        "id": "struct_2",
        "category": "get_file_structure",
        "question": "Show me how the src folder is organized",
        "expected_tool": "get_file_structure",
        "rationale": "Directory structure request"
    },
    {
        "id": "struct_3",
        "category": "get_file_structure",
        "question": "I'm new here. Can you give me a map of the codebase?",
        "expected_tool": "get_file_structure",
        "rationale": "Onboarding/orientation request"
    },
    {
        "id": "struct_4",
        "category": "get_file_structure",
        "question": "What files are in the tests directory?",
        "expected_tool": "get_file_structure",
        "rationale": "Listing contents of directory"
    },
    
    # =========================================================================
    # GET_GIT_HISTORY - Change tracking
    # =========================================================================
    {
        "id": "git_1",
        "category": "get_git_history",
        "question": "What changed in the last few commits?",
        "expected_tool": "get_git_history",
        "rationale": "Recent changes question"
    },
    {
        "id": "git_2",
        "category": "get_git_history",
        "question": "Who last modified the auth.py file?",
        "expected_tool": "get_git_history",
        "rationale": "File history/blame question"
    },
    {
        "id": "git_3",
        "category": "get_git_history",
        "question": "When was the payment module last updated?",
        "expected_tool": "get_git_history",
        "rationale": "Temporal question about changes"
    },
    {
        "id": "git_4",
        "category": "get_git_history",
        "question": "Has anyone touched the database code recently?",
        "expected_tool": "get_git_history",
        "rationale": "Recent modification check"
    },
    
    # =========================================================================
    # NO TOOL - General knowledge / direct answers
    # =========================================================================
    {
        "id": "no_tool_1",
        "category": "no_tool",
        "question": "What's the difference between REST and GraphQL?",
        "expected_tool": None,
        "rationale": "General CS knowledge - not codebase-specific"
    },
    {
        "id": "no_tool_2",
        "category": "no_tool",
        "question": "Can you explain what dependency injection is?",
        "expected_tool": None,
        "rationale": "Concept explanation - general knowledge"
    },
    {
        "id": "no_tool_3",
        "category": "no_tool",
        "question": "What's a good way to structure Python tests?",
        "expected_tool": None,
        "rationale": "Best practice advice - not codebase-specific"
    },
    {
        "id": "no_tool_4",
        "category": "no_tool",
        "question": "How do async/await work in Python?",
        "expected_tool": None,
        "rationale": "Language feature explanation"
    },
    {
        "id": "no_tool_5",
        "category": "no_tool",
        "question": "What are the SOLID principles?",
        "expected_tool": None,
        "rationale": "Software engineering concepts"
    },
    
    # =========================================================================
    # EDGE CASES - Tricky distinctions
    # =========================================================================
    {
        "id": "edge_1",
        "category": "find_references",
        "question": "What parts of the code use the requests library?",
        "expected_tool": "find_references",
        "rationale": "Known library name - find where it's imported/used"
    },
    {
        "id": "edge_2",
        "category": "search_code",
        "question": "How does this app make HTTP requests?",
        "expected_tool": "search_code",
        "rationale": "Conceptual - exploring HOW, not WHERE a specific thing is used"
    },
    {
        "id": "edge_3",
        "category": "get_dependencies",
        "question": "What does app/api.py import from our codebase?",
        "expected_tool": "get_dependencies",
        "rationale": "Imports OF a specific file"
    },
    {
        "id": "edge_4",
        "category": "find_references",
        "question": "What imports app/api.py?",
        "expected_tool": "find_references",
        "rationale": "What imports THIS file = find_references on module name"
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


def clean_json_response(response: str) -> str:
    """Clean up common JSON issues in model responses."""
    # Remove markdown code blocks if present
    response = re.sub(r'^```json\s*', '', response.strip())
    response = re.sub(r'\s*```$', '', response)
    response = re.sub(r'^```\s*', '', response)
    
    # Try to extract JSON object if there's extra text
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        response = json_match.group()
    
    return response.strip()


def fix_json_newlines(response: str) -> str:
    """Fix unescaped newlines in JSON string values."""
    # This regex finds string values and escapes newlines within them
    # Pattern: find "answer": "..." and escape newlines inside
    
    def escape_in_string(match):
        content = match.group(1)
        # Escape actual newlines (not already escaped \n)
        content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return f'"answer": "{content}"'
    
    # Handle "answer": "..." pattern
    response = re.sub(r'"answer":\s*"((?:[^"\\]|\\.)*)"', escape_in_string, response, flags=re.DOTALL)
    
    return response


def parse_response(response: str) -> tuple[dict, Optional[str]]:
    """Parse model response into JSON, with error handling."""
    try:
        return json.loads(response), None
    except json.JSONDecodeError:
        # fallback cleaning
        response = re.sub(r'^```json\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response)
        response = re.sub(r'^```\s*', '', response)
        try:
            return json.loads(response), None
        except json.JSONDecodeError as e:
            return {}, f"Invalid JSON: {str(e)[:50]}"



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
    
    parsed, error = parse_response(response)
    
    if error:
        result["error"] = error
        # Still try to detect tool from raw response for partial credit
        for tool in ["search_code", "read_file", "find_references", 
                     "get_file_structure", "get_git_history", "get_dependencies"]:
            if f'"tool": "{tool}"' in response or f'"tool":"{tool}"' in response:
                result["actual_tool"] = tool
                result["correct_tool"] = (tool == expected_tool)
                break
        if '"tool": null' in response or '"tool":null' in response:
            result["actual_tool"] = None
            result["correct_tool"] = (expected_tool is None)
        return result
    
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
                status = "✓" if result.correct_tool else "✗"
                expected = test["expected_tool"] or "no_tool"
                actual = eval_result["actual_tool"] or "no_tool"
                extra = "" if result.valid_json else " [JSON error]"
                print(f"    {status} Expected: {expected:<20} Got: {actual}{extra}")
            
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
        cat = r.category
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r.correct_tool:
            categories[cat]["correct"] += 1
    
    for cat, stats in sorted(categories.items()):
        pct = 100 * stats["correct"] / stats["total"]
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {cat:<20} {bar} {stats['correct']}/{stats['total']} ({pct:.0f}%)")
    
    # Confusion matrix for tool selection
    print(f"\nTool Confusion (Expected → Actual):")
    confusion = {}
    for r in results:
        expected = r.expected_tool or "no_tool"
        actual = r.actual_tool or "no_tool"
        key = (expected, actual)
        confusion[key] = confusion.get(key, 0) + 1
    
    for (expected, actual), count in sorted(confusion.items()):
        if expected != actual:
            print(f"  {expected:<20} → {actual:<20} ({count}x)")
    
    # Failures
    failures = [r for r in results if not r.correct_tool]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures[:15]:  # Show first 15
            expected = r.expected_tool or "no_tool"
            actual = r.actual_tool or "no_tool"
            print(f"  [{r.test_id}] Expected: {expected}, Got: {actual}")
            if r.error:
                print(f"           Error: {r.error}")
    
    # JSON-specific issues
    json_failures = [r for r in results if not r.valid_json]
    if json_failures:
        print(f"\nJSON Parse Failures ({len(json_failures)}):")
        for r in json_failures[:5]:
            print(f"  [{r.test_id}] {r.error}")
            print(f"           Response preview: {r.raw_response[:100]}...")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("ANALYSIS & RECOMMENDATIONS:")
    print(f"{'='*70}")
    
    accuracy = correct_tool / total
    json_rate = valid_json / total
    
    # JSON format assessment
    if json_rate < 0.9:
        print(f"\n⚠️  JSON Compliance: {json_rate:.0%}")
        print(f"   The model struggles with consistent JSON output.")
        print(f"   → Fine-tuning on structured output format recommended.")
    else:
        print(f"\n✓  JSON Compliance: {json_rate:.0%} (good)")
    
    # Tool selection assessment
    if accuracy >= 0.85:
        print(f"\n✓  Tool Selection: {accuracy:.0%}")
        print(f"   Base model performs well. Fine-tuning may not be needed.")
        print(f"   → Focus on RAG quality and prompt engineering.")
    elif accuracy >= 0.70:
        print(f"\n~  Tool Selection: {accuracy:.0%}")
        print(f"   Decent but has blind spots. Check category breakdown above.")
        print(f"   → Light fine-tuning on weak categories may help.")
    else:
        print(f"\n✗  Tool Selection: {accuracy:.0%}")
        print(f"   Model struggles with tool selection.")
        print(f"   → Fine-tuning recommended, especially on xLAM dataset.")
    
    # Specific category issues
    for cat, stats in categories.items():
        cat_acc = stats["correct"] / stats["total"]
        if cat_acc < 0.6:
            print(f"\n   Low accuracy on '{cat}' ({cat_acc:.0%}):")
            if cat == "find_references":
                print(f"   → Model confuses find_references with search_code")
                print(f"   → Need examples distinguishing 'explore concept' vs 'find usages of X'")
            elif cat == "no_tool":
                print(f"   → Model over-uses tools for general knowledge questions")
                print(f"   → Add more no-tool examples to training data")
    
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
        "by_category": {},
        "results": []
    }
    
    # Category breakdown
    for r in results:
        cat = r.category
        if cat not in output["by_category"]:
            output["by_category"][cat] = {"total": 0, "correct": 0}
        output["by_category"][cat]["total"] += 1
        if r.correct_tool:
            output["by_category"][cat]["correct"] += 1
    
    # Individual results
    for r in results:
        output["results"].append({
            "test_id": r.test_id,
            "category": r.category,
            "question": r.question,
            "expected_tool": r.expected_tool,
            "actual_tool": r.actual_tool,
            "valid_json": r.valid_json,
            "correct_tool": r.correct_tool,
            "raw_response": r.raw_response[:500],  # Truncate
            "error": r.error
        })
    
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
        models = ollama.list()
        model_names = [m.model for m in models.models] if hasattr(models, 'models') else [m['name'] for m in models.get('models', [])]
        
        # Normalize model names for comparison
        normalized_names = [m.replace(':latest', '') for m in model_names]
        target = args.model.replace(':latest', '')
        
        if target not in normalized_names and args.model not in model_names:
            print(f"Model '{args.model}' not found.")
            print(f"Available models: {model_names}")
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
        output_path = Path(f"benchmark_tools_B_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        # output_path = Path(f"benchmark_{args.model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_results(results, args.model, output_path)


if __name__ == "__main__":
    main()