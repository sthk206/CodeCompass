from pathlib import Path
from typing import Callable
import json
from datetime import datetime

from rich.console import Console
from rich.table import Table

from dataclasses import asdict

from evaluation.retrieval.metrics import evaluate_single_query, aggregate_metrics
from evaluation.retrieval.test_cases import RETRIEVAL_QUERIES
from codecompass.retrieval.search import (
        baseline_search, hyde_search, 
        query_expansion_search, query_expansion_search_context, SearchResult,
        query_expansion_context_v2, query_expansion_context_v3
    )

console = Console()

STRATEGIES = [
    ("Baseline", baseline_search),
    ("Query Expansion", query_expansion_search),
    ("Query Expansion with Context", query_expansion_search_context),
    ("HyDE", hyde_search),
    # ("Context v2", query_expansion_context_v2),
    # ("Context v3", query_expansion_context_v3)
]


def eval_strategy(repo_path: Path, search_fn: Callable, limit: int = 5):
        all_metrics = []
        per_query = []
        for q in RETRIEVAL_QUERIES:
            retrieved = search_fn(repo_path, q.query, limit) 
            if retrieved and isinstance(retrieved[0], SearchResult):
                retrieved = [r.id for r in retrieved]
            metrics = evaluate_single_query(retrieved, q.expected, k=limit)
            all_metrics.append(metrics)
            per_query.append({
                  "query": q.query,
                  **asdict(metrics)
            })
        agg =  aggregate_metrics(all_metrics)
        return {
              "per_query": per_query,
              **agg
        }
        
def eval_all(repo_path: Path, limit: int = 5) -> dict:
    results = {}
    
    for name, fn in STRATEGIES:
        console.print(f"\n[cyan]Evaluating: {name}[/cyan]")
        results[name] = eval_strategy(repo_path, fn, limit)
        
        # Quick summary
        r = results[name]
        console.print(f"  Recall@{limit}: {r['avg_recall_at_k']:.3f}")
        console.print(f"  Precision@{limit}: {r['avg_precision_at_k']:.3f}")
        console.print(f"  MRR: {r['avg_mrr']:.3f}")
    
    return results

def print_summary(results: dict, k: int):
    """Print comparison table."""
    table = Table(title=f"\nRetrieval Strategy Comparison (K={k})")
    
    table.add_column("Strategy", style="cyan")
    table.add_column("Recall@K", justify="right")
    table.add_column("Precision@K", justify="right")
    table.add_column("MRR", justify="right")
    
    # Sort by recall
    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_recall_at_k"], reverse=True)
    
    for i, (name, r) in enumerate(sorted_results):
        style = "bold green" if i == 0 else ""
        table.add_row(
            name,
            f"{r['avg_recall_at_k']:.3f}",
            f"{r['avg_precision_at_k']:.3f}",
            f"{r['avg_mrr']:.3f}",
            style=style,
        )
    
    console.print(table)
    
    # Print improvement over baseline
    baseline = results.get("Baseline", {})
    best_name, best = sorted_results[0]
    
    if baseline and best_name != "Baseline":
        baseline_recall = baseline.get("avg_recall_at_k", 0.001)
        improvement = (best["avg_recall_at_k"] - baseline_recall) / baseline_recall * 100
        console.print(f"\n[bold]Best: {best_name}[/bold]")
        console.print(f"[green]+{improvement:.1f}% recall over baseline[/green]")


def main():
    """CLI entry point."""
    import sys
    
    repo_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    repo_path = repo_path.resolve()
    k = 5
    
    console.print(f"\n[bold]Evaluating Retrieval Strategies[/bold]")
    console.print(f"Repository: {repo_path}")
    console.print(f"Test queries: {len(RETRIEVAL_QUERIES)}")
    console.print(f"K: {k}")
    
    # Run evaluation
    results = eval_all(repo_path, limit=k)
    
    # Print summary
    print_summary(results, k)
    
    # Save results
    output_dir = Path("evaluation/retrieval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "repo_path": str(repo_path),
            "k": k,
            "results": results,
        }, f, indent=2)
    
    console.print(f"\n[dim]Results saved to: {output_file}[/dim]")


if __name__ == "__main__":
    main()
