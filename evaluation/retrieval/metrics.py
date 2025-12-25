"""Retrieval evaluation metrics

Relevant IR metrics for RAG quality:
- Recall@K
- Precision@K
- MRR: Mean Reciprocal Rank
- F1@K
Candidate metric
- NDCG@K (not relevant for use case)
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class RetrievalMetrics:
    """Container for all retrieval metrics."""
    recall_at_k: float
    precision_at_k: float
    mrr: float
    f1_at_k: float
    k: int
    
    # Optional: per-result details
    retrieved: list[str] = field(default_factory=list)
    expected: list[str] = field(default_factory=list)
    hits: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "f1_at_k": self.f1_at_k,
            "k": self.k,
            "num_retrieved": len(self.retrieved),
            "num_expected": len(self.expected),
            "num_hits": len(self.hits),
        }


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """
    Calculate Recall@K
    What fraction of relevant (expected) items appear in the top-k results?

    Args:
        retrieved: List of retrieved chunk IDs (in order)
        expected: List of relevant chunk IDs (ground truth)
        k: Number of results to consider
        
    Returns:
        Float between 0 and 1
    """
    if not expected:
        return 1.0  
    
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)
    
    hits = len(retrieved_k & expected_set)
    return hits / len(expected_set)


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """
    Calculate Precision@K
    What fraction of top-k results are relevant?
    
    Args:
        retrieved: List of retrieved chunk IDs (in order)
        expected: List of relevant chunk IDs (ground truth)
        k: Number of results to consider
        
    Returns:
        Float between 0 and 1
    """
    retrieved_k = retrieved[:k]
    
    if not retrieved_k:
        return 0.0
    
    expected_set = set(expected)
    hits = sum(1 for r in retrieved_k if r in expected_set)
    
    return hits / len(retrieved_k)


def mrr(retrieved: list[str], expected: list[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    1 / (position of first relevant result)
    
    Args:
        retrieved: List of retrieved chunk IDs (in order)
        expected: List of relevant chunk IDs (ground truth)
        
    Returns:
        Float between 0 and 1 (1 = first result was relevant)
    """
    expected_set = set(expected)
    
    for i, item in enumerate(retrieved):
        if item in expected_set:
            return 1.0 / (i + 1)
    
    return 0.0

def f1_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """
    Calculate F1 score at K.
    Harmonic mean of precision and recall.
    
    Args:
        retrieved: List of retrieved chunk IDs (in order)
        expected: List of relevant chunk IDs (ground truth)
        k: Number of results to consider
        
    Returns:
        Float between 0 and 1
    """
    p = precision_at_k(retrieved, expected, k)
    r = recall_at_k(retrieved, expected, k)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)


def evaluate_single_query(
    retrieved: list[str], 
    expected: list[str], 
    k: int = 5
) -> RetrievalMetrics:
    """
    Evaluate a single query with all metrics.

    Args:
        retrieved: List of retrieved chunk IDs (in order)
        expected: List of relevant chunk IDs (ground truth)
        k: Number of results to consider
        
    Returns:
        RetrievalMetrics object with all scores
    """
    expected_set = set(expected)
    hits = [r for r in retrieved[:k] if r in expected_set]
    
    return RetrievalMetrics(
        recall_at_k=recall_at_k(retrieved, expected, k),
        precision_at_k=precision_at_k(retrieved, expected, k),
        mrr=mrr(retrieved, expected),
        f1_at_k=f1_at_k(retrieved, expected, k),
        k=k,
        retrieved=retrieved[:k],
        expected=expected,
        hits=hits,
    )


def aggregate_metrics(metrics_list: list[RetrievalMetrics]) -> dict:
    """
    Aggregate metrics across multiple queries.
    
    Args:
        metrics_list: List of RetrievalMetrics from individual queries
        
    Returns:
        Dict with averaged metrics
    """
    if not metrics_list:
        return {}
    
    n = len(metrics_list)
    
    return {
        "avg_recall_at_k": sum(m.recall_at_k for m in metrics_list) / n,
        "avg_precision_at_k": sum(m.precision_at_k for m in metrics_list) / n,
        "avg_mrr": sum(m.mrr for m in metrics_list) / n,
        "avg_f1_at_k": sum(m.f1_at_k for m in metrics_list) / n,
        "num_queries": n,
        "k": metrics_list[0].k,
    }