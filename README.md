# CodeCompass
AI-Powered Repository Onboarding Assistant


## Retrieval Strategy Evaluation

Evaluated retrieval strategies on 19 test queries across 5 categories (feature search, concept search, API search, debug search, natural language).

### Results

| Strategy | Recall@5 | Precision@5 | MRR |
|----------|----------|-------------|-----|
| **HyDE** | **0.733** | 0.295 | 0.671 |
| Query Expansion | 0.718 | 0.295 | 0.737 |
| Baseline (vector search) | 0.644 | 0.263 | 0.754 |
| Query Expansion + Context | 0.428 | 0.189 | 0.399 |

**HyDE** (Hypothetical Document Embedding) performed best, improving recall by **13.9%** over baseline. This approach generates hypothetical code matching the query, then searches for similar real code.

### Context-Aware Expansion: Negative Result

The context-aware strategy (providing repo imports to the LLM) performed surprisingly poorly. I hypothesized the prompt was suboptimal, so I tested 4 variations:

| Variant | Description | Recall@5 |
|---------|-------------|----------|
| v4: Rule-based | No LLM, keyword matching | 0.691 |
| v2: Pick from list | LLM selects relevant imports | 0.665 |
| v3: Minimal prompt | Simple prompt | 0.644 |
| v1: Fewer imports | 15 imports instead of 100 | 0.600 |
| Original | 100 imports, verbose prompt | 0.428 |

**Finding**: Even the best context variant (rule-based, no LLM) underperformed simple query expansion. Adding repository context introduces noise rather than helping retrieval.

**Takeaway**: Simpler strategies (HyDE, query expansion) outperform complex context-aware approaches for code search.