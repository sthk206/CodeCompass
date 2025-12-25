# Validate if all retrieval queries exist in repo index
import sys
from pathlib import Path
import pyarrow as pa

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codecompass.indexing.store import CodeStore
from evaluation.retrieval.test_cases import RETRIEVAL_QUERIES

# Load the index
store = CodeStore(Path("."))

# Convert LanceTable to a list of dicts using Arrow
arrow_table = store.table.to_arrow()
results = arrow_table.to_pylist()  

# Collect all indexed IDs
actual_ids = set(r["id"] for r in results)
print(f"Total chunks indexed: {len(actual_ids)}\n")

# Check for missing IDs
missing = []
for q in RETRIEVAL_QUERIES:
    for expected_id in q.expected:
        if expected_id not in actual_ids:
            missing.append((q.query, expected_id))

if missing:
    print("❌ MISSING CHUNK IDs:")
    for query, chunk_id in missing:
        print(f"  Query: {query}")
        print(f"  Missing: {chunk_id}\n")
else:
    print("✅ All expected chunk IDs exist in index!")
