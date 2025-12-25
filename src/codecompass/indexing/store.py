"""Vector store for code chunks using LanceDB"""

from pathlib import Path
from typing import Optional
import hashlib
import json

import lancedb
from pydantic import BaseModel

from codecompass.config import settings
from codecompass.indexing.chunker import CodeChunk
from codecompass.llm.ollama import embed

class StoredChunk(BaseModel):
    """LanceDB Schema for storing code chunks"""
    id: str
    file_path: str
    name: str
    chunk_type: str
    code: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    parent_class: Optional[str]
    # Searchable text (code / docstring / name)
    search_text: str
    vector: list[float]

class CodeStore:
    """Manages vector store for repo"""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()
        self.db_path = self._get_db_path()
        self.db = lancedb.connect(str(self.db_path))
        self.table_name = "code_chunks"
        self._table = None

    def _get_db_path(self) -> Path:
        """Generate unique database path for current repository."""
        path_hash = hashlib.sha256(str(self.repo_path).encode()).hexdigest()[:8]
        repo_name = self.repo_path.name
        return settings.data_dir / "indices" / f"{repo_name}-{path_hash}"
    
    @property
    def table(self):
        """Get or create the chunks table"""
        if self._table is None:
            if self.table_name in self.db.table_names():
                self._table = self.db.open_table(self.table_name)
            else:
                self._table = None
        return self._table

    def wait_for_index(self, index_name, timeout: float = 5.0):
        import time
        while True:
            indices = self.table.list_indices()

            if indices and any(index.name == index_name for index in indices):
                break
            print(f"⏳ Waiting for {index_name} to be ready...")
            time.sleep(timeout)

        print(f"✅ {index_name} is ready!")


    def index_chunks(self, chunks: list[CodeChunk], show_progress: bool = True) -> int:
        """Index code chunks into vector store"""
        if not chunks:
            return 0
        
        records = []

        from rich.progress import Progress
        with Progress() as progress:
            task = progress.add_task("Embedding chunks...", total=len(chunks))

            for chunk in chunks:
                search_text = self._create_search_text(chunk)
                
                vector = embed(search_text)

                records.append({
                    "id": chunk.id,
                    "file_path": chunk.file_path,
                    "name": chunk.name,
                    "chunk_type": chunk.chunk_type,
                    "code": chunk.code,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "docstring": chunk.docstring or "",
                    "parent_class": chunk.parent_class or "",
                    "search_text": search_text,
                    "vector": vector,
                })

                progress.update(task, advance=1)
                
        self._table = self.db.create_table(
            self.table_name,
            records,
            mode="overwrite"
        )

        # For BM25 hybrid search
        self.table.create_fts_index("search_text")
        # Since create_fts_index is async, we wait to utilize hybrid search
        self.wait_for_index("search_text_idx")



        # Collect all unique imports from chunks
        all_imports = set()
        for chunk in chunks:
            if chunk.imports:
                for imp in chunk.imports:
                    # Extract module name: "from typer import ..." → "typer"
                    if imp.startswith("from "):
                        module = imp.split()[1].split(".")[0]
                    elif imp.startswith("import "):
                        module = imp.split()[1].split(".")[0].split(",")[0]
                    all_imports.add(module)
    
        self._save_metadata(len(records), list(all_imports))

        return len(records)

    def _create_search_text(self, chunk: CodeChunk) -> str:
        """Create searchable text from chunk"""
        parts = [
            f"Name: {chunk.name}",
            f"Type: {chunk.chunk_type}",
        ]

        if chunk.parent_class:
            parts.append(f"Class: {chunk.parent_class}")
        
        if chunk.docstring:
            parts.append(f"Description: {chunk.docstring}")

        if chunk.imports:
            parts.append(f"File imports: {', '.join(chunk.imports)}")
        
        parts.append(f"Code:\n{chunk.code}")
        
        return "\n".join(parts)
    

    def _save_metadata(self, chunk_count: int, imports: list[str] = None):
        """Save index metadata"""
        from datetime import datetime

        metadata = {
            "repo_path": str(self.repo_path),
            "indexed_at": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "imports": imports or []
        }

        metadata_path = self.db_path / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2))

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search for code chunks matching a query"""
        if self.table is None:
            return []
        
        query_vector = embed(query)
        # hybrid search
        results = (
            self.table
            .search(query_type="hybrid", fts_columns="search_text")
            .vector(query_vector)
            .text(query)
            .limit(limit)
            .to_list()
        )
        # vector search
        # results = (
        #     self.table
        #     .search(query_vector)
        #     .limit(limit)
        #     .to_list()
        # )

        return results

    def get_stats(self) -> dict:
        """Get index stats"""
        metadata_path = self.db_path / "metadata.json"

        if not metadata_path.exists():
            return {"status": "not_indexed"}
        metadata = json.loads(metadata_path.read_text())
        return {
            "status": "indexed",
            **metadata
        }

    def is_indexed(self) -> bool:
        """Check if the repository has been indexed"""
        return self.table is not None
    
def index_repository(repo_path: Path) -> int:
    """Index a repo and return number of chunks"""
    from codecompass.indexing.chunker import chunk_repository

    print(f"Indexing repository: {repo_path}")

    chunks = list(chunk_repository(repo_path))
    print(f"Found {len(chunks)} code chunks")

    if not chunks:
        print("No chunks found. Please check this is a Python repository.")
        return 0
    
    store = CodeStore(repo_path)
    count = store.index_chunks(chunks)

    print(f"✅ Indexed {count} chunks")
    return count
