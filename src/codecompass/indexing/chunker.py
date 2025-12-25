from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import json


@dataclass
class CodeChunk:
    """chunk of code extracted from a file"""
    id: str
    file_path: str
    name: str
    chunk_type: str
    code: str
    start_line: int
    end_line: int
    docstring: str | None
    parent_class: str | None
    imports: list[str] | None = None

class PythonChunker:
    """Extract code from Python files with AST parsing"""
    def __init__(self):
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

    def chunk_file(self, file_path: Path, repo_root: Path) -> Iterator[CodeChunk]:
        """Parse a Python file and yield code chunks"""
        try:
            # content = file_path.read_text(encoding="utf-8")
            content_bytes = file_path.read_bytes()
        except UnicodeDecodeError:
            return # skip binary files
        
        tree = self.parser.parse(content_bytes)
        relative_path = str(file_path.relative_to(repo_root))
        imports = self._extract_imports(tree.root_node, content_bytes)
        for chunk in self._extract_chunks(tree.root_node, content_bytes, relative_path):
            chunk.imports = imports  # attach imports to each chunk
            yield chunk
        # yield from self._extract_chunks(tree.root_node, content_bytes, relative_path)

    def _extract_imports(self, node, source: bytes) -> list[str]:
        """Extract all imports from file"""
        imports = []
        for child in node.children:
            if child.type in ("import_statement", "import_from_statement"):
                imports.append(source[child.start_byte:child.end_byte].decode('utf-8'))
        return imports

    def _extract_chunks(self, node, source: bytes, file_path: str, parent_class: str = None):
        """Recurisvely extract chunks from AST nodes"""

        for child in node.children:

            # decorated functions/classes (eg - @app.command())
            if child.type == "decorated_definition":
                # Find the actual function/class inside
                for subchild in child.children:
                    if subchild.type == "function_definition":
                        yield self._make_chunk(
                            subchild, source, file_path, "function", parent_class,
                            node_for_code=child  # Use parent node for full code including decorator
                        )
                    elif subchild.type == "class_definition":
                        class_name = self._get_name(subchild)
                        yield self._make_chunk(
                            subchild, source, file_path, "class", None,
                            node_for_code=child
                        )
                        class_body = self._get_child_by_type(subchild, "block")
                        if class_body:
                            yield from self._extract_chunks(
                                class_body, source, file_path, parent_class=class_name
                            )

            # functions
            elif child.type == "function_definition":
                yield self._make_chunk(child, source, file_path, "function", parent_class)

            # classes
            elif child.type == "class_definition":
                class_name = self._get_name(child)
                yield self._make_chunk(child, source, file_path, "class", None)

                class_body = self._get_child_by_type(child, "block")
                if class_body:
                    yield from self._extract_chunks(
                        class_body, source, file_path, parent_class=class_name
                    )

            elif child.type not in ("function_definition", "class_definition"):
                yield from self._extract_chunks(child, source, file_path, parent_class)
    
    def _make_chunk(self, node, source: bytes, file_path: str, chunk_type: str, parent_class: str = None, node_for_code: any = None):
        """Create CodeChunk from AST node"""
        name = self._get_name(node)

        code_node = node_for_code or node
        code = source[code_node.start_byte:code_node.end_byte].decode('utf-8')
        docstring = self._extract_docstring(code_node, source)

        if parent_class:
            chunk_id = f"{file_path}::{parent_class}.{name}"
            chunk_type = "method"
        else:
            chunk_id = f"{file_path}::{name}"

        return CodeChunk(
            id=chunk_id,
            file_path=file_path,
            name=name,
            chunk_type=chunk_type,
            code=code,
            start_line=code_node.start_point[0] + 1,  # 1-indexed
            end_line=code_node.end_point[0] + 1,
            docstring=docstring,
            parent_class=parent_class,
        )

    def _get_name(self, node) -> str:
        """Extract the name from a function or class definition."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return "<unknown>"
    
    def _get_child_by_type(self, node, type_name: str):
        """Find a child node by type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None
    
    def _extract_docstring(self, node, source: bytes) -> str | None:
        """Extract docstring from a function or class."""
        body = self._get_child_by_type(node, "block")
        if not body or not body.children:
            return None
        
        first_stmt = body.children[0]
        if first_stmt.type == "expression_statement":
            expr = first_stmt.children[0] if first_stmt.children else None
            if expr and expr.type == "string":
                docstring = source[expr.start_byte:expr.end_byte].decode('utf-8')
                # Clean up the docstring (remove quotes)
                return docstring.strip('"""').strip("'''").strip()
        
        return None


def chunk_repository(repo_path: Path) -> Iterator[CodeChunk]:
    """Chunk all Python files in a repository."""
    chunker = PythonChunker()
    
    ignore_patterns = [
        "venv", ".venv", "node_modules", "__pycache__", 
        ".git", "build", "dist", ".eggs", "egg-info", "scratch", "scripts", "evaluation"
    ]

    filtered_files = [
        f for f in repo_path.rglob("*.py")
        if not any(pattern in str(f) for pattern in ignore_patterns)
    ]

    print(f"Found {len(filtered_files)} Python files.")
    
    for file_path in filtered_files:
        yield from chunker.chunk_file(file_path, repo_path)
