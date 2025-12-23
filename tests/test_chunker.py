# tests/test_chunker.py
import tempfile
from pathlib import Path
from codecompass.indexing.chunker import PythonChunker, chunk_repository


def test_chunk_simple_function():
    """Test chunking a simple function."""
    chunker = PythonChunker()
    
    code = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
    
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        
        file_path = Path(f.name)
        chunks = list(chunker.chunk_file(file_path, file_path.parent))
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.name == "hello"
        assert chunk.chunk_type == "function"
        assert chunk.docstring == "Say hello to someone."
        print(f"✅ Function chunk: {chunk.id}")


def test_chunk_class_with_methods():
    """Test chunking a class with methods."""
    chunker = PythonChunker()
    
    code = '''
class Calculator:
    """A simple calculator."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
'''
    
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        
        file_path = Path(f.name)
        chunks = list(chunker.chunk_file(file_path, file_path.parent))
        
        # Should have: 1 class + 2 methods = 3 chunks
        assert len(chunks) == 3
        
        class_chunk = next(c for c in chunks if c.chunk_type == "class")
        assert class_chunk.name == "Calculator"
        
        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        assert len(method_chunks) == 2
        assert all(c.parent_class == "Calculator" for c in method_chunks)
        
        print(f"✅ Class chunk: {class_chunk.id}")
        for m in method_chunks:
            print(f"✅ Method chunk: {m.id}")


if __name__ == "__main__":
    test_chunk_simple_function()
    test_chunk_class_with_methods()
    print("\nAll chunker tests passed!")