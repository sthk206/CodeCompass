VERBOSE = False

def test_ollama_connection():
    print("Testing ollama connection...")
    import ollama

    response = ollama.chat(
        model = "qwen2.5:7b",
        messages = [{"role": "user", "content": "Say 'CodeCompass is ready!'"}]
    )
    if VERBOSE:
        print(response)
        print(response["message"]["content"])

    print("✅ Ollma Connection Test Successful.")

def test_tree_sitter():


    print("Testing Tree-sitter...")
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PY_LANG = Language(tspython.language())
    parser = Parser(PY_LANG)

    code = b'''
import ollama
from codecompass.config import settings

def generate(prompt: str, system: str = None) -> str:
    """Generate response with Ollama"""
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model = settings.chat_model,
        messages = messages
    )
    return response["message"]["content"]

def embed(text: str) -> list[float]:
    """Generate embedding with Ollama"""
    response = ollama.embeddings(
        model = settings.embedding_model,
        prompt = text
    )
    return response["embedding"]

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate batch embeddings with Ollama"""
    return [embed(text) for text in texts]

'''
    tree = parser.parse(code)
    root_node = tree.root_node

    def print_tree(node, indent=0):
        print("  " * indent + f"{node.type}: {node.text.decode(errors='ignore')}")
        for child in node.children:
            print_tree(child, indent + 1)


    if VERBOSE:
        print_tree(root_node)
        print(f"Root node type: {root_node.type}")
        print(f"Children: {[child.type for child in tree.root_node.children]}")
        print("\nTop-level definitions:")
        for child in root_node.children:
            if child.type == "function_definition":
                print(f"Function: {child.child_by_field_name('name').text.decode()}")
            elif child.type == "import_statement":
                print(f"Import: {child.text.decode()}")

    print("✅ Tree-sitter Test Successful.")

def test_lancedb():
    print("Testing LanceDB...")
    import lancedb
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db = lancedb.connect(tmpdir)

        data = [
            {"text": "hello", "vector": [0.1] * 768},
            {"text": "world", "vector": [0.2] * 768},
        ]
        table = db.create_table("test", data)

        results = table.search([0.14] * 768).limit(1).to_list()
        
        if VERBOSE:
            print(results)

    print("✅ LanceDB Test Successful.")

if __name__ == "__main__":
    VERBOSE = True
    test_ollama_connection()
    test_tree_sitter()
    test_lancedb()