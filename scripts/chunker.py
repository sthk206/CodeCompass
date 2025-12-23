from pathlib import Path
from pprint import pprint

# Import your PythonChunker class
from codecompass.indexing.chunker import PythonChunker  # Replace with your actual file/module
from tree_sitter import Parser, Language
import tree_sitter_python as tspython

def test_chunk_file():
    repo_root = Path(".")  # Current directory as repo root
    file_path = Path("src/codecompass/cli.py")  # The sample file


    chunker = PythonChunker()
    chunks = list(chunker.chunk_file(file_path, repo_root))
    

    gen = chunker.chunk_file(file_path, repo_root)  # only create generator once

    pprint(next(gen))  # first chunk
    pprint("~~~~~~~")
    pprint(next(gen))  # second chunk
    pprint("~~~~~~~")
    pprint(next(gen))  # third chunk
    pprint("~~~~~~~")
    pprint(next(gen))  # fourth chunk
    pprint("~~~~~~~")
    pprint(next(gen))  # fifth chunk
    pprint("~~~~~~~")
    pprint(next(gen))  # sixth chunk
    pprint("~~~~~~~")
    

def serialize_node(node, source_bytes):
    return {
        "type": node.type,
        "is_named": node.is_named,
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "start_point": {
            "row": node.start_point.row,
            "column": node.start_point.column,
        },
        "end_point": {
            "row": node.end_point.row,
            "column": node.end_point.column,
        },
        "text": source_bytes[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        ),
        "children": [
            {
                "field": node.field_name_for_child(i),
                "node": serialize_node(child, source_bytes),
            }
            for i, child in enumerate(node.children)
        ],
    }

def make_ast_json_from_file(file_path: Path):
    import json

    language = Language(tspython.language())
    parser = Parser(language)

    file = file_path.read_bytes()
    tree = parser.parse(file)
    ast_dict = serialize_node(tree.root_node, file)
    with open(f'ast_jsons/{str(file_path).replace("/", ":")}.json', 'w', encoding='utf-8') as f:
        json.dump(ast_dict, f, indent=2)


if __name__ == "__main__":
    # test_chunk_file()
    make_ast_json_from_file(Path("src/codecompass/cli.py"))
