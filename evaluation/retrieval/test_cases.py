from enum import Enum
from dataclasses import dataclass

class QueryCategory(Enum):
    FEATURE_SEARCH = "feature_search"      # "command line functions"
    CONCEPT_SEARCH = "concept_search"      # "how embeddings work"
    API_SEARCH = "api_search"              # "function to search code"
    DEBUG_SEARCH = "debug_search"          # "where is X defined"
    NATURAL_LANGUAGE = "natural_language"  # "I want to parse Python files"


@dataclass
class RetrievalQuery:
    query: str
    expected: list[str]
    category: QueryCategory
    description: str

RETRIEVAL_QUERIES = [
    # FEATURE SEARCH: Looking for specific functionality
    RetrievalQuery(
        query="command line functions",
        expected=[
            "src/codecompass/cli.py::hello",
            "src/codecompass/cli.py::index",
            "src/codecompass/cli.py::search",
            "src/codecompass/cli.py::status",
            "src/codecompass/cli.py::chat",
        ],
        category=QueryCategory.FEATURE_SEARCH,
        description="Should find all CLI commands defined with @app.command()",
    ),
    
    RetrievalQuery(
        query="index a repository",
        expected=[
            "src/codecompass/cli.py::index",
            "src/codecompass/indexing/store.py::index_repository",
            "src/codecompass/indexing/store.py::CodeStore.index_chunks",
        ],
        category=QueryCategory.FEATURE_SEARCH,
        description="Should find indexing-related functions",
    ),
    
    RetrievalQuery(
        query="search code in repository",
        expected=[
            "src/codecompass/cli.py::search",
            "src/codecompass/retrieval/search.py::search_code",
            "src/codecompass/indexing/store.py::CodeStore.search",
        ],
        category=QueryCategory.FEATURE_SEARCH,
        description="Should find search-related functions",
    ),
    
    RetrievalQuery(
        query="interactive chat with codebase",
        expected=[
            "src/codecompass/cli.py::chat",
            "src/codecompass/retrieval/rag.py::answer_question",
        ],
        category=QueryCategory.FEATURE_SEARCH,
        description="Should find chat/conversation functions",
    ),

    # CONCEPT SEARCH: Looking for how something is implemented
    RetrievalQuery(
        query="how are embeddings generated",
        expected=[
            "src/codecompass/llm/ollama.py::embed",
            "src/codecompass/llm/ollama.py::embed_batch",
        ],
        category=QueryCategory.CONCEPT_SEARCH,
        description="Should find embedding generation code",
    ),
    
    RetrievalQuery(
        query="parse Python abstract syntax tree",
        expected=[
            "src/codecompass/indexing/chunker.py::PythonChunker",
            "src/codecompass/indexing/chunker.py::PythonChunker._extract_chunks",
            "src/codecompass/indexing/chunker.py::PythonChunker.chunk_file",
        ],
        category=QueryCategory.CONCEPT_SEARCH,
        description="Should find AST parsing code",
    ),
    
    RetrievalQuery(
        query="vector database storage",
        expected=[
            "src/codecompass/indexing/store.py::CodeStore",
            "src/codecompass/indexing/store.py::CodeStore.search",
            "src/codecompass/indexing/store.py::CodeStore.index_chunks",
        ],
        category=QueryCategory.CONCEPT_SEARCH,
        description="Should find vector store implementation",
    ),
    
    RetrievalQuery(
        query="RAG retrieval augmented generation",
        expected=[
            "src/codecompass/retrieval/rag.py::answer_question",
            "src/codecompass/retrieval/search.py::search_code",
        ],
        category=QueryCategory.CONCEPT_SEARCH,
        description="Should find RAG pipeline code",
    ),
    
    RetrievalQuery(
        query="extract functions from source code",
        expected=[
            "src/codecompass/indexing/chunker.py::PythonChunker._extract_chunks",
            "src/codecompass/indexing/chunker.py::PythonChunker._make_chunk",
            "src/codecompass/indexing/chunker.py::chunk_repository",
        ],
        category=QueryCategory.CONCEPT_SEARCH,
        description="Should find code extraction logic",
    ),

    # API SEARCH: Looking for functions/classes to use
    RetrievalQuery(
        query="CodeStore class",
        expected=[
            "src/codecompass/indexing/store.py::CodeStore",
        ],
        category=QueryCategory.API_SEARCH,
        description="Direct class name search",
    ),
    
    RetrievalQuery(
        query="PythonChunker",
        expected=[
            "src/codecompass/indexing/chunker.py::PythonChunker",
        ],
        category=QueryCategory.API_SEARCH,
        description="Direct class name search",
    ),
    
    RetrievalQuery(
        query="generate LLM response",
        expected=[
            "src/codecompass/llm/ollama.py::generate",
        ],
        category=QueryCategory.API_SEARCH,
        description="Should find the generate function",
    ),
    
    RetrievalQuery(
        query="CodeChunk dataclass",
        expected=[
            "src/codecompass/indexing/chunker.py::CodeChunk",
        ],
        category=QueryCategory.API_SEARCH,
        description="Should find the data model",
    ),

    # DEBUG SEARCH: Where is something defined/used
    RetrievalQuery(
        query="where is search_text created",
        expected=[
            "src/codecompass/indexing/store.py::CodeStore._create_search_text",
        ],
        category=QueryCategory.DEBUG_SEARCH,
        description="Should find where search_text field is populated",
    ),
    
    RetrievalQuery(
        query="docstring extraction",
        expected=[
            "src/codecompass/indexing/chunker.py::PythonChunker._extract_docstring",
        ],
        category=QueryCategory.DEBUG_SEARCH,
        description="Should find docstring handling code",
    ),
    
    RetrievalQuery(
        query="configuration settings",
        expected=[
            "src/codecompass/config.py::Settings",
        ],
        category=QueryCategory.DEBUG_SEARCH,
        description="Should find config/settings",
    ),

    # NATURAL LANGUAGE: Casual/conversational queries
    RetrievalQuery(
        query="I want to parse Python files",
        expected=[
            "src/codecompass/indexing/chunker.py::PythonChunker",
            "src/codecompass/indexing/chunker.py::PythonChunker.chunk_file",
            "src/codecompass/indexing/chunker.py::chunk_repository",
        ],
        category=QueryCategory.NATURAL_LANGUAGE,
        description="Natural language query about parsing",
    ),
    
    RetrievalQuery(
        query="how do I ask questions about my code",
        expected=[
            "src/codecompass/cli.py::chat",
            "src/codecompass/retrieval/rag.py::answer_question",
        ],
        category=QueryCategory.NATURAL_LANGUAGE,
        description="Natural language query about usage",
    ),
    
    RetrievalQuery(
        query="save code chunks to database",
        expected=[
            "src/codecompass/indexing/store.py::CodeStore.index_chunks",
            "src/codecompass/indexing/store.py::index_repository",
        ],
        category=QueryCategory.NATURAL_LANGUAGE,
        description="Natural language about storage",
    ),
]
