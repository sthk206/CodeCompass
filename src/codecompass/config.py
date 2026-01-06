from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ollama
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    ollama_ctx_window: int = 16384 # default = 4098; consider decreasing if hardware limitations

    # indexing
    chunk_max_chars: int = 5000 # empirical context for nomic-embed (2048 tokens)
    embedding_dimensions: int = 768


    # paths
    data_dir: Path = Path.home() / ".codecompass"

    class ConfigDict:
        env_prefix = "CODECAMPASS_"
        env_file = ".env"

settings = Settings()

settings.data_dir.mkdir(parents=True, exist_ok=True)    