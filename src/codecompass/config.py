from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ollama
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"

    # indexing
    chunk_max_tokens: int = 512
    embedding_dimensions: int = 768

    # paths
    data_dir: Path = Path.home() / ".codecompass"

    class ConfigDict:
        env_prefix = "CODECAMPASS_"
        env_file = ".env"

settings = Settings()

settings.data_dir.mkdir(parents=True, exist_ok=True)    