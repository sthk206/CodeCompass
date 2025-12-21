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
