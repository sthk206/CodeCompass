import ollama
from codecompass.config import settings

def chat(messages: list[dict], temperature: float = 0.0, model: str | None = None) -> str:
    """Generate a response from a conversation using Ollama"""
    response = ollama.chat(
        model=model or settings.chat_model,
        messages=messages,
        options={"temperature": temperature},
        keep_alive="30m",
    )
    return response["message"]["content"]


def generate(prompt: str, system: str = None, temperature: float = 0.0, model: str | None = None) -> str:
    """Generate a response using Ollama"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    return chat(messages, temperature, model)


def embed(text: str) -> list[float]:
    """Generate embedding with Ollama"""
    response = ollama.embeddings(
        model = settings.embedding_model,
        prompt = text,
        keep_alive="30m",
    )
    return response["embedding"]

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate batch embeddings with Ollama"""
    return [embed(text) for text in texts]

