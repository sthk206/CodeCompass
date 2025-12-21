from codecompass.llm.ollama import generate, embed

def test_generate_hello():
    response = generate("Say hello")
    assert "hello" in response.lower()

def test_embed_length():
    vec = embed("Hello world")
    assert len(vec) == 768