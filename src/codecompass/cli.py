import typer

app = typer.Typer(
    name="codecompass",
    help="AI-powered repository onboarding assistant",
)

@app.command()
def hello():
    """Test command."""
    print("CodeCompass is working!")
    
@app.command()
def hello2():
    """Test command."""
    print("CodeCompass is working 2!")

if __name__ == "__main__":
    app()