# tests/test_store.py
import tempfile
from pathlib import Path
from codecompass.indexing.chunker import CodeChunk
from codecompass.indexing.store import CodeStore, index_repository


def test_index_and_search():
    """Test indexing and searching code."""
    # Create temp repository
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create some Python files
        (repo_path / "auth.py").write_text('''
def login(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    # Check credentials against database
    return check_credentials(username, password)

def logout(session_id: str) -> None:
    """Log out a user and invalidate their session."""
    invalidate_session(session_id)
''')
        
        (repo_path / "users.py").write_text('''
class UserService:
    """Service for managing users."""
    
    def create_user(self, email: str, name: str) -> dict:
        """Create a new user account."""
        return {"email": email, "name": name}
    
    def delete_user(self, user_id: int) -> bool:
        """Delete a user by ID."""
        return True
''')
        
        # Index the repository
        count = index_repository(repo_path)
        print(f"Indexed {count} chunks")
        assert count > 0
        
        # Test search
        store = CodeStore(repo_path)
        
        # Search for authentication
        results = store.search("user authentication login", limit=3)
        print(f"\nSearch 'user authentication login':")
        for r in results:
            print(f"  - {r['name']} ({r['chunk_type']}) in {r['file_path']}")
        
        assert any("login" in r["name"] for r in results)
        
        # Search for user management
        results = store.search("create new user account", limit=3)
        print(f"\nSearch 'create new user account':")
        for r in results:
            print(f"  - {r['name']} ({r['chunk_type']}) in {r['file_path']}")
        
        assert any("create_user" in r["name"] or "UserService" in r["name"] for r in results)
        
        print("\n✅ Store tests passed!")


if __name__ == "__main__":
    test_index_and_search()