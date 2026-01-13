import os
import sys
from models import engine, Base, init_db

def reset_database():
    db_path = "mcp_deepwiki.db"
    
    # Check if database exists
    if os.path.exists(db_path):
        print(f"Deleting existing database: {db_path}")
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"Error deleting database: {e}")
            sys.exit(1)
    
    print("Initializing new database...")
    init_db()
    print("Database reset successfully.")

if __name__ == "__main__":
    confirm = input("This will delete all processing history. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        reset_database()
    else:
        print("Reset aborted.")
