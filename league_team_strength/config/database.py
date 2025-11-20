import pyodbc
from typing import Optional

class DatabaseConfig:
    """Database connection configuration."""
    
    def __init__(self, connection_string: str = None):
        if connection_string:
            self.connection_string = connection_string
        else:
            # Default connection string - update as needed
            self.connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                "SERVER=DESKTOP-J9IV3OH;"
                "DATABASE=fussballDB;"
                "Trusted_Connection=yes;"
            )
    
    def get_connection(self):
        """Get database connection."""
        return pyodbc.connect(self.connection_string)
    
    def test_connection(self) -> bool:
        """Test if database connection works."""
        try:
            conn = self.get_connection()
            conn.close()
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False