"""
Database connection management for MSSQL Server.
"""
import pyodbc
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    server: str
    database: str = "fussballDB"
    driver: str = "{ODBC Driver 17 for SQL Server}"
    trusted_connection: bool = True
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    
    @property
    def connection_string(self) -> str:
        """Build connection string."""
        if self.trusted_connection:
            return (
                f"DRIVER={self.driver};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
                f"Connection Timeout={self.timeout};"
            )
        else:
            return (
                f"DRIVER={self.driver};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"Connection Timeout={self.timeout};"
            )


class DatabaseConnection:
    """
    Manages database connections with connection pooling support.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection: Optional[pyodbc.Connection] = None
    
    def connect(self) -> pyodbc.Connection:
        """Establish database connection."""
        if self._connection is None or self._connection.closed:
            try:
                self._connection = pyodbc.connect(
                    self.config.connection_string,
                    autocommit=False
                )
                logger.info(f"Connected to {self.config.database}")
            except pyodbc.Error as e:
                logger.error(f"Database connection failed: {e}")
                raise
        return self._connection
    
    def close(self):
        """Close database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            logger.info("Database connection closed")
    
    @contextmanager
    def cursor(self) -> Generator[pyodbc.Cursor, None, None]:
        """Context manager for database cursor."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
    
    @contextmanager
    def transaction(self) -> Generator[pyodbc.Connection, None, None]:
        """Context manager for transactions."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed: {e}")
            raise


class QueryExecutor:
    """
    Executes queries and returns results as dictionaries.
    """
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results as list of dicts."""
        with self.db.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            columns = [col[0] for col in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
    
    def execute_query_chunked(
        self,
        query: str,
        params: Optional[tuple] = None,
        chunk_size: int = 10000
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Execute query and yield results in chunks for memory efficiency."""
        with self.db.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            columns = [col[0] for col in cursor.description]
            
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                
                yield [dict(zip(columns, row)) for row in rows]
    
    def execute_scalar(
        self, 
        query: str, 
        params: Optional[tuple] = None
    ) -> Any:
        """Execute query and return single value."""
        with self.db.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            row = cursor.fetchone()
            return row[0] if row else None
    
    def execute_non_query(
        self, 
        query: str, 
        params: Optional[tuple] = None
    ) -> int:
        """Execute INSERT/UPDATE/DELETE and return affected rows."""
        with self.db.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.rowcount
    
    def execute_many(
        self, 
        query: str, 
        params_list: List[tuple]
    ) -> int:
        """Execute query with multiple parameter sets."""
        with self.db.cursor() as cursor:
            cursor.fast_executemany = True
            cursor.executemany(query, params_list)
            return cursor.rowcount


# =============================================================================
# SINGLETON CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """
    Singleton manager for database connections.
    """
    _instance: Optional['ConnectionManager'] = None
    _db: Optional[DatabaseConnection] = None
    _executor: Optional[QueryExecutor] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, config: DatabaseConfig):
        """Initialize the connection manager with config."""
        self._db = DatabaseConnection(config)
        self._executor = QueryExecutor(self._db)
    
    @property
    def db(self) -> DatabaseConnection:
        if self._db is None:
            raise RuntimeError("ConnectionManager not initialized. Call initialize() first.")
        return self._db
    
    @property
    def executor(self) -> QueryExecutor:
        if self._executor is None:
            raise RuntimeError("ConnectionManager not initialized. Call initialize() first.")
        return self._executor
    
    def close(self):
        """Close all connections."""
        if self._db:
            self._db.close()


# Convenience function
def get_connection_manager() -> ConnectionManager:
    """Get the singleton connection manager."""
    return ConnectionManager()