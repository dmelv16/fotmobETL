"""
One-time initialization script.
Run this before first use to set up all database tables.
"""

import logging
from config.database import DatabaseConfig
from core.database_manager import StrengthDatabaseManager
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

def initialize_system(connection_string: str):
    """
    Initialize the strength calculation system.
    Creates all necessary database tables.
    """
    setup_logging()
    
    logger.info("Starting system initialization...")
    
    # Test connection
    db_config = DatabaseConfig(connection_string)
    
    if not db_config.test_connection():
        logger.error("Failed to connect to database. Check your connection string.")
        return False
    
    logger.info("Database connection successful")
    
    # Create tables
    conn = db_config.get_connection()
    db_manager = StrengthDatabaseManager(conn)
    
    try:
        db_manager.create_all_tables()
        logger.info("All database tables created successfully")
        
        # Verify tables exist
        cursor = conn.cursor()
        tables_to_check = [
            'league_strength',
            'team_strength',
            'transfer_performance_analysis',
            'european_competition_results',
            'team_elo_history',
            'league_network_edges',
            'style_profiles'
        ]
        
        logger.info("\nVerifying tables:")
        for table_name in tables_to_check:
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = '{table_name}'
            """)
            exists = cursor.fetchone()[0]
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {table_name}")
        
        conn.close()
        
        logger.info("\nSystem initialization complete!")
        logger.info("You can now run: python main.py process-season <year>")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        conn.close()
        return False


if __name__ == "__main__":
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=fussballDB;"
        "Trusted_Connection=yes;"
    )
    
    success = initialize_system(connection_string)
    
    if not success:
        exit(1)