import json
import pyodbc
from typing import Dict, Any

def parse_momentum_data(raw_json: str) -> Dict[str, Any]:
    """Parse match momentum data from raw JSON string."""
    data = json.loads(raw_json)
    
    # Navigate to momentum data
    if 'api_responses' in data:
        content = data.get('api_responses', {}).get('match_details', {}).get('data', {}).get('content', {})
    else:
        content = data.get('match_details', {}).get('data', {}).get('content', {})
    
    momentum = content.get('momentum', {})
    
    if not momentum:
        return {'match_id': data.get('match_id'), 'momentum_data': []}
    
    match_id = data.get('match_id')
    momentum_data = []
    
    # Get main momentum data
    main_momentum = momentum.get('main', {})
    debug_title = main_momentum.get('debugTitle')
    
    for point in main_momentum.get('data', []):
        momentum_data.append({
            'match_id': match_id,
            'minute': point.get('minute'),
            'value': point.get('value'),
            'debug_title': debug_title
        })
    
    return {
        'match_id': match_id,
        'momentum_data': momentum_data
    }

def create_momentum_table(connection):
    """Create match momentum table."""
    cursor = connection.cursor()
    
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_momentum]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_momentum] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [minute] FLOAT NOT NULL,
                [momentum_value] INT,
                [debug_title] NVARCHAR(200),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    connection.commit()

def process_momentum(connection_string: str, batch_size: int = 100):
    """Process match momentum from fotmob_raw_data table."""
    conn = pyodbc.connect(connection_string)
    
    # Create table if it doesn't exist
    create_momentum_table(conn)
    
    cursor = conn.cursor()
    
    # Get existing match_ids to avoid duplicates
    cursor.execute("SELECT DISTINCT match_id FROM [dbo].[match_momentum]")
    existing_matches = set(row[0] for row in cursor.fetchall())
    
    # Fetch raw data
    query = """
        SELECT id, match_id, raw_json
        FROM [dbo].[fotmob_raw_data]
        WHERE data_type = 'match_details'
        ORDER BY id
    """
    
    cursor.execute(query)
    
    momentum_batch = []
    
    for row in cursor.fetchall():
        source_id, match_id, raw_json = row
        
        if match_id in existing_matches:
            continue
        
        try:
            parsed = parse_momentum_data(raw_json)
            
            if not parsed['momentum_data']:
                continue
            
            for point in parsed['momentum_data']:
                momentum_batch.append((
                    point['match_id'],
                    point['minute'],
                    point['value'],
                    point['debug_title']
                ))
            
            # Insert in batches
            if len(momentum_batch) >= batch_size * 90:
                insert_momentum_batch(conn, momentum_batch)
                momentum_batch = []
                
        except Exception as e:
            print(f"Error processing match_id {match_id}: {e}")
            continue
    
    # Insert remaining records
    if momentum_batch:
        insert_momentum_batch(conn, momentum_batch)
    
    conn.close()

def insert_momentum_batch(conn, momentum_batch):
    """Insert batch of momentum records into database."""
    cursor = conn.cursor()
    
    if momentum_batch:
        query = """
            INSERT INTO [dbo].[match_momentum] (
                match_id, minute, momentum_value, debug_title
            ) VALUES (?, ?, ?, ?)
        """
        cursor.executemany(query, momentum_batch)
    
    conn.commit()
    print(f"Inserted {len(momentum_batch)} momentum points")

if __name__ == "__main__":
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server;"
        "DATABASE=fussballDB;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    process_momentum(connection_string)