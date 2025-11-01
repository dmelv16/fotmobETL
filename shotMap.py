import json
import pyodbc
from typing import Dict, Any

def parse_shotmap_data(raw_json: str) -> Dict[str, Any]:
    """Parse shotmap data from raw JSON string."""
    data = json.loads(raw_json)
    
    # Navigate to shotmap data
    if 'api_responses' in data:
        content = data.get('api_responses', {}).get('match_details', {}).get('data', {}).get('content', {})
    else:
        content = data.get('match_details', {}).get('data', {}).get('content', {})
    
    shotmap = content.get('shotmap', {})
    
    if not shotmap:
        return {'match_id': data.get('match_id'), 'shots': []}
    
    match_id = data.get('match_id')
    shots_data = []
    
    # Process shots array
    for shot in shotmap.get('shots', []):
        if not shot:  # Skip empty objects
            continue
            
        shots_data.append({
            'match_id': match_id,
            'shot_id': shot.get('id'),
            'event_type': shot.get('eventType'),
            'team_id': shot.get('teamId'),
            'player_id': shot.get('playerId'),
            'player_name': shot.get('playerName'),
            'first_name': shot.get('firstName'),
            'last_name': shot.get('lastName'),
            'full_name': shot.get('fullName'),
            'x': shot.get('x'),
            'y': shot.get('y'),
            'minute': shot.get('min'),
            'minute_added': shot.get('minAdded'),
            'is_blocked': shot.get('isBlocked'),
            'is_on_target': shot.get('isOnTarget'),
            'blocked_x': shot.get('blockedX'),
            'blocked_y': shot.get('blockedY'),
            'goal_crossed_y': shot.get('goalCrossedY'),
            'goal_crossed_z': shot.get('goalCrossedZ'),
            'expected_goals': shot.get('expectedGoals'),
            'expected_goals_on_target': shot.get('expectedGoalsOnTarget'),
            'shot_type': shot.get('shotType'),
            'situation': shot.get('situation'),
            'period': shot.get('period'),
            'is_own_goal': shot.get('isOwnGoal'),
            'is_saved_off_line': shot.get('isSavedOffLine'),
            'is_from_inside_box': shot.get('isFromInsideBox'),
            'team_color': shot.get('teamColor')
        })
    
    return {
        'match_id': match_id,
        'shots': shots_data
    }

def create_shotmap_table(connection):
    """Create shotmap table."""
    cursor = connection.cursor()
    
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_shotmap]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_shotmap] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [shot_id] BIGINT,
                [event_type] NVARCHAR(50),
                [team_id] INT,
                [player_id] INT,
                [player_name] NVARCHAR(200),
                [first_name] NVARCHAR(100),
                [last_name] NVARCHAR(100),
                [full_name] NVARCHAR(200),
                [x] FLOAT,
                [y] FLOAT,
                [minute] INT,
                [minute_added] INT,
                [is_blocked] BIT,
                [is_on_target] BIT,
                [blocked_x] FLOAT,
                [blocked_y] FLOAT,
                [goal_crossed_y] FLOAT,
                [goal_crossed_z] FLOAT,
                [expected_goals] FLOAT,
                [expected_goals_on_target] FLOAT,
                [shot_type] NVARCHAR(50),
                [situation] NVARCHAR(50),
                [period] NVARCHAR(50),
                [is_own_goal] BIT,
                [is_saved_off_line] BIT,
                [is_from_inside_box] BIT,
                [team_color] NVARCHAR(20),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    connection.commit()

def process_shotmap(connection_string: str, batch_size: int = 100):
    """Process shotmap data from fotmob_raw_data table."""
    conn = pyodbc.connect(connection_string)
    
    # Create table if it doesn't exist
    create_shotmap_table(conn)
    
    cursor = conn.cursor()
    
    # Get existing match_ids to avoid duplicates
    cursor.execute("SELECT DISTINCT match_id FROM [dbo].[match_shotmap]")
    existing_matches = set(row[0] for row in cursor.fetchall())
    
    # Fetch raw data
    query = """
        SELECT id, match_id, raw_json
        FROM [dbo].[fotmob_raw_data]
        WHERE data_type = 'match_details'
        ORDER BY id
    """
    
    cursor.execute(query)
    
    shotmap_batch = []
    
    for row in cursor.fetchall():
        source_id, match_id, raw_json = row
        
        if match_id in existing_matches:
            continue
        
        try:
            parsed = parse_shotmap_data(raw_json)
            
            if not parsed['shots']:
                continue
            
            for shot in parsed['shots']:
                shotmap_batch.append((
                    shot['match_id'],
                    shot['shot_id'],
                    shot['event_type'],
                    shot['team_id'],
                    shot['player_id'],
                    shot['player_name'],
                    shot['first_name'],
                    shot['last_name'],
                    shot['full_name'],
                    shot['x'],
                    shot['y'],
                    shot['minute'],
                    shot['minute_added'],
                    shot['is_blocked'],
                    shot['is_on_target'],
                    shot['blocked_x'],
                    shot['blocked_y'],
                    shot['goal_crossed_y'],
                    shot['goal_crossed_z'],
                    shot['expected_goals'],
                    shot['expected_goals_on_target'],
                    shot['shot_type'],
                    shot['situation'],
                    shot['period'],
                    shot['is_own_goal'],
                    shot['is_saved_off_line'],
                    shot['is_from_inside_box'],
                    shot['team_color']
                ))
            
            # Insert in batches
            if len(shotmap_batch) >= batch_size * 10:
                insert_shotmap_batch(conn, shotmap_batch)
                shotmap_batch = []
                
        except Exception as e:
            print(f"Error processing match_id {match_id}: {e}")
            continue
    
    # Insert remaining records
    if shotmap_batch:
        insert_shotmap_batch(conn, shotmap_batch)
    
    conn.close()

def insert_shotmap_batch(conn, shotmap_batch):
    """Insert batch of shotmap records into database."""
    cursor = conn.cursor()
    
    if shotmap_batch:
        query = """
            INSERT INTO [dbo].[match_shotmap] (
                match_id, shot_id, event_type, team_id, player_id, player_name,
                first_name, last_name, full_name, x, y, minute, minute_added,
                is_blocked, is_on_target, blocked_x, blocked_y,
                goal_crossed_y, goal_crossed_z, expected_goals,
                expected_goals_on_target, shot_type, situation, period,
                is_own_goal, is_saved_off_line, is_from_inside_box, team_color
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, shotmap_batch)
    
    conn.commit()
    print(f"Inserted {len(shotmap_batch)} shots")

if __name__ == "__main__":
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server;"
        "DATABASE=fussballDB;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    process_shotmap(connection_string)