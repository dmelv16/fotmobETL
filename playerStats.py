import json
import pyodbc
from typing import Dict, Any, List

def parse_player_stats(raw_json: str) -> Dict[str, Any]:
    """Parse player statistics from raw JSON string."""
    data = json.loads(raw_json)
    
    # Navigate to playerStats data
    if 'api_responses' in data:
        content = data.get('api_responses', {}).get('match_details', {}).get('data', {}).get('content', {})
    else:
        content = data.get('match_details', {}).get('data', {}).get('content', {})
    
    player_stats = content.get('playerStats', {})
    
    if not player_stats:
        return {'match_id': data.get('match_id'), 'players': [], 'shotmap': []}
    
    match_id = data.get('match_id')
    players = []
    shotmap_events = []
    
    # Process each player
    for player_id, player_data in player_stats.items():
        # Basic player info
        player_info = {
            'match_id': match_id,
            'player_id': player_data.get('id'),
            'player_name': player_data.get('name'),
            'opta_id': player_data.get('optaId'),
            'team_id': player_data.get('teamId'),
            'team_name': player_data.get('teamName'),
            'is_goalkeeper': player_data.get('isGoalkeeper', False)
        }
        
        # Process stats categories
        for category in player_data.get('stats', []):
            category_title = category.get('title')
            category_key = category.get('key')
            
            for stat_name, stat_data in category.get('stats', {}).items():
                stat = stat_data.get('stat', {})
                
                players.append({
                    **player_info,
                    'category': category_title,
                    'category_key': category_key,
                    'stat_name': stat_name,
                    'stat_key': stat_data.get('key'),
                    'stat_value': stat.get('value'),
                    'stat_total': stat.get('total'),
                    'stat_type': stat.get('type')
                })
        
        # Process shotmap
        for shot in player_data.get('shotmap', []):
            shotmap_events.append({
                'match_id': match_id,
                'shot_id': shot.get('id'),
                'event_type': shot.get('eventType'),
                'team_id': shot.get('teamId'),
                'player_id': shot.get('playerId'),
                'player_name': shot.get('playerName'),
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
                'is_from_inside_box': shot.get('isFromInsideBox')
            })
    
    return {
        'match_id': match_id,
        'players': players,
        'shotmap': shotmap_events
    }

def create_player_stats_tables(connection):
    """Create player statistics tables."""
    cursor = connection.cursor()
    
    # Create player_stats table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[player_stats]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[player_stats] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [player_id] INT,
                [player_name] NVARCHAR(200),
                [opta_id] NVARCHAR(50),
                [team_id] INT,
                [team_name] NVARCHAR(200),
                [is_goalkeeper] BIT,
                [category] NVARCHAR(100),
                [category_key] NVARCHAR(100),
                [stat_name] NVARCHAR(200),
                [stat_key] NVARCHAR(100),
                [stat_value] FLOAT,
                [stat_total] FLOAT,
                [stat_type] NVARCHAR(50),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create shotmap table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[player_shotmap]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[player_shotmap] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [shot_id] BIGINT,
                [event_type] NVARCHAR(50),
                [team_id] INT,
                [player_id] INT,
                [player_name] NVARCHAR(200),
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
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    connection.commit()

def process_player_stats(connection_string: str, batch_size: int = 100):
    """Process player statistics from fotmob_raw_data table."""
    conn = pyodbc.connect(connection_string)
    
    # Create tables if they don't exist
    create_player_stats_tables(conn)
    
    cursor = conn.cursor()
    
    # Get existing match_ids to avoid duplicates
    cursor.execute("SELECT DISTINCT match_id FROM [dbo].[player_stats]")
    existing_matches = set(row[0] for row in cursor.fetchall())
    
    # Fetch raw data
    query = """
        SELECT id, match_id, raw_json
        FROM [dbo].[fotmob_raw_data]
        WHERE data_type = 'match_details'
        ORDER BY id
    """
    
    cursor.execute(query)
    
    stats_batch = []
    shotmap_batch = []
    
    for row in cursor.fetchall():
        source_id, match_id, raw_json = row
        
        if match_id in existing_matches:
            continue
        
        try:
            parsed = parse_player_stats(raw_json)
            
            if not parsed['players']:
                continue
            
            # Prepare player stats records
            for stat in parsed['players']:
                stats_batch.append((
                    stat['match_id'],
                    stat['player_id'],
                    stat['player_name'],
                    stat['opta_id'],
                    stat['team_id'],
                    stat['team_name'],
                    stat['is_goalkeeper'],
                    stat['category'],
                    stat['category_key'],
                    stat['stat_name'],
                    stat['stat_key'],
                    stat['stat_value'],
                    stat['stat_total'],
                    stat['stat_type']
                ))
            
            # Prepare shotmap records
            for shot in parsed['shotmap']:
                shotmap_batch.append((
                    shot['match_id'],
                    shot['shot_id'],
                    shot['event_type'],
                    shot['team_id'],
                    shot['player_id'],
                    shot['player_name'],
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
                    shot['is_from_inside_box']
                ))
            
            # Insert in batches
            if len(stats_batch) >= batch_size * 50:  # Many stats per match
                insert_player_stats_batch(conn, stats_batch, shotmap_batch)
                stats_batch = []
                shotmap_batch = []
                
        except Exception as e:
            print(f"Error processing match_id {match_id}: {e}")
            continue
    
    # Insert remaining records
    if stats_batch or shotmap_batch:
        insert_player_stats_batch(conn, stats_batch, shotmap_batch)
    
    conn.close()

def insert_player_stats_batch(conn, stats_batch, shotmap_batch):
    """Insert batch of player stats records into database."""
    cursor = conn.cursor()
    
    # Insert player stats
    if stats_batch:
        query = """
            INSERT INTO [dbo].[player_stats] (
                match_id, player_id, player_name, opta_id, team_id, team_name,
                is_goalkeeper, category, category_key, stat_name, stat_key,
                stat_value, stat_total, stat_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, stats_batch)
    
    # Insert shotmap data
    if shotmap_batch:
        query = """
            INSERT INTO [dbo].[player_shotmap] (
                match_id, shot_id, event_type, team_id, player_id, player_name,
                x, y, minute, minute_added, is_blocked, is_on_target,
                blocked_x, blocked_y, goal_crossed_y, goal_crossed_z,
                expected_goals, expected_goals_on_target, shot_type,
                situation, period, is_own_goal, is_saved_off_line,
                is_from_inside_box
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, shotmap_batch)
    
    conn.commit()
    print(f"Inserted {len(stats_batch)} player stats and {len(shotmap_batch)} shots")

if __name__ == "__main__":
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server;"
        "DATABASE=fussballDB;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    process_player_stats(connection_string)