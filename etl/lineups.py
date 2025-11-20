import json
import pyodbc
from datetime import datetime
from typing import Dict, Any, List, Optional

def process_single_match(conn, match_id: int, raw_json: str) -> bool:
    """Process a single match's lineup data."""
    try:
        # Create tables if they don't exist
        create_lineup_tables(conn)
        
        cursor = conn.cursor()
        
        # Check if already processed
        cursor.execute("SELECT match_id FROM [dbo].[match_lineups] WHERE match_id = ?", match_id)
        if cursor.fetchone():
            return True  # Already processed
        
        # Parse the data
        parsed = parse_lineup_data(raw_json)
        
        if not parsed['players']:  # Skip if no lineup data
            return False
        
        # Insert lineup record
        cursor.execute("""
            INSERT INTO [dbo].[match_lineups] (
                match_id, lineup_type, source, available_filters
            ) VALUES (?, ?, ?, ?)
        """, parsed['match_id'], parsed['lineup_type'], parsed['source'],
             ','.join(parsed.get('available_filters', [])) if parsed.get('available_filters') else None)
        
        # Insert team stats
        if parsed.get('home_team_stats'):
            stats = parsed['home_team_stats']
            cursor.execute("""
                INSERT INTO [dbo].[match_team_stats] (
                    match_id, team_id, team_name, formation, rating,
                    average_starter_age, total_starter_market_value, is_home_team
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, parsed['match_id'], stats['team_id'], stats['team_name'], stats['formation'],
                 stats['rating'], stats['average_starter_age'], stats['total_starter_market_value'],
                 stats['is_home'])
        
        if parsed.get('away_team_stats'):
            stats = parsed['away_team_stats']
            cursor.execute("""
                INSERT INTO [dbo].[match_team_stats] (
                    match_id, team_id, team_name, formation, rating,
                    average_starter_age, total_starter_market_value, is_home_team
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, parsed['match_id'], stats['team_id'], stats['team_name'], stats['formation'],
                 stats['rating'], stats['average_starter_age'], stats['total_starter_market_value'],
                 stats['is_home'])
        
        # Insert players
        for player in parsed['players']:
            cursor.execute("""
                INSERT INTO [dbo].[match_lineup_players] (
                    match_id, team_id, player_id, player_name, first_name, last_name,
                    short_name, age, position_id, usual_position_id, shirt_number, is_captain,
                    is_home_team, is_starter, formation, country_name, country_code,
                    market_value, rating, fantasy_score, player_of_match,
                    h_layout_x, h_layout_y, h_layout_width, h_layout_height,
                    v_layout_x, v_layout_y, v_layout_width, v_layout_height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, player['match_id'], player['team_id'], player['player_id'], player['player_name'],
                 player['first_name'], player['last_name'], player['short_name'], player['age'],
                 player['position_id'], player['usual_position_id'], player['shirt_number'],
                 player['is_captain'], player['is_home_team'], player['is_starter'], player['formation'],
                 player['country_name'], player['country_code'], player['market_value'], player['rating'],
                 player['fantasy_score'], player['player_of_match'], player['h_layout_x'],
                 player['h_layout_y'], player['h_layout_width'], player['h_layout_height'],
                 player['v_layout_x'], player['v_layout_y'], player['v_layout_width'], player['v_layout_height'])
            
            # Insert substitutions
            for sub in player['substitutions']:
                cursor.execute("""
                    INSERT INTO [dbo].[match_substitutions] (
                        match_id, player_id, player_name, team_id, substitution_time,
                        substitution_type, substitution_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, player['match_id'], player['player_id'], player['player_name'],
                     player['team_id'], sub['time'], sub['type'], sub['reason'])
        
        # Insert coaches
        for coach in parsed['coaches']:
            cursor.execute("""
                INSERT INTO [dbo].[match_coaches] (
                    match_id, team_id, coach_id, coach_name, first_name, last_name,
                    age, is_home_team, country_name, country_code, primary_team_id,
                    primary_team_name, usual_position_id, is_coach
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, coach['match_id'], coach['team_id'], coach['coach_id'], coach['coach_name'],
                 coach['first_name'], coach['last_name'], coach['age'], coach['is_home_team'],
                 coach['country_name'], coach['country_code'], coach['primary_team_id'],
                 coach['primary_team_name'], coach['usual_position_id'], coach['is_coach'])
        
        # Insert unavailable players
        for unavailable in parsed['unavailable']:
            cursor.execute("""
                INSERT INTO [dbo].[match_unavailable_players] (
                    match_id, team_id, player_id, player_name, first_name, last_name,
                    age, country_name, country_code, market_value, is_home_team,
                    injury_id, unavailability_type, expected_return
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, unavailable['match_id'], unavailable['team_id'], unavailable['player_id'],
                 unavailable['player_name'], unavailable['first_name'], unavailable['last_name'],
                 unavailable['age'], unavailable['country_name'], unavailable['country_code'],
                 unavailable['market_value'], unavailable['is_home_team'], unavailable['injury_id'],
                 unavailable['unavailability_type'], unavailable['expected_return'])
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error in process_single_match for match {match_id}: {e}")
        conn.rollback()
        return False
    
def parse_lineup_data(raw_json: str) -> Dict[str, Any]:
    """Parse lineup data from raw JSON string."""
    data = json.loads(raw_json)
    
    # Navigate to lineup data - it's in content.lineup
    if 'api_responses' in data:
        content = data.get('api_responses', {}).get('match_details', {}).get('data', {}).get('content', {})
    else:
        content = data.get('match_details', {}).get('data', {}).get('content', {})
    
    lineup = content.get('lineup', {})
    
    if not lineup:
        return {'match_id': data.get('match_id'), 'players': [], 'coaches': [], 'unavailable': []}
    
    match_id = lineup.get('matchId', data.get('match_id'))
    lineup_type = lineup.get('lineupType')
    source = lineup.get('source')
    available_filters = lineup.get('availableFilters', [])
    
    players = []
    coaches = []
    unavailable_players = []
    
    # Process home team
    home_team = lineup.get('homeTeam', {})
    home_team_stats = {}
    if home_team:
        home_team_id = home_team.get('id')
        home_team_name = home_team.get('name')
        home_formation = home_team.get('formation')
        home_rating = home_team.get('rating')
        
        home_team_stats = {
            'team_id': home_team_id,
            'team_name': home_team_name,
            'formation': home_formation,
            'rating': home_rating,
            'average_starter_age': home_team.get('averageStarterAge'),
            'total_starter_market_value': home_team.get('totalStarterMarketValue'),
            'is_home': True
        }
        
        # Process home starters
        for player in home_team.get('starters', []):
            players.append(parse_player(player, match_id, home_team_id, True, True, home_formation))
        
        # Process home subs
        for player in home_team.get('subs', []):
            players.append(parse_player(player, match_id, home_team_id, True, False, home_formation))
        
        # Process home unavailable players
        for player in home_team.get('unavailable', []):
            unavailable_players.append(parse_unavailable_player(player, match_id, home_team_id, True))
        
        # Process home coach
        if home_team.get('coach'):
            coaches.append(parse_coach(home_team['coach'], match_id, home_team_id, True))
    
    # Process away team
    away_team = lineup.get('awayTeam', {})
    away_team_stats = {}
    if away_team:
        away_team_id = away_team.get('id')
        away_team_name = away_team.get('name')
        away_formation = away_team.get('formation')
        away_rating = away_team.get('rating')
        
        away_team_stats = {
            'team_id': away_team_id,
            'team_name': away_team_name,
            'formation': away_formation,
            'rating': away_rating,
            'average_starter_age': away_team.get('averageStarterAge'),
            'total_starter_market_value': away_team.get('totalStarterMarketValue'),
            'is_home': False
        }
        
        # Process away starters
        for player in away_team.get('starters', []):
            players.append(parse_player(player, match_id, away_team_id, False, True, away_formation))
        
        # Process away subs
        for player in away_team.get('subs', []):
            players.append(parse_player(player, match_id, away_team_id, False, False, away_formation))
        
        # Process away unavailable players
        for player in away_team.get('unavailable', []):
            unavailable_players.append(parse_unavailable_player(player, match_id, away_team_id, False))
        
        # Process away coach
        if away_team.get('coach'):
            coaches.append(parse_coach(away_team['coach'], match_id, away_team_id, False))
    
    return {
        'match_id': match_id,
        'lineup_type': lineup_type,
        'source': source,
        'available_filters': available_filters,
        'home_team_stats': home_team_stats,
        'away_team_stats': away_team_stats,
        'players': players,
        'coaches': coaches,
        'unavailable': unavailable_players
    }

def parse_player(player_data: Dict, match_id: int, team_id: int, is_home: bool, is_starter: bool, formation: str) -> Dict:
    """Parse individual player data."""
    performance = player_data.get('performance', {})
    
    # Parse substitution events
    substitutions = []
    for sub_event in performance.get('substitutionEvents', []):
        substitutions.append({
            'time': sub_event.get('time'),
            'type': sub_event.get('type'),
            'reason': sub_event.get('reason')
        })
    
    # Get layout information
    h_layout = player_data.get('horizontalLayout', {})
    v_layout = player_data.get('verticalLayout', {})
    
    return {
        'match_id': match_id,
        'team_id': team_id,
        'player_id': player_data.get('id'),
        'player_name': player_data.get('name'),
        'first_name': player_data.get('firstName'),
        'last_name': player_data.get('lastName'),
        'short_name': player_data.get('shortName'),
        'age': player_data.get('age'),
        'position_id': player_data.get('positionId'),
        'usual_position_id': player_data.get('usualPlayingPositionId'),
        'shirt_number': player_data.get('shirtNumber'),
        'is_captain': player_data.get('isCaptain', False),
        'is_home_team': is_home,
        'is_starter': is_starter,
        'formation': formation,
        'country_name': player_data.get('countryName'),
        'country_code': player_data.get('countryCode'),
        'market_value': player_data.get('marketValue'),
        'rating': performance.get('rating'),
        'fantasy_score': performance.get('fantasyScore'),
        'player_of_match': performance.get('playerOfTheMatch', False),
        'substitutions': substitutions,
        # Layout positions
        'h_layout_x': h_layout.get('x'),
        'h_layout_y': h_layout.get('y'),
        'h_layout_width': h_layout.get('width'),
        'h_layout_height': h_layout.get('height'),
        'v_layout_x': v_layout.get('x'),
        'v_layout_y': v_layout.get('y'),
        'v_layout_width': v_layout.get('width'),
        'v_layout_height': v_layout.get('height')
    }

def parse_unavailable_player(player_data: Dict, match_id: int, team_id: int, is_home: bool) -> Dict:
    """Parse unavailable player data."""
    unavailability = player_data.get('unavailability', {})
    
    return {
        'match_id': match_id,
        'team_id': team_id,
        'player_id': player_data.get('id'),
        'player_name': player_data.get('name'),
        'first_name': player_data.get('firstName'),
        'last_name': player_data.get('lastName'),
        'age': player_data.get('age'),
        'country_name': player_data.get('countryName'),
        'country_code': player_data.get('countryCode'),
        'market_value': player_data.get('marketValue'),
        'is_home_team': is_home,
        'injury_id': unavailability.get('injuryId'),
        'unavailability_type': unavailability.get('type'),
        'expected_return': unavailability.get('expectedReturn')
    }

def parse_coach(coach_data: Dict, match_id: int, team_id: int, is_home: bool) -> Dict:
    """Parse coach data."""
    return {
        'match_id': match_id,
        'team_id': team_id,
        'coach_id': coach_data.get('id'),
        'coach_name': coach_data.get('name'),
        'first_name': coach_data.get('firstName'),
        'last_name': coach_data.get('lastName'),
        'age': coach_data.get('age'),
        'is_home_team': is_home,
        'country_name': coach_data.get('countryName'),
        'country_code': coach_data.get('countryCode'),
        'primary_team_id': coach_data.get('primaryTeamId'),
        'primary_team_name': coach_data.get('primaryTeamName'),
        'usual_position_id': coach_data.get('usualPlayingPositionId'),
        'is_coach': coach_data.get('isCoach', True)
    }

def create_lineup_tables(connection):
    """Create lineup-related tables."""
    cursor = connection.cursor()
    
    # Create match_lineups table for general lineup info
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_lineups]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_lineups] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [lineup_type] NVARCHAR(50),
                [source] NVARCHAR(50),
                [available_filters] NVARCHAR(500),
                [home_formation] NVARCHAR(20),
                [away_formation] NVARCHAR(20),
                [created_at] DATETIME DEFAULT GETDATE(),
                CONSTRAINT UQ_match_lineups_match_id UNIQUE(match_id)
            );
        END
    """)
    
    # Create match_team_stats table for team-level statistics
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_team_stats]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_team_stats] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [team_id] INT,
                [team_name] NVARCHAR(200),
                [formation] NVARCHAR(20),
                [rating] FLOAT,
                [average_starter_age] FLOAT,
                [total_starter_market_value] BIGINT,
                [is_home_team] BIT,
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create match_lineup_players table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_lineup_players]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_lineup_players] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [team_id] INT,
                [player_id] INT,
                [player_name] NVARCHAR(200),
                [first_name] NVARCHAR(100),
                [last_name] NVARCHAR(100),
                [short_name] NVARCHAR(100),
                [age] INT,
                [position_id] INT,
                [usual_position_id] INT,
                [shirt_number] NVARCHAR(10),
                [is_captain] BIT,
                [is_home_team] BIT,
                [is_starter] BIT,
                [formation] NVARCHAR(20),
                [country_name] NVARCHAR(100),
                [country_code] NVARCHAR(10),
                [market_value] BIGINT,
                [rating] FLOAT,
                [fantasy_score] NVARCHAR(20),
                [player_of_match] BIT,
                [h_layout_x] FLOAT,
                [h_layout_y] FLOAT,
                [h_layout_width] FLOAT,
                [h_layout_height] FLOAT,
                [v_layout_x] FLOAT,
                [v_layout_y] FLOAT,
                [v_layout_width] FLOAT,
                [v_layout_height] FLOAT,
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create match_substitutions table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_substitutions]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_substitutions] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [player_id] INT,
                [player_name] NVARCHAR(200),
                [team_id] INT,
                [substitution_time] INT,
                [substitution_type] NVARCHAR(20),
                [substitution_reason] NVARCHAR(50),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create match_coaches table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_coaches]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_coaches] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [team_id] INT,
                [coach_id] INT,
                [coach_name] NVARCHAR(200),
                [first_name] NVARCHAR(100),
                [last_name] NVARCHAR(100),
                [age] INT,
                [is_home_team] BIT,
                [country_name] NVARCHAR(100),
                [country_code] NVARCHAR(10),
                [primary_team_id] INT,
                [primary_team_name] NVARCHAR(200),
                [usual_position_id] INT,
                [is_coach] BIT,
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create match_unavailable_players table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_unavailable_players]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_unavailable_players] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [team_id] INT,
                [player_id] INT,
                [player_name] NVARCHAR(200),
                [first_name] NVARCHAR(100),
                [last_name] NVARCHAR(100),
                [age] INT,
                [country_name] NVARCHAR(100),
                [country_code] NVARCHAR(10),
                [market_value] BIGINT,
                [is_home_team] BIT,
                [injury_id] INT,
                [unavailability_type] NVARCHAR(50),
                [expected_return] NVARCHAR(100),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    connection.commit()

def process_lineups(connection_string: str, batch_size: int = 100):
    """Process lineup data from fotmob_raw_data table."""
    conn = pyodbc.connect(connection_string)
    
    # Create tables if they don't exist
    create_lineup_tables(conn)
    
    cursor = conn.cursor()
    
    # Get existing match_ids to avoid duplicates
    cursor.execute("SELECT DISTINCT match_id FROM [dbo].[match_lineups]")
    existing_matches = set(row[0] for row in cursor.fetchall())
    
    # Fetch raw data
    query = """
        SELECT id, match_id, raw_json
        FROM [dbo].[fotmob_raw_data]
        WHERE data_type = 'match_details'
        ORDER BY id
    """
    
    cursor.execute(query)
    
    lineups_batch = []
    team_stats_batch = []
    players_batch = []
    substitutions_batch = []
    coaches_batch = []
    unavailable_batch = []
    
    for row in cursor.fetchall():
        source_id, match_id, raw_json = row
        
        if match_id in existing_matches:
            continue
        
        try:
            parsed = parse_lineup_data(raw_json)
            
            if not parsed['players']:  # Skip if no lineup data
                continue
            
            # Prepare lineup record
            lineups_batch.append((
                parsed['match_id'],
                parsed['lineup_type'],
                parsed['source'],
                ','.join(parsed.get('available_filters', [])) if parsed.get('available_filters') else None
            ))
            
            # Prepare team stats records
            if parsed.get('home_team_stats'):
                stats = parsed['home_team_stats']
                team_stats_batch.append((
                    parsed['match_id'],
                    stats['team_id'],
                    stats['team_name'],
                    stats['formation'],
                    stats['rating'],
                    stats['average_starter_age'],
                    stats['total_starter_market_value'],
                    stats['is_home']
                ))
            
            if parsed.get('away_team_stats'):
                stats = parsed['away_team_stats']
                team_stats_batch.append((
                    parsed['match_id'],
                    stats['team_id'],
                    stats['team_name'],
                    stats['formation'],
                    stats['rating'],
                    stats['average_starter_age'],
                    stats['total_starter_market_value'],
                    stats['is_home']
                ))
            
            # Prepare player records
            for player in parsed['players']:
                players_batch.append((
                    player['match_id'],
                    player['team_id'],
                    player['player_id'],
                    player['player_name'],
                    player['first_name'],
                    player['last_name'],
                    player['short_name'],
                    player['age'],
                    player['position_id'],
                    player['usual_position_id'],
                    player['shirt_number'],
                    player['is_captain'],
                    player['is_home_team'],
                    player['is_starter'],
                    player['formation'],
                    player['country_name'],
                    player['country_code'],
                    player['market_value'],
                    player['rating'],
                    player['fantasy_score'],
                    player['player_of_match'],
                    player['h_layout_x'],
                    player['h_layout_y'],
                    player['h_layout_width'],
                    player['h_layout_height'],
                    player['v_layout_x'],
                    player['v_layout_y'],
                    player['v_layout_width'],
                    player['v_layout_height']
                ))
                
                # Prepare substitution records
                for sub in player['substitutions']:
                    substitutions_batch.append((
                        player['match_id'],
                        player['player_id'],
                        player['player_name'],
                        player['team_id'],
                        sub['time'],
                        sub['type'],
                        sub['reason']
                    ))
            
            # Prepare coach records
            for coach in parsed['coaches']:
                coaches_batch.append((
                    coach['match_id'],
                    coach['team_id'],
                    coach['coach_id'],
                    coach['coach_name'],
                    coach['first_name'],
                    coach['last_name'],
                    coach['age'],
                    coach['is_home_team'],
                    coach['country_name'],
                    coach['country_code'],
                    coach['primary_team_id'],
                    coach['primary_team_name'],
                    coach['usual_position_id'],
                    coach['is_coach']
                ))
            
            # Prepare unavailable player records
            for unavailable in parsed['unavailable']:
                unavailable_batch.append((
                    unavailable['match_id'],
                    unavailable['team_id'],
                    unavailable['player_id'],
                    unavailable['player_name'],
                    unavailable['first_name'],
                    unavailable['last_name'],
                    unavailable['age'],
                    unavailable['country_name'],
                    unavailable['country_code'],
                    unavailable['market_value'],
                    unavailable['is_home_team'],
                    unavailable['injury_id'],
                    unavailable['unavailability_type'],
                    unavailable['expected_return']
                ))
            
            # Insert in batches
            if len(lineups_batch) >= batch_size:
                insert_lineup_batch(conn, lineups_batch, team_stats_batch, players_batch, 
                                   substitutions_batch, coaches_batch, unavailable_batch)
                lineups_batch = []
                team_stats_batch = []
                players_batch = []
                substitutions_batch = []
                coaches_batch = []
                unavailable_batch = []
                
        except Exception as e:
            print(f"Error processing match_id {match_id}: {e}")
            continue
    
    # Insert remaining records
    if lineups_batch:
        insert_lineup_batch(conn, lineups_batch, team_stats_batch, players_batch, 
                           substitutions_batch, coaches_batch, unavailable_batch)
    
    conn.close()

def insert_lineup_batch(conn, lineups_batch, team_stats_batch, players_batch, 
                       substitutions_batch, coaches_batch, unavailable_batch):
    """Insert batch of lineup records into database."""
    cursor = conn.cursor()
    
    # Insert lineups
    if lineups_batch:
        query = """
            INSERT INTO [dbo].[match_lineups] (
                match_id, lineup_type, source, available_filters
            ) VALUES (?, ?, ?, ?)
        """
        cursor.executemany(query, lineups_batch)
    
    # Insert team stats
    if team_stats_batch:
        query = """
            INSERT INTO [dbo].[match_team_stats] (
                match_id, team_id, team_name, formation, rating,
                average_starter_age, total_starter_market_value, is_home_team
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, team_stats_batch)
    
    # Insert players
    if players_batch:
        query = """
            INSERT INTO [dbo].[match_lineup_players] (
                match_id, team_id, player_id, player_name, first_name, last_name,
                short_name, age, position_id, usual_position_id, shirt_number, is_captain,
                is_home_team, is_starter, formation, country_name, country_code,
                market_value, rating, fantasy_score, player_of_match,
                h_layout_x, h_layout_y, h_layout_width, h_layout_height,
                v_layout_x, v_layout_y, v_layout_width, v_layout_height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, players_batch)
    
    # Insert substitutions
    if substitutions_batch:
        query = """
            INSERT INTO [dbo].[match_substitutions] (
                match_id, player_id, player_name, team_id, substitution_time,
                substitution_type, substitution_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, substitutions_batch)
    
    # Insert coaches
    if coaches_batch:
        query = """
            INSERT INTO [dbo].[match_coaches] (
                match_id, team_id, coach_id, coach_name, first_name, last_name,
                age, is_home_team, country_name, country_code, primary_team_id,
                primary_team_name, usual_position_id, is_coach
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, coaches_batch)
    
    # Insert unavailable players
    if unavailable_batch:
        query = """
            INSERT INTO [dbo].[match_unavailable_players] (
                match_id, team_id, player_id, player_name, first_name, last_name,
                age, country_name, country_code, market_value, is_home_team,
                injury_id, unavailability_type, expected_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, unavailable_batch)
    
    conn.commit()
    print(f"Inserted {len(lineups_batch)} matches with {len(players_batch)} players, "
          f"{len(substitutions_batch)} substitutions, {len(coaches_batch)} coaches, "
          f"and {len(unavailable_batch)} unavailable players")

if __name__ == "__main__":
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server;"
        "DATABASE=fussballDB;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    process_lineups(connection_string)