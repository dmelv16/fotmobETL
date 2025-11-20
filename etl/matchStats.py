import json
import pyodbc
from datetime import datetime
from typing import Dict, Any, List, Optional

def process_single_match(conn, match_id: int, raw_json: str) -> bool:
    """Process a single match's statistics."""
    try:
        create_stats_tables(conn)
        cursor = conn.cursor()
        
        # Check if already processed
        cursor.execute("SELECT match_id FROM [dbo].[match_stats] WHERE match_id = ?", match_id)
        if cursor.fetchone():
            return True
        
        parsed = parse_stats_data(raw_json)
        if not parsed['stats']:
            return False
        
        # Insert individual stats
        for stat in parsed['stats']:
            cursor.execute("""
                INSERT INTO [dbo].[match_stats] (
                    match_id, period, category, category_key, stat_name,
                    stat_key, home_value, away_value, format, display_type, highlighted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, stat['match_id'], stat['period'], stat['category'], stat['category_key'],
                 stat['stat_name'], stat['stat_key'],
                 str(stat['home_value']) if stat['home_value'] is not None else None,
                 str(stat['away_value']) if stat['away_value'] is not None else None,
                 stat['format'], stat['type'], stat['highlighted'])
        
        # Insert summary
        summary = build_summary(parsed['stats'], match_id)
        if summary:
            cursor.execute("""
                INSERT INTO [dbo].[match_stats_summary] (
                    match_id, home_possession, away_possession,
                    home_xg, away_xg, home_xg_open_play, away_xg_open_play,
                    home_xg_set_play, away_xg_set_play, home_xgot, away_xgot,
                    home_total_shots, away_total_shots, home_shots_on_target, away_shots_on_target,
                    home_shots_off_target, away_shots_off_target, home_blocked_shots, away_blocked_shots,
                    home_shots_inside_box, away_shots_inside_box, home_shots_outside_box, away_shots_outside_box,
                    home_big_chances, away_big_chances, home_big_chances_missed, away_big_chances_missed,
                    home_passes, away_passes, home_accurate_passes, away_accurate_passes,
                    home_pass_accuracy_pct, away_pass_accuracy_pct,
                    home_own_half_passes, away_own_half_passes,
                    home_opposition_half_passes, away_opposition_half_passes,
                    home_accurate_long_balls, away_accurate_long_balls,
                    home_accurate_crosses, away_accurate_crosses,
                    home_touches_opp_box, away_touches_opp_box,
                    home_tackles, away_tackles, home_interceptions, away_interceptions,
                    home_blocks, away_blocks, home_clearances, away_clearances,
                    home_keeper_saves, away_keeper_saves,
                    home_duels_won, away_duels_won, home_ground_duels_won, away_ground_duels_won,
                    home_aerial_duels_won, away_aerial_duels_won,
                    home_successful_dribbles, away_successful_dribbles,
                    home_corners, away_corners, home_offsides, away_offsides,
                    home_fouls, away_fouls, home_yellow_cards, away_yellow_cards,
                    home_red_cards, away_red_cards
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, *summary)
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error in process_single_match for match {match_id}: {e}")
        conn.rollback()
        return False


def parse_stats_data(raw_json: str) -> Dict[str, Any]:
    """Parse match statistics from raw JSON string."""
    data = json.loads(raw_json)
    
    # Navigate to stats data - it's in content.stats
    if 'api_responses' in data:
        content = data.get('api_responses', {}).get('match_details', {}).get('data', {}).get('content', {})
    else:
        content = data.get('match_details', {}).get('data', {}).get('content', {})
    
    stats = content.get('stats', {})
    
    if not stats:
        return {'match_id': data.get('match_id'), 'stats': []}
    
    match_id = data.get('match_id')
    parsed_stats = []
    
    # Process each period (All, FirstHalf, SecondHalf, etc.)
    periods = stats.get('Periods', {})
    
    for period_name, period_data in periods.items():
        if not period_data or 'stats' not in period_data:
            continue
        
        # Process each stat category in the period
        for category in period_data.get('stats', []):
            category_title = category.get('title')
            category_key = category.get('key')
            
            # Process individual stats in the category
            for stat in category.get('stats', []):
                stat_title = stat.get('title')
                stat_key = stat.get('key')
                stat_values = stat.get('stats', [])
                
                # Extract home and away values
                home_value = stat_values[0] if len(stat_values) > 0 else None
                away_value = stat_values[1] if len(stat_values) > 1 else None
                
                parsed_stats.append({
                    'match_id': match_id,
                    'period': period_name,
                    'category': category_title,
                    'category_key': category_key,
                    'stat_name': stat_title,
                    'stat_key': stat_key,
                    'home_value': home_value,
                    'away_value': away_value,
                    'format': stat.get('format'),
                    'type': stat.get('type'),
                    'highlighted': stat.get('highlighted')
                })
    
    return {
        'match_id': match_id,
        'stats': parsed_stats
    }

def create_stats_tables(connection):
    """Create match statistics tables."""
    cursor = connection.cursor()
    
    # Create match_stats table (detailed stats)
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_stats]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_stats] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [period] NVARCHAR(50),
                [category] NVARCHAR(100),
                [category_key] NVARCHAR(100),
                [stat_name] NVARCHAR(200),
                [stat_key] NVARCHAR(100),
                [home_value] NVARCHAR(200),
                [away_value] NVARCHAR(200),
                [format] NVARCHAR(50),
                [display_type] NVARCHAR(50),
                [highlighted] NVARCHAR(20),
                [created_at] DATETIME DEFAULT GETDATE(),
                INDEX IX_match_stats_match_id (match_id),
                INDEX IX_match_stats_period (period),
                INDEX IX_match_stats_stat_key (stat_key)
            );
        END
    """)
    
    # Create comprehensive summary table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_stats_summary]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_stats_summary] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL UNIQUE,
                
                -- Possession
                [home_possession] INT,
                [away_possession] INT,
                
                -- Expected Goals
                [home_xg] FLOAT,
                [away_xg] FLOAT,
                [home_xg_open_play] FLOAT,
                [away_xg_open_play] FLOAT,
                [home_xg_set_play] FLOAT,
                [away_xg_set_play] FLOAT,
                [home_xgot] FLOAT,
                [away_xgot] FLOAT,
                
                -- Shots
                [home_total_shots] INT,
                [away_total_shots] INT,
                [home_shots_on_target] INT,
                [away_shots_on_target] INT,
                [home_shots_off_target] INT,
                [away_shots_off_target] INT,
                [home_blocked_shots] INT,
                [away_blocked_shots] INT,
                [home_shots_inside_box] INT,
                [away_shots_inside_box] INT,
                [home_shots_outside_box] INT,
                [away_shots_outside_box] INT,
                [home_big_chances] INT,
                [away_big_chances] INT,
                [home_big_chances_missed] INT,
                [away_big_chances_missed] INT,
                
                -- Passes
                [home_passes] INT,
                [away_passes] INT,
                [home_accurate_passes] INT,
                [away_accurate_passes] INT,
                [home_pass_accuracy_pct] INT,
                [away_pass_accuracy_pct] INT,
                [home_own_half_passes] INT,
                [away_own_half_passes] INT,
                [home_opposition_half_passes] INT,
                [away_opposition_half_passes] INT,
                [home_accurate_long_balls] INT,
                [away_accurate_long_balls] INT,
                [home_accurate_crosses] INT,
                [away_accurate_crosses] INT,
                [home_touches_opp_box] INT,
                [away_touches_opp_box] INT,
                
                -- Defence
                [home_tackles] INT,
                [away_tackles] INT,
                [home_interceptions] INT,
                [away_interceptions] INT,
                [home_blocks] INT,
                [away_blocks] INT,
                [home_clearances] INT,
                [away_clearances] INT,
                [home_keeper_saves] INT,
                [away_keeper_saves] INT,
                
                -- Duels
                [home_duels_won] INT,
                [away_duels_won] INT,
                [home_ground_duels_won] INT,
                [away_ground_duels_won] INT,
                [home_aerial_duels_won] INT,
                [away_aerial_duels_won] INT,
                [home_successful_dribbles] INT,
                [away_successful_dribbles] INT,
                
                -- Set Pieces & Discipline
                [home_corners] INT,
                [away_corners] INT,
                [home_offsides] INT,
                [away_offsides] INT,
                [home_fouls] INT,
                [away_fouls] INT,
                [home_yellow_cards] INT,
                [away_yellow_cards] INT,
                [home_red_cards] INT,
                [away_red_cards] INT,
                
                [created_at] DATETIME DEFAULT GETDATE(),
                INDEX IX_match_stats_summary_match_id (match_id)
            );
        END
    """)
    
    connection.commit()

def extract_numeric_value(value) -> Optional[float]:
    """Extract numeric value from string formats like '408 (85%)'."""
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Extract first number from strings like "408 (85%)"
        import re
        match = re.search(r'(\d+\.?\d*)', value)
        if match:
            return float(match.group(1))
    
    return None

def extract_percentage(value) -> Optional[int]:
    """Extract percentage from string formats like '408 (85%)'."""
    if value is None or not isinstance(value, str):
        return None
    
    import re
    match = re.search(r'\((\d+)%\)', value)
    if match:
        return int(match.group(1))
    
    return None

def build_summary(stats_list: List[Dict], match_id: int) -> Optional[tuple]:
    """Build a comprehensive summary record from stats list."""
    summary = {
        'match_id': match_id,
        # Possession
        'home_possession': None,
        'away_possession': None,
        # Expected Goals
        'home_xg': None,
        'away_xg': None,
        'home_xg_open_play': None,
        'away_xg_open_play': None,
        'home_xg_set_play': None,
        'away_xg_set_play': None,
        'home_xgot': None,
        'away_xgot': None,
        # Shots
        'home_total_shots': None,
        'away_total_shots': None,
        'home_shots_on_target': None,
        'away_shots_on_target': None,
        'home_shots_off_target': None,
        'away_shots_off_target': None,
        'home_blocked_shots': None,
        'away_blocked_shots': None,
        'home_shots_inside_box': None,
        'away_shots_inside_box': None,
        'home_shots_outside_box': None,
        'away_shots_outside_box': None,
        'home_big_chances': None,
        'away_big_chances': None,
        'home_big_chances_missed': None,
        'away_big_chances_missed': None,
        # Passes
        'home_passes': None,
        'away_passes': None,
        'home_accurate_passes': None,
        'away_accurate_passes': None,
        'home_pass_accuracy_pct': None,
        'away_pass_accuracy_pct': None,
        'home_own_half_passes': None,
        'away_own_half_passes': None,
        'home_opposition_half_passes': None,
        'away_opposition_half_passes': None,
        'home_accurate_long_balls': None,
        'away_accurate_long_balls': None,
        'home_accurate_crosses': None,
        'away_accurate_crosses': None,
        'home_touches_opp_box': None,
        'away_touches_opp_box': None,
        # Defence
        'home_tackles': None,
        'away_tackles': None,
        'home_interceptions': None,
        'away_interceptions': None,
        'home_blocks': None,
        'away_blocks': None,
        'home_clearances': None,
        'away_clearances': None,
        'home_keeper_saves': None,
        'away_keeper_saves': None,
        # Duels
        'home_duels_won': None,
        'away_duels_won': None,
        'home_ground_duels_won': None,
        'away_ground_duels_won': None,
        'home_aerial_duels_won': None,
        'away_aerial_duels_won': None,
        'home_successful_dribbles': None,
        'away_successful_dribbles': None,
        # Set Pieces & Discipline
        'home_corners': None,
        'away_corners': None,
        'home_offsides': None,
        'away_offsides': None,
        'home_fouls': None,
        'away_fouls': None,
        'home_yellow_cards': None,
        'away_yellow_cards': None,
        'home_red_cards': None,
        'away_red_cards': None
    }
    
    # Map stat keys to summary fields
    stat_mapping = {
        # Possession
        'BallPossesion': ('home_possession', 'away_possession', extract_numeric_value),
        # Expected Goals
        'expected_goals': ('home_xg', 'away_xg', extract_numeric_value),
        'expected_goals_open_play': ('home_xg_open_play', 'away_xg_open_play', extract_numeric_value),
        'expected_goals_set_play': ('home_xg_set_play', 'away_xg_set_play', extract_numeric_value),
        'expected_goals_on_target': ('home_xgot', 'away_xgot', extract_numeric_value),
        # Shots
        'total_shots': ('home_total_shots', 'away_total_shots', extract_numeric_value),
        'ShotsOnTarget': ('home_shots_on_target', 'away_shots_on_target', extract_numeric_value),
        'ShotsOffTarget': ('home_shots_off_target', 'away_shots_off_target', extract_numeric_value),
        'blocked_shots': ('home_blocked_shots', 'away_blocked_shots', extract_numeric_value),
        'shots_inside_box': ('home_shots_inside_box', 'away_shots_inside_box', extract_numeric_value),
        'shots_outside_box': ('home_shots_outside_box', 'away_shots_outside_box', extract_numeric_value),
        'big_chance': ('home_big_chances', 'away_big_chances', extract_numeric_value),
        'big_chance_missed_title': ('home_big_chances_missed', 'away_big_chances_missed', extract_numeric_value),
        # Passes
        'passes': ('home_passes', 'away_passes', extract_numeric_value),
        'own_half_passes': ('home_own_half_passes', 'away_own_half_passes', extract_numeric_value),
        'opposition_half_passes': ('home_opposition_half_passes', 'away_opposition_half_passes', extract_numeric_value),
        'long_balls_accurate': ('home_accurate_long_balls', 'away_accurate_long_balls', extract_numeric_value),
        'accurate_crosses': ('home_accurate_crosses', 'away_accurate_crosses', extract_numeric_value),
        'touches_opp_box': ('home_touches_opp_box', 'away_touches_opp_box', extract_numeric_value),
        # Defence
        'matchstats.headers.tackles': ('home_tackles', 'away_tackles', extract_numeric_value),
        'interceptions': ('home_interceptions', 'away_interceptions', extract_numeric_value),
        'shot_blocks': ('home_blocks', 'away_blocks', extract_numeric_value),
        'clearances': ('home_clearances', 'away_clearances', extract_numeric_value),
        'keeper_saves': ('home_keeper_saves', 'away_keeper_saves', extract_numeric_value),
        # Duels
        'duel_won': ('home_duels_won', 'away_duels_won', extract_numeric_value),
        'ground_duels_won': ('home_ground_duels_won', 'away_ground_duels_won', extract_numeric_value),
        'aerials_won': ('home_aerial_duels_won', 'away_aerial_duels_won', extract_numeric_value),
        'dribbles_succeeded': ('home_successful_dribbles', 'away_successful_dribbles', extract_numeric_value),
        # Set Pieces & Discipline
        'corners': ('home_corners', 'away_corners', extract_numeric_value),
        'Offsides': ('home_offsides', 'away_offsides', extract_numeric_value),
        'fouls': ('home_fouls', 'away_fouls', extract_numeric_value),
        'yellow_cards': ('home_yellow_cards', 'away_yellow_cards', extract_numeric_value),
        'red_cards': ('home_red_cards', 'away_red_cards', extract_numeric_value)
    }
    
    # Special handling for accurate_passes (includes count and percentage)
    for stat in stats_list:
        if stat['period'] == 'All' and stat['stat_key'] == 'accurate_passes':
            summary['home_accurate_passes'] = extract_numeric_value(stat['home_value'])
            summary['away_accurate_passes'] = extract_numeric_value(stat['away_value'])
            summary['home_pass_accuracy_pct'] = extract_percentage(stat['home_value'])
            summary['away_pass_accuracy_pct'] = extract_percentage(stat['away_value'])
    
    # Fill summary from 'All' period stats
    for stat in stats_list:
        if stat['period'] == 'All' and stat['stat_key'] in stat_mapping:
            home_field, away_field, extractor = stat_mapping[stat['stat_key']]
            summary[home_field] = extractor(stat['home_value'])
            summary[away_field] = extractor(stat['away_value'])
    
    # Only return if we have at least some stats
    if summary['home_possession'] is not None or summary['home_total_shots'] is not None:
        return tuple(summary.values())
    
    return None

def process_stats(connection_string: str, batch_size: int = 100):
    """Process match statistics from fotmob_raw_data table."""
    conn = pyodbc.connect(connection_string)
    
    # Create tables if they don't exist
    create_stats_tables(conn)
    
    cursor = conn.cursor()
    
    # Get existing match_ids to avoid duplicates
    cursor.execute("SELECT DISTINCT match_id FROM [dbo].[match_stats]")
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
    summary_batch = []
    processed_count = 0
    
    for row in cursor.fetchall():
        source_id, match_id, raw_json = row
        
        if match_id in existing_matches:
            continue
        
        try:
            parsed = parse_stats_data(raw_json)
            
            if not parsed['stats']:  # Skip if no stats data
                continue
            
            # Prepare individual stats records
            for stat in parsed['stats']:
                stats_batch.append((
                    stat['match_id'],
                    stat['period'],
                    stat['category'],
                    stat['category_key'],
                    stat['stat_name'],
                    stat['stat_key'],
                    str(stat['home_value']) if stat['home_value'] is not None else None,
                    str(stat['away_value']) if stat['away_value'] is not None else None,
                    stat['format'],
                    stat['type'],
                    stat['highlighted']
                ))
            
            # Build summary record from 'All' period stats
            summary = build_summary(parsed['stats'], match_id)
            if summary:
                summary_batch.append(summary)
            
            processed_count += 1
            
            # Insert in batches
            if len(stats_batch) >= batch_size * 10:  # Stats have many rows per match
                insert_stats_batch(conn, stats_batch, summary_batch)
                stats_batch = []
                summary_batch = []
                
        except Exception as e:
            print(f"Error processing match_id {match_id}: {e}")
            continue
    
    # Insert remaining records
    if stats_batch:
        insert_stats_batch(conn, stats_batch, summary_batch)
    
    print(f"Successfully processed {processed_count} matches")
    conn.close()

def insert_stats_batch(conn, stats_batch, summary_batch):
    """Insert batch of stats records into database."""
    cursor = conn.cursor()
    
    # Insert individual stats
    if stats_batch:
        query = """
            INSERT INTO [dbo].[match_stats] (
                match_id, period, category, category_key, stat_name,
                stat_key, home_value, away_value, format, display_type, highlighted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, stats_batch)
    
    # Insert summary stats
    if summary_batch:
        query = """
            INSERT INTO [dbo].[match_stats_summary] (
                match_id, home_possession, away_possession,
                home_xg, away_xg, home_xg_open_play, away_xg_open_play,
                home_xg_set_play, away_xg_set_play, home_xgot, away_xgot,
                home_total_shots, away_total_shots, home_shots_on_target, away_shots_on_target,
                home_shots_off_target, away_shots_off_target, home_blocked_shots, away_blocked_shots,
                home_shots_inside_box, away_shots_inside_box, home_shots_outside_box, away_shots_outside_box,
                home_big_chances, away_big_chances, home_big_chances_missed, away_big_chances_missed,
                home_passes, away_passes, home_accurate_passes, away_accurate_passes,
                home_pass_accuracy_pct, away_pass_accuracy_pct,
                home_own_half_passes, away_own_half_passes,
                home_opposition_half_passes, away_opposition_half_passes,
                home_accurate_long_balls, away_accurate_long_balls,
                home_accurate_crosses, away_accurate_crosses,
                home_touches_opp_box, away_touches_opp_box,
                home_tackles, away_tackles, home_interceptions, away_interceptions,
                home_blocks, away_blocks, home_clearances, away_clearances,
                home_keeper_saves, away_keeper_saves,
                home_duels_won, away_duels_won, home_ground_duels_won, away_ground_duels_won,
                home_aerial_duels_won, away_aerial_duels_won,
                home_successful_dribbles, away_successful_dribbles,
                home_corners, away_corners, home_offsides, away_offsides,
                home_fouls, away_fouls, home_yellow_cards, away_yellow_cards,
                home_red_cards, away_red_cards
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, summary_batch)
    
    conn.commit()
    print(f"Inserted {len(stats_batch)} stat records for {len(summary_batch)} matches")

if __name__ == "__main__":
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server;"
        "DATABASE=fussballDB;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    process_stats(connection_string)