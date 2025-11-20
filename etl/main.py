import json
import pyodbc
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction_log.txt'),
        logging.StreamHandler()
    ]
)

def create_tracking_tables(connection):
    """Create tables to track data availability by league and season."""
    cursor = connection.cursor()
    
    # Table to track which leagues/seasons have which data types
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[data_availability]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[data_availability] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [league_id] INT,
                [league_name] NVARCHAR(200),
                [season_year] INT,
                [data_type] NVARCHAR(50),
                [total_matches] INT,
                [matches_with_data] INT,
                [availability_percentage] FLOAT,
                [sample_fields_found] NVARCHAR(MAX),
                [first_seen] DATETIME,
                [last_updated] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Table to track extraction status
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[extraction_status]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[extraction_status] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT,
                [league_id] INT,
                [season_year] INT,
                [has_match_details] BIT DEFAULT 0,
                [has_lineups] BIT DEFAULT 0,
                [has_stats] BIT DEFAULT 0,
                [has_momentum] BIT DEFAULT 0,
                [has_player_stats] BIT DEFAULT 0,
                [has_shotmap] BIT DEFAULT 0,
                [has_events] BIT DEFAULT 0,
                [extraction_time] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    connection.commit()

def check_data_exists(json_data: Dict, path: str) -> bool:
    """Check if a data path exists in the JSON."""
    try:
        keys = path.split('.')
        current = json_data
        for key in keys:
            if isinstance(current, dict):
                if key not in current:
                    return False
                current = current[key]
            else:
                return False
        return current is not None and (not isinstance(current, (dict, list)) or len(current) > 0)
    except:
        return False

def analyze_json_structure(raw_json: str, match_id: int) -> Dict[str, bool]:
    """Analyze what data is available in the JSON without extracting."""
    try:
        data = json.loads(raw_json)
        
        # Navigate to the main content
        if 'api_responses' in data:
            base_path = data.get('api_responses', {}).get('match_details', {}).get('data', {})
        else:
            base_path = data.get('match_details', {}).get('data', {})
        
        content = base_path.get('content', {})
        
        # Check what data exists
        structure = {
            'has_match_details': bool(base_path.get('general') or base_path.get('header')),
            'has_lineups': check_data_exists(content, 'lineup.homeTeam.starters') or 
                          check_data_exists(content, 'lineup.awayTeam.starters'),
            'has_stats': check_data_exists(content, 'stats.Periods'),
            'has_momentum': check_data_exists(content, 'momentum.main.data'),
            'has_player_stats': bool(content.get('playerStats')),
            'has_shotmap': check_data_exists(content, 'shotmap.shots'),
            'has_events': check_data_exists(content, 'matchFacts.events.events')
        }
        
        return structure
    except Exception as e:
        logging.error(f"Error analyzing JSON structure for match {match_id}: {str(e)}")
        return {k: False for k in ['has_match_details', 'has_lineups', 'has_stats', 
                                   'has_momentum', 'has_player_stats', 'has_shotmap', 'has_events']}

def process_match_selective(conn, match_id: int, raw_json: str, json_structure: Dict) -> Dict[str, bool]:
    """Process only the data types that exist in the JSON."""
    results = {}
    
    # Process match details if available
    if json_structure.get('has_match_details'):
        try:
            from matchDetails import process_single_match
            result = process_single_match(conn, match_id, raw_json)
            results['match_details'] = result
        except Exception as e:
            logging.error(f"Error processing match_details for {match_id}: {e}")
            results['match_details'] = False
    
    # Process lineups if available
    if json_structure.get('has_lineups'):
        try:
            from lineups import process_single_match
            result = process_single_match(conn, match_id, raw_json)
            results['lineups'] = result
        except Exception as e:
            logging.error(f"Error processing lineups for {match_id}: {e}")
            results['lineups'] = False
    
    # Process stats if available
    if json_structure.get('has_stats'):
        try:
            from matchStats import process_single_match
            result = process_single_match(conn, match_id, raw_json)
            results['stats'] = result
        except Exception as e:
            logging.error(f"Error processing stats for {match_id}: {e}")
            results['stats'] = False
    
    # Process momentum if available
    if json_structure.get('has_momentum'):
        try:
            from momentum import process_single_match
            result = process_single_match(conn, match_id, raw_json)
            results['momentum'] = result
        except Exception as e:
            logging.error(f"Error processing momentum for {match_id}: {e}")
            results['momentum'] = False
    
    # Process player stats if available
    if json_structure.get('has_player_stats'):
        try:
            from playerStats import process_single_match
            result = process_single_match(conn, match_id, raw_json)
            results['player_stats'] = result
        except Exception as e:
            logging.error(f"Error processing player_stats for {match_id}: {e}")
            results['player_stats'] = False
    
    # Process shotmap if available
    if json_structure.get('has_shotmap'):
        try:
            from shotMap import process_single_match
            result = process_single_match(conn, match_id, raw_json)
            results['shotmap'] = result
        except Exception as e:
            logging.error(f"Error processing shotmap for {match_id}: {e}")
            results['shotmap'] = False
    
    return results

def update_data_availability(conn, league_id: int, league_name: str, season_year: int):
    """Update data availability statistics for a league/season."""
    cursor = conn.cursor()
    
    # Get all extraction statuses for this league/season
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CAST(has_match_details AS INT)) as with_match_details,
            SUM(CAST(has_lineups AS INT)) as with_lineups,
            SUM(CAST(has_stats AS INT)) as with_stats,
            SUM(CAST(has_momentum AS INT)) as with_momentum,
            SUM(CAST(has_player_stats AS INT)) as with_player_stats,
            SUM(CAST(has_shotmap AS INT)) as with_shotmap,
            SUM(CAST(has_events AS INT)) as with_events
        FROM [dbo].[extraction_status]
        WHERE league_id = ? AND season_year = ?
    """, league_id, season_year)
    
    row = cursor.fetchone()
    if row and row[0] > 0:
        total = row[0]
        data_types = [
            ('match_details', row[1]),
            ('lineups', row[2]),
            ('stats', row[3]),
            ('momentum', row[4]),
            ('player_stats', row[5]),
            ('shotmap', row[6]),
            ('events', row[7])
        ]
        
        for data_type, count in data_types:
            percentage = (count / total * 100) if total > 0 else 0
            
            cursor.execute("""
                IF EXISTS (SELECT 1 FROM [dbo].[data_availability] 
                          WHERE league_id = ? AND season_year = ? AND data_type = ?)
                    UPDATE [dbo].[data_availability]
                    SET matches_with_data = ?, total_matches = ?, 
                        availability_percentage = ?, last_updated = GETDATE()
                    WHERE league_id = ? AND season_year = ? AND data_type = ?
                ELSE
                    INSERT INTO [dbo].[data_availability]
                    (league_id, league_name, season_year, data_type, total_matches, 
                     matches_with_data, availability_percentage, first_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, league_id, season_year, data_type, count, total, percentage,
                 league_id, season_year, data_type, league_id, league_name,
                 season_year, data_type, total, count, percentage)
    
    conn.commit()

def main(connection_string: str, batch_size: int = 100):
    """Main extraction orchestration function."""
    conn = pyodbc.connect(connection_string)
    
    # Create tracking tables
    logging.info("Creating tracking tables...")
    create_tracking_tables(conn)
    
    # Get matches to process - either not in extraction_status OR has incomplete data
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.id, f.match_id, f.league_id, f.league_name, f.season_year, f.raw_json
        FROM [dbo].[fotmob_raw_data] f
        LEFT JOIN [dbo].[extraction_status] e ON f.match_id = e.match_id
        WHERE f.data_type = 'match_details' 
        AND (e.match_id IS NULL OR e.has_match_details = 0)
        ORDER BY f.league_id, f.season_year, f.match_id
    """)
    
    matches = cursor.fetchall()
    total_matches = len(matches)
    logging.info(f"Found {total_matches} matches to process")
    
    if total_matches == 0:
        logging.info("No new matches to process")
        conn.close()
        return
    
    current_league = None
    current_season = None
    processed = 0
    
    for source_id, match_id, league_id, league_name, season_year, raw_json in matches:
        try:
            # Update stats when league/season changes
            if current_league != league_id or current_season != season_year:
                if current_league is not None:
                    update_data_availability(conn, current_league, league_name, current_season)
                current_league = league_id
                current_season = season_year
                logging.info(f"Processing league {league_name} ({league_id}), season {season_year}")
            
            # First, analyze what data exists in this JSON
            json_structure = analyze_json_structure(raw_json, match_id)
            
            # Log what we found
            available_data = [k.replace('has_', '') for k, v in json_structure.items() if v]
            if available_data:
                logging.debug(f"Match {match_id} has: {', '.join(available_data)}")
            
            # Process only the data that exists
            results = process_match_selective(conn, match_id, raw_json, json_structure)
            
            # Update extraction status based on what was actually processed
            # Check if record exists
            cursor.execute("SELECT id FROM [dbo].[extraction_status] WHERE match_id = ?", match_id)
            exists = cursor.fetchone()
            
            if exists:
                # Update existing record - set flags to 1 only if processing was successful
                cursor.execute("""
                    UPDATE [dbo].[extraction_status] 
                    SET has_match_details = CASE WHEN ? = 1 THEN 1 ELSE has_match_details END,
                        has_lineups = CASE WHEN ? = 1 THEN 1 ELSE has_lineups END,
                        has_stats = CASE WHEN ? = 1 THEN 1 ELSE has_stats END,
                        has_momentum = CASE WHEN ? = 1 THEN 1 ELSE has_momentum END,
                        has_player_stats = CASE WHEN ? = 1 THEN 1 ELSE has_player_stats END,
                        has_shotmap = CASE WHEN ? = 1 THEN 1 ELSE has_shotmap END,
                        extraction_time = GETDATE()
                    WHERE match_id = ?
                """, 
                     1 if results.get('match_details', False) else 0,
                     1 if results.get('lineups', False) else 0,
                     1 if results.get('stats', False) else 0,
                     1 if results.get('momentum', False) else 0,
                     1 if results.get('player_stats', False) else 0,
                     1 if results.get('shotmap', False) else 0,
                     match_id)
            else:
                # Insert new record with actual success flags
                cursor.execute("""
                    INSERT INTO [dbo].[extraction_status] 
                    (match_id, league_id, season_year, has_match_details, has_lineups, 
                     has_stats, has_momentum, has_player_stats, has_shotmap, has_events)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, match_id, league_id, season_year,
                     1 if results.get('match_details', False) else 0,
                     1 if results.get('lineups', False) else 0,
                     1 if results.get('stats', False) else 0,
                     1 if results.get('momentum', False) else 0,
                     1 if results.get('player_stats', False) else 0,
                     1 if results.get('shotmap', False) else 0,
                     json_structure.get('has_events', False))
            conn.commit()
            
            processed += 1
            
            # Progress update
            if processed % batch_size == 0:
                logging.info(f"Processed {processed}/{total_matches} matches ({processed/total_matches*100:.1f}%)")
                
        except Exception as e:
            logging.error(f"Error processing match {match_id}: {str(e)}")
            # Still record that we attempted this match
            cursor.execute("""
                INSERT INTO [dbo].[extraction_status] 
                (match_id, league_id, season_year)
                VALUES (?, ?, ?)
            """, match_id, league_id, season_year)
            conn.commit()
            continue
    
    # Final availability update
    if current_league is not None:
        update_data_availability(conn, current_league, league_name, current_season)
    
    # Print summary
    logging.info("\n=== Extraction Complete ===")
    logging.info(f"Total matches processed: {processed}")
    
    # Show data availability by league
    cursor.execute("""
        SELECT 
            league_name,
            season_year,
            data_type,
            availability_percentage
        FROM [dbo].[data_availability]
        WHERE availability_percentage > 0
        ORDER BY league_name, season_year, availability_percentage DESC
    """)
    
    current_league_print = None
    for row in cursor.fetchall():
        if current_league_print != row[0]:
            logging.info(f"\n{row[0]} - Season {row[1]}:")
            current_league_print = row[0]
        logging.info(f"  {row[2]}: {row[3]:.1f}%")
    
    conn.close()

if __name__ == "__main__":
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=fussballDB;"
        "Trusted_Connection=yes;"
    )
    
    main(connection_string)