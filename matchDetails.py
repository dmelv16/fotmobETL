import json
import pyodbc
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

def parse_match_details(raw_json: str) -> Dict[str, Any]:
    """Parse match details from raw JSON string."""
    data = json.loads(raw_json)
    
    # Handle both direct match_details and api_responses.match_details structure
    if 'match_details' in data:
        if 'api_responses' in data:
            match_details = data.get('api_responses', {}).get('match_details', {})
        else:
            match_details = data.get('match_details', {})
    else:
        match_details = data
    
    match_data = match_details.get('data', {})
    general = match_data.get('general', {})
    header = match_data.get('header', {})
    status = header.get('status', {})
    teams = header.get('teams', [])
    halfs = status.get('halfs', {})
    reason = status.get('reason', {})
    team_colors = general.get('teamColors', {})
    
    # Parse half times with custom date format
    def parse_half_time(time_str: str) -> Optional[datetime]:
        if time_str:
            try:
                return datetime.strptime(time_str, '%d.%m.%Y %H:%M:%S')
            except:
                return None
        return None
    
    # Parse events
    events = header.get('events', {})
    
    def parse_events(events_dict):
        """Parse events and flatten the structure."""
        parsed_events = []
        
        # Process home team goals
        home_goals = events_dict.get('homeTeamGoals', {})
        for player_goals in home_goals.values():
            for goal in player_goals:
                parsed_events.append({
                    'type': 'goal',
                    'is_home': True,
                    'time': goal.get('time'),
                    'time_str': goal.get('timeStr'),
                    'player_id': goal.get('player', {}).get('id'),
                    'player_name': goal.get('player', {}).get('name'),
                    'player_first_name': goal.get('firstName'),
                    'player_last_name': goal.get('lastName'),
                    'assist_str': goal.get('assistStr'),
                    'home_score': goal.get('homeScore'),
                    'away_score': goal.get('awayScore'),
                    'new_score_home': goal.get('newScore', [None, None])[0],
                    'new_score_away': goal.get('newScore', [None, None])[1] if len(goal.get('newScore', [])) > 1 else None,
                    'own_goal': goal.get('ownGoal'),
                    'penalty_shootout': goal.get('isPenaltyShootoutEvent'),
                    'goal_description': goal.get('goalDescription')
                })
        
        # Process away team goals
        away_goals = events_dict.get('awayTeamGoals', {})
        for player_goals in away_goals.values():
            for goal in player_goals:
                parsed_events.append({
                    'type': 'goal',
                    'is_home': False,
                    'time': goal.get('time'),
                    'time_str': goal.get('timeStr'),
                    'player_id': goal.get('player', {}).get('id'),
                    'player_name': goal.get('player', {}).get('name'),
                    'player_first_name': goal.get('firstName'),
                    'player_last_name': goal.get('lastName'),
                    'assist_str': goal.get('assistStr'),
                    'home_score': goal.get('homeScore'),
                    'away_score': goal.get('awayScore'),
                    'new_score_home': goal.get('newScore', [None, None])[0],
                    'new_score_away': goal.get('newScore', [None, None])[1] if len(goal.get('newScore', [])) > 1 else None,
                    'own_goal': goal.get('ownGoal'),
                    'penalty_shootout': goal.get('isPenaltyShootoutEvent'),
                    'goal_description': goal.get('goalDescription')
                })
        
        # Process red cards if needed
        # Add more event types as needed
        
        return parsed_events
    
    parsed_events = parse_events(events)
    
    # Parse content/matchFacts data
    content = match_data.get('content', {})
    match_facts = content.get('matchFacts', {})
    info_box = match_facts.get('infoBox', {})
    
    # Parse Q&A data
    qa_data = []
    for qa in match_facts.get('QAData', []):
        qa_data.append({
            'question': qa.get('question'),
            'answer': qa.get('answer')
        })
    
    return {
        'match_id': data.get('match_id'),
        'match_name': general.get('matchName'),
        'league_id': general.get('leagueId'),
        'league_name': general.get('leagueName'),
        'parent_league_id': general.get('parentLeagueId'),
        'coverage_level': general.get('coverageLevel'),
        'match_time_utc': general.get('matchTimeUTCDate'),
        'started': general.get('started'),
        'finished': general.get('finished'),
        'cancelled': status.get('cancelled'),
        'awarded': status.get('awarded'),
        'home_team_id': general.get('homeTeam', {}).get('id'),
        'home_team_name': general.get('homeTeam', {}).get('name'),
        'home_team_score': teams[0].get('score') if len(teams) > 0 else None,
        'away_team_id': general.get('awayTeam', {}).get('id'),
        'away_team_name': general.get('awayTeam', {}).get('name'),
        'away_team_score': teams[1].get('score') if len(teams) > 1 else None,
        'score_str': status.get('scoreStr'),
        'status_reason_short': reason.get('short'),
        'status_reason_long': reason.get('long'),
        'number_of_home_red_cards': status.get('numberOfHomeRedCards'),
        'number_of_away_red_cards': status.get('numberOfAwayRedCards'),
        'first_half_started': parse_half_time(halfs.get('firstHalfStarted')),
        'first_half_ended': parse_half_time(halfs.get('firstHalfEnded')),
        'second_half_started': parse_half_time(halfs.get('secondHalfStarted')),
        'second_half_ended': parse_half_time(halfs.get('secondHalfEnded')),
        'who_lost_on_penalties': status.get('whoLostOnPenalties'),
        'who_lost_on_aggregated': status.get('whoLostOnAggregated'),
        # Team colors
        'home_color_dark': team_colors.get('darkMode', {}).get('home'),
        'away_color_dark': team_colors.get('darkMode', {}).get('away'),
        'home_color_light': team_colors.get('lightMode', {}).get('home'),
        'away_color_light': team_colors.get('lightMode', {}).get('away'),
        'home_font_dark': team_colors.get('fontDarkMode', {}).get('home'),
        'away_font_dark': team_colors.get('fontDarkMode', {}).get('away'),
        'home_font_light': team_colors.get('fontLightMode', {}).get('home'),
        'away_font_light': team_colors.get('fontLightMode', {}).get('away'),
        'events': events_data,  # Updated events from the proper location
        # Content/MatchFacts data
        'stadium_name': info_box.get('Stadium', {}).get('name') if isinstance(info_box.get('Stadium'), dict) else None,
        'stadium_country': info_box.get('Stadium', {}).get('country') if isinstance(info_box.get('Stadium'), dict) else None,
        'attendance': info_box.get('Attendance'),
        'referee': info_box.get('Referee'),
        'tournament_id': info_box.get('Tournament', {}).get('id') if isinstance(info_box.get('Tournament'), dict) else None,
        'tournament_name': info_box.get('Tournament', {}).get('leagueName') if isinstance(info_box.get('Tournament'), dict) else None,
        'qa_data': qa_data
    }

def create_tables(connection):
    """Create match_details, match_team_colors, match_events, and match_facts tables if they don't exist."""
    cursor = connection.cursor()
    
    # Create match_details table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_details]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_details] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [source_id] INT,
                [match_id] INT NOT NULL,
                [match_name] NVARCHAR(500),
                [league_id] INT,
                [league_name] NVARCHAR(200),
                [parent_league_id] INT,
                [coverage_level] NVARCHAR(50),
                [match_time_utc] DATETIME2,
                [started] BIT,
                [finished] BIT,
                [cancelled] BIT,
                [awarded] BIT,
                [home_team_id] INT,
                [home_team_name] NVARCHAR(200),
                [home_team_score] INT,
                [away_team_id] INT,
                [away_team_name] NVARCHAR(200),
                [away_team_score] INT,
                [score_str] NVARCHAR(20),
                [status_reason_short] NVARCHAR(50),
                [status_reason_long] NVARCHAR(100),
                [number_of_home_red_cards] INT,
                [number_of_away_red_cards] INT,
                [first_half_started] DATETIME2,
                [first_half_ended] DATETIME2,
                [second_half_started] DATETIME2,
                [second_half_ended] DATETIME2,
                [who_lost_on_penalties] NVARCHAR(200),
                [who_lost_on_aggregated] NVARCHAR(200),
                [created_at] DATETIME DEFAULT GETDATE(),
                [updated_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create match_team_colors table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_team_colors]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_team_colors] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [home_color_dark] NVARCHAR(20),
                [away_color_dark] NVARCHAR(20),
                [home_color_light] NVARCHAR(20),
                [away_color_light] NVARCHAR(20),
                [home_font_dark] NVARCHAR(50),
                [away_font_dark] NVARCHAR(50),
                [home_font_light] NVARCHAR(50),
                [away_font_light] NVARCHAR(50),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create match_events table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_events]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_events] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [event_type] NVARCHAR(50),
                [is_home_team] BIT,
                [time_minute] INT,
                [time_str] NVARCHAR(20),
                [player_id] INT,
                [player_name] NVARCHAR(200),
                [player_first_name] NVARCHAR(100),
                [player_last_name] NVARCHAR(100),
                [assist_player] NVARCHAR(200),
                [home_score_before] INT,
                [away_score_before] INT,
                [home_score_after] INT,
                [away_score_after] INT,
                [own_goal] BIT,
                [penalty_shootout] BIT,
                [goal_description] NVARCHAR(500),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create match_facts table
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_facts]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_facts] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [stadium_name] NVARCHAR(200),
                [stadium_country] NVARCHAR(100),
                [attendance] INT,
                [referee] NVARCHAR(200),
                [tournament_id] INT,
                [tournament_name] NVARCHAR(200),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    # Create match_qa table for Q&A data
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[match_qa]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[match_qa] (
                [id] INT IDENTITY(1,1) PRIMARY KEY,
                [match_id] INT NOT NULL,
                [question] NVARCHAR(MAX),
                [answer] NVARCHAR(MAX),
                [created_at] DATETIME DEFAULT GETDATE()
            );
        END
    """)
    
    connection.commit()

def process_match_details(connection_string: str, batch_size: int = 100):
    """Process match details from fotmob_raw_data table."""
    conn = pyodbc.connect(connection_string)
    
    # Create tables if they don't exist
    create_tables(conn)
    
    cursor = conn.cursor()
    
    # Get existing match_ids to avoid duplicates
    cursor.execute("SELECT match_id FROM [dbo].[match_details]")
    existing_matches = set(row[0] for row in cursor.fetchall())
    
    # Fetch raw data
    query = """
        SELECT id, match_id, raw_json
        FROM [dbo].[fotmob_raw_data]
        WHERE data_type = 'match_details'
        ORDER BY id
    """
    
    cursor.execute(query)
    
    match_details_batch = []
    team_colors_batch = []
    events_batch = []
    match_facts_batch = []
    qa_batch = []
    
    for row in cursor.fetchall():
        source_id, match_id, raw_json = row
        
        if match_id in existing_matches:
            continue
        
        try:
            parsed = parse_match_details(raw_json)
            
            # Prepare match details record
            match_details_batch.append((
                source_id,
                parsed['match_id'],
                parsed['match_name'],
                parsed['league_id'],
                parsed['league_name'],
                parsed['parent_league_id'],
                parsed['coverage_level'],
                parsed['match_time_utc'],
                parsed['started'],
                parsed['finished'],
                parsed['cancelled'],
                parsed['awarded'],
                parsed['home_team_id'],
                parsed['home_team_name'],
                parsed['home_team_score'],
                parsed['away_team_id'],
                parsed['away_team_name'],
                parsed['away_team_score'],
                parsed['score_str'],
                parsed['status_reason_short'],
                parsed['status_reason_long'],
                parsed['number_of_home_red_cards'],
                parsed['number_of_away_red_cards'],
                parsed['first_half_started'],
                parsed['first_half_ended'],
                parsed['second_half_started'],
                parsed['second_half_ended'],
                parsed['who_lost_on_penalties'],
                parsed['who_lost_on_aggregated']
            ))
            
            # Prepare team colors record
            team_colors_batch.append((
                parsed['match_id'],
                parsed['home_color_dark'],
                parsed['away_color_dark'],
                parsed['home_color_light'],
                parsed['away_color_light'],
                parsed['home_font_dark'],
                parsed['away_font_dark'],
                parsed['home_font_light'],
                parsed['away_font_light']
            ))
            
            # Prepare events records
            for event in parsed.get('events', []):
                events_batch.append((
                    parsed['match_id'],
                    event['type'],
                    event['is_home'],
                    event['time'],
                    event['time_str'],
                    event['player_id'],
                    event['player_name'],
                    event['player_first_name'],
                    event['player_last_name'],
                    event['assist_str'],
                    event['home_score'],
                    event['away_score'],
                    event['new_score_home'],
                    event['new_score_away'],
                    event['own_goal'],
                    event['penalty_shootout'],
                    event['goal_description']
                ))
            
            # Prepare match facts record
            match_facts_batch.append((
                parsed['match_id'],
                parsed['stadium_name'],
                parsed['stadium_country'],
                parsed['attendance'],
                parsed['referee'],
                parsed['tournament_id'],
                parsed['tournament_name']
            ))
            
            # Prepare Q&A records
            for qa in parsed.get('qa_data', []):
                if qa.get('question') and qa.get('answer'):
                    qa_batch.append((
                        parsed['match_id'],
                        qa['question'],
                        qa['answer']
                    ))
            
            # Insert in batches
            if len(match_details_batch) >= batch_size:
                insert_batch(conn, match_details_batch, team_colors_batch, events_batch, match_facts_batch, qa_batch)
                match_details_batch = []
                team_colors_batch = []
                events_batch = []
                match_facts_batch = []
                qa_batch = []
                
        except Exception as e:
            print(f"Error processing match_id {match_id}: {e}")
            continue
    
    # Insert remaining records
    if match_details_batch:
        insert_batch(conn, match_details_batch, team_colors_batch, events_batch, match_facts_batch, qa_batch)
    
    conn.close()

def insert_batch(conn, match_details_batch, team_colors_batch, events_batch, match_facts_batch, qa_batch):
    """Insert batch of records into database."""
    cursor = conn.cursor()
    
    # Insert match details
    insert_query = """
        INSERT INTO [dbo].[match_details] (
            source_id, match_id, match_name, league_id, league_name,
            parent_league_id, coverage_level, match_time_utc, started, finished,
            cancelled, awarded, home_team_id, home_team_name, home_team_score,
            away_team_id, away_team_name, away_team_score, score_str,
            status_reason_short, status_reason_long, number_of_home_red_cards,
            number_of_away_red_cards, first_half_started, first_half_ended,
            second_half_started, second_half_ended, who_lost_on_penalties,
            who_lost_on_aggregated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(insert_query, match_details_batch)
    
    # Insert team colors
    colors_query = """
        INSERT INTO [dbo].[match_team_colors] (
            match_id, home_color_dark, away_color_dark, home_color_light,
            away_color_light, home_font_dark, away_font_dark, home_font_light,
            away_font_light
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(colors_query, team_colors_batch)
    
    # Insert events
    if events_batch:
        events_query = """
            INSERT INTO [dbo].[match_events] (
                match_id, event_type, is_home_team, time_minute, time_str,
                player_id, player_name, player_first_name, player_last_name,
                assist_player, home_score_before, away_score_before,
                home_score_after, away_score_after, own_goal, penalty_shootout,
                goal_description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(events_query, events_batch)
    
    # Insert match facts
    if match_facts_batch:
        facts_query = """
            INSERT INTO [dbo].[match_facts] (
                match_id, stadium_name, stadium_country, attendance,
                referee, tournament_id, tournament_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(facts_query, match_facts_batch)
    
    # Insert Q&A data
    if qa_batch:
        qa_query = """
            INSERT INTO [dbo].[match_qa] (
                match_id, question, answer
            ) VALUES (?, ?, ?)
        """
        cursor.executemany(qa_query, qa_batch)
    
    conn.commit()
    print(f"Inserted {len(match_details_batch)} match records with {len(events_batch)} events and {len(qa_batch)} Q&A items")

if __name__ == "__main__":
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server;"
        "DATABASE=fussballDB;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    process_match_details(connection_string)