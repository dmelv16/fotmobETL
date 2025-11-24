import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from config.settings import (
    ELO_K_FACTOR_LEAGUE, ELO_K_FACTOR_CUP, ELO_K_FACTOR_EUROPEAN,
    ELO_HOME_ADVANTAGE, ELO_BASE_RATING
)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*SQLAlchemy.*')

try:
    from config.settings import REVERSION_RATE
except ImportError:
    REVERSION_RATE = 0.33

from config.league_mappings import is_european_competition

logger = logging.getLogger(__name__)

class EloRatingSystem:
    """
    Elo rating system for teams.
    Tracks ratings match-by-match and provides current ratings.
    """
    
    def __init__(self, connection, data_loader, league_registry=None):
        self.conn = connection
        self.data_loader = data_loader
        self.league_registry = league_registry
        self.team_ratings = {}  # {team_id: current_elo}
        self.last_processed_season = None
    
    def initialize_season_ratings(self, season_year: int, league_id: int = None, force_reload: bool = False):
        """
        Initialize team ratings for a season.
        Uses previous season's final ratings or defaults to base rating.
        
        Args:
            season_year: Season to initialize
            league_id: Optional league filter
            force_reload: If True, always reload from DB (default: carry forward if sequential)
        """
        # If we're processing sequentially and already have ratings, just ensure new teams are added
        if not force_reload and self.last_processed_season == season_year - 1 and len(self.team_ratings) > 0:
            logger.info(f"Carrying forward ratings from season {season_year - 1}")
            self._add_new_teams_for_season(season_year, league_id)
            self._apply_season_reversion()
            return
        
        # Otherwise, load from database
        logger.info(f"Initializing ratings for season {season_year} from database")
        
        if league_id:
            query = """
                SELECT DISTINCT team_id, team_name
                FROM (
                    SELECT home_team_id as team_id, home_team_name as team_name
                    FROM [dbo].[match_details]
                    WHERE YEAR(match_time_utc) = ? AND parent_league_id = ?
                    UNION
                    SELECT away_team_id as team_id, away_team_name as team_name
                    FROM [dbo].[match_details]
                    WHERE YEAR(match_time_utc) = ? AND parent_league_id = ?
                ) teams
            """
            params = [season_year, league_id, season_year, league_id]
        else:
            query = """
                SELECT DISTINCT team_id, team_name
                FROM (
                    SELECT home_team_id as team_id, home_team_name as team_name
                    FROM [dbo].[match_details]
                    WHERE YEAR(match_time_utc) = ?
                    UNION
                    SELECT away_team_id as team_id, away_team_name as team_name
                    FROM [dbo].[match_details]
                    WHERE YEAR(match_time_utc) = ?
                ) teams
            """
            params = [season_year, season_year]
        
        teams_df = pd.read_sql(query, self.conn, params=params)
        
        # Clear existing ratings
        self.team_ratings = {}
        
        # Get previous season's final ratings
        prev_season = season_year - 1
        cursor = self.conn.cursor()
        
        # Calculate retention rate once
        retention_rate = 1.0 - REVERSION_RATE
        
        for _, row in teams_df.iterrows():
            team_id = row['team_id']
            
            # FIXED: Add tuple for parameters
            cursor.execute("""
                SELECT TOP 1 elo_after
                FROM [dbo].[team_elo_history]
                WHERE team_id = ? AND season_year = ?
                ORDER BY match_date DESC
            """, (team_id, prev_season))
            
            prev_rating = cursor.fetchone()
            
            if prev_rating:
                # Regress towards mean
                self.team_ratings[team_id] = (prev_rating[0] * retention_rate) + (ELO_BASE_RATING * REVERSION_RATE)
            else:
                # New team or no history
                self.team_ratings[team_id] = ELO_BASE_RATING
    
    def _add_new_teams_for_season(self, season_year: int, league_id: int = None):
        """Add any new teams that appear in this season but weren't in our ratings dict."""
        if league_id:
            query = """
                SELECT DISTINCT team_id
                FROM (
                    SELECT home_team_id as team_id FROM [dbo].[match_details]
                    WHERE YEAR(match_time_utc) = ? AND parent_league_id = ?
                    UNION
                    SELECT away_team_id as team_id FROM [dbo].[match_details]
                    WHERE YEAR(match_time_utc) = ? AND parent_league_id = ?
                ) teams
            """
            params = [season_year, league_id, season_year, league_id]
        else:
            query = """
                SELECT DISTINCT team_id
                FROM (
                    SELECT home_team_id as team_id FROM [dbo].[match_details]
                    WHERE YEAR(match_time_utc) = ?
                    UNION
                    SELECT away_team_id as team_id FROM [dbo].[match_details]
                    WHERE YEAR(match_time_utc) = ?
                ) teams
            """
            params = [season_year, season_year]
        
        teams_df = pd.read_sql(query, self.conn, params=params)
        
        cursor = self.conn.cursor()
        prev_season = season_year - 1
        
        for _, row in teams_df.iterrows():
            team_id = row['team_id']
            if team_id not in self.team_ratings:
                # Check if they have history from previous season
                cursor.execute("""
                    SELECT TOP 1 elo_after
                    FROM [dbo].[team_elo_history]
                    WHERE team_id = ? AND season_year = ?
                    ORDER BY match_date DESC
                """, (team_id, prev_season))
                
                prev_rating = cursor.fetchone()
                
                if prev_rating:
                    retention_rate = 1.0 - REVERSION_RATE
                    self.team_ratings[team_id] = (prev_rating[0] * retention_rate) + (ELO_BASE_RATING * REVERSION_RATE)
                else:
                    self.team_ratings[team_id] = ELO_BASE_RATING
    
    def _apply_season_reversion(self):
        """Apply reversion to mean for all existing teams."""
        retention_rate = 1.0 - REVERSION_RATE
        for team_id in self.team_ratings:
            current_rating = self.team_ratings[team_id]
            self.team_ratings[team_id] = (current_rating * retention_rate) + (ELO_BASE_RATING * REVERSION_RATE)
    
    def get_k_factor(self, competition_id: int) -> float:
        """Determine K factor based on competition importance."""
        if self.league_registry:
            comp_info = self.league_registry.get_league_info(competition_id)
            comp_type = comp_info.get('competition_type')
            
            if comp_type == 'continental':
                return ELO_K_FACTOR_EUROPEAN
            elif comp_type == 'cup':
                return ELO_K_FACTOR_CUP
        
        if is_european_competition(competition_id):
            return ELO_K_FACTOR_EUROPEAN
        
        return ELO_K_FACTOR_LEAGUE
    
    def expected_result(self, rating_a: float, rating_b: float, is_home: bool = False) -> float:
        """Calculate expected result probability (0-1)."""
        rating_diff = rating_b - rating_a
        if is_home:
            rating_diff -= ELO_HOME_ADVANTAGE
        
        return 1 / (1 + 10 ** (rating_diff / 400))
    
    def update_ratings(self, home_team_id: int, away_team_id: int,
                      home_score: int, away_score: int, k_factor: float) -> Tuple[float, float]:
        """
        Update Elo ratings based on match result.
        """
        home_rating = self.team_ratings.get(home_team_id, ELO_BASE_RATING)
        away_rating = self.team_ratings.get(away_team_id, ELO_BASE_RATING)
        
        if home_score > away_score:
            actual_result = 1.0
        elif home_score < away_score:
            actual_result = 0.0
        else:
            actual_result = 0.5
        
        expected = self.expected_result(home_rating, away_rating, is_home=True)
        
        goal_diff = abs(home_score - away_score)
        if goal_diff > 0:
            mov_multiplier = 1 + np.log(goal_diff)
        else:
            mov_multiplier = 1.0
            
        current_k = k_factor * mov_multiplier
        
        home_change = current_k * (actual_result - expected)
        away_change = current_k * ((1 - actual_result) - (1 - expected))
        
        self.team_ratings[home_team_id] = home_rating + home_change
        self.team_ratings[away_team_id] = away_rating + away_change
        
        return home_change, away_change
    
    def process_season(self, season_year: int, league_id: int = None,
                      save_to_db: bool = True):
        """
        Process all matches for a season and update Elo ratings.
        
        IMPORTANT: Matches must be processed in chronological order.
        """
        # Only force reload if jumping to non-sequential season
        force_reload = (self.last_processed_season is not None and 
                       self.last_processed_season != season_year - 1)
        
        self.initialize_season_ratings(season_year, league_id, force_reload=force_reload)
        
        matches_df = self.data_loader.get_all_matches(season_year, league_id)
        
        if len(matches_df) == 0:
            logger.warning(f"No matches found for season {season_year}")
            return
        
        # CRITICAL: Ensure matches are sorted chronologically
        if 'match_time_utc' in matches_df.columns:
            matches_df = matches_df.sort_values('match_time_utc')
            logger.info(f"Processing {len(matches_df)} matches for season {season_year} in chronological order")
        else:
            logger.warning("match_time_utc not found - matches may not be in chronological order!")
        
        cursor = self.conn.cursor()
        
        batch_data = []
        BATCH_SIZE = 1000
        matches_processed = 0
        
        for _, match in matches_df.iterrows():
            # Check for nulls in critical fields
            if pd.isna(match['match_id']) or pd.isna(match['home_team_id']) or pd.isna(match['away_team_id']):
                continue

            # Safe casting
            match_id = int(match['match_id'])
            home_team_id = int(match['home_team_id'])
            away_team_id = int(match['away_team_id'])
            
            # Handle scores safely
            if pd.isna(match['home_team_score']) or pd.isna(match['away_team_score']):
                continue
            home_score = int(match['home_team_score'])
            away_score = int(match['away_team_score'])
            
            # Handle potential pandas Timestamp
            match_date = match['match_time_utc'].to_pydatetime() if hasattr(match['match_time_utc'], 'to_pydatetime') else match['match_time_utc']
            
            # Safe competition ID handling
            if 'parent_league_id' in match and pd.notnull(match['parent_league_id']):
                competition_id = int(match['parent_league_id'])
            elif pd.notnull(match['league_id']):
                competition_id = int(match['league_id'])
            else:
                continue

            # Get ratings before match
            home_elo_before = float(self.team_ratings.get(home_team_id, ELO_BASE_RATING))
            away_elo_before = float(self.team_ratings.get(away_team_id, ELO_BASE_RATING))
            
            # Calculate expected result
            expected = float(self.expected_result(home_elo_before, away_elo_before, is_home=True))
            
            # Determine actual result
            if home_score > away_score:
                actual = 1.0
            elif home_score < away_score:
                actual = 0.0
            else:
                actual = 0.5
            
            # Get base K-factor
            k_factor = float(self.get_k_factor(competition_id))
            
            # Update ratings (this modifies self.team_ratings)
            home_change, away_change = self.update_ratings(
                home_team_id, away_team_id, home_score, away_score, k_factor
            )
            
            home_change = float(home_change)
            away_change = float(away_change)
            
            matches_processed += 1
            
            if save_to_db:
                # Home team record
                batch_data.append((
                    home_team_id, match_id, season_year, match_date,
                    home_elo_before, away_team_id, away_elo_before, 1,
                    competition_id, k_factor, actual, expected,
                    home_elo_before + home_change, home_change
                ))
                
                # Away team record
                batch_data.append((
                    away_team_id, match_id, season_year, match_date,
                    away_elo_before, home_team_id, home_elo_before, 0,
                    competition_id, k_factor, 1.0 - actual, 1.0 - expected,
                    away_elo_before + away_change, away_change
                ))
                
                if len(batch_data) >= BATCH_SIZE:
                    self._save_batch(cursor, batch_data)
                    batch_data = []
        
        if save_to_db and batch_data:
            self._save_batch(cursor, batch_data)
            self.conn.commit()
            logger.info(f"Saved Elo history for {matches_processed} matches in season {season_year}")
        
        # Mark this season as processed
        self.last_processed_season = season_year
            
    def _save_batch(self, cursor, batch_data):
        """Helper to execute batch inserts."""
        query = """
            INSERT INTO [dbo].[team_elo_history] (
                team_id, match_id, season_year, match_date,
                elo_before, opponent_id, opponent_elo_before, is_home,
                competition_id, k_factor, actual_result, expected_result,
                elo_after, elo_change
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, batch_data)

    def get_team_rating(self, team_id: int, as_of_date: Optional[str] = None) -> float:
        """
        Get team's current Elo rating, optionally as of a specific date.
        
        Args:
            team_id: Team identifier
            as_of_date: Get rating as of this date (YYYY-MM-DD format)
        """
        team_id = int(team_id)
        
        if as_of_date:
            try:
                query = """
                    SELECT TOP 1 elo_after
                    FROM [dbo].[team_elo_history]
                    WHERE team_id = ?
                    AND match_date <= ?
                    ORDER BY match_date DESC, match_id DESC
                """
                cursor = self.conn.cursor()
                # FIXED: Add tuple
                cursor.execute(query, (team_id, as_of_date))
                row = cursor.fetchone()
                
                if row:
                    return float(row[0])
                else:
                    return float(ELO_BASE_RATING)
            except Exception as e:
                logger.warning(f"Could not retrieve historical Elo for team {team_id}: {e}")
                # Fall through to current rating
        
        # Current rating from in-memory dict
        return float(self.team_ratings.get(team_id, ELO_BASE_RATING))
    
    def get_all_ratings(self) -> Dict[int, float]:
        """Get all current team ratings."""
        return self.team_ratings.copy()