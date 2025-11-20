import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from config.settings import (
    ELO_K_FACTOR_LEAGUE, ELO_K_FACTOR_CUP, ELO_K_FACTOR_EUROPEAN,
    ELO_HOME_ADVANTAGE, ELO_BASE_RATING
)
from config.league_mappings import is_european_competition

logger = logging.getLogger(__name__)

class EloRatingSystem:
    """
    Elo rating system for teams.
    Tracks ratings match-by-match and provides current ratings.
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader
        self.team_ratings = {}  # {team_id: current_elo}
    
    def initialize_season_ratings(self, season_year: int, league_id: int = None):
        """
        Initialize team ratings for a season.
        Uses previous season's final ratings or defaults to base rating.
        """
        # Get teams for this season
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
        if league_id:
            query = query.replace("UNION", "AND league_id = ? UNION").replace(
                "teams", "AND league_id = ? ) teams"
            )
            params = [season_year, league_id, season_year, league_id]
        
        teams_df = pd.read_sql(query, self.conn, params=params)
        
        # Try to get previous season's final ratings
        prev_season = season_year - 1
        cursor = self.conn.cursor()
        
        for _, row in teams_df.iterrows():
            team_id = row['team_id']
            
            # Check if we have a rating from previous season
            cursor.execute("""
                SELECT TOP 1 elo_after
                FROM [dbo].[team_elo_history]
                WHERE team_id = ? AND season_year = ?
                ORDER BY match_date DESC
            """, team_id, prev_season)
            
            prev_rating = cursor.fetchone()
            
            if prev_rating:
                # Regress towards mean slightly (33% regression)
                self.team_ratings[team_id] = prev_rating[0] * 0.67 + ELO_BASE_RATING * 0.33
            else:
                self.team_ratings[team_id] = ELO_BASE_RATING
        
        logger.info(f"Initialized {len(self.team_ratings)} team ratings for season {season_year}")
    
    def get_k_factor(self, competition_id: int) -> float:
        """Determine K factor based on competition importance."""
        if is_european_competition(competition_id):
            return ELO_K_FACTOR_EUROPEAN
        # Could add logic to detect cups vs league
        return ELO_K_FACTOR_LEAGUE
    
    def expected_result(self, rating_a: float, rating_b: float, is_home: bool = False) -> float:
        """
        Calculate expected result for team A against team B.
        Returns probability of win (0-1).
        """
        rating_diff = rating_b - rating_a
        if is_home:
            rating_diff -= ELO_HOME_ADVANTAGE
        
        expected = 1 / (1 + 10 ** (rating_diff / 400))
        return expected
    
    def update_ratings(self, home_team_id: int, away_team_id: int,
                      home_score: int, away_score: int, k_factor: float) -> Tuple[float, float]:
        """
        Update Elo ratings based on match result.
        Returns: (home_elo_change, away_elo_change)
        """
        home_rating = self.team_ratings.get(home_team_id, ELO_BASE_RATING)
        away_rating = self.team_ratings.get(away_team_id, ELO_BASE_RATING)
        
        # Determine actual result
        if home_score > away_score:
            actual_result = 1.0
        elif home_score < away_score:
            actual_result = 0.0
        else:
            actual_result = 0.5
        
        # Calculate expected result
        expected = self.expected_result(home_rating, away_rating, is_home=True)
        
        # Update ratings
        home_change = k_factor * (actual_result - expected)
        away_change = k_factor * ((1 - actual_result) - (1 - expected))
        
        self.team_ratings[home_team_id] = home_rating + home_change
        self.team_ratings[away_team_id] = away_rating + away_change
        
        return home_change, away_change
    
    def process_season(self, season_year: int, league_id: int = None,
                      save_to_db: bool = True):
        """
        Process all matches for a season and update Elo ratings.
        Optionally saves history to database.
        """
        logger.info(f"Processing Elo ratings for season {season_year}, league {league_id}")
        
        # Initialize ratings
        self.initialize_season_ratings(season_year, league_id)
        
        # Get all matches in chronological order
        matches_df = self.data_loader.get_all_matches(season_year, league_id)
        
        if len(matches_df) == 0:
            logger.warning(f"No matches found for season {season_year}, league {league_id}")
            return
        
        cursor = self.conn.cursor()
        
        for _, match in matches_df.iterrows():
            match_id = match['match_id']
            home_team_id = match['home_team_id']
            away_team_id = match['away_team_id']
            home_score = match['home_team_score']
            away_score = match['away_team_score']
            competition_id = match['league_id']
            match_date = match['match_time_utc']
            
            # Get ratings before match
            home_elo_before = self.team_ratings[home_team_id]
            away_elo_before = self.team_ratings[away_team_id]
            
            # Calculate expected result
            expected = self.expected_result(home_elo_before, away_elo_before, is_home=True)
            
            # Determine actual result
            if home_score > away_score:
                actual = 1.0
            elif home_score < away_score:
                actual = 0.0
            else:
                actual = 0.5
            
            # Update ratings
            k_factor = self.get_k_factor(competition_id)
            home_change, away_change = self.update_ratings(
                home_team_id, away_team_id, home_score, away_score, k_factor
            )
            
            # Save to database if requested
            if save_to_db:
                # Home team record
                cursor.execute("""
                    INSERT INTO [dbo].[team_elo_history] (
                        team_id, match_id, season_year, match_date,
                        elo_before, opponent_id, opponent_elo_before, is_home,
                        competition_id, k_factor, actual_result, expected_result,
                        elo_after, elo_change
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, home_team_id, match_id, season_year, match_date,
                     home_elo_before, away_team_id, away_elo_before, 1,
                     competition_id, k_factor, actual, expected,
                     home_elo_before + home_change, home_change)
                
                # Away team record
                cursor.execute("""
                    INSERT INTO [dbo].[team_elo_history] (
                        team_id, match_id, season_year, match_date,
                        elo_before, opponent_id, opponent_elo_before, is_home,
                        competition_id, k_factor, actual_result, expected_result,
                        elo_after, elo_change
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, away_team_id, match_id, season_year, match_date,
                     away_elo_before, home_team_id, home_elo_before, 0,
                     competition_id, k_factor, 1 - actual, 1 - expected,
                     away_elo_before + away_change, away_change)
        
        if save_to_db:
            self.conn.commit()
            logger.info(f"Saved Elo history for {len(matches_df)} matches")
    
    def get_team_rating(self, team_id: int) -> float:
        """Get current Elo rating for a team."""
        return self.team_ratings.get(team_id, ELO_BASE_RATING)
    
    def get_all_ratings(self) -> Dict[int, float]:
        """Get all current team ratings."""
        return self.team_ratings.copy()