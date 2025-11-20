import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from config.settings import CHAMPIONS_LEAGUE_WEIGHT, EUROPA_LEAGUE_WEIGHT, CONFERENCE_LEAGUE_WEIGHT
from config.league_mappings import EUROPEAN_COMPETITIONS

logger = logging.getLogger(__name__)

class EuropeanResultsAnalyzer:
    """
    Analyzes European competition results to infer domestic league strength.
    Uses head-to-head results between teams from different leagues.
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader
        self.league_ratings = {}  # {league_id: rating}
    
    def get_competition_weight(self, competition_id: int) -> float:
        """Get weight for competition importance."""
        comp_info = EUROPEAN_COMPETITIONS.get(competition_id, {})
        return comp_info.get('weight', 0.8)
    
    def initialize_league_ratings(self, league_ids: List[int], base_ratings: Dict[int, float] = None):
        """
        Initialize league ratings.
        Can use previous ratings or base estimates.
        """
        if base_ratings:
            self.league_ratings = base_ratings.copy()
        else:
            # Start all at 1500 (Elo-style)
            for league_id in league_ids:
                self.league_ratings[league_id] = 1500.0
    
    def get_team_domestic_league(self, team_id: int, season_year: int) -> Optional[int]:
        """
        Determine which domestic league a team plays in.
        """
        query = """
            SELECT TOP 1 league_id
            FROM [dbo].[match_details]
            WHERE (home_team_id = ? OR away_team_id = ?)
              AND YEAR(match_time_utc) = ?
              AND league_id NOT IN (42, 73, 848)  -- Exclude European competitions
            GROUP BY league_id
            ORDER BY COUNT(*) DESC
        """
        
        df = pd.read_sql(query, self.conn, params=[team_id, team_id, season_year])
        
        if len(df) > 0:
            return df['league_id'].iloc[0]
        return None
    
    def expected_result(self, rating_a: float, rating_b: float, is_home: bool = False) -> float:
        """
        Calculate expected result using Elo formula.
        """
        rating_diff = rating_b - rating_a
        if is_home:
            rating_diff -= 100  # Home advantage
        
        return 1 / (1 + 10 ** (rating_diff / 400))
    
    def update_league_ratings_from_match(self, home_league_id: int, away_league_id: int,
                                        home_score: int, away_score: int,
                                        competition_weight: float, k_factor: float = 30):
        """
        Update league ratings based on a match result.
        """
        home_rating = self.league_ratings.get(home_league_id, 1500.0)
        away_rating = self.league_ratings.get(away_league_id, 1500.0)
        
        # Determine actual result
        if home_score > away_score:
            actual = 1.0
        elif home_score < away_score:
            actual = 0.0
        else:
            actual = 0.5
        
        # Expected result
        expected = self.expected_result(home_rating, away_rating, is_home=True)
        
        # Update with competition weighting
        weighted_k = k_factor * competition_weight
        
        home_change = weighted_k * (actual - expected)
        away_change = weighted_k * ((1 - actual) - (1 - expected))
        
        self.league_ratings[home_league_id] = home_rating + home_change
        self.league_ratings[away_league_id] = away_rating + away_change
    
    def process_season(self, season_year: int, save_to_db: bool = True) -> Dict[int, float]:
        """
        Process all European matches for a season and update league ratings.
        
        Returns: Dict of final league ratings
        """
        logger.info(f"Processing European results for season {season_year}")
        
        # Get all European matches
        matches_df = self.data_loader.get_european_matches(season_year)
        
        if len(matches_df) == 0:
            logger.warning(f"No European matches found for season {season_year}")
            return {}
        
        # Get domestic leagues for all participating teams
        team_leagues = {}
        for team_id in pd.concat([matches_df['home_team_id'], matches_df['away_team_id']]).unique():
            league_id = self.get_team_domestic_league(team_id, season_year)
            if league_id:
                team_leagues[team_id] = league_id
        
        # Initialize ratings for leagues that appear in European competitions
        unique_leagues = set(team_leagues.values())
        self.initialize_league_ratings(list(unique_leagues))
        
        processed_matches = []
        
        # Process each match
        for _, match in matches_df.iterrows():
            home_team_id = match['home_team_id']
            away_team_id = match['away_team_id']
            
            # Get domestic leagues
            home_league = team_leagues.get(home_team_id)
            away_league = team_leagues.get(away_team_id)
            
            # Skip if can't determine leagues or same league
            if not home_league or not away_league or home_league == away_league:
                continue
            
            competition_id = match['competition_id']
            competition_weight = self.get_competition_weight(competition_id)
            
            home_score = match['home_team_score']
            away_score = match['away_team_score']
            
            # Calculate expected result before update
            home_rating_before = self.league_ratings[home_league]
            away_rating_before = self.league_ratings[away_league]
            expected = self.expected_result(home_rating_before, away_rating_before, is_home=True)
            
            # Determine actual result
            if home_score > away_score:
                actual = 1.0
            elif home_score < away_score:
                actual = 0.0
            else:
                actual = 0.5
            
            # Update ratings
            self.update_league_ratings_from_match(
                home_league, away_league, home_score, away_score, competition_weight
            )
            
            # Save match record
            if save_to_db:
                processed_matches.append({
                    'match_id': match['match_id'],
                    'season_year': season_year,
                    'competition_id': competition_id,
                    'competition_name': match['competition_name'],
                    'competition_weight': competition_weight,
                    'home_team_id': home_team_id,
                    'home_team_name': match['home_team_name'],
                    'home_league_id': home_league,
                    'home_score': home_score,
                    'home_xg': match['home_xg'],
                    'away_team_id': away_team_id,
                    'away_team_name': match['away_team_name'],
                    'away_league_id': away_league,
                    'away_score': away_score,
                    'away_xg': match['away_xg'],
                    'match_date': match['match_time_utc'],
                    'expected_result': expected,
                    'actual_result': actual,
                    'surprise_factor': abs(actual - expected)
                })
        
        logger.info(f"Processed {len(processed_matches)} inter-league European matches")
        
        # Save to database
        if save_to_db and processed_matches:
            self._save_matches_to_db(processed_matches)
            self._save_league_network(season_year)
        
        return self.league_ratings
    
    def _save_matches_to_db(self, matches: List[Dict]):
        """Save European match results to database."""
        cursor = self.conn.cursor()
        
        for match in matches:
            # Check if exists
            cursor.execute("""
                SELECT id FROM [dbo].[european_competition_results]
                WHERE match_id = ?
            """, match['match_id'])
            
            if cursor.fetchone():
                continue  # Already exists
            
            cursor.execute("""
                INSERT INTO [dbo].[european_competition_results] (
                    match_id, season_year, competition_id, competition_name, competition_weight,
                    home_team_id, home_team_name, home_league_id, home_league_name,
                    home_score, home_xg,
                    away_team_id, away_team_name, away_league_id, away_league_name,
                    away_score, away_xg,
                    match_date, result, expected_result, actual_result, surprise_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 
                         CASE WHEN ? > ? THEN 'home' WHEN ? < ? THEN 'away' ELSE 'draw' END,
                         ?, ?, ?)
            """, match['match_id'], match['season_year'], match['competition_id'], 
                 match['competition_name'], match['competition_weight'],
                 match['home_team_id'], match['home_team_name'], match['home_league_id'],
                 match['home_score'], match['home_xg'],
                 match['away_team_id'], match['away_team_name'], match['away_league_id'],
                 match['away_score'], match['away_xg'],
                 match['match_date'],
                 match['home_score'], match['away_score'], match['home_score'], match['away_score'],
                 match['expected_result'], match['actual_result'], match['surprise_factor'])
        
        self.conn.commit()
        logger.info(f"Saved {len(matches)} European match records")
    
    def _save_league_network(self, season_year: int):
        """Save league relationships derived from European results to network table."""
        cursor = self.conn.cursor()
        
        # Get all league pairs that played each other
        cursor.execute("""
            SELECT 
                home_league_id as league_a,
                away_league_id as league_b,
                COUNT(*) as match_count,
                AVG(CASE WHEN result = 'home' THEN 1.0 
                         WHEN result = 'draw' THEN 0.5 
                         ELSE 0.0 END) as league_a_win_rate
            FROM [dbo].[european_competition_results]
            WHERE season_year = ?
            GROUP BY home_league_id, away_league_id
        """, season_year)
        
        pairs = cursor.fetchall()
        
        for pair in pairs:
            league_a, league_b, match_count, win_rate = pair
            
            # Calculate strength gap from win rate
            # If win rate is 0.7, league A is stronger
            # Convert to multiplier: league B relative to A
            if win_rate > 0.5:
                gap_multiplier = 1 / (win_rate / 0.5)  # B is weaker
            elif win_rate < 0.5:
                gap_multiplier = (0.5 / win_rate)  # B is stronger
            else:
                gap_multiplier = 1.0
            
            confidence = min(match_count / 10, 1.0)  # Max confidence at 10 matches
            
            # Update or insert
            cursor.execute("""
                IF EXISTS (SELECT 1 FROM [dbo].[league_network_edges]
                          WHERE league_a_id = ? AND league_b_id = ? AND season_year = ?)
                    UPDATE [dbo].[league_network_edges]
                    SET european_matches = ?,
                        strength_gap_estimate = ?,
                        gap_confidence = ?,
                        gap_method = 'european',
                        last_updated = GETDATE()
                    WHERE league_a_id = ? AND league_b_id = ? AND season_year = ?
                ELSE
                    INSERT INTO [dbo].[league_network_edges] (
                        league_a_id, league_b_id, season_year, european_matches,
                        strength_gap_estimate, gap_confidence, gap_method
                    ) VALUES (?, ?, ?, ?, ?, ?, 'european')
            """, league_a, league_b, season_year, match_count, gap_multiplier, confidence,
                 league_a, league_b, season_year, league_a, league_b, season_year,
                 match_count, gap_multiplier, confidence)
        
        self.conn.commit()
        logger.info(f"Saved {len(pairs)} European-based league network edges")
    
    def get_league_ratings_normalized(self, scale_min: float = 0, scale_max: float = 100) -> Dict[int, float]:
        """
        Get league ratings normalized to a 0-100 scale.
        """
        if not self.league_ratings:
            return {}
        
        ratings = np.array(list(self.league_ratings.values()))
        min_rating = ratings.min()
        max_rating = ratings.max()
        
        normalized = {}
        for league_id, rating in self.league_ratings.items():
            normalized_value = ((rating - min_rating) / (max_rating - min_rating)) * (scale_max - scale_min) + scale_min
            normalized[league_id] = normalized_value
        
        return normalized