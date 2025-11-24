"""
Repository classes for persisting ratings and attributes to the database.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

from database.connection import QueryExecutor


class RatingRepository:
    """Repository for entity ratings."""
    
    def __init__(self, executor: QueryExecutor):
        self.executor = executor
    
    # =========================================================================
    # PLAYER RATINGS
    # =========================================================================
    
    def save_player_rating(
        self,
        player_id: int,
        match_id: int,
        rating: float,
        rating_change: float,
        k_factor: float,
        data_tier: int,
        competition_multiplier: float
    ):
        """Save a player rating snapshot."""
        query = """
            INSERT INTO player_ratings 
                (player_id, match_id, rating, rating_change, k_factor_used,
                 data_tier, competition_multiplier, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE())
        """
        self.executor.execute_non_query(
            query, 
            (player_id, match_id, rating, rating_change, k_factor,
             data_tier, competition_multiplier)
        )
    
    def save_player_ratings_batch(self, ratings: List[Dict[str, Any]]):
        """Batch save player ratings."""
        query = """
            INSERT INTO player_ratings 
                (player_id, match_id, rating, rating_change, k_factor_used,
                 data_tier, competition_multiplier, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE())
        """
        params = [
            (r['player_id'], r['match_id'], r['rating'], r['rating_change'],
             r['k_factor'], r['data_tier'], r['competition_multiplier'])
            for r in ratings
        ]
        self.executor.execute_many(query, params)
    
    def get_player_current_rating(self, player_id: int) -> Optional[float]:
        """Get most recent rating for a player."""
        query = """
            SELECT TOP 1 rating
            FROM player_ratings
            WHERE player_id = ?
            ORDER BY created_at DESC
        """
        result = self.executor.execute_scalar(query, (player_id,))
        return float(result) if result else None
    
    def get_player_rating_history(
        self, 
        player_id: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get rating history for a player."""
        query = f"""
            SELECT TOP {limit} match_id, rating, rating_change, 
                   k_factor_used, data_tier, created_at
            FROM player_ratings
            WHERE player_id = ?
            ORDER BY created_at DESC
        """
        return self.executor.execute_query(query, (player_id,))
    
    # =========================================================================
    # TEAM RATINGS
    # =========================================================================
    
    def save_team_rating(
        self,
        team_id: int,
        match_id: int,
        rating: float,
        rating_change: float,
        expected_result: float,
        actual_result: float,
        k_factor: float,
        data_tier: int
    ):
        """Save a team rating snapshot."""
        query = """
            INSERT INTO team_ratings 
                (team_id, match_id, rating, rating_change, expected_result,
                 actual_result, k_factor_used, data_tier, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
        """
        self.executor.execute_non_query(
            query,
            (team_id, match_id, rating, rating_change, expected_result,
             actual_result, k_factor, data_tier)
        )
    
    def get_team_current_rating(self, team_id: int) -> Optional[float]:
        """Get most recent rating for a team."""
        query = """
            SELECT TOP 1 rating
            FROM team_ratings
            WHERE team_id = ?
            ORDER BY created_at DESC
        """
        result = self.executor.execute_scalar(query, (team_id,))
        return float(result) if result else None
    
    def get_all_current_team_ratings(self) -> Dict[int, float]:
        """Get current ratings for all teams."""
        query = """
            WITH latest AS (
                SELECT team_id, rating,
                       ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY created_at DESC) as rn
                FROM team_ratings
            )
            SELECT team_id, rating FROM latest WHERE rn = 1
        """
        results = self.executor.execute_query(query)
        return {r['team_id']: r['rating'] for r in results}
    
    # =========================================================================
    # COACH RATINGS
    # =========================================================================
    
    def save_coach_rating(
        self,
        coach_id: int,
        match_id: int,
        rating: float,
        rating_change: float,
        k_factor: float,
        data_tier: int
    ):
        """Save a coach rating snapshot."""
        query = """
            INSERT INTO coach_ratings 
                (coach_id, match_id, rating, rating_change, 
                 k_factor_used, data_tier, created_at)
            VALUES (?, ?, ?, ?, ?, ?, GETDATE())
        """
        self.executor.execute_non_query(
            query,
            (coach_id, match_id, rating, rating_change, k_factor, data_tier)
        )
    
    def get_coach_current_rating(self, coach_id: int) -> Optional[float]:
        """Get most recent rating for a coach."""
        query = """
            SELECT TOP 1 rating
            FROM coach_ratings
            WHERE coach_id = ?
            ORDER BY created_at DESC
        """
        result = self.executor.execute_scalar(query, (coach_id,))
        return float(result) if result else None
    
    # =========================================================================
    # LEAGUE RATINGS
    # =========================================================================
    
    def save_league_rating(
        self,
        league_id: int,
        rating: float,
        team_count: int,
        avg_team_rating: float,
        rating_std_dev: float,
        season_year: str
    ):
        """Save a league rating snapshot."""
        query = """
            INSERT INTO league_ratings 
                (league_id, rating, team_count, avg_team_rating,
                 rating_std_dev, season_year, created_at)
            VALUES (?, ?, ?, ?, ?, ?, GETDATE())
        """
        self.executor.execute_non_query(
            query,
            (league_id, rating, team_count, avg_team_rating,
             rating_std_dev, season_year)
        )
    
    def get_league_current_rating(self, league_id: int) -> Optional[float]:
        """Get most recent rating for a league."""
        query = """
            SELECT TOP 1 rating
            FROM league_ratings
            WHERE league_id = ?
            ORDER BY created_at DESC
        """
        result = self.executor.execute_scalar(query, (league_id,))
        return float(result) if result else None


class AttributeRepository:
    """Repository for entity attributes."""
    
    def __init__(self, executor: QueryExecutor):
        self.executor = executor
    
    # =========================================================================
    # PLAYER ATTRIBUTES
    # =========================================================================
    
    def save_player_attribute(
        self,
        player_id: int,
        attribute_code: str,
        value: float,
        confidence: float,
        matches_used: int,
        data_tiers_used: str  # JSON array
    ):
        """Save or update a player attribute."""
        # Use MERGE for upsert
        query = """
            MERGE INTO player_attributes AS target
            USING (SELECT ? as player_id, ? as attribute_code) AS source
            ON target.player_id = source.player_id 
               AND target.attribute_code = source.attribute_code
            WHEN MATCHED THEN
                UPDATE SET value = ?, confidence = ?, matches_used = ?,
                           data_tiers_used = ?, updated_at = GETDATE()
            WHEN NOT MATCHED THEN
                INSERT (player_id, attribute_code, value, confidence, 
                        matches_used, data_tiers_used, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, GETDATE(), GETDATE());
        """
        self.executor.execute_non_query(
            query,
            (player_id, attribute_code, value, confidence, matches_used,
             data_tiers_used, player_id, attribute_code, value, confidence,
             matches_used, data_tiers_used)
        )
    
    def save_player_attributes_batch(
        self, 
        player_id: int, 
        attributes: Dict[str, Dict]
    ):
        """Save multiple attributes for a player."""
        for attr_code, attr_data in attributes.items():
            self.save_player_attribute(
                player_id=player_id,
                attribute_code=attr_code,
                value=attr_data['value'],
                confidence=attr_data['confidence'],
                matches_used=attr_data['matches_used'],
                data_tiers_used=str(attr_data.get('data_tiers_used', []))
            )
    
    def get_player_attributes(self, player_id: int) -> Dict[str, Dict]:
        """Get all attributes for a player."""
        query = """
            SELECT attribute_code, value, confidence, matches_used,
                   data_tiers_used, updated_at
            FROM player_attributes
            WHERE player_id = ?
        """
        results = self.executor.execute_query(query, (player_id,))
        
        return {
            r['attribute_code']: {
                'value': r['value'],
                'confidence': r['confidence'],
                'matches_used': r['matches_used'],
                'data_tiers_used': r['data_tiers_used'],
                'updated_at': r['updated_at']
            }
            for r in results
        }
    
    def get_attribute_leaderboard(
        self, 
        attribute_code: str, 
        limit: int = 100,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get top players for a specific attribute."""
        query = f"""
            SELECT TOP {limit} pa.player_id, pa.value, pa.confidence,
                   mlp.player_name, mlp.usual_position_id
            FROM player_attributes pa
            JOIN (
                SELECT player_id, MAX(player_name) as player_name,
                       MAX(usual_position_id) as usual_position_id
                FROM match_lineup_players
                GROUP BY player_id
            ) mlp ON pa.player_id = mlp.player_id
            WHERE pa.attribute_code = ? AND pa.confidence >= ?
            ORDER BY pa.value DESC
        """
        return self.executor.execute_query(query, (attribute_code, min_confidence))
    
    # =========================================================================
    # TEAM ATTRIBUTES
    # =========================================================================
    
    def save_team_attribute(
        self,
        team_id: int,
        attribute_code: str,
        value: float,
        matches_used: int
    ):
        """Save or update a team attribute."""
        query = """
            MERGE INTO team_attributes AS target
            USING (SELECT ? as team_id, ? as attribute_code) AS source
            ON target.team_id = source.team_id 
               AND target.attribute_code = source.attribute_code
            WHEN MATCHED THEN
                UPDATE SET value = ?, matches_used = ?, updated_at = GETDATE()
            WHEN NOT MATCHED THEN
                INSERT (team_id, attribute_code, value, matches_used, 
                        created_at, updated_at)
                VALUES (?, ?, ?, ?, GETDATE(), GETDATE());
        """
        self.executor.execute_non_query(
            query,
            (team_id, attribute_code, value, matches_used,
             team_id, attribute_code, value, matches_used)
        )
    
    def get_team_attributes(self, team_id: int) -> Dict[str, Dict]:
        """Get all attributes for a team."""
        query = """
            SELECT attribute_code, value, matches_used, updated_at
            FROM team_attributes
            WHERE team_id = ?
        """
        results = self.executor.execute_query(query, (team_id,))
        
        return {
            r['attribute_code']: {
                'value': r['value'],
                'matches_used': r['matches_used'],
                'updated_at': r['updated_at']
            }
            for r in results
        }
    
    # =========================================================================
    # COACH ATTRIBUTES
    # =========================================================================
    
    def save_coach_attribute(
        self,
        coach_id: int,
        attribute_code: str,
        value: float,
        matches_used: int
    ):
        """Save or update a coach attribute."""
        query = """
            MERGE INTO coach_attributes AS target
            USING (SELECT ? as coach_id, ? as attribute_code) AS source
            ON target.coach_id = source.coach_id 
               AND target.attribute_code = source.attribute_code
            WHEN MATCHED THEN
                UPDATE SET value = ?, matches_used = ?, updated_at = GETDATE()
            WHEN NOT MATCHED THEN
                INSERT (coach_id, attribute_code, value, matches_used,
                        created_at, updated_at)
                VALUES (?, ?, ?, ?, GETDATE(), GETDATE());
        """
        self.executor.execute_non_query(
            query,
            (coach_id, attribute_code, value, matches_used,
             coach_id, attribute_code, value, matches_used)
        )
    
    def get_coach_attributes(self, coach_id: int) -> Dict[str, Dict]:
        """Get all attributes for a coach."""
        query = """
            SELECT attribute_code, value, matches_used, updated_at
            FROM coach_attributes
            WHERE coach_id = ?
        """
        results = self.executor.execute_query(query, (coach_id,))
        
        return {
            r['attribute_code']: {
                'value': r['value'],
                'matches_used': r['matches_used'],
                'updated_at': r['updated_at']
            }
            for r in results
        }


class ProcessingStateRepository:
    """Repository for tracking processing state."""
    
    def __init__(self, executor: QueryExecutor):
        self.executor = executor
    
    def mark_match_processed(
        self,
        match_id: int,
        processing_type: str,  # 'ratings', 'attributes', 'full'
        success: bool,
        error_message: Optional[str] = None
    ):
        """Mark a match as processed."""
        query = """
            INSERT INTO processing_log
                (match_id, processing_type, success, error_message, processed_at)
            VALUES (?, ?, ?, ?, GETDATE())
        """
        self.executor.execute_non_query(
            query, (match_id, processing_type, success, error_message)
        )
    
    def get_unprocessed_matches(
        self, 
        processing_type: str = 'ratings'
    ) -> List[int]:
        """Get matches that haven't been processed yet."""
        query = """
            SELECT md.match_id
            FROM match_details md
            LEFT JOIN processing_log pl 
                ON md.match_id = pl.match_id AND pl.processing_type = ?
            WHERE md.finished = 1 AND pl.match_id IS NULL
            ORDER BY md.match_time_utc
        """
        results = self.executor.execute_query(query, (processing_type,))
        return [r['match_id'] for r in results]
    
    def get_last_processed_date(
        self, 
        processing_type: str = 'ratings'
    ) -> Optional[datetime]:
        """Get the date of the last processed match."""
        query = """
            SELECT TOP 1 md.match_time_utc
            FROM processing_log pl
            JOIN match_details md ON pl.match_id = md.match_id
            WHERE pl.processing_type = ? AND pl.success = 1
            ORDER BY md.match_time_utc DESC
        """
        result = self.executor.execute_scalar(query, (processing_type,))
        return result
    
    def reset_processing_state(self, processing_type: Optional[str] = None):
        """Clear processing log (for reprocessing)."""
        if processing_type:
            query = "DELETE FROM processing_log WHERE processing_type = ?"
            self.executor.execute_non_query(query, (processing_type,))
        else:
            self.executor.execute_non_query("TRUNCATE TABLE processing_log")