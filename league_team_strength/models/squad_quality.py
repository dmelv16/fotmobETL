import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SquadQualityEvaluator:
    """
    Evaluates team strength based on aggregated player quality.
    Uses player ratings, market values, and performance metrics.
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader
    
    def calculate_team_squad_quality(self, team_id: int, season_year: int) -> Optional[Dict]:
        """
        Calculate team's squad quality based on player aggregation.
        
        Uses:
        - Player ratings from matches
        - Market values
        - Player ages (prime vs declining)
        - Minutes distribution (squad depth)
        """
        # Get all players who played for team in season
        query = """
            SELECT 
                mlp.player_id,
                mlp.player_name,
                mlp.age,
                mlp.position_id,
                COUNT(*) as appearances,
                AVG(mlp.rating) as avg_rating,
                MAX(mlp.market_value) as market_value,
                SUM(CASE WHEN mlp.is_starter = 1 THEN 1 ELSE 0 END) as starts
            FROM [dbo].[match_lineup_players] mlp
            INNER JOIN [dbo].[match_details] md ON mlp.match_id = md.match_id
            WHERE mlp.team_id = ?
              AND YEAR(md.match_time_utc) = ?
              AND md.finished = 1
            GROUP BY mlp.player_id, mlp.player_name, mlp.age, mlp.position_id
            HAVING COUNT(*) >= 3  -- Minimum appearances to count
        """
        
        df = pd.read_sql(query, self.conn, params=[team_id, season_year])
        
        if len(df) == 0:
            return None
        
        # Calculate individual player quality scores
        df['quality_score'] = self._calculate_player_quality_score(df)
        
        # Weight by appearances (regular players matter more)
        total_appearances = df['appearances'].sum()
        df['weight'] = df['appearances'] / total_appearances
        
        # Calculate weighted average quality
        weighted_quality = (df['quality_score'] * df['weight']).sum()
        
        # Calculate squad depth (how evenly distributed are appearances)
        appearance_gini = self._calculate_gini_coefficient(df['appearances'].values)
        depth_score = 1 - appearance_gini  # Lower Gini = more depth
        
        # Calculate age profile
        avg_age = (df['age'] * df['weight']).sum()
        
        # Adjust quality for age profile
        age_adjustment = self._get_age_profile_adjustment(avg_age)
        
        # Calculate market value (if available)
        total_market_value = df['market_value'].sum()
        
        # Final squad quality rating (normalized to ~1500 scale)
        base_rating = weighted_quality * 10  # Scale up player scores
        depth_adjustment = depth_score * 100  # Bonus for depth
        age_adjustment_points = age_adjustment * 50
        
        final_rating = base_rating + depth_adjustment + age_adjustment_points
        
        return {
            'team_id': team_id,
            'season_year': season_year,
            'squad_size': len(df),
            'avg_player_rating': df['avg_rating'].mean(),
            'weighted_avg_rating': (df['avg_rating'] * df['weight']).sum(),
            'squad_depth_score': depth_score,
            'avg_squad_age': avg_age,
            'total_market_value': total_market_value,
            'squad_quality_rating': final_rating
        }
    
    def _calculate_player_quality_score(self, player_df: pd.DataFrame) -> pd.Series:
        """
        Calculate individual player quality scores.
        Combines rating and market value (if available).
        """
        # Normalize ratings (typically 0-10 scale in FotMob)
        normalized_rating = player_df['avg_rating'] * 10
        
        # Normalize market values (log scale to handle huge variance)
        if player_df['market_value'].notna().any():
            # Fill missing with median
            market_values = player_df['market_value'].fillna(player_df['market_value'].median())
            # Log transform (add 1 to avoid log(0))
            log_mv = np.log10(market_values + 1)
            # Normalize to 0-100
            normalized_mv = ((log_mv - log_mv.min()) / (log_mv.max() - log_mv.min())) * 100
        else:
            normalized_mv = 50  # Default if no market value data
        
        # Combine (weight rating more heavily - it's performance-based)
        quality_score = 0.7 * normalized_rating + 0.3 * normalized_mv
        
        return quality_score
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient (measure of inequality).
        Used to measure appearance distribution (squad depth).
        
        Returns: 0 = perfect equality, 1 = perfect inequality
        """
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        
        return gini
    
    def _get_age_profile_adjustment(self, avg_age: float) -> float:
        """
        Adjustment factor based on squad age profile.
        Returns: -1 to +1 (adjustment multiplier)
        """
        if avg_age < 23:
            return -0.2  # Very young squad, inexperienced
        elif avg_age < 25:
            return 0.3  # Young and improving
        elif avg_age < 27:
            return 0.5  # Prime age
        elif avg_age < 29:
            return 0.3  # Still good
        elif avg_age < 31:
            return 0.0  # Neutral
        else:
            return -0.3  # Aging squad
    
    def calculate_league_teams_squad_quality(self, league_id: int, season_year: int) -> pd.DataFrame:
        """
        Calculate squad quality for all teams in a league.
        """
        team_ids = self.data_loader.get_league_teams_for_season(league_id, season_year)
        
        results = []
        
        for team_id in team_ids:
            quality = self.calculate_team_squad_quality(team_id, season_year)
            if quality:
                results.append(quality)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Normalize to 0-100 scale
        min_rating = df['squad_quality_rating'].min()
        max_rating = df['squad_quality_rating'].max()
        
        if max_rating > min_rating:
            df['squad_quality_normalized'] = ((df['squad_quality_rating'] - min_rating) / (max_rating - min_rating)) * 100
        else:
            df['squad_quality_normalized'] = 50.0
        
        return df