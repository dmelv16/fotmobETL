import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SquadQualityEvaluator:
    """
    Evaluates team strength based on aggregated player quality.
    Robustly handles missing data (ratings, market values, ages).
    Returns RAW ratings without normalization.
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader
    
    def calculate_team_squad_quality(self, team_id: int, season_year: int) -> Optional[Dict]:
        """
        Calculate team's squad quality based on player aggregation.
        Returns RAW rating without normalization or clipping.
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
        
        # Determine what data we have available
        data_availability = self._assess_data_availability(df)
        
        # Calculate quality scores using available data
        df['quality_score'] = self._calculate_player_quality_score(df, data_availability)
        
        # Weight by appearances (regular players matter more)
        total_appearances = df['appearances'].sum()
        df['weight'] = df['appearances'] / total_appearances
        
        # Calculate weighted average quality
        weighted_quality = (df['quality_score'] * df['weight']).sum()
        
        # Calculate squad depth (how evenly distributed are appearances)
        appearance_gini = self._calculate_gini_coefficient(df['appearances'].values)
        depth_score = 1 - appearance_gini  # Lower Gini = more depth
        
        # Calculate age profile (handle missing ages)
        avg_age, age_data_completeness = self._calculate_average_age(df)
        age_adjustment = self._get_age_profile_adjustment(avg_age) if avg_age else 0
        
        # Calculate market value (if available)
        total_market_value = df['market_value'].sum() if data_availability['has_market_value'] else None
        
        # Final squad quality rating - RAW SCALE (no normalization)
        # Components contribute additively
        base_rating = weighted_quality * 10  # Scale: ~0-1000
        depth_adjustment = depth_score * 100  # Scale: 0-100
        age_adjustment_points = age_adjustment * 50 if avg_age else 0  # Scale: -15 to +25
        
        raw_rating = base_rating + depth_adjustment + age_adjustment_points
        
        # Calculate confidence based on data availability
        confidence = self._calculate_confidence_score(data_availability, len(df))
        
        return {
            'team_id': team_id,
            'season_year': season_year,
            'squad_size': len(df),
            'avg_player_rating': df['avg_rating'].mean() if data_availability['has_ratings'] else None,
            'weighted_avg_rating': (df['avg_rating'] * df['weight']).sum() if data_availability['has_ratings'] else None,
            'squad_depth_score': depth_score,
            'avg_squad_age': avg_age,
            'age_data_completeness': age_data_completeness,
            'total_market_value': total_market_value,
            'squad_quality_rating': float(raw_rating),  # RAW, no normalization
            'calculation_confidence': confidence,
            'data_sources_used': self._get_data_sources_description(data_availability)
        }
    
    def _assess_data_availability(self, player_df: pd.DataFrame) -> Dict[str, any]:
        """
        Assess what data is available for this team/season.
        Returns a dict describing data completeness.
        """
        total_players = len(player_df)
        
        # Check ratings
        rating_count = player_df['avg_rating'].notna().sum()
        has_ratings = rating_count > 0
        rating_completeness = rating_count / total_players if total_players > 0 else 0
        
        # Check market values
        mv_count = player_df['market_value'].notna().sum()
        has_market_value = mv_count > 0
        mv_completeness = mv_count / total_players if total_players > 0 else 0
        
        # Check ages
        age_count = player_df['age'].notna().sum()
        has_ages = age_count > 0
        age_completeness = age_count / total_players if total_players > 0 else 0
        
        return {
            'has_ratings': has_ratings,
            'rating_completeness': rating_completeness,
            'has_market_value': has_market_value,
            'mv_completeness': mv_completeness,
            'has_ages': has_ages,
            'age_completeness': age_completeness,
            'total_players': total_players
        }
    
    def _calculate_player_quality_score(self, player_df: pd.DataFrame, 
                                       data_availability: Dict) -> pd.Series:
        """
        Calculate individual player quality scores using whatever data is available.
        Adapts calculation method based on data completeness.
        """
        components = []
        weights = []
        
        # Component 1: Appearance-based quality (always available)
        appearance_score = self._calculate_appearance_score(player_df)
        components.append(appearance_score)
        weights.append(0.4)
        
        # Component 2: Starter ratio (always available)
        starter_score = self._calculate_starter_score(player_df)
        components.append(starter_score)
        weights.append(0.3)
        
        # Component 3: Performance rating (if available)
        if data_availability['has_ratings']:
            rating_score = self._calculate_rating_score(player_df, data_availability['rating_completeness'])
            components.append(rating_score)
            weights.append(0.5 * data_availability['rating_completeness'])
        
        # Component 4: Market value (if available)
        if data_availability['has_market_value']:
            mv_score = self._calculate_market_value_score(player_df, data_availability['mv_completeness'])
            components.append(mv_score)
            weights.append(0.3 * data_availability['mv_completeness'])
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Combine components
        quality_score = sum(comp * weight for comp, weight in zip(components, normalized_weights))
        
        return quality_score
    
    def _calculate_appearance_score(self, player_df: pd.DataFrame) -> pd.Series:
        """
        Score based on number of appearances (normalized 0-100).
        More appearances = more important player.
        """
        max_apps = player_df['appearances'].max()
        if max_apps == 0:
            return pd.Series([50.0] * len(player_df), index=player_df.index)
        
        return (player_df['appearances'] / max_apps) * 100
    
    def _calculate_starter_score(self, player_df: pd.DataFrame) -> pd.Series:
        """
        Score based on starter ratio (starts / appearances).
        Normalized to 0-100 scale.
        """
        starter_ratio = player_df['starts'] / player_df['appearances']
        return starter_ratio * 100
    
    def _calculate_rating_score(self, player_df: pd.DataFrame, completeness: float) -> pd.Series:
        """
        Score based on performance ratings.
        Handles missing ratings by using median imputation.
        """
        ratings = player_df['avg_rating'].copy()
        
        # Impute missing ratings with median (conservative approach)
        if ratings.isna().any():
            median_rating = ratings.median()
            if pd.isna(median_rating):
                median_rating = 6.5
            ratings = ratings.fillna(median_rating)
        
        # Ratings typically range from 0-10, normalize to 0-100
        min_rating = ratings.min()
        max_rating = ratings.max()
        
        if max_rating > min_rating:
            normalized = ((ratings - min_rating) / (max_rating - min_rating)) * 100
        else:
            normalized = pd.Series([50.0] * len(ratings), index=ratings.index)
        
        return normalized
    
    def _calculate_market_value_score(self, player_df: pd.DataFrame, completeness: float) -> pd.Series:
        """
        Score based on market values.
        Uses log scale to handle extreme variance.
        """
        market_values = player_df['market_value'].copy()
        
        # Impute missing values with median
        if market_values.isna().any():
            median_mv = market_values.median()
            if pd.isna(median_mv):
                return pd.Series([50.0] * len(market_values), index=market_values.index)
            market_values = market_values.fillna(median_mv)
        
        # Log transform to handle huge variance
        log_mv = np.log10(market_values + 1)
        
        # Normalize to 0-100
        min_log = log_mv.min()
        max_log = log_mv.max()
        
        if max_log > min_log:
            normalized = ((log_mv - min_log) / (max_log - min_log)) * 100
        else:
            normalized = pd.Series([50.0] * len(log_mv), index=log_mv.index)
        
        return normalized
    
    def _calculate_average_age(self, player_df: pd.DataFrame) -> Tuple[Optional[float], float]:
        """
        Calculate weighted average age, handling missing data.
        Returns (avg_age, completeness_ratio).
        """
        ages = player_df['age'].copy()
        age_count = ages.notna().sum()
        
        if age_count == 0:
            return None, 0.0
        
        completeness = age_count / len(player_df)
        
        # Weight by appearances (only for players with known ages)
        valid_ages = player_df[ages.notna()].copy()
        total_apps = valid_ages['appearances'].sum()
        
        if total_apps == 0:
            return ages.mean(), completeness
        
        weighted_age = (valid_ages['age'] * valid_ages['appearances']).sum() / total_apps
        
        return weighted_age, completeness
    
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
    
    def _calculate_confidence_score(self, data_availability: Dict, squad_size: int) -> float:
        """
        Calculate confidence in the squad quality calculation.
        Based on data availability and squad size.
        
        Returns: 0.0 to 1.0
        """
        # Start with base confidence
        confidence = 0.5
        
        # Boost for having ratings (most important data)
        if data_availability['has_ratings']:
            confidence += 0.3 * data_availability['rating_completeness']
        
        # Boost for having market values
        if data_availability['has_market_value']:
            confidence += 0.15 * data_availability['mv_completeness']
        
        # Boost for having ages
        if data_availability['has_ages']:
            confidence += 0.05 * data_availability['age_completeness']
        
        # Adjust for squad size (more players = more confidence)
        if squad_size >= 20:
            confidence *= 1.0
        elif squad_size >= 15:
            confidence *= 0.9
        elif squad_size >= 10:
            confidence *= 0.8
        else:
            confidence *= 0.6
        
        return min(1.0, confidence)
    
    def _get_data_sources_description(self, data_availability: Dict) -> str:
        """
        Generate a human-readable description of what data was used.
        """
        sources = []
        
        sources.append("appearances")
        sources.append("starter_status")
        
        if data_availability['has_ratings']:
            completeness = data_availability['rating_completeness']
            sources.append(f"ratings({completeness:.0%})")
        
        if data_availability['has_market_value']:
            completeness = data_availability['mv_completeness']
            sources.append(f"market_values({completeness:.0%})")
        
        if data_availability['has_ages']:
            completeness = data_availability['age_completeness']
            sources.append(f"ages({completeness:.0%})")
        
        return ", ".join(sources)
    
    def calculate_league_teams_squad_quality(self, league_id: int, season_year: int) -> pd.DataFrame:
        """
        Calculate squad quality for all teams in a league.
        Returns RAW ratings without normalization.
        """
        team_ids = self.data_loader.get_league_teams_for_season(league_id, season_year)
        
        results = []
        
        for team_id in team_ids:
            try:
                quality = self.calculate_team_squad_quality(team_id, season_year)
                if quality:
                    results.append(quality)
            except Exception as e:
                logger.error(f"Error calculating squad quality for team {team_id}: {e}", exc_info=True)
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)