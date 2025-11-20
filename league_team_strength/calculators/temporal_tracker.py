import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TemporalStrengthTracker:
    """
    Tracks how league and team strengths evolve over time.
    Detects trends, regime changes, and temporal patterns.
    """
    
    def __init__(self, connection):
        self.conn = connection
    
    def get_league_strength_history(self, league_id: int, 
                                    start_season: int, 
                                    end_season: int) -> pd.DataFrame:
        """
        Get historical league strength data.
        """
        query = """
            SELECT 
                season_year,
                overall_strength,
                transfer_matrix_score,
                european_results_score,
                calculation_confidence,
                trend_1yr,
                trend_3yr
            FROM [dbo].[league_strength]
            WHERE league_id = ?
              AND season_year BETWEEN ? AND ?
            ORDER BY season_year
        """
        
        return pd.read_sql(query, self.conn, params=[league_id, start_season, end_season])
    
    def get_team_strength_evolution(self, team_id: int, season_year: int) -> pd.DataFrame:
        """
        Get team's strength evolution within a season (if multiple snapshots).
        """
        query = """
            SELECT 
                as_of_date,
                overall_strength,
                elo_rating,
                xg_performance_rating,
                squad_quality_rating,
                coaching_effect,
                matches_played,
                league_position
            FROM [dbo].[team_strength]
            WHERE team_id = ?
              AND season_year = ?
            ORDER BY as_of_date
        """
        
        return pd.read_sql(query, self.conn, params=[team_id, season_year])
    
    def detect_league_breakout_seasons(self, min_strength_increase: float = 5.0) -> pd.DataFrame:
        """
        Identify leagues that had breakout seasons (significant strength increases).
        
        Returns: DataFrame with leagues and their breakout details
        """
        query = """
            SELECT 
                ls1.league_id,
                ls1.league_name,
                ls1.season_year,
                ls1.overall_strength as current_strength,
                ls0.overall_strength as previous_strength,
                ls1.overall_strength - ls0.overall_strength as strength_increase,
                ls1.european_results_score,
                ls1.transfer_matrix_score
            FROM [dbo].[league_strength] ls1
            INNER JOIN [dbo].[league_strength] ls0 
                ON ls1.league_id = ls0.league_id 
                AND ls1.season_year = ls0.season_year + 1
            WHERE ls1.overall_strength - ls0.overall_strength >= ?
            ORDER BY strength_increase DESC
        """
        
        return pd.read_sql(query, self.conn, params=[min_strength_increase])
    
    def detect_team_regime_changes(self, team_id: int) -> List[Dict]:
        """
        Detect significant changes in team strength (new coach, major signings, etc.)
        
        Returns: List of detected regime changes
        """
        # Get all team strength records
        query = """
            SELECT 
                season_year,
                overall_strength,
                coaching_effect,
                squad_quality_rating
            FROM [dbo].[team_strength]
            WHERE team_id = ?
            ORDER BY season_year, as_of_date
        """
        
        df = pd.read_sql(query, self.conn, params=[team_id])
        
        if len(df) < 2:
            return []
        
        # Detect significant changes
        changes = []
        
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            strength_change = curr['overall_strength'] - prev['overall_strength']
            
            # Significant change threshold
            if abs(strength_change) > 10:
                change_type = 'improvement' if strength_change > 0 else 'decline'
                
                # Determine likely cause
                coaching_change = abs(curr['coaching_effect'] - prev['coaching_effect']) > 5
                squad_change = abs(curr['squad_quality_rating'] - prev['squad_quality_rating']) > 10
                
                likely_cause = []
                if coaching_change:
                    likely_cause.append('coaching')
                if squad_change:
                    likely_cause.append('squad_overhaul')
                if not likely_cause:
                    likely_cause.append('performance_shift')
                
                changes.append({
                    'season': curr['season_year'],
                    'type': change_type,
                    'magnitude': abs(strength_change),
                    'likely_cause': ', '.join(likely_cause)
                })
        
        return changes
    
    def calculate_strength_volatility(self, league_id: int, 
                                     start_season: int, 
                                     end_season: int) -> float:
        """
        Calculate how volatile a league's strength has been (standard deviation).
        
        Returns: Volatility score (higher = more volatile)
        """
        history = self.get_league_strength_history(league_id, start_season, end_season)
        
        if len(history) < 2:
            return 0.0
        
        return float(history['overall_strength'].std())
    
    def project_league_strength(self, league_id: int, 
                               target_season: int,
                               method: str = 'linear') -> Optional[float]:
        """
        Project future league strength based on historical trends.
        
        Methods: 'linear', 'exponential_smoothing'
        
        Returns: Projected strength score
        """
        # Get recent history (last 5 seasons)
        current_season = target_season - 1
        start_season = current_season - 4
        
        history = self.get_league_strength_history(league_id, start_season, current_season)
        
        if len(history) < 3:
            logger.warning(f"Insufficient history for projection (league {league_id})")
            return None
        
        if method == 'linear':
            # Simple linear regression
            X = history['season_year'].values
            y = history['overall_strength'].values
            
            # Fit line
            slope, intercept = np.polyfit(X, y, 1)
            
            # Project
            projection = slope * target_season + intercept
            
            return float(projection)
        
        elif method == 'exponential_smoothing':
            # Exponential smoothing (weight recent seasons more)
            alpha = 0.3
            
            smoothed = [history['overall_strength'].iloc[0]]
            
            for i in range(1, len(history)):
                value = history['overall_strength'].iloc[i]
                smoothed_value = alpha * value + (1 - alpha) * smoothed[-1]
                smoothed.append(smoothed_value)
            
            # Calculate trend
            trend = (smoothed[-1] - smoothed[-3]) / 2  # Average change over last 2 seasons
            
            # Project
            projection = smoothed[-1] + trend
            
            return float(projection)
        
        else:
            logger.error(f"Unknown projection method: {method}")
            return None