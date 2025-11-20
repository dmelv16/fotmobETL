import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class XGPerformanceRater:
    """
    Evaluates team strength based on xG performance.
    Provides a cleaner signal than results (less variance/luck).
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader
    
    def calculate_team_xg_rating(self, team_id: int, season_year: int,
                                 as_of_date: Optional[str] = None) -> Optional[Dict]:
        """
        Calculate team's xG-based performance rating.
        
        Returns: Dict with xG stats and derived rating
        """
        # Build query to get team's matches
        query = """
            SELECT 
                md.match_id,
                md.match_time_utc,
                CASE WHEN md.home_team_id = ? THEN 1 ELSE 0 END as is_home,
                CASE WHEN md.home_team_id = ? THEN md.home_team_score ELSE md.away_team_score END as goals_for,
                CASE WHEN md.home_team_id = ? THEN md.away_team_score ELSE md.home_team_score END as goals_against,
                CASE WHEN md.home_team_id = ? THEN mss.home_xg ELSE mss.away_xg END as xg_for,
                CASE WHEN md.home_team_id = ? THEN mss.away_xg ELSE mss.home_xg END as xg_against
            FROM [dbo].[match_details] md
            LEFT JOIN [dbo].[match_stats_summary] mss ON md.match_id = mss.match_id
            WHERE (md.home_team_id = ? OR md.away_team_id = ?)
              AND YEAR(md.match_time_utc) = ?
              AND md.finished = 1
        """
        
        params = [team_id] * 7 + [season_year]
        
        if as_of_date:
            query += " AND md.match_time_utc <= ?"
            params.append(as_of_date)
        
        query += " ORDER BY md.match_time_utc"
        
        df = pd.read_sql(query, self.conn, params=params)
        
        if len(df) < 5:  # Minimum matches needed
            return None
        
        # Remove matches without xG data
        df = df.dropna(subset=['xg_for', 'xg_against'])
        
        if len(df) < 5:
            return None
        
        # Calculate metrics
        matches_played = len(df)
        
        total_xg_for = df['xg_for'].sum()
        total_xg_against = df['xg_against'].sum()
        
        avg_xg_for = df['xg_for'].mean()
        avg_xg_against = df['xg_against'].mean()
        
        xg_difference = avg_xg_for - avg_xg_against
        
        # Calculate xG overperformance (actual goals vs expected)
        total_goals_for = df['goals_for'].sum()
        xg_overperformance = total_goals_for - total_xg_for
        
        # Calculate rating (normalized to ~1500 scale like Elo)
        # Base rating on xG difference per game
        base_rating = 1500 + (xg_difference * 200)  # Each xG difference worth ~200 points
        
        # Adjust for overperformance (but weight it less - it's often luck)
        overperformance_adjustment = (xg_overperformance / matches_played) * 50
        
        final_rating = base_rating + overperformance_adjustment
        
        return {
            'team_id': team_id,
            'season_year': season_year,
            'matches_played': matches_played,
            'xg_for': total_xg_for,
            'xg_against': total_xg_against,
            'avg_xg_for_per_game': avg_xg_for,
            'avg_xg_against_per_game': avg_xg_against,
            'xg_difference_per_game': xg_difference,
            'xg_overperformance': xg_overperformance,
            'xg_rating': final_rating
        }
    
    def calculate_league_teams_xg_ratings(self, league_id: int, season_year: int) -> pd.DataFrame:
        """
        Calculate xG ratings for all teams in a league.
        
        Returns: DataFrame with team ratings
        """
        # Get all teams in league
        team_ids = self.data_loader.get_league_teams_for_season(league_id, season_year)
        
        results = []
        
        for team_id in team_ids:
            rating = self.calculate_team_xg_rating(team_id, season_year)
            if rating:
                results.append(rating)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Normalize to 0-100 scale
        min_rating = df['xg_rating'].min()
        max_rating = df['xg_rating'].max()
        
        if max_rating > min_rating:
            df['xg_rating_normalized'] = ((df['xg_rating'] - min_rating) / (max_rating - min_rating)) * 100
        else:
            df['xg_rating_normalized'] = 50.0
        
        return df