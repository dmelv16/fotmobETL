import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime
from config.settings import TEAM_STRENGTH_WEIGHTS, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX

logger = logging.getLogger(__name__)

class TeamStrengthCalculator:
    """
    Master calculator for team strength scores.
    Combines Elo, xG performance, squad quality, and coaching effect.
    """
    
    def __init__(self, connection, elo_system, xg_rater, squad_evaluator, data_loader):
        self.conn = connection
        self.elo_system = elo_system
        self.xg_rater = xg_rater
        self.squad_evaluator = squad_evaluator
        self.data_loader = data_loader
    
    def calculate_team_strength(self, team_id: int, season_year: int,
                               as_of_date: Optional[str] = None,
                               save_to_db: bool = True) -> Optional[Dict]:
        """
        Calculate comprehensive team strength score.
        
        Components:
        1. Elo rating (match results)
        2. xG performance (underlying quality)
        3. Squad quality (player aggregation)
        4. Coaching effect (over/underperformance)
        
        Returns: Dict with component scores and composite strength
        """
        logger.info(f"Calculating team strength for team {team_id}, season {season_year}")
        
        # Component 1: Elo Rating
        elo_rating = self.elo_system.get_team_rating(team_id)
        
        if elo_rating == 1500:  # Default rating = no data
            elo_component = None
        else:
            # Normalize Elo to 0-100 scale (typical range: 1200-1800)
            elo_component = ((elo_rating - 1200) / 600) * 100
            elo_component = np.clip(elo_component, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX)
        
        # Component 2: xG Performance
        xg_data = self.xg_rater.calculate_team_xg_rating(team_id, season_year, as_of_date)
        
        if xg_data:
            xg_rating = xg_data['xg_rating']
            # Normalize (typical range: 1200-1800)
            xg_component = ((xg_rating - 1200) / 600) * 100
            xg_component = np.clip(xg_component, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX)
        else:
            xg_component = None
        
        # Component 3: Squad Quality
        squad_data = self.squad_evaluator.calculate_team_squad_quality(team_id, season_year)
        
        if squad_data:
            squad_rating = squad_data['squad_quality_rating']
            # Normalize (typical range: 1200-1800)
            squad_component = ((squad_rating - 1200) / 600) * 100
            squad_component = np.clip(squad_component, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX)
        else:
            squad_component = None
        
        # Component 4: Coaching Effect
        coaching_effect = self._calculate_coaching_effect(team_id, season_year, 
                                                          elo_component, xg_component, 
                                                          squad_component)
        
        # Calculate composite strength
        weights = TEAM_STRENGTH_WEIGHTS
        
        components = {
            'elo': (elo_component, weights['elo_rating']),
            'xg': (xg_component, weights['xg_performance']),
            'squad': (squad_component, weights['squad_quality']),
            'coaching': (coaching_effect, weights['coaching_effect'])
        }
        
        # Weighted average (adjust for missing components)
        total_weight = 0
        weighted_sum = 0
        
        for comp_name, (score, weight) in components.items():
            if score is not None:
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            logger.warning(f"No valid components for team {team_id}, season {season_year}")
            return None
        
        overall_strength = weighted_sum / total_weight
        overall_strength = np.clip(overall_strength, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX)
        
        # Get additional context
        context = self._get_team_context(team_id, season_year)
        
        result = {
            'team_id': team_id,
            'team_name': context.get('team_name'),
            'league_id': context.get('league_id'),
            'season_year': season_year,
            'as_of_date': as_of_date or datetime.now().date(),
            'elo_rating': elo_component,
            'xg_performance_rating': xg_component,
            'squad_quality_rating': squad_component,
            'coaching_effect': coaching_effect,
            'overall_strength': overall_strength,
            'matches_played': context.get('matches_played', 0),
            'league_position': context.get('league_position'),
            'points': context.get('points'),
            'goals_for': context.get('goals_for'),
            'goals_against': context.get('goals_against'),
            'xg_for': xg_data.get('xg_for') if xg_data else None,
            'xg_against': xg_data.get('xg_against') if xg_data else None
        }
        
        logger.info(f"Team {team_id} strength: {overall_strength:.1f}")
        
        if save_to_db:
            self._save_to_database(result)
        
        return result
    
    def _calculate_coaching_effect(self, team_id: int, season_year: int,
                                   elo_score: Optional[float],
                                   xg_score: Optional[float],
                                   squad_score: Optional[float]) -> Optional[float]:
        """
        Calculate coaching effect = actual performance vs expected from squad/xG.
        
        Positive value = team overperforming (good coaching)
        Negative value = team underperforming (poor coaching or bad luck)
        """
        if elo_score is None:
            return None
        
        # Expected performance is average of xG and squad quality
        expected_components = [s for s in [xg_score, squad_score] if s is not None]
        
        if not expected_components:
            return 0.0
        
        expected_performance = np.mean(expected_components)
        
        # Coaching effect = actual (Elo) - expected
        coaching_effect = elo_score - expected_performance
        
        # Normalize to similar scale as other components
        # Typical range: -20 to +20
        coaching_effect = np.clip(coaching_effect, -20, 20)
        
        return float(coaching_effect)
    
    def _get_team_context(self, team_id: int, season_year: int) -> Dict:
        """
        Get team context (league position, points, etc.)
        """
        # Get team's matches
        query = """
            SELECT TOP 1
                md.league_id,
                md.home_team_name,
                COUNT(*) as matches_played,
                SUM(CASE 
                    WHEN md.home_team_id = ? AND md.home_team_score > md.away_team_score THEN 3
                    WHEN md.away_team_id = ? AND md.away_team_score > md.home_team_score THEN 3
                    WHEN md.home_team_score = md.away_team_score THEN 1
                    ELSE 0
                END) as points,
                SUM(CASE WHEN md.home_team_id = ? THEN md.home_team_score ELSE md.away_team_score END) as goals_for,
                SUM(CASE WHEN md.home_team_id = ? THEN md.away_team_score ELSE md.home_team_score END) as goals_against
            FROM [dbo].[match_details] md
            WHERE (md.home_team_id = ? OR md.away_team_id = ?)
              AND YEAR(md.match_time_utc) = ?
              AND md.finished = 1
            GROUP BY md.league_id, md.home_team_name
        """
        
        params = [team_id] * 6 + [season_year]
        df = pd.read_sql(query, self.conn, params=params)
        
        if len(df) == 0:
            return {}
        
        row = df.iloc[0]
        
        # Get league position (simplified - would need full league table logic)
        league_position = self._get_league_position(team_id, row['league_id'], season_year, row['points'])
        
        return {
            'team_name': row['home_team_name'],
            'league_id': row['league_id'],
            'matches_played': row['matches_played'],
            'points': row['points'],
            'goals_for': row['goals_for'],
            'goals_against': row['goals_against'],
            'league_position': league_position
        }
    
    def _get_league_position(self, team_id: int, league_id: int, 
                           season_year: int, team_points: int) -> Optional[int]:
        """
        Determine team's position in league standings.
        """
        # Get all teams' points in league
        query = """
            SELECT 
                CASE WHEN md.home_team_id = teams.team_id THEN md.home_team_id ELSE md.away_team_id END as team_id,
                SUM(CASE 
                    WHEN md.home_team_id = teams.team_id AND md.home_team_score > md.away_team_score THEN 3
                    WHEN md.away_team_id = teams.team_id AND md.away_team_score > md.home_team_score THEN 3
                    WHEN md.home_team_score = md.away_team_score THEN 1
                    ELSE 0
                END) as points
            FROM [dbo].[match_details] md
            CROSS APPLY (
                SELECT md.home_team_id as team_id
                UNION
                SELECT md.away_team_id
            ) teams
            WHERE md.league_id = ?
              AND YEAR(md.match_time_utc) = ?
              AND md.finished = 1
            GROUP BY CASE WHEN md.home_team_id = teams.team_id THEN md.home_team_id ELSE md.away_team_id END
            ORDER BY points DESC
        """
        
        df = pd.read_sql(query, self.conn, params=[league_id, season_year])
        
        if len(df) == 0:
            return None
        
        # Find position
        for i, row in df.iterrows():
            if row['team_id'] == team_id:
                return i + 1
        
        return None
    
    def _save_to_database(self, result: Dict):
        """Save team strength to database."""
        cursor = self.conn.cursor()
        
        # Insert (team strength is time-series, so we always insert new records)
        cursor.execute("""
            INSERT INTO [dbo].[team_strength] (
                team_id, team_name, league_id, season_year, as_of_date,
                elo_rating, xg_performance_rating, squad_quality_rating,
                coaching_effect, overall_strength, matches_played,
                league_position, points, goals_for, goals_against,
                xg_for, xg_against
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, result['team_id'], result['team_name'], result['league_id'],
             result['season_year'], result['as_of_date'],
             result['elo_rating'], result['xg_performance_rating'],
             result['squad_quality_rating'], result['coaching_effect'],
             result['overall_strength'], result['matches_played'],
             result['league_position'], result['points'],
             result['goals_for'], result['goals_against'],
             result['xg_for'], result['xg_against'])
        
        self.conn.commit()
        logger.info(f"Saved team strength for team {result['team_id']}, season {result['season_year']}")
    
    def calculate_league_teams(self, league_id: int, season_year: int) -> pd.DataFrame:
        """
        Calculate strength for all teams in a league.
        
        Returns: DataFrame sorted by strength
        """
        team_ids = self.data_loader.get_league_teams_for_season(league_id, season_year)
        
        results = []
        
        for team_id in team_ids:
            try:
                result = self.calculate_team_strength(team_id, season_year, save_to_db=True)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error calculating strength for team {team_id}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('overall_strength', ascending=False)
        
        return df