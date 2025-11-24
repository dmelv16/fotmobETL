import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import warnings
# Suppress the specific SQLAlchemy warning from pandas
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*SQLAlchemy.*')
logger = logging.getLogger(__name__)

class FotMobDataLoader:
    """Load and prepare data from existing FotMob tables."""
    
    def __init__(self, connection):
        self.conn = connection
    
    def get_all_matches(self, season_year: Optional[int] = None, 
                       league_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load match details with flexible filtering.
        """
        query = """
            SELECT 
                md.match_id,
                md.league_id,
                md.parent_league_id,
                md.league_name,
                YEAR(md.match_time_utc) as season_year,
                md.home_team_id,
                md.home_team_name,
                md.home_team_score,
                md.away_team_id,
                md.away_team_name,
                md.away_team_score,
                md.match_time_utc,
                md.finished,
                md.cancelled
            FROM [dbo].[match_details] md
            WHERE md.finished = 1 AND md.cancelled = 0
        """
        
        params = []
        if season_year:
            query += " AND YEAR(md.match_time_utc) = ?"
            params.append(season_year)
        
        if league_id:
            query += " AND md.parent_league_id = ?"
            params.append(league_id)
        
        query += " ORDER BY md.match_time_utc, md.match_id"
        
        if params:
            return pd.read_sql(query, self.conn, params=params)
        return pd.read_sql(query, self.conn)
    
    def get_match_stats(self, match_ids: List[int] = None) -> pd.DataFrame:
        """Load match statistics including xG."""
        query = """
            SELECT 
                match_id,
                home_xg,
                away_xg,
                home_total_shots,
                away_total_shots,
                home_shots_on_target,
                away_shots_on_target,
                home_possession,
                away_possession,
                home_passes,
                away_passes,
                home_pass_accuracy_pct,
                away_pass_accuracy_pct,
                home_tackles,
                away_tackles,
                home_fouls,
                away_fouls
            FROM [dbo].[match_stats_summary]
        """
        
        if match_ids:
            placeholders = ','.join(['?'] * len(match_ids))
            query += f" WHERE match_id IN ({placeholders})"
            return pd.read_sql(query, self.conn, params=match_ids)
        
        return pd.read_sql(query, self.conn)
    
    def get_player_match_performance(self, player_id: int, 
                                    start_season: int, 
                                    end_season: int) -> pd.DataFrame:
        """
        Get a player's match-level performance across seasons.
        """
        query = """
            SELECT 
                mlp.match_id,
                YEAR(md.match_time_utc) as season_year,
                md.parent_league_id as league_id,
                md.league_name,
                mlp.team_id,
                mlp.player_id,
                mlp.player_name,
                mlp.age,
                mlp.position_id,
                mlp.is_starter,
                mlp.rating,
                
                (SELECT COUNT(*) 
                FROM [dbo].[match_events] me 
                WHERE me.match_id = mlp.match_id 
                AND me.player_id = mlp.player_id 
                AND me.event_type = 'goal') as goals,
                
                0 as assists
                
            FROM [dbo].[match_lineup_players] mlp
            INNER JOIN [dbo].[match_details] md ON mlp.match_id = md.match_id
            
            WHERE mlp.player_id = ?
            AND YEAR(md.match_time_utc) BETWEEN ? AND ?
            ORDER BY YEAR(md.match_time_utc), mlp.match_id
        """
        
        return pd.read_sql(query, self.conn, params=[player_id, start_season, end_season])
    
    def get_team_season_summary(self, team_id: int, season_year: int) -> Dict:
        """Get aggregated season stats for a team."""
        query = """
            SELECT 
                COUNT(*) as matches_played,
                SUM(CASE WHEN md.home_team_id = ? THEN md.home_team_score ELSE md.away_team_score END) as goals_for,
                SUM(CASE WHEN md.home_team_id = ? THEN md.away_team_score ELSE md.home_team_score END) as goals_against,
                SUM(CASE 
                    WHEN md.home_team_id = ? AND mss.home_xg IS NOT NULL THEN mss.home_xg
                    WHEN md.away_team_id = ? AND mss.away_xg IS NOT NULL THEN mss.away_xg
                    ELSE 0
                END) as xg_for,
                SUM(CASE 
                    WHEN md.home_team_id = ? AND mss.away_xg IS NOT NULL THEN mss.away_xg
                    WHEN md.away_team_id = ? AND mss.home_xg IS NOT NULL THEN mss.home_xg
                    ELSE 0
                END) as xg_against
            FROM [dbo].[match_details] md
            LEFT JOIN [dbo].[match_stats_summary] mss ON md.match_id = mss.match_id
            WHERE (md.home_team_id = ? OR md.away_team_id = ?)
              AND YEAR(md.match_time_utc) = ?
              AND md.finished = 1
        """
        
        params = [team_id] * 8 + [season_year]
        df = pd.read_sql(query, self.conn, params=params)
        
        if len(df) == 0:
            return None
        
        return df.iloc[0].to_dict()
    
    def get_league_teams_for_season(self, league_id: int, season_year: int) -> List[int]:
        """Get all teams that played in a league during a season."""
        query = """
            SELECT DISTINCT team_id
            FROM (
                SELECT home_team_id as team_id
                FROM [dbo].[match_details]
                WHERE parent_league_id = ? AND YEAR(match_time_utc) = ?
                UNION
                SELECT away_team_id as team_id
                FROM [dbo].[match_details]
                WHERE parent_league_id = ? AND YEAR(match_time_utc) = ?
            ) teams
        """
        
        df = pd.read_sql(query, self.conn, params=[league_id, season_year, league_id, season_year])
        return df['team_id'].tolist()
    
    def get_transfers_between_leagues(self, source_league_id: int, 
                                     dest_league_id: int,
                                     start_season: int,
                                     end_season: int) -> pd.DataFrame:
        """Identify players who moved between two specific leagues."""
        query = """
            WITH player_seasons AS (
                SELECT DISTINCT
                    mlp.player_id,
                    mlp.player_name,
                    mlp.position_id,
                    md.parent_league_id as league_id,
                    YEAR(md.match_time_utc) as season_year,
                    mlp.team_id,
                    AVG(mlp.age) as avg_age,
                    COUNT(*) as appearances
                FROM [dbo].[match_lineup_players] mlp
                INNER JOIN [dbo].[match_details] md ON mlp.match_id = md.match_id
                WHERE md.parent_league_id IN (?, ?)
                  AND YEAR(md.match_time_utc) BETWEEN ? AND ?
                GROUP BY mlp.player_id, mlp.player_name, mlp.position_id, 
                         md.parent_league_id, YEAR(md.match_time_utc), mlp.team_id
            )
            
            SELECT 
                source.player_id,
                source.player_name,
                source.position_id,
                source.league_id as source_league_id,
                source.season_year as source_season,
                source.team_id as source_team_id,
                source.avg_age as source_age,
                source.appearances as source_appearances,
                
                dest.league_id as dest_league_id,
                dest.season_year as dest_season,
                dest.team_id as dest_team_id,
                dest.avg_age as dest_age,
                dest.appearances as dest_appearances
                
            FROM player_seasons source
            INNER JOIN player_seasons dest 
                ON source.player_id = dest.player_id
                AND dest.season_year = source.season_year + 1
                AND dest.league_id != source.league_id
            
            WHERE source.league_id = ? AND dest.league_id = ?
              AND source.appearances >= 10
              AND dest.appearances >= 5
        """
        
        params = [source_league_id, dest_league_id, start_season, end_season,
                 source_league_id, dest_league_id]
        
        return pd.read_sql(query, self.conn, params=params)

    def get_all_transfers(self, start_season: int, end_season: int) -> pd.DataFrame:
        """
        Identify ALL players who moved between ANY leagues in the given window.
        Optimized to replace O(N^2) pair queries.
        """
        query = """
            WITH player_seasons AS (
                SELECT DISTINCT
                    mlp.player_id,
                    mlp.player_name,
                    mlp.position_id,
                    md.parent_league_id as league_id,
                    YEAR(md.match_time_utc) as season_year,
                    mlp.team_id,
                    AVG(mlp.age) as avg_age,
                    COUNT(*) as appearances
                FROM [dbo].[match_lineup_players] mlp
                INNER JOIN [dbo].[match_details] md ON mlp.match_id = md.match_id
                WHERE YEAR(md.match_time_utc) BETWEEN ? AND ?
                GROUP BY mlp.player_id, mlp.player_name, mlp.position_id, 
                         md.parent_league_id, YEAR(md.match_time_utc), mlp.team_id
            )
            
            SELECT 
                source.player_id,
                source.player_name,
                source.position_id,
                source.league_id as source_league_id,
                source.season_year as source_season,
                source.team_id as source_team_id,
                source.avg_age as source_age,
                source.appearances as source_appearances,
                
                dest.league_id as dest_league_id,
                dest.season_year as dest_season,
                dest.team_id as dest_team_id,
                dest.avg_age as dest_age,
                dest.appearances as dest_appearances
                
            FROM player_seasons source
            INNER JOIN player_seasons dest 
                ON source.player_id = dest.player_id
                AND dest.season_year = source.season_year + 1
                AND dest.league_id != source.league_id
            
            WHERE source.appearances >= 10
              AND dest.appearances >= 5
        """
        return pd.read_sql(query, self.conn, params=[start_season, end_season])

    def get_european_matches(self, season_year: int) -> pd.DataFrame:
        """Get all European competition matches for a season."""
        from config.league_mappings import EUROPEAN_COMPETITIONS
        
        competition_ids = list(EUROPEAN_COMPETITIONS.keys())
        
        if not competition_ids:
            return pd.DataFrame()
        
        placeholders = ','.join(['?'] * len(competition_ids))
        
        query = f"""
            SELECT 
                md.match_id,
                md.parent_league_id as competition_id,
                md.league_name as competition_name,
                md.home_team_id,
                md.home_team_name,
                md.home_team_score,
                md.away_team_id,
                md.away_team_name,
                md.away_team_score,
                md.match_time_utc,
                mss.home_xg,
                mss.away_xg
            FROM [dbo].[match_details] md
            LEFT JOIN [dbo].[match_stats_summary] mss ON md.match_id = mss.match_id
            WHERE md.parent_league_id IN ({placeholders})
              AND YEAR(md.match_time_utc) = ?
              AND md.finished = 1
        """
        
        params = competition_ids + [season_year]
        return pd.read_sql(query, self.conn, params=params)