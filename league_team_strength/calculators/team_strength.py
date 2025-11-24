import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime, date, timedelta
from config.settings import TEAM_STRENGTH_WEIGHTS

logger = logging.getLogger(__name__)

class TeamStrengthCalculator:
    """
    Team strength calculator using unified Elo scale (500-2500).
    All ratings stored on same scale for direct comparison.
    Mean ~1500, typical range 1000-2000, extremes 500-2500.
    """
    
    CONTINENTAL_COMPETITIONS = [42, 10611, 73, 10613, 10216, 10615, 297, 10043, 45, 10618, 299, 9682]
    CUP_DIVISION_LEVEL = 'Cup'
    
    def __init__(self, connection, elo_system, xg_rater, squad_evaluator, data_loader):
        self.conn = connection
        self.elo_system = elo_system
        self.xg_rater = xg_rater
        self.squad_evaluator = squad_evaluator
        self.data_loader = data_loader
    
    def calculate_league_teams(self, league_id: int, season_year: int,
                              snapshot_frequency: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate team strengths for all teams in a league.
        """
        if snapshot_frequency:
            return self._calculate_league_teams_with_timeline(
                league_id, season_year, snapshot_frequency
            )
        
        return self._calculate_league_teams_vectorized(league_id, season_year)
    
    def _calculate_league_teams_vectorized(self, league_id: int, season_year: int,
                                        as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Vectorized calculation for all teams in a league at once.
        Returns all ratings normalized to 0-100 percentile scale.
        """
        league_id = int(league_id)
        season_year = int(season_year)
        
        if as_of_date is None:
            as_of_date = f"{season_year}-06-30"
        
        team_ids = self.data_loader.get_league_teams_for_season(league_id, season_year)
        
        if not team_ids:
            return pd.DataFrame()
        
        # Get domestic league IDs for all teams at once
        domestic_leagues = self._get_bulk_domestic_leagues(team_ids, season_year)
        
        # Filter to only teams that have a domestic league
        valid_team_ids = [tid for tid in team_ids if tid in domestic_leagues and domestic_leagues[tid] is not None]
        
        if not valid_team_ids:
            return pd.DataFrame()
        
        # Fetch all RAW ratings
        elo_ratings = self._get_bulk_elo_ratings(valid_team_ids, as_of_date)
        xg_ratings = self._get_bulk_xg_ratings(valid_team_ids, season_year, as_of_date)
        squad_ratings = self._get_bulk_squad_ratings(valid_team_ids, season_year)
        
        # Fetch context data
        contexts = self._get_bulk_team_contexts(valid_team_ids, season_year, as_of_date)
        xg_stats = self._get_bulk_xg_stats(valid_team_ids, season_year, as_of_date)
        
        # Normalize to 0-100 percentile scale (ensures equal influence regardless of original scale)
        elo_components = self._percentile_normalize_vectorized(elo_ratings, season_year, 'elo')
        xg_components = self._percentile_normalize_vectorized(xg_ratings, season_year, 'xg')
        squad_components = self._percentile_normalize_vectorized(squad_ratings, season_year, 'squad')
        
        # Calculate coaching effect (on 0-100 scale)
        coaching_effects = self._calculate_coaching_effects_vectorized(
            elo_components, xg_components, squad_components
        )
        
        # Calculate composite strength (on 0-100 scale)
        overall_strengths = self._calculate_composite_strengths_vectorized(
            elo_components, xg_components, squad_components, coaching_effects
        )
        
        # Build results with normalized values
        results = self._build_results_dataframe(
            valid_team_ids, contexts, domestic_leagues, xg_stats,
            elo_ratings, xg_ratings, squad_ratings,  # RAW values for reference
            elo_components, xg_components, squad_components,  # NORMALIZED (0-100)
            coaching_effects, overall_strengths,  # NORMALIZED (0-100)
            season_year, as_of_date
        )
        
        return results
    
    def _percentile_normalize_vectorized(self, ratings: pd.Series, season_year: int, 
                                        component_type: str) -> pd.Series:
        """
        Normalize ratings to 0-100 percentile scale using actual distribution.
        NO CLIPPING - preserves true relative positions.
        
        Args:
            ratings: Series of raw ratings
            season_year: Season year for reference data
            component_type: 'elo', 'xg', or 'squad'
        
        Returns:
            Series of percentile scores (0-100)
        """
        # Handle default Elo (1500) as missing data
        if component_type == 'elo':
            ratings = ratings.replace(1500, np.nan)
        
        # Get reference distribution from database
        reference_values = self._get_reference_distribution(season_year, component_type)
        
        if not reference_values or len(reference_values) == 0:
            # No reference data - use z-score normalization with assumed distribution
            return self._zscore_normalize_vectorized(ratings, component_type)
        
        # Calculate percentile for each rating
        normalized = ratings.copy()
        
        for idx in ratings.index:
            if pd.isna(ratings[idx]):
                normalized[idx] = np.nan
                continue
            
            value = ratings[idx]
            # Percentile = % of reference values below this value
            percentile = (sum(1 for r in reference_values if r < value) / len(reference_values)) * 100
            normalized[idx] = percentile
        
        return normalized
    
    def _zscore_normalize_vectorized(self, ratings: pd.Series, component_type: str) -> pd.Series:
        """
        Fallback normalization using z-scores when no reference data exists.
        """
        # Typical distributions for each component
        typical_params = {
            'elo': {'mean': 1500, 'std': 150},
            'xg': {'mean': 1500, 'std': 150},
            'squad': {'mean': 400, 'std': 100}
        }
        
        params = typical_params.get(component_type, {'mean': 50, 'std': 20})
        
        # Z-score: (value - mean) / std
        z_scores = (ratings - params['mean']) / params['std']
        
        # Convert to 0-100 scale: z=0 (mean) => 50, z=±2 => 10/90
        normalized = 50 + (z_scores * 20)
        
        return normalized
    
    def _get_reference_distribution(self, season_year: int, component_type: str) -> List[float]:
        """
        Get reference distribution of ratings for normalization.
        Currently returns empty list to use z-score normalization.
        """
        # Always use z-score normalization - no reference data stored
        return []
    
    def _calculate_coaching_effects_vectorized(self, elo_scores: pd.Series,
                                               xg_scores: pd.Series,
                                               squad_scores: pd.Series) -> pd.Series:
        """
        Calculate coaching effect = actual performance - expected performance.
        Expected = average of xG and squad components.
        NO CLIPPING on coaching effect.
        """
        # Expected performance (mean of xg and squad where available)
        expected = pd.DataFrame({'xg': xg_scores, 'squad': squad_scores}).mean(axis=1, skipna=True)
        
        # Coaching effect = actual (elo) - expected
        coaching = elo_scores - expected
        
        # Set to NaN where elo is NaN
        coaching[elo_scores.isna()] = np.nan
        
        # If no expected (no xg or squad), set to 0
        coaching[(expected.isna()) & (elo_scores.notna())] = 0.0
        
        # NO CLIPPING - let extreme coaching effects show through
        return coaching
    
    def _calculate_composite_strengths_vectorized(self, elo_comp: pd.Series,
                                                   xg_comp: pd.Series,
                                                   squad_comp: pd.Series,
                                                   coaching_comp: pd.Series) -> pd.Series:
        """
        Calculate composite strength using weighted average.
        NO CLIPPING on final result.
        """
        weights = TEAM_STRENGTH_WEIGHTS
        
        # Create DataFrame of components
        components = pd.DataFrame({
            'elo': elo_comp * weights['elo_rating'],
            'xg': xg_comp * weights['xg_performance'],
            'squad': squad_comp * weights['squad_quality'],
            'coaching': coaching_comp * weights['coaching_effect']
        })
        
        # Weighted sum
        weighted_sum = components.sum(axis=1, skipna=True)
        
        # Total weight (only for non-NaN components)
        weight_values = pd.Series({
            'elo': weights['elo_rating'],
            'xg': weights['xg_performance'],
            'squad': weights['squad_quality'],
            'coaching': weights['coaching_effect']
        })
        
        total_weights = components.notna().mul(weight_values).sum(axis=1)
        
        # Weighted average
        overall = weighted_sum / total_weights
        
        # Replace inf/-inf with NaN
        overall = overall.replace([np.inf, -np.inf], np.nan)
        
        # NO CLIPPING - preserve true values
        return overall
    
    def _build_results_dataframe(self, team_ids: List[int], contexts: pd.DataFrame,
                                 domestic_leagues: Dict[int, Optional[int]],
                                 xg_stats: pd.DataFrame,
                                 elo_raw: pd.Series, xg_raw: pd.Series, squad_raw: pd.Series,
                                 elo_comp: pd.Series, xg_comp: pd.Series, squad_comp: pd.Series,
                                 coaching_comp: pd.Series, overall_comp: pd.Series,
                                 season_year: int, as_of_date: str) -> pd.DataFrame:
        """
        Build results DataFrame with BOTH raw and normalized values.
        """
        results = []
        
        for team_id in team_ids:
            if team_id not in contexts.index:
                continue
            
            ctx = contexts.loc[team_id]
            domestic_league_id = domestic_leagues.get(team_id)
            
            if domestic_league_id is None:
                continue
            
            def safe_get(series, tid):
                if tid not in series.index:
                    return None
                val = series.loc[tid]
                return None if pd.isna(val) else float(val)
            
            # Get xG stats
            xg_for = None
            xg_against = None
            if team_id in xg_stats.index:
                xg_row = xg_stats.loc[team_id]
                xg_for = None if pd.isna(xg_row['xg_for']) else float(xg_row['xg_for'])
                xg_against = None if pd.isna(xg_row['xg_against']) else float(xg_row['xg_against'])
            
            result = {
                'team_id': team_id,
                'team_name': ctx['team_name'],
                'league_id': domestic_league_id,
                'season_year': season_year,
                'as_of_date': as_of_date,
                # NORMALIZED VALUES (0-100 percentile scale)
                'elo_rating': safe_get(elo_comp, team_id),
                'xg_performance_rating': safe_get(xg_comp, team_id),
                'squad_quality_rating': safe_get(squad_comp, team_id),
                'coaching_effect': safe_get(coaching_comp, team_id),
                'overall_strength': safe_get(overall_comp, team_id),
                # Context
                'matches_played': int(ctx['matches_played']),
                'league_position': None if pd.isna(ctx.get('league_position')) else int(ctx['league_position']),
                'points': int(ctx['points']) if pd.notna(ctx['points']) else 0,
                'goals_for': int(ctx['goals_for']) if pd.notna(ctx['goals_for']) else 0,
                'goals_against': int(ctx['goals_against']) if pd.notna(ctx['goals_against']) else 0,
                'xg_for': xg_for,
                'xg_against': xg_against
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    # ========== BULK DATA FETCHING METHODS ==========
    
    def _get_bulk_domestic_leagues(self, team_ids: List[int], season_year: int) -> Dict[int, Optional[int]]:
        """Get domestic league IDs for multiple teams at once."""
        if not team_ids:
            return {}
        
        continental_ids_str = ','.join(map(str, self.CONTINENTAL_COMPETITIONS))
        team_ids_str = ','.join(map(str, team_ids))
        
        query = f"""
            WITH LeagueMatches AS (
                SELECT 
                    team_id,
                    league_id,
                    DivisionLevel,
                    COUNT(*) as match_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY team_id 
                        ORDER BY 
                            CASE 
                                WHEN DivisionLevel = '1' THEN 1
                                WHEN DivisionLevel = '2' THEN 2
                                WHEN DivisionLevel = '3' THEN 3
                                ELSE 99
                            END,
                            COUNT(*) DESC
                    ) as rn
                FROM (
                    SELECT 
                        md.home_team_id as team_id, 
                        COALESCE(md.parent_league_id, md.league_id) as league_id
                    FROM [dbo].[match_details] md
                    WHERE md.home_team_id IN ({team_ids_str})
                    AND YEAR(md.match_time_utc) = ?
                    AND md.finished = 1
                    AND COALESCE(md.parent_league_id, md.league_id) IS NOT NULL
                    AND COALESCE(md.parent_league_id, md.league_id) NOT IN ({continental_ids_str})
                    
                    UNION ALL
                    
                    SELECT 
                        md.away_team_id as team_id, 
                        COALESCE(md.parent_league_id, md.league_id) as league_id
                    FROM [dbo].[match_details] md
                    WHERE md.away_team_id IN ({team_ids_str})
                    AND YEAR(md.match_time_utc) = ?
                    AND md.finished = 1
                    AND COALESCE(md.parent_league_id, md.league_id) IS NOT NULL
                    AND COALESCE(md.parent_league_id, md.league_id) NOT IN ({continental_ids_str})
                ) matches
                LEFT JOIN [dbo].[LeagueDivisions] ld 
                    ON matches.league_id = ld.LeagueID
                WHERE COALESCE(ld.DivisionLevel, '') != ?
                AND ld.DivisionLevel IS NOT NULL 
                AND ld.DivisionLevel NOT IN ('Cup', 'Continental')
                GROUP BY team_id, league_id, DivisionLevel
            )
            SELECT team_id, league_id
            FROM LeagueMatches
            WHERE rn = 1
        """
        
        params = [season_year, season_year, self.CUP_DIVISION_LEVEL]
        df = pd.read_sql(query, self.conn, params=params)
        
        result = {}
        for _, row in df.iterrows():
            result[int(row['team_id'])] = int(row['league_id']) if pd.notna(row['league_id']) else None
        
        for team_id in team_ids:
            if team_id not in result:
                result[team_id] = None
        
        return result
    
    def _get_bulk_xg_stats(self, team_ids: List[int], season_year: int, 
                        as_of_date: str) -> pd.DataFrame:
        """Get xG for/against stats for multiple teams at once."""
        if not team_ids:
            return pd.DataFrame()
        
        team_ids_str = ','.join(map(str, team_ids))
        
        query = f"""
            SELECT 
                team_id,
                SUM(xg_for) as xg_for,
                SUM(xg_against) as xg_against
            FROM (
                SELECT 
                    md.home_team_id as team_id,
                    ISNULL(ms.home_xg, 0) as xg_for,
                    ISNULL(ms.away_xg, 0) as xg_against
                FROM [dbo].[match_details] md
                LEFT JOIN [dbo].[match_stats_summary] ms ON md.match_id = ms.match_id
                WHERE md.home_team_id IN ({team_ids_str})
                AND YEAR(md.match_time_utc) = ?
                AND CAST(md.match_time_utc AS DATE) <= ?
                AND md.finished = 1
                
                UNION ALL
                
                SELECT 
                    md.away_team_id as team_id,
                    ISNULL(ms.away_xg, 0) as xg_for,
                    ISNULL(ms.home_xg, 0) as xg_against
                FROM [dbo].[match_details] md
                LEFT JOIN [dbo].[match_stats_summary] ms ON md.match_id = ms.match_id
                WHERE md.away_team_id IN ({team_ids_str})
                AND YEAR(md.match_time_utc) = ?
                AND CAST(md.match_time_utc AS DATE) <= ?
                AND md.finished = 1
            ) combined
            GROUP BY team_id
        """
        
        df = pd.read_sql(query, self.conn, 
                        params=[season_year, as_of_date, season_year, as_of_date])
        
        return df.set_index('team_id')

    def _get_bulk_elo_ratings(self, team_ids: List[int], as_of_date: str) -> pd.Series:
        """Get Elo ratings for multiple teams at once."""
        ratings = {}
        for team_id in team_ids:
            rating = self.elo_system.get_team_rating(team_id, as_of_date=as_of_date)
            ratings[team_id] = np.nan if rating is None else rating
        return pd.Series(ratings, dtype=float)

    def _get_bulk_xg_ratings(self, team_ids: List[int], season_year: int, 
                            as_of_date: str) -> pd.Series:
        """Get xG ratings for multiple teams at once."""
        ratings = {}
        for team_id in team_ids:
            xg_data = self.xg_rater.calculate_team_xg_rating(team_id, season_year, as_of_date)
            ratings[team_id] = xg_data['xg_rating'] if xg_data and 'xg_rating' in xg_data else np.nan
        return pd.Series(ratings, dtype=float)

    def _get_bulk_squad_ratings(self, team_ids: List[int], season_year: int) -> pd.Series:
        """Get squad quality ratings for multiple teams at once."""
        ratings = {}
        for team_id in team_ids:
            squad_data = self.squad_evaluator.calculate_team_squad_quality(team_id, season_year)
            ratings[team_id] = squad_data['squad_quality_rating'] if squad_data and 'squad_quality_rating' in squad_data else np.nan
        return pd.Series(ratings, dtype=float)
    
    def _get_bulk_team_contexts(self, team_ids: List[int], season_year: int, 
                               as_of_date: str) -> pd.DataFrame:
        """Get contexts for all teams in a single query."""
        if not team_ids:
            return pd.DataFrame()
        
        team_ids_str = ','.join(map(str, team_ids))
        domestic_leagues = self._get_bulk_domestic_leagues(team_ids, season_year)
        unique_league_ids = list(set([lid for lid in domestic_leagues.values() if lid is not None]))
        
        if not unique_league_ids:
            return pd.DataFrame()
        
        stats_query = f"""
            SELECT 
                team_id,
                MAX(team_name) as team_name,
                COUNT(*) as matches_played,
                SUM(points) as points,
                SUM(goals_for) as goals_for,
                SUM(goals_against) as goals_against
            FROM (
                SELECT 
                    md.home_team_id as team_id,
                    md.home_team_name as team_name,
                    CASE 
                        WHEN md.home_team_score > md.away_team_score THEN 3
                        WHEN md.home_team_score = md.away_team_score THEN 1
                        ELSE 0
                    END as points,
                    md.home_team_score as goals_for,
                    md.away_team_score as goals_against
                FROM [dbo].[match_details] md
                WHERE md.home_team_id IN ({team_ids_str})
                  AND YEAR(md.match_time_utc) = ?
                  AND CAST(md.match_time_utc AS DATE) <= ?
                  AND md.finished = 1
                
                UNION ALL
                
                SELECT 
                    md.away_team_id as team_id,
                    md.away_team_name as team_name,
                    CASE 
                        WHEN md.away_team_score > md.home_team_score THEN 3
                        WHEN md.away_team_score = md.home_team_score THEN 1
                        ELSE 0
                    END as points,
                    md.away_team_score as goals_for,
                    md.home_team_score as goals_against
                FROM [dbo].[match_details] md
                WHERE md.away_team_id IN ({team_ids_str})
                  AND YEAR(md.match_time_utc) = ?
                  AND CAST(md.match_time_utc AS DATE) <= ?
                  AND md.finished = 1
            ) combined
            GROUP BY team_id
        """
        
        stats_df = pd.read_sql(stats_query, self.conn, 
                              params=[season_year, as_of_date, season_year, as_of_date])
        
        league_positions = self._get_bulk_league_positions(
            team_ids, domestic_leagues, unique_league_ids, season_year, as_of_date
        )
        
        stats_df['league_position'] = stats_df['team_id'].map(league_positions)
        
        return stats_df.set_index('team_id')
    
    def _get_bulk_league_positions(self, team_ids: List[int], 
                                domestic_leagues: Dict[int, Optional[int]],
                                unique_league_ids: List[int],
                                season_year: int, as_of_date: str) -> Dict[int, Optional[int]]:
        """Calculate league positions for all teams at once."""
        if not unique_league_ids:
            return {}
        
        league_ids_str = ','.join(map(str, unique_league_ids))
        
        query = f"""
            SELECT 
                league_id,
                team_id,
                SUM(points) as points,
                SUM(goals_for) - SUM(goals_against) as goal_difference
            FROM (
                SELECT 
                    COALESCE(md.parent_league_id, md.league_id) as league_id,
                    md.home_team_id as team_id,
                    CASE 
                        WHEN md.home_team_score > md.away_team_score THEN 3
                        WHEN md.home_team_score = md.away_team_score THEN 1
                        ELSE 0
                    END as points,
                    md.home_team_score as goals_for,
                    md.away_team_score as goals_against
                FROM [dbo].[match_details] md
                WHERE COALESCE(md.parent_league_id, md.league_id) IN ({league_ids_str})
                AND YEAR(md.match_time_utc) = ?
                AND CAST(md.match_time_utc AS DATE) <= ?
                AND md.finished = 1
                
                UNION ALL
                
                SELECT 
                    COALESCE(md.parent_league_id, md.league_id) as league_id,
                    md.away_team_id as team_id,
                    CASE 
                        WHEN md.away_team_score > md.home_team_score THEN 3
                        WHEN md.away_team_score = md.home_team_score THEN 1
                        ELSE 0
                    END as points,
                    md.away_team_score as goals_for,
                    md.home_team_score as goals_against
                FROM [dbo].[match_details] md
                WHERE COALESCE(md.parent_league_id, md.league_id) IN ({league_ids_str})
                AND YEAR(md.match_time_utc) = ?
                AND CAST(md.match_time_utc AS DATE) <= ?
                AND md.finished = 1
            ) combined
            GROUP BY league_id, team_id
            ORDER BY league_id, points DESC, goal_difference DESC
        """
        
        standings_df = pd.read_sql(query, self.conn, 
                                params=[season_year, as_of_date, season_year, as_of_date])
        
        standings_df['position'] = standings_df.groupby('league_id').cumcount() + 1
        
        positions = {}
        for team_id in team_ids:
            league_id = domestic_leagues.get(team_id)
            if league_id is None:
                positions[team_id] = None
                continue
            
            team_row = standings_df[
                (standings_df['team_id'] == team_id) & 
                (standings_df['league_id'] == league_id)
            ]
            
            if len(team_row) > 0:
                positions[team_id] = int(team_row.iloc[0]['position'])
            else:
                positions[team_id] = None
        
        return positions
    
    # ========== DATABASE METHODS ==========
    
    def _bulk_save_to_database(self, results_df: pd.DataFrame):
        """Bulk insert/update using executemany for SQL Server."""
        if len(results_df) == 0:
            return
        
        cursor = self.conn.cursor()
        
        records = []
        for _, row in results_df.iterrows():
            records.append((
                int(row['team_id']),
                str(row['team_name']) if pd.notna(row['team_name']) else None,
                int(row['league_id']),
                int(row['season_year']),
                str(row['as_of_date']),
                # NORMALIZED values
                float(row['elo_rating']) if pd.notna(row['elo_rating']) else None,
                float(row['xg_performance_rating']) if pd.notna(row['xg_performance_rating']) else None,
                float(row['squad_quality_rating']) if pd.notna(row['squad_quality_rating']) else None,
                float(row['coaching_effect']) if pd.notna(row['coaching_effect']) else None,
                float(row['overall_strength']) if pd.notna(row['overall_strength']) else None,
                int(row['matches_played']) if pd.notna(row['matches_played']) else 0,
                int(row['league_position']) if pd.notna(row['league_position']) else None,
                int(row['points']) if pd.notna(row['points']) else 0,
                int(row['goals_for']) if pd.notna(row['goals_for']) else 0,
                int(row['goals_against']) if pd.notna(row['goals_against']) else 0,
                float(row['xg_for']) if pd.notna(row['xg_for']) else None,
                float(row['xg_against']) if pd.notna(row['xg_against']) else None
            ))
        
        merge_query = """
            MERGE [dbo].[team_strength] AS target
            USING (SELECT ? AS team_id, ? AS team_name, ? AS league_id, ? AS season_year, 
                        ? AS as_of_date,
                        ? AS elo_rating, ? AS xg_performance_rating, ? AS squad_quality_rating,
                        ? AS coaching_effect, ? AS overall_strength,
                        ? AS matches_played, ? AS league_position, ? AS points,
                        ? AS goals_for, ? AS goals_against, ? AS xg_for, ? AS xg_against) AS source
            ON target.team_id = source.team_id 
            AND target.league_id = source.league_id
            AND target.season_year = source.season_year
            AND target.as_of_date = source.as_of_date
            WHEN MATCHED THEN
                UPDATE SET
                    team_name = source.team_name,
                    elo_rating = source.elo_rating,
                    xg_performance_rating = source.xg_performance_rating,
                    squad_quality_rating = source.squad_quality_rating,
                    coaching_effect = source.coaching_effect,
                    overall_strength = source.overall_strength,
                    matches_played = source.matches_played,
                    league_position = source.league_position,
                    points = source.points,
                    goals_for = source.goals_for,
                    goals_against = source.goals_against,
                    xg_for = source.xg_for,
                    xg_against = source.xg_against,
                    last_updated = GETDATE()
            WHEN NOT MATCHED THEN
                INSERT (team_id, team_name, league_id, season_year, as_of_date,
                    elo_rating, xg_performance_rating, squad_quality_rating,
                    coaching_effect, overall_strength, matches_played,
                    league_position, points, goals_for, goals_against,
                    xg_for, xg_against)
                VALUES (source.team_id, source.team_name, source.league_id, 
                    source.season_year, source.as_of_date,
                    source.elo_rating, source.xg_performance_rating, source.squad_quality_rating,
                    source.coaching_effect, source.overall_strength, source.matches_played,
                    source.league_position, source.points, 
                    source.goals_for, source.goals_against,
                    source.xg_for, source.xg_against);
        """
        
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            cursor.executemany(merge_query, batch)
        
        self.conn.commit()
    
    # ========== HELPER METHODS (Unchanged) ==========
    
    def _calculate_league_teams_with_timeline(self, league_id: int, season_year: int,
                                             snapshot_frequency: str) -> pd.DataFrame:
        """Generate timeline snapshots for all teams in a league."""
        team_ids = self.data_loader.get_league_teams_for_season(int(league_id), int(season_year))
        
        if not team_ids:
            return pd.DataFrame()
        
        snapshot_dates = self._generate_snapshot_dates(season_year, snapshot_frequency)
        
        if not snapshot_dates:
            return pd.DataFrame()
        
        all_results = []
        
        for as_of_date in snapshot_dates:
            date_results = self._calculate_league_teams_vectorized(
                league_id, season_year, as_of_date=as_of_date
            )
            
            if len(date_results) > 0:
                all_results.append(date_results)
        
        if not all_results:
            return pd.DataFrame()
        
        df = pd.concat(all_results, ignore_index=True)
        df = df[df['matches_played'] > 0]
        df = df.sort_values(['as_of_date', 'overall_strength'], ascending=[True, False])
        
        if len(df) > 0:
            self._bulk_save_to_database(df)
        
        return df
    
    def _generate_snapshot_dates(self, season_year: int, snapshot_frequency: str) -> List[str]:
        """Generate snapshot dates for a season."""
        # Use Jan 1 to Dec 31 for the season year
        season_start = date(season_year, 1, 1)
        season_end = date(season_year, 12, 31)
        
        dates = []
        
        if snapshot_frequency == 'monthly':
            current_date = season_start
            while current_date <= season_end:
                dates.append(current_date.strftime('%Y-%m-%d'))
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
        
        elif snapshot_frequency == 'quarterly':
            current_date = season_start
            while current_date <= season_end:
                dates.append(current_date.strftime('%Y-%m-%d'))
                new_month = current_date.month + 3
                if new_month > 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=new_month - 12, day=1)
                else:
                    current_date = current_date.replace(month=new_month, day=1)
        
        elif snapshot_frequency == 'biweekly':
            current_date = season_start
            while current_date <= season_end:
                dates.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=14)
        
        else:
            raise ValueError(f"Unknown snapshot_frequency: {snapshot_frequency}")
        
        return dates