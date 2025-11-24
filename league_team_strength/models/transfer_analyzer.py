import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from config.settings import MIN_TRANSFERS_FOR_LEAGUE_PAIR, TRANSFER_RECENCY_WEIGHT

logger = logging.getLogger(__name__)

class TransferPerformanceAnalyzer:
    """
    Analyzes player performance changes across transfers to infer league strength gaps.
    Uses multiple metrics beyond just goals to capture all player types.
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader
    
    def calculate_player_transfer_impact(self, player_id: int, 
                                        source_season: int, 
                                        dest_season: int,
                                        source_league_id: int,
                                        dest_league_id: int,
                                        source_team_id: int = None,
                                        dest_team_id: int = None) -> Optional[Dict]:
        """
        Calculate a single player's performance change across a transfer.
        Uses multiple metrics to capture performance of all player types.
        """
        # Get player performance in both seasons
        perf_df = self.data_loader.get_player_match_performance(
            player_id, source_season, dest_season
        )
        
        if len(perf_df) == 0:
            return None
        
        # Filter for specific league - handle both parent_league_id and league_id columns
        league_col = 'parent_league_id' if 'parent_league_id' in perf_df.columns else 'league_id'
        
        source_df = perf_df[
            (perf_df['season_year'] == source_season) & 
            (perf_df[league_col] == source_league_id)
        ]
        
        dest_df = perf_df[
            (perf_df['season_year'] == dest_season) & 
            (perf_df[league_col] == dest_league_id)
        ]
        
        # If we have team IDs, filter by those too (more precise)
        if source_team_id:
            source_df = source_df[source_df['team_id'] == source_team_id]
        if dest_team_id:
            dest_df = dest_df[dest_df['team_id'] == dest_team_id]
        
        if len(source_df) == 0 or len(dest_df) == 0:
            return None

        # Calculate playing time
        source_starts = source_df['is_starter'].sum()
        source_apps = len(source_df)
        dest_starts = dest_df['is_starter'].sum()
        dest_apps = len(dest_df)
        
        # Estimate minutes (starters ~75 min, subs ~20 min on average)
        source_minutes = (source_starts * 75) + ((source_apps - source_starts) * 20)
        dest_minutes = (dest_starts * 75) + ((dest_apps - dest_starts) * 20)
        
        # Minimum thresholds - must play meaningful minutes
        if source_minutes < 450 or dest_minutes < 270:  # ~5 full games and ~3 full games
            return None
        
        # Calculate performance metrics
        source_metrics = self._calculate_performance_metrics(source_df, source_minutes)
        dest_metrics = self._calculate_performance_metrics(dest_df, dest_minutes)
        
        if source_metrics is None or dest_metrics is None:
            return None
        
        # Calculate percentage changes for each metric
        changes = {}
        for metric in ['goals_per90', 'assists_per90', 'rating', 'start_rate']:
            if source_metrics[metric] > 0:
                pct_change = ((dest_metrics[metric] - source_metrics[metric]) / source_metrics[metric]) * 100
            else:
                # Handle zero baseline
                if dest_metrics[metric] > 0:
                    pct_change = 100.0  # Improvement from nothing
                else:
                    pct_change = 0.0
            changes[f'{metric}_change_pct'] = pct_change
        
        # Composite performance change (weighted by importance)
        weights = {
            'goals_per90_change_pct': 0.30,
            'assists_per90_change_pct': 0.20,
            'rating_change_pct': 0.35,
            'start_rate_change_pct': 0.15
        }
        
        composite_change = sum(changes[k] * weights[k] for k in weights.keys())
        
        # Get player info
        player_info = source_df.iloc[0] if len(source_df) > 0 else dest_df.iloc[0]
        age = dest_df['age'].iloc[0] if not dest_df.empty and pd.notna(dest_df['age'].iloc[0]) else None
        
        return {
            'player_id': player_id,
            'player_name': player_info.get('player_name', 'Unknown'),
            'age_at_transfer': age,
            'source_league_id': source_league_id,
            'source_season': source_season,
            'source_minutes': source_minutes,
            'source_apps': source_apps,
            'source_metrics': source_metrics,
            'dest_league_id': dest_league_id,
            'dest_season': dest_season,
            'dest_minutes': dest_minutes,
            'dest_apps': dest_apps,
            'dest_metrics': dest_metrics,
            'metric_changes': changes,
            'composite_change_pct': composite_change
        }
    
    def _calculate_performance_metrics(self, match_df: pd.DataFrame, total_minutes: int) -> Optional[Dict]:
        """
        Calculate per-90 performance metrics from match data.
        """
        if total_minutes == 0:
            return None
        
        # Goals and assists per 90
        goals = match_df['goals'].sum() if 'goals' in match_df.columns else 0
        assists = match_df['assists'].sum() if 'assists' in match_df.columns else 0
        
        goals_per90 = (goals / total_minutes) * 90
        assists_per90 = (assists / total_minutes) * 90
        
        # Average rating (if available)
        if 'rating' in match_df.columns:
            ratings = match_df['rating'].dropna()
            avg_rating = ratings.mean() if len(ratings) > 0 else 6.5
        else:
            avg_rating = 6.5  # Neutral default
        
        # Starting rate (% of games started)
        start_rate = match_df['is_starter'].mean() * 100 if 'is_starter' in match_df.columns else 50.0
        
        return {
            'goals_per90': goals_per90,
            'assists_per90': assists_per90,
            'rating': avg_rating,
            'start_rate': start_rate
        }
    
    def analyze_league_pair_transfers(self, source_league_id: int, 
                                     dest_league_id: int,
                                     start_season: int,
                                     end_season: int,
                                     transfers_df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Analyze transfers between two leagues to estimate strength gap.
        
        Gap interpretation:
        - gap > 1.0: Destination league is stronger
        - gap < 1.0: Source league is stronger
        - gap = 1.0: Leagues are equal
        """
        # Get transfers if not provided
        if transfers_df is None:
            transfers_df = self.data_loader.get_transfers_between_leagues(
                source_league_id, dest_league_id, start_season, end_season
            )
        
        if len(transfers_df) < MIN_TRANSFERS_FOR_LEAGUE_PAIR:
            return None
        
        performance_changes = []
        
        for _, transfer in transfers_df.iterrows():
            # Get team IDs if available for more precise filtering
            source_team_id = transfer.get('source_team_id')
            dest_team_id = transfer.get('dest_team_id')
            
            impact = self.calculate_player_transfer_impact(
                transfer['player_id'],
                transfer['source_season'],
                transfer['dest_season'],
                source_league_id,
                dest_league_id,
                source_team_id,
                dest_team_id
            )
            
            if impact:
                # Apply age adjustment
                age = impact['age_at_transfer']
                age_adjustment = self._get_age_adjustment(age)
                
                # CRITICAL FIX: Age adjustment should be ADDED to observed change
                # If young player naturally improves 10% from age, and we observe 20% improvement,
                # the league difficulty contributed 10% improvement (20% - 10%)
                adjusted_change = impact['composite_change_pct'] - age_adjustment
                
                performance_changes.append({
                    'player_id': impact['player_id'],
                    'raw_change_pct': impact['composite_change_pct'],
                    'age_adjusted_change_pct': adjusted_change,
                    'season': impact['dest_season'],
                    'minutes': impact['dest_minutes']
                })
        
        if len(performance_changes) < MIN_TRANSFERS_FOR_LEAGUE_PAIR:
            return None
        
        # Statistical analysis
        changes_df = pd.DataFrame(performance_changes)
        
        # Apply recency weighting
        max_season = changes_df['season'].max()
        changes_df['recency_weight'] = np.exp(-TRANSFER_RECENCY_WEIGHT * (max_season - changes_df['season']))
        
        # Weighted average change
        weighted_avg_change = np.average(
            changes_df['age_adjusted_change_pct'], 
            weights=changes_df['recency_weight']
        )
        
        # CRITICAL FIX: Proper gap calculation
        # If players improve by X% on average when transferring, destination league is easier
        # If players decline by X% on average, destination league is harder
        #
        # Gap formula: strength_dest / strength_source
        # Performance change = (performance_dest / performance_source - 1) * 100
        # If performance_dest = performance_source * (strength_source / strength_dest)
        # Then: strength_dest / strength_source = 1 / (1 + performance_change/100)
        
        # Clip extreme values
        safe_change = np.clip(weighted_avg_change, -80, 200)
        
        # Calculate gap
        if safe_change >= 0:
            # Performance improved -> destination is easier -> gap < 1
            gap = 1.0 / (1.0 + (safe_change / 100.0))
        else:
            # Performance declined -> destination is harder -> gap > 1
            gap = 1.0 + (abs(safe_change) / 100.0)
        
        # Confidence calculation
        sample_size_confidence = min(len(performance_changes) / 20.0, 1.0)
        
        if len(changes_df) > 1:
            std_dev = changes_df['age_adjusted_change_pct'].std()
            # Lower std dev = higher confidence
            consistency_confidence = 1.0 / (1.0 + (std_dev / 40.0))
        else:
            consistency_confidence = 0.5
            std_dev = 0.0
        
        overall_confidence = (sample_size_confidence * 0.7) + (consistency_confidence * 0.3)
        
        return {
            'source_league_id': source_league_id,
            'dest_league_id': dest_league_id,
            'gap_multiplier': gap,
            'confidence': overall_confidence,
            'sample_size': len(performance_changes),
            'avg_performance_change_pct': weighted_avg_change,
            'std_dev': std_dev,
            'interpretation': self._interpret_gap(gap, weighted_avg_change)
        }
    
    def _interpret_gap(self, gap: float, avg_change: float) -> str:
        """Generate human-readable interpretation of gap."""
        if gap > 1.2:
            return f"Destination ~{((gap-1)*100):.0f}% stronger (players declined {abs(avg_change):.1f}%)"
        elif gap > 1.05:
            return f"Destination slightly stronger (players declined {abs(avg_change):.1f}%)"
        elif gap < 0.8:
            return f"Source ~{((1/gap-1)*100):.0f}% stronger (players improved {avg_change:.1f}%)"
        elif gap < 0.95:
            return f"Source slightly stronger (players improved {avg_change:.1f}%)"
        else:
            return "Leagues roughly equal"
    
    def _get_age_adjustment(self, age: Optional[int]) -> float:
        """
        Get expected performance improvement/decline based on age alone.
        Positive values = player expected to improve
        Negative values = player expected to decline
        """
        if age is None:
            return 0.0  # Unknown age = no adjustment
        
        # Age curve for footballers
        if age < 20:
            return 12.0  # Rapid improvement expected
        elif age < 22:
            return 8.0
        elif age < 24:
            return 5.0
        elif age < 26:
            return 2.0  # Peak/slight improvement
        elif age < 28:
            return 0.0  # Peak years, stable
        elif age < 30:
            return -2.0  # Slight decline
        elif age < 32:
            return -5.0
        else:
            return -10.0  # Noticeable decline
    
    def build_transfer_matrix(self, season_year: int, 
                             league_ids: List[int],
                             lookback_seasons: int = 3) -> pd.DataFrame:
        """
        Build a matrix of all league pair relationships based on transfers.
        """
        start_season = season_year - lookback_seasons
        end_season = season_year
        
        # Fetch all transfers in one query (optimization)
        all_transfers = self.data_loader.get_all_transfers(start_season, end_season)
        
        if len(all_transfers) == 0:
            logger.warning("No transfers found in the given period.")
            return pd.DataFrame()
        
        # Filter for valid leagues
        valid_leagues = set(league_ids)
        all_transfers = all_transfers[
            all_transfers['source_league_id'].isin(valid_leagues) &
            all_transfers['dest_league_id'].isin(valid_leagues)
        ]
        
        results = []
        
        # Group by league pair
        grouped = all_transfers.groupby(['source_league_id', 'dest_league_id'])
        
        for (source, dest), pair_transfers in grouped:
            if len(pair_transfers) < MIN_TRANSFERS_FOR_LEAGUE_PAIR:
                continue
            
            if source == dest:
                continue
            
            # Analyze this pair
            analysis = self.analyze_league_pair_transfers(
                source, dest, start_season, end_season, transfers_df=pair_transfers
            )
            
            if analysis:
                # Add transfer counts
                analysis['transfer_count'] = len(pair_transfers)
                results.append(analysis)
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def save_to_database(self, transfer_matrix: pd.DataFrame, season_year: int):
        """Save transfer analysis results to database."""
        if len(transfer_matrix) == 0:
            return
        
        cursor = self.conn.cursor()
        
        for _, row in transfer_matrix.iterrows():
            # Check if edge exists
            cursor.execute("""
                SELECT id FROM [dbo].[league_network_edges]
                WHERE league_a_id = ? AND league_b_id = ? AND season_year = ?
            """, (int(row['source_league_id']), int(row['dest_league_id']), int(season_year)))
            
            exists = cursor.fetchone()
            
            if exists:
                # Update existing
                cursor.execute("""
                    UPDATE [dbo].[league_network_edges]
                    SET strength_gap_estimate = ?,
                        gap_confidence = ?,
                        gap_method = 'transfer',
                        total_transfers = ?,
                        transfer_count_a_to_b = ?,
                        last_updated = GETDATE()
                    WHERE league_a_id = ? AND league_b_id = ? AND season_year = ?
                """, (
                    float(row['gap_multiplier']),
                    float(row['confidence']),
                    int(row['sample_size']),
                    int(row['transfer_count']),
                    int(row['source_league_id']),
                    int(row['dest_league_id']),
                    int(season_year)
                ))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO [dbo].[league_network_edges] (
                        league_a_id, league_b_id, season_year,
                        total_transfers, transfer_count_a_to_b,
                        strength_gap_estimate, gap_confidence, gap_method,
                        last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'transfer', GETDATE())
                """, (
                    int(row['source_league_id']),
                    int(row['dest_league_id']),
                    int(season_year),
                    int(row['sample_size']),
                    int(row['transfer_count']),
                    float(row['gap_multiplier']),
                    float(row['confidence'])
                ))
        
        self.conn.commit()
        logger.info(f"Saved {len(transfer_matrix)} transfer-based league relationships")