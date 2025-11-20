import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from config.settings import MIN_TRANSFERS_FOR_LEAGUE_PAIR, TRANSFER_RECENCY_WEIGHT, MAX_AGE_CHANGE_FOR_TRANSFER

logger = logging.getLogger(__name__)

class TransferPerformanceAnalyzer:
    """
    Analyzes player performance changes across transfers to infer league strength gaps.
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader
        self.league_gaps = {}  # {(league_a, league_b): gap_multiplier}
    
    def calculate_player_transfer_impact(self, player_id: int, 
                                        source_season: int, 
                                        dest_season: int) -> Optional[Dict]:
        """
        Calculate a single player's performance change across a transfer.
        
        Returns: Dict with performance metrics in both leagues, or None if insufficient data
        """
        # Get player performance in both seasons
        perf_df = self.data_loader.get_player_match_performance(
            player_id, source_season, dest_season
        )
        
        if len(perf_df) == 0:
            return None
        
        source_df = perf_df[perf_df['season_year'] == source_season]
        dest_df = perf_df[perf_df['season_year'] == dest_season]
        
        if len(source_df) < 10 or len(dest_df) < 5:  # Minimum sample size
            return None
        
        # Calculate per-90 metrics (simplified - you'd expand this)
        source_matches = len(source_df)
        dest_matches = len(dest_df)
        
        # Assuming ~90 minutes per match for starters (you'd calculate actual minutes)
        source_minutes = source_matches * 90
        dest_minutes = dest_matches * 90
        
        source_goals = source_df['goals'].sum()
        dest_goals = dest_df['goals'].sum()
        
        source_g90 = (source_goals / source_minutes) * 90 if source_minutes > 0 else 0
        dest_g90 = (dest_goals / dest_minutes) * 90 if dest_minutes > 0 else 0
        
        # Calculate performance change percentage
        if source_g90 > 0:
            goals_change_pct = ((dest_g90 - source_g90) / source_g90) * 100
        else:
            goals_change_pct = None
        
        return {
            'player_id': player_id,
            'player_name': source_df['player_name'].iloc[0],
            'position_id': source_df['position_id'].iloc[0],
            'source_league_id': source_df['league_id'].iloc[0],
            'source_season': source_season,
            'source_matches': source_matches,
            'source_goals_per90': source_g90,
            'dest_league_id': dest_df['league_id'].iloc[0],
            'dest_season': dest_season,
            'dest_matches': dest_matches,
            'dest_goals_per90': dest_g90,
            'goals_change_pct': goals_change_pct,
            'age_at_transfer': dest_df['age'].iloc[0]
        }
    
    def analyze_league_pair_transfers(self, source_league_id: int, 
                                     dest_league_id: int,
                                     start_season: int,
                                     end_season: int) -> Dict:
        """
        Analyze all transfers between two leagues to estimate strength gap.
        
        Returns: Dict with gap estimate, confidence, and sample size
        """
        logger.info(f"Analyzing transfers: League {source_league_id} -> {dest_league_id}, "
                   f"seasons {start_season}-{end_season}")
        
        # Get all transfers between these leagues
        transfers_df = self.data_loader.get_transfers_between_leagues(
            source_league_id, dest_league_id, start_season, end_season
        )
        
        if len(transfers_df) < MIN_TRANSFERS_FOR_LEAGUE_PAIR:
            logger.warning(f"Insufficient transfers ({len(transfers_df)}) between leagues "
                         f"{source_league_id} and {dest_league_id}")
            return None
        
        # Calculate performance change for each transfer
        performance_changes = []
        
        for _, transfer in transfers_df.iterrows():
            impact = self.calculate_player_transfer_impact(
                transfer['player_id'],
                transfer['source_season'],
                transfer['dest_season']
            )
            
            if impact and impact['goals_change_pct'] is not None:
                # Age adjustment - expect natural decline/improvement based on age
                age = impact['age_at_transfer']
                age_adjustment = self._get_age_adjustment(age)
                
                # Adjust performance change for expected age-based change
                adjusted_change = impact['goals_change_pct'] - age_adjustment
                
                performance_changes.append({
                    'player_id': impact['player_id'],
                    'raw_change_pct': impact['goals_change_pct'],
                    'age_adjusted_change_pct': adjusted_change,
                    'season': impact['dest_season']
                })
        
        if len(performance_changes) < MIN_TRANSFERS_FOR_LEAGUE_PAIR:
            return None
        
        # Convert to DataFrame for analysis
        changes_df = pd.DataFrame(performance_changes)
        
        # Apply recency weighting (more recent transfers matter more)
        max_season = changes_df['season'].max()
        changes_df['recency_weight'] = np.exp(
            -TRANSFER_RECENCY_WEIGHT * (max_season - changes_df['season'])
        )
        
        # Calculate weighted average performance change
        weighted_avg_change = np.average(
            changes_df['age_adjusted_change_pct'],
            weights=changes_df['recency_weight']
        )
        
        # Convert percentage change to difficulty multiplier
        # If players perform 20% worse on average, dest league is 1.25x harder (1 / 0.8)
        if weighted_avg_change < 0:
            # Performance dropped - destination is harder
            difficulty_multiplier = 1 / (1 + weighted_avg_change / 100)
        else:
            # Performance improved - destination is easier
            difficulty_multiplier = 1 / (1 + weighted_avg_change / 100)
        
        # Calculate confidence based on sample size and consistency
        sample_size_confidence = min(len(performance_changes) / 20, 1.0)  # Max at 20 transfers
        
        # Consistency: lower std dev = higher confidence
        std_dev = changes_df['age_adjusted_change_pct'].std()
        consistency_confidence = 1 / (1 + std_dev / 50)  # Normalize std dev
        
        overall_confidence = (sample_size_confidence + consistency_confidence) / 2
        
        result = {
            'source_league_id': source_league_id,
            'dest_league_id': dest_league_id,
            'gap_multiplier': difficulty_multiplier,
            'confidence': overall_confidence,
            'sample_size': len(performance_changes),
            'avg_performance_change_pct': weighted_avg_change,
            'std_dev': std_dev
        }
        
        logger.info(f"League gap estimate: {difficulty_multiplier:.3f} "
                   f"(confidence: {overall_confidence:.2f}, n={len(performance_changes)})")
        
        return result
    
    def _get_age_adjustment(self, age: int) -> float:
        """
        Get expected performance change based on age (in percentage points).
        This is a simplified curve - you'd calibrate from your data.
        """
        if age < 23:
            return 5.0  # Expect 5% improvement (young players developing)
        elif age < 27:
            return 2.0  # Expect slight improvement (entering prime)
        elif age < 30:
            return 0.0  # Stable (prime years)
        elif age < 33:
            return -3.0  # Expect slight decline
        else:
            return -8.0  # Expect noticeable decline
    
    def build_transfer_matrix(self, season_year: int, 
                             league_ids: List[int],
                             lookback_seasons: int = 3) -> pd.DataFrame:
        """
        Build a matrix of all league pair relationships based on transfers.
        
        Returns: DataFrame with columns [source_league, dest_league, gap_multiplier, confidence]
        """
        results = []
        
        start_season = season_year - lookback_seasons
        end_season = season_year
        
        # Analyze all pairs of leagues
        for i, source_league in enumerate(league_ids):
            for dest_league in league_ids[i+1:]:  # Avoid duplicates and self-pairs
                
                # Analyze both directions
                forward = self.analyze_league_pair_transfers(
                    source_league, dest_league, start_season, end_season
                )
                
                if forward:
                    results.append(forward)
                
                backward = self.analyze_league_pair_transfers(
                    dest_league, source_league, start_season, end_season
                )
                
                if backward:
                    results.append(backward)
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def save_to_database(self, transfer_matrix: pd.DataFrame, season_year: int):
        """Save transfer analysis results to database."""
        if len(transfer_matrix) == 0:
            return
        
        cursor = self.conn.cursor()
        
        for _, row in transfer_matrix.iterrows():
            # Check if record exists
            cursor.execute("""
                SELECT id FROM [dbo].[league_network_edges]
                WHERE league_a_id = ? AND league_b_id = ? AND season_year = ?
            """, row['source_league_id'], row['dest_league_id'], season_year)
            
            existing = cursor.fetchone()
            
            if existing:
                # Update
                cursor.execute("""
                    UPDATE [dbo].[league_network_edges]
                    SET strength_gap_estimate = ?,
                        gap_confidence = ?,
                        gap_method = 'transfer',
                        total_transfers = ?,
                        last_updated = GETDATE()
                    WHERE league_a_id = ? AND league_b_id = ? AND season_year = ?
                """, row['gap_multiplier'], row['confidence'], row['sample_size'],
                     row['source_league_id'], row['dest_league_id'], season_year)
            else:
                # Insert
                cursor.execute("""
                    INSERT INTO [dbo].[league_network_edges] (
                        league_a_id, league_b_id, season_year,
                        total_transfers, strength_gap_estimate, gap_confidence, gap_method
                    ) VALUES (?, ?, ?, ?, ?, ?, 'transfer')
                """, row['source_league_id'], row['dest_league_id'], season_year,
                     row['sample_size'], row['gap_multiplier'], row['confidence'])
        
        self.conn.commit()
        logger.info(f"Saved {len(transfer_matrix)} transfer-based league relationships")