import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from config.settings import LEAGUE_STRENGTH_WEIGHTS, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX
from config.league_mappings import get_league_info, TIER_1_LEAGUES, TIER_2_LEAGUES, TIER_2_5_LEAGUES

logger = logging.getLogger(__name__)

class LeagueStrengthCalculator:
    """
    Master calculator that combines all league strength components into final ratings.
    """
    
    def __init__(self, connection, transfer_analyzer, european_analyzer, data_loader):
        self.conn = connection
        self.transfer_analyzer = transfer_analyzer
        self.european_analyzer = european_analyzer
        self.data_loader = data_loader
    
    def calculate_league_strength(self, league_id: int, season_year: int,
                                  save_to_db: bool = True) -> Optional[Dict]:
        """
        Calculate comprehensive league strength score.
        
        Combines:
        1. Transfer performance matrix
        2. European competition results
        3. Network inference (for leagues without direct data)
        4. Historical consistency
        
        Returns: Dict with component scores and final composite strength
        """
        logger.info(f"Calculating league strength for league {league_id}, season {season_year}")
        
        league_info = get_league_info(league_id)
        
        # Component 1: Transfer Matrix Score
        transfer_score = self._get_transfer_matrix_score(league_id, season_year)
        
        # Component 2: European Results Score
        european_score = self._get_european_results_score(league_id, season_year)
        
        # Component 3: Network Inference Score
        network_score = self._get_network_inference_score(league_id, season_year)
        
        # Component 4: Historical Consistency Score
        historical_score = self._get_historical_consistency_score(league_id, season_year)
        
        # Calculate composite score
        weights = LEAGUE_STRENGTH_WEIGHTS
        
        # Build component scores dict (with None handling)
        components = {
            'transfer': (transfer_score, weights['transfer_matrix']),
            'european': (european_score, weights['european_results']),
            'network': (network_score, weights['network_inference']),
            'historical': (historical_score, weights['historical_consistency'])
        }
        
        # Calculate weighted average, adjusting weights for missing components
        total_weight = 0
        weighted_sum = 0
        
        for comp_name, (score, weight) in components.items():
            if score is not None:
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            logger.warning(f"No valid components for league {league_id}, season {season_year}")
            # Fall back to base strength from config
            overall_strength = league_info.get('base_strength', 50)
            calculation_confidence = 0.1
        else:
            # Normalize by actual total weight
            overall_strength = weighted_sum / total_weight
            
            # Calculate confidence (more components = higher confidence)
            components_available = sum(1 for score, _ in components.values() if score is not None)
            calculation_confidence = components_available / len(components)
        
        # Ensure within bounds
        overall_strength = np.clip(overall_strength, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX)
        
        result = {
            'league_id': league_id,
            'league_name': league_info['name'],
            'season_year': season_year,
            'tier': league_info['tier'],
            'transfer_matrix_score': transfer_score,
            'european_results_score': european_score,
            'network_inference_score': network_score,
            'historical_consistency_score': historical_score,
            'overall_strength': overall_strength,
            'calculation_confidence': calculation_confidence,
            'sample_size': self._get_total_sample_size(league_id, season_year)
        }
        
        # Calculate trends if historical data exists
        trend_1yr, trend_3yr = self._calculate_trends(league_id, season_year)
        result['trend_1yr'] = trend_1yr
        result['trend_3yr'] = trend_3yr
        
        logger.info(f"League {league_id} strength: {overall_strength:.1f} "
                   f"(confidence: {calculation_confidence:.2f})")
        
        if save_to_db:
            self._save_to_database(result)
        
        return result
    
    def _get_transfer_matrix_score(self, league_id: int, season_year: int) -> Optional[float]:
        """
        Get strength score derived from transfer performance analysis.
        Uses the league network edges table.
        """
        cursor = self.conn.cursor()
        
        # Get all transfer-based relationships involving this league
        cursor.execute("""
            SELECT 
                league_a_id,
                league_b_id,
                strength_gap_estimate,
                gap_confidence,
                total_transfers
            FROM [dbo].[league_network_edges]
            WHERE (league_a_id = ? OR league_b_id = ?)
              AND season_year = ?
              AND gap_method = 'transfer'
              AND total_transfers >= ?
        """, league_id, league_id, season_year, 5)
        
        edges = cursor.fetchall()
        
        if not edges:
            return None
        
        # Calculate relative strength based on known relationships
        # Use anchor leagues (Tier 1) as reference points
        anchor_strengths = {lid: info['base_strength'] for lid, info in TIER_1_LEAGUES.items()}
        
        strength_estimates = []
        
        for edge in edges:
            league_a, league_b, gap, confidence, sample_size = edge
            
            if league_a == league_id:
                # This league is A, gap tells us about B relative to A
                if league_b in anchor_strengths:
                    # We know B's strength, can infer A's
                    # If gap = 1.2 (B is 20% harder than A), and B = 90, then A ≈ 75
                    inferred_strength = anchor_strengths[league_b] / gap
                    strength_estimates.append((inferred_strength, confidence))
            
            elif league_b == league_id:
                # This league is B
                if league_a in anchor_strengths:
                    # A is known, gap tells us about B
                    inferred_strength = anchor_strengths[league_a] * gap
                    strength_estimates.append((inferred_strength, confidence))
        
        if not strength_estimates:
            # No anchors found, use base estimate from config
            league_info = get_league_info(league_id)
            return league_info.get('base_strength')
        
        # Weighted average of estimates
        strengths = np.array([s[0] for s in strength_estimates])
        confidences = np.array([s[1] for s in strength_estimates])
        
        weighted_strength = np.average(strengths, weights=confidences)
        
        return float(weighted_strength)
    
    def _get_european_results_score(self, league_id: int, season_year: int) -> Optional[float]:
        """
        Get strength score from European competition performance.
        """
        cursor = self.conn.cursor()
        
        # Get European competition results for teams from this league
        cursor.execute("""
            SELECT 
                result,
                expected_result,
                actual_result,
                competition_weight
            FROM [dbo].[european_competition_results]
            WHERE (home_league_id = ? OR away_league_id = ?)
              AND season_year = ?
        """, league_id, league_id, season_year)
        
        results = cursor.fetchall()
        
        if not results:
            return None
        
        # Calculate performance vs expectation
        total_weight = 0
        weighted_performance = 0
        
        for result, expected, actual, comp_weight in results:
            # How much did teams from this league overperform?
            performance = actual - expected
            weighted_performance += performance * comp_weight
            total_weight += comp_weight
        
        if total_weight == 0:
            return None
        
        avg_performance = weighted_performance / total_weight
        
        # Convert performance metric to strength score
        # Positive performance = exceeding expectations = stronger league
        # Scale: -0.5 to +0.5 performance → ±10 strength points around base
        league_info = get_league_info(league_id)
        base_strength = league_info.get('base_strength', 70)
        
        adjustment = avg_performance * 20  # Scale factor
        final_strength = base_strength + adjustment
        
        return float(np.clip(final_strength, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX))
    
    def _get_network_inference_score(self, league_id: int, season_year: int) -> Optional[float]:
        """
        Infer strength using network graph traversal.
        For leagues without direct transfer/European data, use transitive relationships.
        """
        cursor = self.conn.cursor()
        
        # Get all edges in the network
        cursor.execute("""
            SELECT 
                league_a_id,
                league_b_id,
                strength_gap_estimate,
                gap_confidence
            FROM [dbo].[league_network_edges]
            WHERE season_year = ?
              AND strength_gap_estimate IS NOT NULL
        """, season_year)
        
        edges = cursor.fetchall()
        
        if not edges:
            return None
        
        # Build adjacency graph
        graph = {}
        for league_a, league_b, gap, confidence in edges:
            if league_a not in graph:
                graph[league_a] = []
            if league_b not in graph:
                graph[league_b] = []
            
            graph[league_a].append((league_b, gap, confidence))
            # Reverse direction (inverse gap)
            graph[league_b].append((league_a, 1/gap if gap != 0 else 1.0, confidence))
        
        # Use BFS to find paths to anchor leagues
        anchor_strengths = {lid: info['base_strength'] for lid, info in TIER_1_LEAGUES.items()}
        
        if league_id in anchor_strengths:
            return anchor_strengths[league_id]
        
        if league_id not in graph:
            return None
        
        # Find shortest paths to any anchor league
        from collections import deque
        
        visited = {league_id}
        queue = deque([(league_id, 1.0, 1.0)])  # (node, cumulative_gap, cumulative_confidence)
        
        strength_estimates = []
        
        while queue and len(strength_estimates) < 5:  # Limit to 5 paths for efficiency
            current, cum_gap, cum_confidence = queue.popleft()
            
            if current in anchor_strengths and current != league_id:
                # Found an anchor - infer strength
                inferred_strength = anchor_strengths[current] / cum_gap
                strength_estimates.append((inferred_strength, cum_confidence))
                continue
            
            # Explore neighbors
            if current in graph:
                for neighbor, gap, confidence in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_gap = cum_gap * gap
                        new_confidence = cum_confidence * confidence * 0.9  # Decay confidence with distance
                        
                        if new_confidence > 0.1:  # Only pursue if confidence remains reasonable
                            queue.append((neighbor, new_gap, new_confidence))
        
        if not strength_estimates:
            return None
        
        # Weighted average
        strengths = np.array([s[0] for s in strength_estimates])
        confidences = np.array([s[1] for s in strength_estimates])
        
        weighted_strength = np.average(strengths, weights=confidences)
        
        return float(weighted_strength)
    
    def _get_historical_consistency_score(self, league_id: int, season_year: int) -> Optional[float]:
        """
        Get strength based on historical performance (smoothing year-to-year volatility).
        """
        cursor = self.conn.cursor()
        
        # Get previous seasons' strength scores
        cursor.execute("""
            SELECT TOP 3
                season_year,
                overall_strength
            FROM [dbo].[league_strength]
            WHERE league_id = ?
              AND season_year < ?
            ORDER BY season_year DESC
        """, league_id, season_year)
        
        historical = cursor.fetchall()
        
        if not historical:
            return None
        
        # Weight recent seasons more heavily (exponential decay)
        weights = [0.5, 0.3, 0.2]  # Most recent, 2 years ago, 3 years ago
        
        weighted_strength = sum(
            strength * weights[i] 
            for i, (year, strength) in enumerate(historical)
        )
        
        return float(weighted_strength)
    
    def _calculate_trends(self, league_id: int, season_year: int) -> tuple:
        """
        Calculate 1-year and 3-year strength trends.
        Returns: (trend_1yr, trend_3yr) in percentage points
        """
        cursor = self.conn.cursor()
        
        # Get current and historical strengths
        cursor.execute("""
            SELECT 
                season_year,
                overall_strength
            FROM [dbo].[league_strength]
            WHERE league_id = ?
              AND season_year <= ?
            ORDER BY season_year DESC
        """, league_id, season_year)
        
        history = cursor.fetchall()
        
        if len(history) < 2:
            return None, None
        
        strengths_dict = {year: strength for year, strength in history}
        
        # 1-year trend
        trend_1yr = None
        if season_year in strengths_dict and season_year - 1 in strengths_dict:
            trend_1yr = strengths_dict[season_year] - strengths_dict[season_year - 1]
        
        # 3-year trend (linear regression slope)
        trend_3yr = None
        if len(history) >= 4:
            years = np.array([h[0] for h in history[:4]])
            strengths = np.array([h[1] for h in history[:4]])
            
            # Simple linear regression
            slope = np.polyfit(years, strengths, 1)[0]
            trend_3yr = slope
        
        return trend_1yr, trend_3yr
    
    def _get_total_sample_size(self, league_id: int, season_year: int) -> int:
        """
        Get total sample size used in calculation (transfers + European matches).
        """
        cursor = self.conn.cursor()
        
        # Transfers
        cursor.execute("""
            SELECT COALESCE(SUM(total_transfers), 0)
            FROM [dbo].[league_network_edges]
            WHERE (league_a_id = ? OR league_b_id = ?)
              AND season_year = ?
        """, league_id, league_id, season_year)
        
        transfers = cursor.fetchone()[0]
        
        # European matches
        cursor.execute("""
            SELECT COUNT(*)
            FROM [dbo].[european_competition_results]
            WHERE (home_league_id = ? OR away_league_id = ?)
              AND season_year = ?
        """, league_id, league_id, season_year)
        
        european = cursor.fetchone()[0]
        
        return int(transfers + european)
    
    def _save_to_database(self, result: Dict):
        """Save league strength to database."""
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute("""
            SELECT id FROM [dbo].[league_strength]
            WHERE league_id = ? AND season_year = ?
        """, result['league_id'], result['season_year'])
        
        existing = cursor.fetchone()
        
        if existing:
            # Update
            cursor.execute("""
                UPDATE [dbo].[league_strength]
                SET league_name = ?,
                    tier = ?,
                    transfer_matrix_score = ?,
                    european_results_score = ?,
                    network_inference_score = ?,
                    historical_consistency_score = ?,
                    overall_strength = ?,
                    calculation_confidence = ?,
                    sample_size = ?,
                    trend_1yr = ?,
                    trend_3yr = ?,
                    last_updated = GETDATE()
                WHERE league_id = ? AND season_year = ?
            """, result['league_name'], result['tier'],
                 result['transfer_matrix_score'], result['european_results_score'],
                 result['network_inference_score'], result['historical_consistency_score'],
                 result['overall_strength'], result['calculation_confidence'],
                 result['sample_size'], result['trend_1yr'], result['trend_3yr'],
                 result['league_id'], result['season_year'])
        else:
            # Insert
            cursor.execute("""
                INSERT INTO [dbo].[league_strength] (
                    league_id, league_name, season_year, tier,
                    transfer_matrix_score, european_results_score,
                    network_inference_score, historical_consistency_score,
                    overall_strength, calculation_confidence, sample_size,
                    trend_1yr, trend_3yr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, result['league_id'], result['league_name'], result['season_year'],
                 result['tier'], result['transfer_matrix_score'], result['european_results_score'],
                 result['network_inference_score'], result['historical_consistency_score'],
                 result['overall_strength'], result['calculation_confidence'],
                 result['sample_size'], result['trend_1yr'], result['trend_3yr'])
        
        self.conn.commit()
        logger.info(f"Saved league strength for league {result['league_id']}, season {result['season_year']}")
    
    def calculate_all_leagues(self, season_year: int, league_ids: List[int] = None) -> pd.DataFrame:
        """
        Calculate strength for all leagues in a season.
        
        Returns: DataFrame with all league strengths
        """
        if league_ids is None:
            # Get all leagues that have data
            league_ids = list(TIER_1_LEAGUES.keys()) + list(TIER_2_LEAGUES.keys()) + list(TIER_2_5_LEAGUES.keys())
        
        results = []
        
        for league_id in league_ids:
            try:
                result = self.calculate_league_strength(league_id, season_year, save_to_db=True)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error calculating strength for league {league_id}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('overall_strength', ascending=False)
        
        return df