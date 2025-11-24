import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from config.settings import LEAGUE_STRENGTH_WEIGHTS, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX
from config.league_mappings import LeagueRegistry

logger = logging.getLogger(__name__)

class LeagueStrengthCalculator:
    """
    Master calculator that combines all league strength components into final ratings.
    """
    
    def __init__(self, connection, transfer_analyzer, european_analyzer, data_loader, league_registry):
        self.conn = connection
        self.transfer_analyzer = transfer_analyzer
        self.european_analyzer = european_analyzer
        self.data_loader = data_loader
        self.league_registry = league_registry
    
    def calculate_league_strength(self, league_id: int, season_year: int,
                                  save_to_db: bool = True) -> Optional[Dict]:
        league_id = int(league_id)
        season_year = int(season_year)

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM [dbo].[match_details] 
            WHERE parent_league_id = ? AND YEAR(match_time_utc) = ?
        """, (league_id, season_year))
        
        match_count = cursor.fetchone()[0]
        if match_count == 0:
            return None
                   
        league_info = self.league_registry.get_league_info(league_id)
        
        # Components
        transfer_score = self._get_transfer_matrix_score(league_id, season_year)
        european_score = self._get_european_results_score(league_id, season_year)
        network_score = self._get_network_inference_score(league_id, season_year)
        historical_score = self._get_historical_consistency_score(league_id, season_year)
        
        # Calculate composite score
        weights = LEAGUE_STRENGTH_WEIGHTS
        
        components = {
            'transfer': (transfer_score, weights['transfer_matrix']),
            'european': (european_score, weights['european_results']),
            'network': (network_score, weights['network_inference']),
            'historical': (historical_score, weights['historical_consistency'])
        }
        
        total_weight = 0
        weighted_sum = 0

        for comp_name, (score, weight) in components.items():
            if score is not None:
                weighted_sum += score * weight
                total_weight += weight

        # If we have no data points, we abort
        if total_weight == 0:
            return None
        
        # Calculate strength and confidence based on available components
        overall_strength = weighted_sum / total_weight
        components_available = sum(1 for score, _ in components.values() if score is not None)
        calculation_confidence = components_available / len(components)
        
        overall_strength = np.clip(overall_strength, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX)
        
        result = {
            'league_id': league_id,
            'league_name': league_info['name'],
            'season_year': season_year,
            'tier': self.league_registry.get_league_tier(league_id),
            'transfer_matrix_score': float(transfer_score) if transfer_score else None,
            'european_results_score': float(european_score) if european_score else None,
            'network_inference_score': float(network_score) if network_score else None,
            'historical_consistency_score': float(historical_score) if historical_score else None,
            'overall_strength': float(overall_strength),
            'calculation_confidence': float(calculation_confidence),
            'sample_size': self._get_total_sample_size(league_id, season_year)
        }
        
        trend_1yr, trend_3yr = self._calculate_trends(league_id, season_year)
        result['trend_1yr'] = float(trend_1yr) if trend_1yr is not None else None
        result['trend_3yr'] = float(trend_3yr) if trend_3yr is not None else None
        
        if save_to_db:
            self._save_to_database(result)
        
        return result
    
    def _get_transfer_matrix_score(self, league_id: int, season_year: int) -> Optional[float]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT league_a_id, league_b_id, strength_gap_estimate, gap_confidence
            FROM [dbo].[league_network_edges]
            WHERE (league_a_id = ? OR league_b_id = ?)
              AND season_year = ? AND gap_method = 'transfer' AND total_transfers >= ?
        """, (int(league_id), int(league_id), int(season_year), 5))
        
        edges = cursor.fetchall()
        if not edges: 
            return None
        
        anchor_leagues = self._get_anchor_leagues(season_year - 1)
        if not anchor_leagues:
            return None
            
        strength_estimates = []
        
        for edge in edges:
            la, lb, gap, conf = edge
            
            # CRITICAL FIX: Properly interpret gap direction
            # gap = strength_a / strength_b (la is gap times stronger than lb)
            
            if la == league_id and lb in anchor_leagues:
                # We are league_a, they are league_b
                # gap = our_strength / their_strength
                # Therefore: our_strength = gap * their_strength
                estimated_strength = gap * anchor_leagues[lb]
                strength_estimates.append((estimated_strength, conf))
                
            elif lb == league_id and la in anchor_leagues:
                # We are league_b, they are league_a  
                # gap = their_strength / our_strength
                # Therefore: our_strength = their_strength / gap
                estimated_strength = anchor_leagues[la] / gap if gap > 0.01 else anchor_leagues[la] / 0.01
                strength_estimates.append((estimated_strength, conf))
        
        if not strength_estimates: 
            return None
        
        strengths = np.array([s[0] for s in strength_estimates])
        confs = np.array([s[1] for s in strength_estimates])
        
        result = float(np.average(strengths, weights=confs))
        
        # Sanity check - clip to reasonable bounds
        result = np.clip(result, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX)
        
        return result
    
    def _get_anchor_leagues(self, target_season: int) -> Dict[int, float]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT TOP 20 ls.league_id, ls.overall_strength
            FROM [dbo].[league_strength] ls
            INNER JOIN [dbo].[LeagueDivisions] ld ON ls.league_id = ld.LeagueID
            WHERE ls.season_year = ? AND ld.DivisionLevel = '1' AND ls.calculation_confidence > 0.5
            ORDER BY ls.overall_strength DESC
        """, (int(target_season),))
        
        results = cursor.fetchall()
        if results: 
            return {row[0]: float(row[1]) for row in results}
        
        # Fallback for first season - use reasonable defaults for top leagues
        top_divisions = self.league_registry.get_top_division_leagues()
        return {lid: 70.0 for lid in top_divisions[:20]}
    
    def _get_european_results_score(self, league_id: int, season_year: int) -> Optional[float]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT result, expected_result, actual_result, competition_weight
            FROM [dbo].[european_competition_results]
            WHERE (home_league_id = ? OR away_league_id = ?) AND season_year = ?
        """, (int(league_id), int(league_id), int(season_year)))
        
        results = cursor.fetchall()
        
        # CRITICAL FIX: Return None instead of 0 when no data
        if not results: 
            return None
        
        total_weight = 0
        weighted_performance = 0
        for _, expected, actual, weight in results:
            weighted_performance += (actual - expected) * weight
            total_weight += weight
        
        if total_weight == 0: 
            return None
        
        avg_performance = weighted_performance / total_weight
        league_info = self.league_registry.get_league_info(league_id)
        base_strength = league_info.get('base_strength') or 50.0
        
        # Performance adjustment scaled appropriately
        # avg_performance ranges roughly -1 to +1, scale to +/- 20 points
        result = base_strength + (avg_performance * 20)
        
        return float(np.clip(result, STRENGTH_SCALE_MIN, STRENGTH_SCALE_MAX))

    def _has_historical_data(self, season_year: int) -> bool:
        """Check if we have real historical data (not just fallbacks)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM [dbo].[league_strength]
            WHERE season_year = ?
        """, (int(season_year),))
        
        count = cursor.fetchone()[0]
        return count > 0
    
    def _get_network_inference_score(self, league_id: int, season_year: int) -> Optional[float]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT league_a_id, league_b_id, strength_gap_estimate, gap_confidence
            FROM [dbo].[league_network_edges]
            WHERE season_year = ? AND strength_gap_estimate IS NOT NULL
        """, (int(season_year),))
        
        edges = cursor.fetchall()
        if not edges: 
            return None
        
        # Build bidirectional graph
        graph = {}
        for la, lb, gap, conf in edges:
            if la not in graph: 
                graph[la] = []
            if lb not in graph: 
                graph[lb] = []
            
            safe_gap = max(gap, 0.01)
            graph[la].append((lb, safe_gap, conf))
            graph[lb].append((la, 1.0 / safe_gap, conf))
        
        anchor_strengths = self._get_anchor_leagues(season_year - 1)
        if not anchor_strengths:
            return None
        
        # **NEW: Check if this league actually has edges in the graph**
        if league_id not in graph:
            return None
        
        # **NEW: Check if this league has direct connections to any anchors**
        # If using fallback anchors (first season), require direct connection
        has_real_data = self._has_historical_data(season_year - 1)
        
        if not has_real_data:
            # First season - only trust direct connections
            direct_neighbors = [neighbor for neighbor, _, _ in graph.get(league_id, [])]
            has_direct_anchor = any(n in anchor_strengths for n in direct_neighbors)
            
            if not has_direct_anchor:
                return None  # Don't infer strength through multiple hops in first season
    
    def _get_historical_consistency_score(self, league_id: int, season_year: int) -> Optional[float]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT TOP 3 overall_strength FROM [dbo].[league_strength]
            WHERE league_id = ? AND season_year < ? ORDER BY season_year DESC
        """, (int(league_id), int(season_year)))
        
        historical = cursor.fetchall()
        if not historical: 
            return None
        
        vals = [h[0] for h in historical]
        weights = [0.5, 0.3, 0.2][:len(vals)]
        norm_weights = [w/sum(weights) for w in weights]
        return float(sum(v * w for v, w in zip(vals, norm_weights)))
    
    def _calculate_trends(self, league_id: int, season_year: int) -> tuple:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT season_year, overall_strength FROM [dbo].[league_strength]
            WHERE league_id = ? AND season_year < ? ORDER BY season_year DESC
        """, (int(league_id), int(season_year)))
        
        history = cursor.fetchall()
        if len(history) < 2: 
            return None, None
        
        strengths = {h[0]: h[1] for h in history}
        
        # Calculate 1-year trend from the two most recent historical seasons
        most_recent_year = history[0][0]
        prev_year = most_recent_year - 1
        trend_1yr = strengths[most_recent_year] - strengths[prev_year] if prev_year in strengths else None
        
        # Calculate 3-year trend
        trend_3yr = None
        if len(history) >= 3:
            yrs = np.array([h[0] for h in history[:3]])
            strs = np.array([h[1] for h in history[:3]])
            trend_3yr = np.polyfit(yrs, strs, 1)[0]
            
        return trend_1yr, trend_3yr
    
    def _get_total_sample_size(self, league_id: int, season_year: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COALESCE(SUM(total_transfers), 0) 
            FROM [dbo].[league_network_edges] 
            WHERE (league_a_id = ? OR league_b_id = ?) AND season_year = ?
        """, (int(league_id), int(league_id), int(season_year)))
        t = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM [dbo].[european_competition_results] 
            WHERE (home_league_id = ? OR away_league_id = ?) AND season_year = ?
        """, (int(league_id), int(league_id), int(season_year)))
        e = cursor.fetchone()[0]
        
        return int(t + e)
    
    def _save_to_database(self, result: Dict):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM [dbo].[league_strength] 
            WHERE league_id = ? AND season_year = ?
        """, (result['league_id'], result['season_year']))
        exists = cursor.fetchone()
        
        params = [
            result['league_name'], result['tier'], result['transfer_matrix_score'],
            result['european_results_score'], result['network_inference_score'],
            result['historical_consistency_score'], result['overall_strength'],
            result['calculation_confidence'], result['sample_size'],
            result['trend_1yr'], result['trend_3yr'],
            result['league_id'], result['season_year']
        ]
        
        if exists:
            cursor.execute("""
                UPDATE [dbo].[league_strength] 
                SET league_name=?, tier=?, transfer_matrix_score=?, european_results_score=?, 
                    network_inference_score=?, historical_consistency_score=?, overall_strength=?, 
                    calculation_confidence=?, sample_size=?, trend_1yr=?, trend_3yr=?, 
                    last_updated=GETDATE()
                WHERE league_id=? AND season_year=?
            """, params)
        else:
            cursor.execute("""
                INSERT INTO [dbo].[league_strength] 
                (league_name, tier, transfer_matrix_score, european_results_score, 
                 network_inference_score, historical_consistency_score, overall_strength, 
                 calculation_confidence, sample_size, trend_1yr, trend_3yr, league_id, season_year)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, params)
        self.conn.commit()
    
    def calculate_all_leagues(self, season_year: int, league_ids: List[int] = None) -> pd.DataFrame:
        if league_ids is None:
            league_ids = self.league_registry.get_domestic_leagues()
        
        results = []
        for league_id in league_ids:
            try:
                result = self.calculate_league_strength(int(league_id), int(season_year), save_to_db=True)
                if result: 
                    results.append(result)
            except Exception as e:
                logger.error(f"Error calculating strength for league {league_id}: {e}", exc_info=True)
        
        if not results: 
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values('overall_strength', ascending=False)