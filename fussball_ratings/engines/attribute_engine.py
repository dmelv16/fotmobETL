"""
Attribute calculation engine for converting raw stats to 0-20 scale.
Updated with proper implementations replacing placeholder logic.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import math
from collections import defaultdict
from processors.match_processor import CoachMatchStats, TeamMatchStats
from config.constants import (
    DataTier, PositionGroup, ATTR_MIN, ATTR_MAX, ATTR_DEFAULT,
    MIN_MATCHES_FOR_ATTR, PERCENTILE_TO_ATTR, MATCH_DECAY_WEIGHTS
)
from config.attribute_definitions import (
    AttributeDefinition, CalculationMethod, ALL_PLAYER_ATTRIBUTES
)
from config.coach_team_attributes import (
    COACH_ATTRIBUTES, TEAM_ATTRIBUTES, LEAGUE_ATTRIBUTES
)
from models.entities import Player, AttributeScore
from processors.data_handlers import DataTierProcessor
from processors.match_processor import PlayerMatchStats

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default when denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def safe_avg(values: list, default: float = 0.0) -> float:
    """Safe average that returns default for empty lists."""
    if not values:
        return default
    return sum(values) / len(values)

@dataclass
class StatAccumulator:
    """Accumulates stats across matches with decay weighting."""
    values: List[Tuple[float, float, int]] = field(default_factory=list)
    
    def add(self, value: float, weight: float, match_id: int):
        self.values.append((value, weight, match_id))
    
    @property
    def weighted_sum(self) -> float:
        return sum(v * w for v, w, _ in self.values) if self.values else 0.0
    
    @property
    def total_weight(self) -> float:
        return sum(w for _, w, _ in self.values)
    
    @property
    def weighted_average(self) -> Optional[float]:
        return self.weighted_sum / self.total_weight if self.total_weight else None
    
    @property
    def match_count(self) -> int:
        return len(self.values)
    
    @property
    def raw_values(self) -> List[float]:
        return [v for v, _, _ in self.values]


class PercentileCalculator:
    """Calculates percentiles within a population for normalization."""
    
    def __init__(self):
        self.stat_distributions: Dict[str, Dict[PositionGroup, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
    
    def record_stat(self, stat_key: str, value: float, position_group: PositionGroup):
        self.stat_distributions[stat_key][position_group].append(value)
    
    def get_percentile(self, stat_key: str, value: float, position_group: PositionGroup) -> float:
        values = self.stat_distributions[stat_key].get(position_group, [])
        if not values:
            return 50.0
        below = sum(1 for v in values if v < value)
        equal = sum(1 for v in values if v == value)
        return safe_divide(below + 0.5 * equal, len(values), 0.5) * 100
    
    def percentile_to_attribute(self, percentile: float) -> float:
        percentiles = sorted(PERCENTILE_TO_ATTR.keys())
        for i, p in enumerate(percentiles):
            if percentile <= p:
                if i == 0:
                    return float(PERCENTILE_TO_ATTR[p])
                lower_p, upper_p = percentiles[i - 1], p
                lower_attr, upper_attr = PERCENTILE_TO_ATTR[lower_p], PERCENTILE_TO_ATTR[upper_p]
                ratio = (percentile - lower_p) / (upper_p - lower_p)
                return lower_attr + ratio * (upper_attr - lower_attr)
        return float(ATTR_MAX)


class DecayCalculator:
    """Calculates time-based decay weights for historical data."""
    
    @staticmethod
    def get_match_decay(matches_ago: int) -> float:
        for threshold, weight in sorted(MATCH_DECAY_WEIGHTS.items()):
            if matches_ago <= threshold:
                return weight
        return 0.1
    
    @staticmethod
    def get_recency_weight(match_date: datetime, current_date: datetime, half_life_days: int = 180) -> float:
        days_ago = (current_date - match_date).days
        return math.exp(-0.693 * days_ago / half_life_days)


# =============================================================================
# BASE CALCULATOR (unchanged)
# =============================================================================

class BaseAttributeCalculator(ABC):
    def __init__(self, percentile_calc: PercentileCalculator):
        self.percentile_calc = percentile_calc
    
    @abstractmethod
    def calculate(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats],
                  position_group: PositionGroup, current_match_count: int) -> Optional[AttributeScore]:
        pass
    
    def has_required_data(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> bool:
        if len(match_stats) < definition.min_matches:
            return False
        valid_matches = [ms for ms in match_stats if ms.data_tier.value >= definition.min_tier.value]
        return len(valid_matches) >= definition.min_matches
    
    def get_stat_value(self, match_stat: PlayerMatchStats, stat_key: str, default: Optional[float] = None) -> Optional[float]:
        return match_stat.stats.get(stat_key, default)


# =============================================================================
# UPDATED CONTEXTUAL CALCULATOR - NO MORE PLACEHOLDERS
# =============================================================================

class ContextualCalculator(BaseAttributeCalculator):
    """Calculate context-dependent attributes with proper implementations."""
    
    def calculate(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats],
                  position_group: PositionGroup, current_match_count: int) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        calc_map = {
            'CNS': self._calc_consistency,
            'CON': self._calc_concentration,
            'VER': self._calc_versatility,
            'ADA': self._calc_adaptability,
            'BIG': self._calc_big_game,
            'LEA': self._calc_leadership,
        }
        
        calc_func = calc_map.get(definition.code)
        return calc_func(definition, match_stats) if calc_func else None
        
    def _calc_consistency(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        ratings = [ms.rating for ms in match_stats if ms.data_tier.value >= definition.min_tier.value and ms.rating]
        tiers_used = set(ms.data_tier for ms in match_stats if ms.rating)
        
        if len(ratings) < definition.min_matches:
            return None
        
        avg = safe_avg(ratings, 7.0)  # Default to average rating
        variance = safe_divide(sum((r - avg) ** 2 for r in ratings), len(ratings), 0.0)
        std_dev = math.sqrt(variance)
        
        value = max(ATTR_MIN, min(ATTR_MAX, 20 - (std_dev * 10)))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(ratings), definition.min_matches * 2, 0.0)),
            matches_used=len(ratings),
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
        
    def _calc_concentration(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        ratings, errors = [], 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            if ms.rating:
                ratings.append(ms.rating)
                tiers_used.add(ms.data_tier)
            errors += ms.stats.get('error_led_to_goal', 0) or 0
        
        if len(ratings) < definition.min_matches:
            return None
        
        avg = safe_avg(ratings, 7.0)
        std_dev = math.sqrt(safe_divide(sum((r - avg) ** 2 for r in ratings), len(ratings), 0.0))
        error_rate = safe_divide(errors, len(ratings), 0.0)
        
        value = max(ATTR_MIN, min(ATTR_MAX, 20 - (std_dev * 8) - (error_rate * 20)))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(ratings), definition.min_matches * 2, 0.0)),
            matches_used=len(ratings),
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_versatility(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        positions_played = defaultdict(list)
        tiers_used = set()
        
        for ms in match_stats:
            if ms.position_id is not None:
                rating = ms.rating if ms.rating else 6.5
                positions_played[ms.position_id].append(rating)
                tiers_used.add(ms.data_tier)
        
        if len(match_stats) < definition.min_matches:
            return None
        
        positions_with_good_performance = 0
        for pos_id, ratings in positions_played.items():
            if len(ratings) >= 3:
                avg_rating = safe_avg(ratings, 6.0)
                if avg_rating >= 6.0:
                    positions_with_good_performance += 1
        
        value = max(ATTR_MIN, min(ATTR_MAX, 5 + (positions_with_good_performance * 3)))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(match_stats), definition.min_matches * 2, 0.0)),
            matches_used=len(match_stats),
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_adaptability(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        league_performance: Dict[int, List[float]] = defaultdict(list)
        country_performance: Dict[str, List[float]] = defaultdict(list)
        tiers_used = set()
        
        for ms in match_stats:
            if ms.rating is None:
                continue
            tiers_used.add(ms.data_tier)
            if ms.league_id is not None:
                league_performance[ms.league_id].append(ms.rating)
            if ms.country_code:
                country_performance[ms.country_code].append(ms.rating)
        
        if not league_performance:
            return None
        
        leagues_with_good_data = [
            (league_id, ratings) 
            for league_id, ratings in league_performance.items() 
            if len(ratings) >= 5
        ]
        
        if len(leagues_with_good_data) < 1:
            return AttributeScore(
                value=10,
                confidence=0.3,
                matches_used=len(match_stats),
                last_updated=datetime.now(),
                data_tiers_used=list(tiers_used)
            )
        
        league_averages = [safe_avg(ratings, 6.5) for _, ratings in leagues_with_good_data]
        
        if len(leagues_with_good_data) == 1:
            avg = league_averages[0]
            value = 8 + (avg - 6.0) * 3
        else:
            overall_avg = safe_avg(league_averages, 6.5)
            variance = safe_divide(sum((a - overall_avg) ** 2 for a in league_averages), len(league_averages), 0.0)
            std_dev = math.sqrt(variance)
            
            league_bonus = min(3, len(leagues_with_good_data) - 1)
            consistency_score = max(0, 4 - std_dev * 4)
            value = 8 + league_bonus + consistency_score + (overall_avg - 6.5) * 2
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(match_stats), definition.min_matches * 1.5, 0.0)),
            matches_used=len(match_stats),
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_big_game(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        big_game_ratings = []
        regular_game_ratings = []
        tiers_used = set()
        
        for ms in match_stats:
            if ms.rating is None:
                continue
            tiers_used.add(ms.data_tier)
            if ms.own_team_rating is not None and ms.opponent_team_rating is not None:
                rating_diff = ms.opponent_team_rating - ms.own_team_rating
                if rating_diff > 0.2:
                    big_game_ratings.append(ms.rating)
                else:
                    regular_game_ratings.append(ms.rating)
            else:
                regular_game_ratings.append(ms.rating)
        
        total_matches = len(big_game_ratings) + len(regular_game_ratings)
        
        if total_matches < definition.min_matches:
            return None
        
        if not big_game_ratings:
            avg_rating = safe_avg(regular_game_ratings, 6.5)
            value = 10 + (avg_rating - 6.5) * 3
            confidence = 0.4
        else:
            avg_big = safe_avg(big_game_ratings, 6.5)
            avg_regular = safe_avg(regular_game_ratings, avg_big)
            performance_lift = avg_big - avg_regular
            value = 10 + (avg_big - 6.5) * 3 + (performance_lift * 2)
            confidence = min(1.0, safe_divide(len(big_game_ratings), 10, 0.0))
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=total_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_leadership(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        results_when_starting = {'W': 0, 'D': 0, 'L': 0}
        captain_matches = 0
        tiers_used = set()
        
        for ms in match_stats:
            tiers_used.add(ms.data_tier)
            if ms.is_starter and ms.match_result:
                results_when_starting[ms.match_result] += 1
                if ms.stats.get('is_captain', False):
                    captain_matches += 1
        
        total_starts = sum(results_when_starting.values())
        
        if total_starts < definition.min_matches:
            return None
        
        win_rate = safe_divide(results_when_starting['W'], total_starts, 0.0)
        ppg = safe_divide(results_when_starting['W'] * 3 + results_when_starting['D'], total_starts, 1.0)
        captain_rate = safe_divide(captain_matches, total_starts, 0.0)
        captain_bonus = captain_rate * 3
        
        value = 10 + (ppg - 1.4) * 6.67 + captain_bonus
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(total_starts, definition.min_matches * 2, 0.0)),
            matches_used=total_starts,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


# =============================================================================
# UPDATED INVERSE CALCULATOR - NATURAL FITNESS FIXED
# =============================================================================

class InverseCalculator(BaseAttributeCalculator):
    """Calculate inverse statistics (lower is better, like injuries)."""
    
    def calculate(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats],
                  position_group: PositionGroup, current_match_count: int) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        if definition.code == 'NFI':
            return self._calc_natural_fitness(definition, match_stats)
        return None
    
    def _calc_natural_fitness(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        tiers_used = set(ms.data_tier for ms in match_stats)
        
        injured_matches = sum(1 for ms in match_stats if ms.was_injured)
        total_matches = len(match_stats)
        
        if total_matches < definition.min_matches:
            return None
        
        injury_rate = safe_divide(injured_matches, total_matches, 0.0)
        unique_injuries = len(set(ms.injury_id for ms in match_stats if ms.injury_id is not None))
        
        value = 18 - (injury_rate * 40) - (unique_injuries * 0.5)
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        confidence = min(1.0, safe_divide(total_matches, definition.min_matches * 2, 0.0))
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=total_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


# =============================================================================
# UPDATED COACH ATTRIBUTE CALCULATOR - ALL PLACEHOLDERS FIXED
# =============================================================================

class CoachAttributeCalculator:
    """Calculates coach attributes from match history - fully implemented."""
    
    def calculate_all_attributes(self, coach: Any, match_history: List[CoachMatchStats]) -> Dict[str, AttributeScore]:
        results = {}
        for attr_code, definition in COACH_ATTRIBUTES.items():
            score = self._calculate_attribute(definition, match_history)
            if score:
                results[attr_code] = score
        return results
    
    def _calculate_attribute(self, definition: AttributeDefinition, match_history: List[CoachMatchStats]) -> Optional[AttributeScore]:
        valid_matches = [m for m in match_history if m.data_tier.value >= definition.min_tier.value]
        
        if len(valid_matches) < definition.min_matches:
            return None
        
        calc_map = {
            'ATT': self._calc_attacking_tactics,
            'DEF': self._calc_defensive_tactics,
            'DIS': self._calc_discipline,
            'ADP': self._calc_adaptability,
            'GMG': self._calc_game_management,
            'ROT': self._calc_squad_rotation,
            'FOR': self._calc_formation_mastery,
            'SUB': self._calc_substitution_impact,
            'PDV': self._calc_player_development,
            'YDV': self._calc_youth_development,
        }
        
        calc_func = calc_map.get(definition.code)
        return calc_func(valid_matches) if calc_func else None
    
    def _calc_attacking_tactics(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        
        total_goals = sum(m.team_stats.get('team_score', 0) or 0 for m in matches)
        total_xg = sum(m.team_stats.get('xg', 0) or 0 for m in matches if m.team_stats.get('xg') is not None)
        matches_with_xg = sum(1 for m in matches if m.team_stats.get('xg') is not None)
        
        if matches_with_xg >= len(matches) // 2 and matches_with_xg > 0:
            diff_per_match = safe_divide(total_goals - total_xg, matches_with_xg, 0.0)
            value = 10 + diff_per_match * 6.67
        else:
            value = 10 + (safe_divide(total_goals, len(matches), 1.5) - 1.5) * 5
        
        return AttributeScore(
            value=max(ATTR_MIN, min(ATTR_MAX, value)),
            confidence=min(1.0, safe_divide(len(matches), 30, 0.0)),
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_defensive_tactics(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        
        total_conceded = sum(m.team_stats.get('team_score_conceded', 0) or 0 for m in matches)
        total_xg_against = sum(m.team_stats.get('xg_conceded', 0) or 0 for m in matches if m.team_stats.get('xg_conceded') is not None)
        matches_with_xg = sum(1 for m in matches if m.team_stats.get('xg_conceded') is not None)
        
        if matches_with_xg >= len(matches) // 2 and matches_with_xg > 0:
            diff_per_match = safe_divide(total_xg_against - total_conceded, matches_with_xg, 0.0)
            value = 10 + diff_per_match * 6.67
        else:
            value = 10 + (1.0 - safe_divide(total_conceded, len(matches), 1.0)) * 10
        
        return AttributeScore(
            value=max(ATTR_MIN, min(ATTR_MAX, value)),
            confidence=min(1.0, safe_divide(len(matches), 30, 0.0)),
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_discipline(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        
        total_cost = sum(
            (m.team_stats.get('yellow_cards', 0) or 0) + 
            (m.team_stats.get('red_cards', 0) or 0) * 3 + 
            (m.team_stats.get('fouls', 0) or 0) * 0.1 
            for m in matches
        )
        avg_cost = safe_divide(total_cost, len(matches), 3.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (3 - avg_cost) * 3))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(matches), 30, 0.0)),
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_adaptability(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        
        comebacks = 0
        blown_leads = 0
        matches_with_events = 0
        
        for m in matches:
            if not m.goal_events:
                continue
            
            matches_with_events += 1
            was_losing = False
            was_winning = False
            final_result = m.team_stats.get('result', 'D')
            
            for event in m.goal_events:
                home_score = event.get('home_score_after', 0)
                away_score = event.get('away_score_after', 0)
                
                if m.is_home_team:
                    own_score, opp_score = home_score, away_score
                else:
                    own_score, opp_score = away_score, home_score
                
                if own_score < opp_score:
                    was_losing = True
                elif own_score > opp_score:
                    was_winning = True
            
            if was_losing and final_result in ('W', 'D'):
                comebacks += 1
            if was_winning and final_result in ('L', 'D'):
                blown_leads += 1
        
        if matches_with_events < 10:
            wins = sum(1 for m in matches if m.team_stats.get('result') == 'W')
            value = 10 + (safe_divide(wins, len(matches), 0.4) - 0.4) * 20
            confidence = 0.5
        else:
            comeback_rate = safe_divide(comebacks, matches_with_events, 0.0)
            blown_lead_rate = safe_divide(blown_leads, matches_with_events, 0.0)
            value = 10 + (comeback_rate * 15) - (blown_lead_rate * 10)
            confidence = min(1.0, safe_divide(matches_with_events, 25, 0.0))
        
        return AttributeScore(
            value=max(ATTR_MIN, min(ATTR_MAX, value)),
            confidence=confidence,
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_game_management(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        
        late_goal_impact = 0
        matches_with_events = 0
        
        for m in matches:
            if not m.goal_events:
                continue
            
            matches_with_events += 1
            
            score_at_75 = (0, 0)
            for event in m.goal_events:
                if event.get('minute', 0) <= 75:
                    if m.is_home_team:
                        score_at_75 = (event.get('home_score_after', 0), event.get('away_score_after', 0))
                    else:
                        score_at_75 = (event.get('away_score_after', 0), event.get('home_score_after', 0))
            
            own_75, opp_75 = score_at_75
            final_own = m.team_stats.get('team_score', 0) or 0
            final_opp = m.team_stats.get('team_score_conceded', 0) or 0
            
            diff_at_75 = own_75 - opp_75
            diff_final = final_own - final_opp
            
            if diff_final > diff_at_75:
                late_goal_impact += 1
            elif diff_final < diff_at_75:
                late_goal_impact -= 1
        
        if matches_with_events < 10:
            wins = sum(1 for m in matches if m.team_stats.get('result') == 'W')
            value = 10 + (safe_divide(wins, len(matches), 0.4) - 0.4) * 15
            confidence = 0.4
        else:
            impact_rate = safe_divide(late_goal_impact, matches_with_events, 0.0)
            value = 10 + impact_rate * 8
            confidence = min(1.0, safe_divide(matches_with_events, 25, 0.0))
        
        return AttributeScore(
            value=max(ATTR_MIN, min(ATTR_MAX, value)),
            confidence=confidence,
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_squad_rotation(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        if len(matches) < 2:
            return None
        
        all_players = set()
        for m in matches:
            all_players.update(m.lineup_player_ids)
        
        avg_unique = safe_divide(len(all_players), len(matches), 12.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg_unique - 12) * 2))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(matches), 40, 0.0)),
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_formation_mastery(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        formation_results = defaultdict(lambda: {'W': 0, 'D': 0, 'L': 0, 'total': 0})
        
        for m in matches:
            if m.formation:
                result = m.team_stats.get('result', 'D')
                formation_results[m.formation][result] += 1
                formation_results[m.formation]['total'] += 1
        
        if not formation_results:
            return None
        
        best_win_rate = 0
        formations_mastered = 0
        
        for formation, results in formation_results.items():
            if results['total'] >= 5:
                win_rate = safe_divide(results['W'], results['total'], 0.0)
                if win_rate > 0.5:
                    formations_mastered += 1
                best_win_rate = max(best_win_rate, win_rate)
        
        value = max(ATTR_MIN, min(ATTR_MAX, 8 + formations_mastered * 2 + best_win_rate * 4))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(matches), 40, 0.0)),
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_substitution_impact(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        
        positive_impact_subs = 0
        negative_impact_subs = 0
        total_subs = 0
        
        for m in matches:
            for sub in m.substitutions:
                sub_minute = sub.get('minute', 45)
                score_at_sub = sub.get('score_before', (0, 0))
                
                goals_after = [e for e in m.goal_events if e.get('minute', 0) > sub_minute]
                
                if not goals_after:
                    continue
                
                total_subs += 1
                
                final_event = goals_after[-1]
                if m.is_home_team:
                    own_before = score_at_sub[0] if isinstance(score_at_sub, tuple) else sub.get('home_score_before', 0)
                    opp_before = score_at_sub[1] if isinstance(score_at_sub, tuple) else sub.get('away_score_before', 0)
                    own_after = final_event.get('home_score_after', 0)
                    opp_after = final_event.get('away_score_after', 0)
                else:
                    own_before = score_at_sub[1] if isinstance(score_at_sub, tuple) else sub.get('away_score_before', 0)
                    opp_before = score_at_sub[0] if isinstance(score_at_sub, tuple) else sub.get('home_score_before', 0)
                    own_after = final_event.get('away_score_after', 0)
                    opp_after = final_event.get('home_score_after', 0)
                
                diff_before = own_before - opp_before
                diff_after = own_after - opp_after
                
                if diff_after > diff_before:
                    positive_impact_subs += 1
                elif diff_after < diff_before:
                    negative_impact_subs += 1
        
        if total_subs < 10:
            wins = sum(1 for m in matches if m.team_stats.get('result') == 'W')
            value = 10 + (safe_divide(wins, len(matches), 0.4) - 0.4) * 15
            confidence = 0.4
        else:
            impact_rate = safe_divide(positive_impact_subs - negative_impact_subs, total_subs, 0.0)
            value = 10 + impact_rate * 10
            confidence = min(1.0, safe_divide(total_subs, 50, 0.0))
        
        return AttributeScore(
            value=max(ATTR_MIN, min(ATTR_MAX, value)),
            confidence=confidence,
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_player_development(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        sorted_matches = sorted(matches, key=lambda m: m.match_date or datetime.min)
        
        if len(sorted_matches) < 10:
            return None
        
        player_first_ratings: Dict[int, List[float]] = defaultdict(list)
        player_last_ratings: Dict[int, List[float]] = defaultdict(list)
        
        midpoint = len(sorted_matches) // 2
        
        for i, m in enumerate(sorted_matches):
            for player_id, rating in m.player_ratings.items():
                if i < midpoint:
                    player_first_ratings[player_id].append(rating)
                else:
                    player_last_ratings[player_id].append(rating)
        
        improvements = []
        for player_id in player_first_ratings:
            if player_id in player_last_ratings:
                if len(player_first_ratings[player_id]) >= 3 and len(player_last_ratings[player_id]) >= 3:
                    first_avg = safe_avg(player_first_ratings[player_id], 6.5)
                    last_avg = safe_avg(player_last_ratings[player_id], 6.5)
                    improvements.append(last_avg - first_avg)
        
        if not improvements:
            return AttributeScore(
                value=10, confidence=0.3, matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        avg_improvement = safe_avg(improvements, 0.0)
        value = 10 + avg_improvement * 16.67
        
        return AttributeScore(
            value=max(ATTR_MIN, min(ATTR_MAX, value)),
            confidence=min(1.0, safe_divide(len(improvements), 15, 0.0)),
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )
    
    def _calc_youth_development(self, matches: List[CoachMatchStats]) -> Optional[AttributeScore]:
        sorted_matches = sorted(matches, key=lambda m: m.match_date or datetime.min)
        
        if len(sorted_matches) < 10:
            return None
        
        young_first_ratings: Dict[int, List[float]] = defaultdict(list)
        young_last_ratings: Dict[int, List[float]] = defaultdict(list)
        
        midpoint = len(sorted_matches) // 2
        
        for i, m in enumerate(sorted_matches):
            for player_id, rating in m.player_ratings.items():
                age = m.player_ages.get(player_id, 30)
                if age < 23:
                    if i < midpoint:
                        young_first_ratings[player_id].append(rating)
                    else:
                        young_last_ratings[player_id].append(rating)
        
        improvements = []
        for player_id in young_first_ratings:
            if player_id in young_last_ratings:
                if len(young_first_ratings[player_id]) >= 2 and len(young_last_ratings[player_id]) >= 2:
                    first_avg = safe_avg(young_first_ratings[player_id], 6.5)
                    last_avg = safe_avg(young_last_ratings[player_id], 6.5)
                    improvements.append(last_avg - first_avg)
        
        if not improvements:
            return AttributeScore(
                value=10, confidence=0.2, matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        avg_improvement = safe_avg(improvements, 0.0)
        value = 10 + avg_improvement * 12
        
        return AttributeScore(
            value=max(ATTR_MIN, min(ATTR_MAX, value)),
            confidence=min(1.0, safe_divide(len(improvements), 10, 0.0)),
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )


# =============================================================================
# UPDATED TEAM ATTRIBUTE CALCULATOR - SQUAD DEPTH FIXED
# =============================================================================

class TeamAttributeCalculator:
    """Calculates team attributes from match history - fully implemented."""
    
    def calculate_all_attributes(self, team: Any, match_history: List[TeamMatchStats]) -> Dict[str, AttributeScore]:
        results = {}
        for attr_code, definition in TEAM_ATTRIBUTES.items():
            score = self._calculate_attribute(definition, match_history)
            if score:
                results[attr_code] = score
        return results
    
    def _calculate_attribute(self, definition: AttributeDefinition, match_history: List[TeamMatchStats]) -> Optional[AttributeScore]:
        valid = [m for m in match_history if m.data_tier.value >= definition.min_tier.value]
        
        if len(valid) < definition.min_matches:
            return None
        
        calc_map = {
            'ATK': self._calc_attack,
            'DEF': self._calc_defense,
            'POS': self._calc_possession,
            'PRS': self._calc_pressing,
            'PHY': self._calc_physicality,
            'DIS': self._calc_discipline,
            'SPA': self._calc_set_piece_attack,
            'SPD': self._calc_set_piece_defense,
            'CON': self._calc_consistency,
            'BIG': self._calc_big_game,
            'DEP': self._calc_squad_depth,
        }
        
        calc_func = calc_map.get(definition.code)
        return calc_func(valid) if calc_func else None
    
    def _calc_attack(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        avg_goals = safe_divide(sum(m.stats.get('team_score', 0) or 0 for m in matches), len(matches), 1.5)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg_goals - 1.5) * 5))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(matches), 20, 0.0)), 
                            matches_used=len(matches), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_defense(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        avg_conceded = safe_divide(sum(m.stats.get('team_score_conceded', 0) or 0 for m in matches), len(matches), 1.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (1.0 - avg_conceded) * 10))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(matches), 20, 0.0)), 
                            matches_used=len(matches), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_possession(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        poss_values = [m.stats.get('possession') for m in matches if m.stats.get('possession')]
        if not poss_values:
            return None
        avg = safe_avg(poss_values, 50.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg - 50) * 0.4))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(poss_values), 20, 0.0)), 
                            matches_used=len(poss_values), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_pressing(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        avg = safe_divide(sum((m.stats.get('tackles', 0) or 0) + (m.stats.get('interceptions', 0) or 0) for m in matches), len(matches), 25.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg - 25) * 0.4))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(matches), 20, 0.0)), 
                            matches_used=len(matches), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_physicality(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        avg = safe_divide(sum((m.stats.get('duels_won', 0) or 0) + (m.stats.get('aerial_duels_won', 0) or 0) for m in matches), len(matches), 30.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg - 30) * 0.3))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(matches), 20, 0.0)), 
                            matches_used=len(matches), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_discipline(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        avg_cost = safe_divide(
            sum((m.stats.get('yellow_cards', 0) or 0) + (m.stats.get('red_cards', 0) or 0) * 3 + 
                (m.stats.get('fouls', 0) or 0) * 0.1 for m in matches), 
            len(matches), 3.0
        )
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (3 - avg_cost) * 3))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(matches), 20, 0.0)), 
                            matches_used=len(matches), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_set_piece_attack(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        vals = [m.stats.get('xg_set_play') for m in matches if m.stats.get('xg_set_play') is not None]
        if not vals:
            return None
        avg = safe_avg(vals, 0.3)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg - 0.3) * 20))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(vals), 30, 0.0)), 
                            matches_used=len(vals), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_set_piece_defense(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        vals = [m.stats.get('xg_set_play_conceded') for m in matches if m.stats.get('xg_set_play_conceded') is not None]
        if not vals:
            return None
        avg = safe_avg(vals, 0.3)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (0.3 - avg) * 20))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(vals), 30, 0.0)), 
                            matches_used=len(vals), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_consistency(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        points = [m.stats.get('points', 1) or 1 for m in matches]
        if len(points) < 5:
            return None
        avg = safe_avg(points, 1.0)
        std_dev = math.sqrt(safe_divide(sum((p - avg) ** 2 for p in points), len(points), 0.0))
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (1.0 - std_dev) * 6))
        return AttributeScore(value=value, confidence=min(1.0, safe_divide(len(matches), 40, 0.0)), 
                            matches_used=len(matches), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_big_game(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        if not matches:
            return None
        
        big_games = [m for m in matches if m.opponent_rating and m.opponent_rating > (m.stats.get('own_rating', 0) or 0)]
        
        if len(big_games) < 5:
            wins = sum(1 for m in matches if m.stats.get('result') == 'W')
            value = 10 + (safe_divide(wins, len(matches), 0.4) - 0.4) * 20
            confidence = 0.5
        else:
            wins = sum(1 for m in big_games if m.stats.get('result') == 'W')
            draws = sum(1 for m in big_games if m.stats.get('result') == 'D')
            ppg = safe_divide(wins * 3 + draws, len(big_games), 1.0)
            value = 10 + (ppg - 1.0) * 5
            confidence = min(1.0, safe_divide(len(big_games), 15, 0.0))
        
        return AttributeScore(value=max(ATTR_MIN, min(ATTR_MAX, value)), confidence=confidence, 
                            matches_used=len(matches), last_updated=datetime.now(), 
                            data_tiers_used=list(set(m.data_tier for m in matches)))

    def _calc_squad_depth(self, matches: List[TeamMatchStats]) -> Optional[AttributeScore]:
        if len(matches) < 10:
            return None
        
        all_players = set()
        all_starters = set()
        
        for m in matches:
            all_players.update(m.lineup_player_ids)
            all_starters.update(m.starter_player_ids)
        
        player_appearances: Dict[int, int] = defaultdict(int)
        for m in matches:
            for pid in m.lineup_player_ids:
                player_appearances[pid] += 1
        
        regulars = sum(1 for pid, apps in player_appearances.items() if apps >= len(matches) * 0.2)
        
        value = 8 + (regulars - 11) * 1.5
        
        if len(all_starters) > 15:
            value += (len(all_starters) - 15) * 0.3
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(matches), 30, 0.0)),
            matches_used=len(matches),
            last_updated=datetime.now(),
            data_tiers_used=list(set(m.data_tier for m in matches))
        )


# =============================================================================
# REMAINING CALCULATORS (unchanged from original)
# =============================================================================

class Per90Calculator(BaseAttributeCalculator):
    """Calculate per-90-minute statistics."""
    
    def calculate(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats],
                position_group: PositionGroup, current_match_count: int) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        total_actual, total_expected, match_count = 0.0, 0.0, 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            actual = self.get_stat_value(ms, definition.primary_stats[0].stat_key, 0.0)
            if len(definition.primary_stats) <= 1:
                continue
            
            expected = self.get_stat_value(ms, definition.primary_stats[1].stat_key)
            if expected is None:
                continue
            
            total_actual += actual
            total_expected += expected
            match_count += 1
            tiers_used.add(ms.data_tier)
        
        if match_count == 0 or total_expected == 0:
            return None
        
        differential = safe_divide(total_actual - total_expected, total_expected, 0.0)
        percentile = self.percentile_calc.get_percentile(f"{definition.code}_diff", differential, position_group)
        
        return AttributeScore(
            value=self.percentile_calc.percentile_to_attribute(percentile),
            confidence=min(1.0, safe_divide(match_count, definition.min_matches * 2, 0.0)),
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class PercentageCalculator(BaseAttributeCalculator):
    """Calculate success rate percentages."""
    
    def calculate(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats],
                  position_group: PositionGroup, current_match_count: int) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        total_success, total_attempts, match_count = 0.0, 0.0, 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            success = self.get_stat_value(ms, definition.primary_stats[0].stat_key, 0.0)
            attempts = success
            
            if len(definition.primary_stats) > 1:
                attempts_val = self.get_stat_value(ms, definition.primary_stats[1].stat_key)
                if attempts_val and attempts_val > 0:
                    attempts = attempts_val
            
            if success is not None and attempts > 0:
                total_success += success
                total_attempts += attempts
                match_count += 1
                tiers_used.add(ms.data_tier)
        
        if total_attempts == 0:
            return None
        
        percentile = self.percentile_calc.get_percentile(
            f"{definition.code}_rate", total_success / total_attempts, position_group
        )
        
        return AttributeScore(
            value=self.percentile_calc.percentile_to_attribute(percentile),
            confidence=min(1.0, match_count / (definition.min_matches * 2)),
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class CompositeCalculator(BaseAttributeCalculator):
    """Calculate attributes from multiple weighted stats."""
    
    def calculate(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats],
                  position_group: PositionGroup, current_match_count: int) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        stat_accumulators = {sr.stat_key: StatAccumulator() for sr in definition.primary_stats + definition.secondary_stats}
        tiers_used = set()
        
        for i, ms in enumerate(match_stats):
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            weight = DecayCalculator.get_match_decay(current_match_count - i)
            
            for sr in definition.primary_stats + definition.secondary_stats:
                value = self.get_stat_value(ms, sr.stat_key)
                if value is not None:
                    if ms.minutes_played > 0:
                        value = (value / ms.minutes_played) * 90
                    stat_accumulators[sr.stat_key].add(value, weight, ms.match_id)
            
            tiers_used.add(ms.data_tier)
        
        component_scores, total_weight = [], 0.0
        
        for sr in definition.primary_stats:
            acc = stat_accumulators[sr.stat_key]
            if acc.match_count == 0:
                if not sr.is_optional:
                    return None
                continue
            
            percentile = self.percentile_calc.get_percentile(sr.stat_key, acc.weighted_average, position_group)
            weight = definition.stat_weights.get(sr.stat_key, sr.weight)
            component_scores.append((percentile, weight))
            total_weight += weight
        
        if not component_scores or total_weight == 0:
            return None
        
        final_percentile = sum(p * w for p, w in component_scores) / total_weight
        if not definition.higher_is_better:
            final_percentile = 100 - final_percentile
        
        match_count = max(acc.match_count for acc in stat_accumulators.values())
        
        return AttributeScore(
            value=self.percentile_calc.percentile_to_attribute(final_percentile),
            confidence=min(1.0, match_count / (definition.min_matches * 2)),
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class DifferentialCalculator(BaseAttributeCalculator):
    """Calculate based on expected vs actual."""
    
    def calculate(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats],
                  position_group: PositionGroup, current_match_count: int) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        total_actual, total_expected, match_count = 0.0, 0.0, 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            actual = self.get_stat_value(ms, definition.primary_stats[0].stat_key, 0.0)
            if len(definition.primary_stats) <= 1:
                continue
            
            expected = self.get_stat_value(ms, definition.primary_stats[1].stat_key)
            if expected is None:
                continue
            
            total_actual += actual
            total_expected += expected
            match_count += 1
            tiers_used.add(ms.data_tier)
        
        if match_count == 0 or total_expected == 0:
            return None
        
        differential = (total_actual - total_expected) / total_expected
        percentile = self.percentile_calc.get_percentile(f"{definition.code}_diff", differential, position_group)
        
        return AttributeScore(
            value=self.percentile_calc.percentile_to_attribute(percentile),
            confidence=min(1.0, match_count / (definition.min_matches * 2)),
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class AggregateCalculator(BaseAttributeCalculator):
    """Calculate aggregate statistics."""
    
    def calculate(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats],
                  position_group: PositionGroup, current_match_count: int) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        if definition.code == 'STA':
            return self._calc_stamina(definition, match_stats)
        return None
    
    def _calc_stamina(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        minutes = [ms.minutes_played for ms in match_stats if ms.data_tier.value >= definition.min_tier.value and ms.minutes_played > 0]
        tiers_used = set(ms.data_tier for ms in match_stats if ms.minutes_played > 0)
        
        if len(minutes) < definition.min_matches:
            return None
        
        avg_minutes = sum(minutes) / len(minutes)
        value = max(ATTR_MIN, min(ATTR_MAX, 5 + (avg_minutes / 90) * 10))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, len(minutes) / (definition.min_matches * 2)),
            matches_used=len(minutes),
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


# =============================================================================
# MAIN ATTRIBUTE ENGINE
# =============================================================================

class AttributeEngine:
    """Main engine for calculating all attributes."""
    
    def __init__(self):
        self.percentile_calc = PercentileCalculator()
        self._bootstrapped = False
        self.calculators = {
            CalculationMethod.PER_90: Per90Calculator(self.percentile_calc),
            CalculationMethod.PERCENTAGE: PercentageCalculator(self.percentile_calc),
            CalculationMethod.COMPOSITE: CompositeCalculator(self.percentile_calc),
            CalculationMethod.DIFFERENTIAL: DifferentialCalculator(self.percentile_calc),
            CalculationMethod.CONTEXTUAL: ContextualCalculator(self.percentile_calc),
            CalculationMethod.AGGREGATE: AggregateCalculator(self.percentile_calc),
            CalculationMethod.INVERSE: InverseCalculator(self.percentile_calc),
        }
        
        self.coach_calculator = CoachAttributeCalculator()
        self.team_calculator = TeamAttributeCalculator()

    def bootstrap_percentiles(
        self, 
        historical_stats: Dict[int, List[PlayerMatchStats]],
        player_registry: Dict[int, 'Player']
    ) -> None:
        """
        Pre-populate percentile distributions from historical data.
        Call this BEFORE calculating attributes for the first time.
        """
        if self._bootstrapped:
            return
        
        for player_id, match_stats_list in historical_stats.items():
            player = player_registry.get(player_id)
            if not player:
                continue
            
            for ms in match_stats_list:
                if ms.minutes_played < 10:
                    continue
                    
                for stat_key, value in ms.stats.items():
                    if value is not None and value > 0:
                        per_90 = (value / ms.minutes_played) * 90 if ms.minutes_played > 0 else value
                        self.percentile_calc.record_stat(
                            stat_key, 
                            per_90, 
                            ms.position_group or player.position_group
                        )
        
        self._bootstrapped = True
        print(f"Bootstrapped percentiles with {len(historical_stats)} players")

    def calculate_player_attributes(
        self, 
        player: 'Player', 
        match_stats_history: List[PlayerMatchStats],
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Dict[str, AttributeScore]:
        """Calculate all attributes for a player with league normalization."""
        results = {}
        
        if player.is_goalkeeper:
            from config.attribute_definitions import GOALKEEPER_ATTRIBUTES
            attributes = GOALKEEPER_ATTRIBUTES
        else:
            attributes = ALL_PLAYER_ATTRIBUTES
        
        # Determine player's primary league
        league_counts: Dict[int, int] = {}
        for ms in match_stats_history:
            if ms.league_id:
                league_counts[ms.league_id] = league_counts.get(ms.league_id, 0) + 1
        
        primary_league_id = max(league_counts, key=league_counts.get) if league_counts else None
        
        for attr_code, definition in attributes.items():
            if definition.applicable_positions and player.position_group not in definition.applicable_positions:
                continue
            if definition.exclude_positions and player.position_group in definition.exclude_positions:
                continue
            
            calculator = self.calculators.get(definition.calculation_method)
            if calculator:
                score = calculator.calculate(
                    definition, 
                    match_stats_history, 
                    player.position_group, 
                    player.matches_played
                )
                if score:
                    # Apply league normalization if available
                    if league_ratings and primary_league_id:
                        score.value = self.normalize_by_league_strength(
                            score.value,
                            primary_league_id,
                            league_ratings
                        )
                        # Re-clamp after normalization
                        score.value = max(ATTR_MIN, min(ATTR_MAX, score.value))
                    
                    results[attr_code] = score
        
        return results
    
    def calculate_coach_attributes(self, coach: Any, match_stats_history: List[CoachMatchStats]) -> Dict[str, AttributeScore]:
        return self.coach_calculator.calculate_all_attributes(coach, match_stats_history)
    
    def calculate_team_attributes(self, team: Any, match_stats_history: List[TeamMatchStats]) -> Dict[str, AttributeScore]:
        return self.team_calculator.calculate_all_attributes(team, match_stats_history)
    
    def update_population_stats(self, all_players: Dict[int, Player], all_match_stats: Dict[int, List[PlayerMatchStats]]):
        for player_id, player in all_players.items():
            if player_id not in all_match_stats:
                continue
            
            for ms in all_match_stats[player_id]:
                for stat_key, value in ms.stats.items():
                    if value is not None:
                        per_90 = safe_divide(value * 90, ms.minutes_played, value) if ms.minutes_played > 0 else value
                        self.percentile_calc.record_stat(stat_key, per_90, player.position_group)

    def normalize_by_league_strength(
        self, 
        raw_value: float, 
        player_league_id: int,
        league_ratings: Dict[int, float],
        base_league_rating: float = 1000
    ) -> float:
        """Adjust attribute value based on league difficulty."""
        league_rating = league_ratings.get(player_league_id, base_league_rating)
        multiplier = league_rating / base_league_rating
        
        # Soft adjustment: 10% bonus/penalty per 200 rating points difference
        adjustment = (multiplier - 1) * 0.5
        return raw_value * (1 + adjustment)