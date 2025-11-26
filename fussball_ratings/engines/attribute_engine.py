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
    MIN_MATCHES_FOR_ATTR, PERCENTILE_TO_ATTR, MATCH_DECAY_WEIGHTS,
    QUALITY_ADJUSTMENT_WEIGHT, REFERENCE_RATING, MIN_RATING_DIFF_THRESHOLD
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

def adjust_stat_for_quality(
    raw_value: float,
    quality_multiplier: float,
    weight: float = QUALITY_ADJUSTMENT_WEIGHT
) -> float:
    """
    Adjust a stat value based on quality of competition.
    
    Args:
        raw_value: The raw stat value
        quality_multiplier: From MatchQualityContext (>1 = harder, <1 = easier)
        weight: How much to weight the adjustment (0-1)
    
    Returns:
        Blended value between raw and quality-adjusted
    """
    if weight <= 0:
        return raw_value
    
    adjusted = raw_value * quality_multiplier
    return raw_value * (1 - weight) + adjusted * weight


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

@dataclass
class MatchQualityContext:
    """Quality context for a single match."""
    match_id: int
    opponent_rating: float
    league_rating: float
    own_rating: float
    
    @property
    def combined_quality(self) -> float:
        """Combined quality: 60% opponent, 40% league."""
        return (self.opponent_rating * 0.6) + (self.league_rating * 0.4)
    
    @property
    def quality_multiplier(self) -> float:
        """Multiplier for stat adjustment (>1 = harder, <1 = easier)."""
        return self.combined_quality / REFERENCE_RATING
    
    @classmethod
    def from_player_match_stats(
        cls,
        pms: 'PlayerMatchStats',
        league_ratings: Dict[int, float]
    ) -> 'MatchQualityContext':
        """Create from PlayerMatchStats - ADD THIS METHOD."""
        league_rating = REFERENCE_RATING
        if pms.league_id and pms.league_id in league_ratings:
            league_rating = league_ratings[pms.league_id]
        
        own_rating = pms.own_team_rating or REFERENCE_RATING
        opponent_rating = pms.opponent_team_rating or REFERENCE_RATING
        
        return cls(
            match_id=pms.match_id,
            opponent_rating=opponent_rating,
            league_rating=league_rating,
            own_rating=own_rating
        )
    
    @classmethod
    def from_coach_match_stats(
        cls,
        cms: 'CoachMatchStats',
        league_ratings: Dict[int, float]
    ) -> 'MatchQualityContext':
        """Create from CoachMatchStats."""
        league_id = cms.team_stats.get('league_id')
        league_rating = league_ratings.get(league_id, REFERENCE_RATING) if league_id else REFERENCE_RATING
        own_rating = cms.team_stats.get('team_rating') or REFERENCE_RATING
        opponent_rating = cms.team_stats.get('opponent_rating') or REFERENCE_RATING
        
        return cls(
            match_id=cms.match_id,
            opponent_rating=opponent_rating,
            league_rating=league_rating,
            own_rating=own_rating
        )
    
    @classmethod
    def from_team_match_stats(
        cls,
        tms: 'TeamMatchStats',
        league_ratings: Dict[int, float]
    ) -> 'MatchQualityContext':
        """Create from TeamMatchStats."""
        league_id = tms.stats.get('league_id')
        league_rating = league_ratings.get(league_id, REFERENCE_RATING) if league_id else REFERENCE_RATING
        own_rating = tms.stats.get('own_rating') or tms.stats.get('team_rating') or REFERENCE_RATING
        opponent_rating = tms.opponent_rating or REFERENCE_RATING
        
        return cls(
            match_id=tms.match_id,
            opponent_rating=opponent_rating,
            league_rating=league_rating,
            own_rating=own_rating
        )


    
class PercentileCalculator:
    """
    Calculates percentiles within a population for normalization.
    Now with quality stratification using LIVE ratings.
    """
    
    def __init__(self):
        # Standard distributions (for backward compatibility)
        self.stat_distributions: Dict[str, Dict[PositionGroup, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Quality-stratified distributions
        # stat_key -> quality_tier -> position_group -> values
        self.quality_distributions: Dict[str, Dict[str, Dict[PositionGroup, List[Tuple[float, float]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
    
    def _get_quality_tier(self, avg_quality: float) -> str:
        """
        Determine quality tier based on average combined quality.
        Uses relative thresholds based on REFERENCE_RATING.
        """
        ratio = avg_quality / REFERENCE_RATING
        
        if ratio >= 1.15:
            return 'elite'      # Top leagues, strong opponents
        elif ratio >= 0.95:
            return 'good'       # Solid leagues
        elif ratio >= 0.75:
            return 'average'    # Mid-tier leagues
        else:
            return 'weak'       # Lower leagues
    
    def record_stat(
        self, 
        stat_key: str, 
        value: float, 
        position_group: PositionGroup,
        quality_context: Optional[MatchQualityContext] = None
    ):
        """Record a stat value with optional quality context."""
        # Always record to standard distribution (backward compatibility)
        self.stat_distributions[stat_key][position_group].append(value)
        
        # Also record to quality-stratified distribution if context provided
        if quality_context:
            tier = self._get_quality_tier(quality_context.combined_quality)
            # Store (value, quality_multiplier) tuple
            self.quality_distributions[stat_key][tier][position_group].append(
                (value, quality_context.quality_multiplier)
            )
    
    def get_percentile(
        self, 
        stat_key: str, 
        value: float, 
        position_group: PositionGroup,
        quality_context: Optional[MatchQualityContext] = None
    ) -> float:
        """
        Get percentile with optional quality-aware comparison.
        
        If quality_context is provided:
        - Compares primarily within same quality tier (70%)
        - Also compares globally for context (30%)
        """
        # Standard global comparison
        global_values = self.stat_distributions[stat_key].get(position_group, [])
        
        if not global_values:
            return 50.0
        
        global_percentile = self._calculate_percentile(value, global_values)
        
        # If no quality context, return global percentile
        if not quality_context:
            return global_percentile
        
        # Quality-aware comparison
        tier = self._get_quality_tier(quality_context.combined_quality)
        tier_data = self.quality_distributions[stat_key].get(tier, {}).get(position_group, [])
        
        if len(tier_data) < 20:
            # Not enough data in tier, fall back to global
            return global_percentile
        
        # Extract just values from (value, multiplier) tuples
        tier_values = [v for v, _ in tier_data]
        tier_percentile = self._calculate_percentile(value, tier_values)
        
        # Blend: 70% same-tier, 30% global
        return tier_percentile * 0.7 + global_percentile * 0.3
    
    def _calculate_percentile(self, value: float, distribution: List[float]) -> float:
        """Calculate raw percentile within a distribution."""
        if not distribution:
            return 50.0
        below = sum(1 for v in distribution if v < value)
        equal = sum(1 for v in distribution if v == value)
        return (below + 0.5 * equal) / len(distribution) * 100
    
    def percentile_to_attribute(self, percentile: float) -> float:
        """Convert percentile to 1-20 attribute scale."""
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

class BaseAttributeCalculator:
    """Base calculator with quality-awareness."""
    
    def __init__(self, percentile_calc: PercentileCalculator):
        self.percentile_calc = percentile_calc
    
    def calculate(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup, 
        current_match_count: int,
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Optional['AttributeScore']:
        """Override in subclasses."""
        raise NotImplementedError
    
    def has_required_data(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats]
    ) -> bool:
        if len(match_stats) < definition.min_matches:
            return False
        valid_matches = [
            ms for ms in match_stats 
            if ms.data_tier.value >= definition.min_tier.value
        ]
        return len(valid_matches) >= definition.min_matches
    
    def get_stat_value(
        self, 
        match_stat: PlayerMatchStats, 
        stat_key: str, 
        default: Optional[float] = None
    ) -> Optional[float]:
        return match_stat.stats.get(stat_key, default)
    
    def get_quality_context(
        self,
        match_stat: PlayerMatchStats,
        league_ratings: Optional[Dict[int, float]]
    ) -> MatchQualityContext:
        """Extract quality context from a player match stat."""
        return MatchQualityContext.from_player_match_stats(
            match_stat, 
            league_ratings or {}
        )

class ContextualCalculator(BaseAttributeCalculator):
    """Calculate context-dependent attributes with proper implementations."""
    
    def calculate(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup, 
        current_match_count: int,
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Optional[AttributeScore]:
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
        if calc_func:
            if definition.code == 'BIG':
                return calc_func(definition, match_stats, league_ratings)
            return calc_func(definition, match_stats)
        return None
        
    def _calc_consistency(self, definition: AttributeDefinition, match_stats: List[PlayerMatchStats]) -> Optional[AttributeScore]:
        ratings = [ms.rating for ms in match_stats if ms.data_tier.value >= definition.min_tier.value and ms.rating]
        tiers_used = set(ms.data_tier for ms in match_stats if ms.rating)
        
        if len(ratings) < definition.min_matches:
            return None
        
        avg = safe_avg(ratings, 7.0)
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
    
    def _calc_big_game(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Optional[AttributeScore]:
        """Big game performance - uses league_ratings for better opponent comparison."""
        big_game_ratings = []
        regular_game_ratings = []
        tiers_used = set()
        
        for ms in match_stats:
            if ms.rating is None:
                continue
            tiers_used.add(ms.data_tier)
            
            # Determine if "big game" using available ratings
            is_big_game = False
            
            if ms.own_team_rating is not None and ms.opponent_team_rating is not None:
                # Direct comparison from match data
                rating_diff = ms.opponent_team_rating - ms.own_team_rating
                is_big_game = rating_diff > 50  # Opponent rated 50+ higher
            elif league_ratings and ms.league_id:
                # Use league rating as proxy for game importance
                league_rating = league_ratings.get(ms.league_id, REFERENCE_RATING)
                is_big_game = league_rating > REFERENCE_RATING * 1.1  # Top league
            
            if is_big_game:
                big_game_ratings.append(ms.rating)
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
    """Calculate inverse statistics (no quality adjustment needed for injuries)."""
    
    def calculate(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup, 
        current_match_count: int,
        league_ratings: Optional[Dict[int, float]] = None  # Accept but don't use
    ) -> Optional[AttributeScore]:
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
    """Calculates coach attributes from match history with quality adjustment."""
    
    def calculate_all_attributes(
        self, 
        coach: Any, 
        match_history: List[CoachMatchStats],
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Dict[str, AttributeScore]:
        """Calculate all applicable attributes for a coach."""
        results = {}
        
        for attr_code, definition in COACH_ATTRIBUTES.items():
            score = self._calculate_attribute(definition, match_history, league_ratings)
            if score:
                results[attr_code] = score
        
        return results
    
    def _calculate_attribute(
        self,
        definition: AttributeDefinition,
        match_history: List[CoachMatchStats],
        league_ratings: Optional[Dict[int, float]]
    ) -> Optional[AttributeScore]:
        """Calculate a single attribute."""
        valid_matches = [
            m for m in match_history
            if m.data_tier.value >= definition.min_tier.value
        ]
        
        if len(valid_matches) < definition.min_matches:
            return None
        
        # Build quality contexts
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]] = []
        for m in valid_matches:
            if league_ratings:
                ctx = MatchQualityContext.from_coach_match_stats(m, league_ratings)
            else:
                ctx = None
            matches_with_quality.append((m, ctx))
        
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
            'AST': self._calc_attacking_style,
            'DST': self._calc_defensive_style,
        }
        
        calc_func = calc_map.get(definition.code)
        if calc_func:
            return calc_func(matches_with_quality, definition)
        return None
    
    def _calc_attacking_tactics(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Attacking output vs league average."""
        total_goals = 0.0
        total_xg = 0.0
        matches_with_xg = 0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for cms, quality_ctx in matches_with_quality:
            goals = cms.team_stats.get('team_score', 0) or 0
            xg = cms.team_stats.get('xg')
            
            if quality_ctx:
                adjusted_goals = adjust_stat_for_quality(goals, quality_ctx.quality_multiplier)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted_goals = goals
            
            total_goals += adjusted_goals
            
            if xg is not None:
                if quality_ctx:
                    adjusted_xg = adjust_stat_for_quality(xg, quality_ctx.quality_multiplier)
                else:
                    adjusted_xg = xg
                total_xg += adjusted_xg
                matches_with_xg += 1
            
            tiers_used.add(cms.data_tier)
        
        num_matches = len(matches_with_quality)
        
        if matches_with_xg >= num_matches // 2 and matches_with_xg > 0:
            diff_per_match = safe_divide(total_goals - total_xg, matches_with_xg, 0.0)
            value = 10 + diff_per_match * 6.67
        else:
            avg_goals = safe_divide(total_goals, num_matches, 1.5)
            value = 10 + (avg_goals - 1.5) * 5
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        avg_quality = safe_divide(total_quality, quality_count, REFERENCE_RATING) if quality_count > 0 else REFERENCE_RATING
        quality_factor = min(1.0, avg_quality / REFERENCE_RATING)
        base_confidence = min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0))
        confidence = base_confidence * (0.8 + 0.2 * quality_factor)
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_defensive_tactics(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Defensive solidity vs league average."""
        total_conceded = 0.0
        total_xg_against = 0.0
        matches_with_xg = 0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for cms, quality_ctx in matches_with_quality:
            conceded = cms.team_stats.get('team_score_conceded', 0) or 0
            xg_against = cms.team_stats.get('xg_conceded')
            
            # Inverse adjustment - conceding less vs strong teams is impressive
            if quality_ctx:
                inverse_mult = safe_divide(REFERENCE_RATING, quality_ctx.combined_quality, 1.0)
                adjusted_conceded = adjust_stat_for_quality(conceded, inverse_mult)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted_conceded = conceded
            
            total_conceded += adjusted_conceded
            
            if xg_against is not None:
                if quality_ctx:
                    adjusted_xg = adjust_stat_for_quality(xg_against, inverse_mult)
                else:
                    adjusted_xg = xg_against
                total_xg_against += adjusted_xg
                matches_with_xg += 1
            
            tiers_used.add(cms.data_tier)
        
        num_matches = len(matches_with_quality)
        
        if matches_with_xg >= num_matches // 2 and matches_with_xg > 0:
            diff_per_match = safe_divide(total_xg_against - total_conceded, matches_with_xg, 0.0)
            value = 10 + diff_per_match * 6.67
        else:
            avg_conceded = safe_divide(total_conceded, num_matches, 1.0)
            value = 10 + (1.0 - avg_conceded) * 10
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        avg_quality = safe_divide(total_quality, quality_count, REFERENCE_RATING) if quality_count > 0 else REFERENCE_RATING
        quality_factor = min(1.0, avg_quality / REFERENCE_RATING)
        base_confidence = min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0))
        confidence = base_confidence * (0.8 + 0.2 * quality_factor)
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_discipline(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Card and foul management - no quality adjustment (absolute metric)."""
        total_cost = 0.0
        tiers_used = set()
        
        for cms, _ in matches_with_quality:
            cost = (
                (cms.team_stats.get('yellow_cards', 0) or 0) + 
                (cms.team_stats.get('red_cards', 0) or 0) * 3 + 
                (cms.team_stats.get('fouls', 0) or 0) * 0.1
            )
            total_cost += cost
            tiers_used.add(cms.data_tier)
        
        num_matches = len(matches_with_quality)
        avg_cost = safe_divide(total_cost, num_matches, 3.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (3 - avg_cost) * 3))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0)),
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_adaptability(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Points gained when trailing - quality adjusted."""
        comebacks = 0
        blown_leads = 0
        matches_with_events = 0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for cms, quality_ctx in matches_with_quality:
            if not cms.goal_events:
                continue
            
            matches_with_events += 1
            tiers_used.add(cms.data_tier)
            
            if quality_ctx:
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            
            was_losing = False
            was_winning = False
            final_result = cms.team_stats.get('result', 'D')
            
            for event in cms.goal_events:
                home_score = event.get('home_score_after', 0)
                away_score = event.get('away_score_after', 0)
                
                if cms.is_home_team:
                    own_score, opp_score = home_score, away_score
                else:
                    own_score, opp_score = away_score, home_score
                
                if own_score < opp_score:
                    was_losing = True
                elif own_score > opp_score:
                    was_winning = True
            
            if was_losing and final_result in ('W', 'D'):
                # Comeback worth more vs strong opponent
                if quality_ctx and quality_ctx.quality_multiplier > 1.0:
                    comebacks += quality_ctx.quality_multiplier
                else:
                    comebacks += 1
            if was_winning and final_result in ('L', 'D'):
                blown_leads += 1
        
        num_matches = len(matches_with_quality)
        
        if matches_with_events < 10:
            wins = sum(1 for cms, _ in matches_with_quality if cms.team_stats.get('result') == 'W')
            value = 10 + (safe_divide(wins, num_matches, 0.4) - 0.4) * 20
            confidence = 0.5
        else:
            comeback_rate = safe_divide(comebacks, matches_with_events, 0.0)
            blown_lead_rate = safe_divide(blown_leads, matches_with_events, 0.0)
            value = 10 + (comeback_rate * 15) - (blown_lead_rate * 10)
            confidence = min(1.0, safe_divide(matches_with_events, 25, 0.0))
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_game_management(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Late game results (75+ min)."""
        late_goal_impact = 0.0
        matches_with_events = 0
        tiers_used = set()
        
        for cms, quality_ctx in matches_with_quality:
            if not cms.goal_events:
                continue
            
            matches_with_events += 1
            tiers_used.add(cms.data_tier)
            
            score_at_75 = (0, 0)
            for event in cms.goal_events:
                if event.get('minute', 0) <= 75:
                    if cms.is_home_team:
                        score_at_75 = (event.get('home_score_after', 0), event.get('away_score_after', 0))
                    else:
                        score_at_75 = (event.get('away_score_after', 0), event.get('home_score_after', 0))
            
            own_75, opp_75 = score_at_75
            final_own = cms.team_stats.get('team_score', 0) or 0
            final_opp = cms.team_stats.get('team_score_conceded', 0) or 0
            
            diff_at_75 = own_75 - opp_75
            diff_final = final_own - final_opp
            
            # Quality adjustment - late goals vs strong teams worth more
            multiplier = quality_ctx.quality_multiplier if quality_ctx else 1.0
            
            if diff_final > diff_at_75:
                late_goal_impact += 1 * multiplier
            elif diff_final < diff_at_75:
                late_goal_impact -= 1
        
        num_matches = len(matches_with_quality)
        
        if matches_with_events < 10:
            wins = sum(1 for cms, _ in matches_with_quality if cms.team_stats.get('result') == 'W')
            value = 10 + (safe_divide(wins, num_matches, 0.4) - 0.4) * 15
            confidence = 0.4
        else:
            impact_rate = safe_divide(late_goal_impact, matches_with_events, 0.0)
            value = 10 + impact_rate * 8
            confidence = min(1.0, safe_divide(matches_with_events, 25, 0.0))
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_squad_rotation(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Lineup variation management - no quality adjustment needed."""
        if len(matches_with_quality) < 2:
            return None
        
        all_players = set()
        tiers_used = set()
        
        for cms, _ in matches_with_quality:
            all_players.update(cms.lineup_player_ids)
            tiers_used.add(cms.data_tier)
        
        num_matches = len(matches_with_quality)
        avg_unique = safe_divide(len(all_players), num_matches, 12.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg_unique - 12) * 2))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0)),
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_formation_mastery(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Success rate by formation - quality weighted."""
        formation_results: Dict[str, Dict[str, float]] = defaultdict(lambda: {'W': 0, 'D': 0, 'L': 0, 'total': 0, 'quality': 0})
        tiers_used = set()
        
        for cms, quality_ctx in matches_with_quality:
            formation = cms.formation
            if not formation:
                continue
            
            result = cms.team_stats.get('result', 'D')
            tiers_used.add(cms.data_tier)
            
            # Weight wins by quality of opposition
            if result == 'W' and quality_ctx:
                formation_results[formation]['W'] += quality_ctx.quality_multiplier
            else:
                formation_results[formation][result] += 1
            
            formation_results[formation]['total'] += 1
            if quality_ctx:
                formation_results[formation]['quality'] += quality_ctx.combined_quality
        
        if not formation_results:
            return None
        
        best_win_rate = 0.0
        formations_mastered = 0
        
        for formation, results in formation_results.items():
            if results['total'] >= 5:
                win_rate = safe_divide(results['W'], results['total'], 0.0)
                if win_rate > 0.5:
                    formations_mastered += 1
                best_win_rate = max(best_win_rate, win_rate)
        
        value = max(ATTR_MIN, min(ATTR_MAX, 8 + formations_mastered * 2 + best_win_rate * 4))
        
        num_matches = len(matches_with_quality)
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0)),
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_substitution_impact(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Effect of substitutions on results."""
        positive_impact = 0.0
        negative_impact = 0.0
        total_subs = 0
        tiers_used = set()
        
        for cms, quality_ctx in matches_with_quality:
            tiers_used.add(cms.data_tier)
            
            for sub in cms.substitutions:
                sub_minute = sub.get('minute', 45)
                
                goals_after = [e for e in cms.goal_events if e.get('minute', 0) > sub_minute]
                if not goals_after:
                    continue
                
                total_subs += 1
                
                # Get score change after sub
                score_before = sub.get('score_before', (0, 0))
                if isinstance(score_before, tuple):
                    own_before, opp_before = score_before
                else:
                    own_before, opp_before = 0, 0
                
                final_event = goals_after[-1]
                if cms.is_home_team:
                    own_after = final_event.get('home_score_after', 0)
                    opp_after = final_event.get('away_score_after', 0)
                else:
                    own_after = final_event.get('away_score_after', 0)
                    opp_after = final_event.get('home_score_after', 0)
                
                diff_before = own_before - opp_before
                diff_after = own_after - opp_after
                
                multiplier = quality_ctx.quality_multiplier if quality_ctx else 1.0
                
                if diff_after > diff_before:
                    positive_impact += 1 * multiplier
                elif diff_after < diff_before:
                    negative_impact += 1
        
        num_matches = len(matches_with_quality)
        
        if total_subs < 10:
            wins = sum(1 for cms, _ in matches_with_quality if cms.team_stats.get('result') == 'W')
            value = 10 + (safe_divide(wins, num_matches, 0.4) - 0.4) * 15
            confidence = 0.4
        else:
            impact_rate = safe_divide(positive_impact - negative_impact, total_subs, 0.0)
            value = 10 + impact_rate * 10
            confidence = min(1.0, safe_divide(total_subs, 50, 0.0))
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_player_development(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Player rating improvements under management."""
        sorted_matches = sorted(
            matches_with_quality, 
            key=lambda x: x[0].match_date or datetime.min
        )
        
        if len(sorted_matches) < 10:
            return None
        
        tiers_used = set()
        player_first_ratings: Dict[int, List[float]] = defaultdict(list)
        player_last_ratings: Dict[int, List[float]] = defaultdict(list)
        
        midpoint = len(sorted_matches) // 2
        
        for i, (cms, _) in enumerate(sorted_matches):
            tiers_used.add(cms.data_tier)
            for player_id, rating in cms.player_ratings.items():
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
        
        num_matches = len(matches_with_quality)
        
        if not improvements:
            return AttributeScore(
                value=10, confidence=0.3, matches_used=num_matches,
                last_updated=datetime.now(), data_tiers_used=list(tiers_used)
            )
        
        avg_improvement = safe_avg(improvements, 0.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + avg_improvement * 16.67))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(improvements), 15, 0.0)),
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_youth_development(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """U23 player progression."""
        sorted_matches = sorted(
            matches_with_quality, 
            key=lambda x: x[0].match_date or datetime.min
        )
        
        if len(sorted_matches) < 10:
            return None
        
        tiers_used = set()
        young_first_ratings: Dict[int, List[float]] = defaultdict(list)
        young_last_ratings: Dict[int, List[float]] = defaultdict(list)
        
        midpoint = len(sorted_matches) // 2
        
        for i, (cms, _) in enumerate(sorted_matches):
            tiers_used.add(cms.data_tier)
            for player_id, rating in cms.player_ratings.items():
                age = cms.player_ages.get(player_id, 30)
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
        
        num_matches = len(matches_with_quality)
        
        if not improvements:
            return AttributeScore(
                value=10, confidence=0.2, matches_used=num_matches,
                last_updated=datetime.now(), data_tiers_used=list(tiers_used)
            )
        
        avg_improvement = safe_avg(improvements, 0.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + avg_improvement * 12))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(improvements), 10, 0.0)),
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_attacking_style(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Possession-based vs direct attacking approach."""
        total_possession = 0.0
        total_opp_half = 0.0
        total_own_half = 0.0
        valid_count = 0
        tiers_used = set()
        
        for cms, _ in matches_with_quality:
            possession = cms.team_stats.get('possession')
            opp_half = cms.team_stats.get('opposition_half_passes', 0) or 0
            own_half = cms.team_stats.get('own_half_passes', 0) or 0
            
            if possession is not None:
                total_possession += possession
                total_opp_half += opp_half
                total_own_half += own_half
                valid_count += 1
                tiers_used.add(cms.data_tier)
        
        if valid_count < definition.min_matches:
            return None
        
        avg_possession = safe_divide(total_possession, valid_count, 50.0)
        pass_ratio = safe_divide(total_opp_half, total_own_half + total_opp_half, 0.5)
        
        # 1 = ultra-direct, 20 = ultra-possession
        value = 5 + (avg_possession - 40) * 0.25 + (pass_ratio - 0.4) * 15
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(valid_count, definition.min_matches * 2, 0.0)),
            matches_used=valid_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_defensive_style(
        self, 
        matches_with_quality: List[Tuple[CoachMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """High press vs low block defensive approach."""
        total_tackles = 0.0
        total_interceptions = 0.0
        total_possession = 0.0
        valid_count = 0
        tiers_used = set()
        
        for cms, _ in matches_with_quality:
            tackles = cms.team_stats.get('tackles', 0) or 0
            interceptions = cms.team_stats.get('interceptions', 0) or 0
            possession = cms.team_stats.get('possession', 50) or 50
            
            total_tackles += tackles
            total_interceptions += interceptions
            total_possession += possession
            valid_count += 1
            tiers_used.add(cms.data_tier)
        
        if valid_count < definition.min_matches:
            return None
        
        avg_def_actions = safe_divide(total_tackles + total_interceptions, valid_count, 20.0)
        avg_possession = safe_divide(total_possession, valid_count, 50.0)
        
        # 1 = deep block, 20 = gegenpressing
        # High pressing teams: more tackles/interceptions + less possession (counterpress)
        # OR more tackles/interceptions + high possession (high line)
        press_intensity = avg_def_actions / 25.0  # Normalize around 25 actions/game
        value = 10 + (press_intensity - 1.0) * 8
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(valid_count, definition.min_matches * 2, 0.0)),
            matches_used=valid_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


# =============================================================================
# UPDATED TEAM ATTRIBUTE CALCULATOR - SQUAD DEPTH FIXED
# =============================================================================

class TeamAttributeCalculator:
    """Calculates team attributes from match history with quality adjustment."""
    
    def calculate_all_attributes(
        self, 
        team: Any, 
        match_history: List[TeamMatchStats],
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Dict[str, AttributeScore]:
        """Calculate all applicable attributes for a team."""
        results = {}
        
        for attr_code, definition in TEAM_ATTRIBUTES.items():
            score = self._calculate_attribute(definition, match_history, league_ratings)
            if score:
                results[attr_code] = score
        
        return results
    
    def _calculate_attribute(
        self,
        definition: AttributeDefinition,
        match_history: List[TeamMatchStats],
        league_ratings: Optional[Dict[int, float]]
    ) -> Optional[AttributeScore]:
        """Calculate a single attribute."""
        valid_matches = [
            m for m in match_history
            if m.data_tier.value >= definition.min_tier.value
        ]
        
        if len(valid_matches) < definition.min_matches:
            return None
        
        # Build quality contexts
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]] = []
        for m in valid_matches:
            if league_ratings:
                ctx = MatchQualityContext.from_team_match_stats(m, league_ratings)
            else:
                ctx = None
            matches_with_quality.append((m, ctx))
        
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
        if calc_func:
            return calc_func(matches_with_quality, definition)
        return None
    
    def _calc_attack(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Goals and chances created - quality adjusted."""
        total_goals = 0.0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for tms, quality_ctx in matches_with_quality:
            goals = tms.stats.get('team_score', 0) or 0
            
            if quality_ctx:
                adjusted_goals = adjust_stat_for_quality(goals, quality_ctx.quality_multiplier)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted_goals = goals
            
            total_goals += adjusted_goals
            tiers_used.add(tms.data_tier)
        
        num_matches = len(matches_with_quality)
        avg_goals = safe_divide(total_goals, num_matches, 1.5)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg_goals - 1.5) * 5))
        
        avg_quality = safe_divide(total_quality, quality_count, REFERENCE_RATING) if quality_count > 0 else REFERENCE_RATING
        quality_factor = min(1.0, avg_quality / REFERENCE_RATING)
        base_confidence = min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0))
        confidence = base_confidence * (0.8 + 0.2 * quality_factor)
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_defense(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Goals and chances conceded - inverse quality adjusted."""
        total_conceded = 0.0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for tms, quality_ctx in matches_with_quality:
            conceded = tms.stats.get('team_score_conceded', 0) or 0
            
            if quality_ctx:
                inverse_mult = safe_divide(REFERENCE_RATING, quality_ctx.combined_quality, 1.0)
                adjusted_conceded = adjust_stat_for_quality(conceded, inverse_mult)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted_conceded = conceded
            
            total_conceded += adjusted_conceded
            tiers_used.add(tms.data_tier)
        
        num_matches = len(matches_with_quality)
        avg_conceded = safe_divide(total_conceded, num_matches, 1.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (1.0 - avg_conceded) * 10))
        
        avg_quality = safe_divide(total_quality, quality_count, REFERENCE_RATING) if quality_count > 0 else REFERENCE_RATING
        quality_factor = min(1.0, avg_quality / REFERENCE_RATING)
        base_confidence = min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0))
        confidence = base_confidence * (0.8 + 0.2 * quality_factor)
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_possession(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Ball retention style - no quality adjustment (absolute metric)."""
        poss_values = []
        tiers_used = set()
        
        for tms, _ in matches_with_quality:
            poss = tms.stats.get('possession')
            if poss is not None:
                poss_values.append(poss)
                tiers_used.add(tms.data_tier)
        
        if len(poss_values) < definition.min_matches:
            return None
        
        avg = safe_avg(poss_values, 50.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg - 50) * 0.4))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(poss_values), definition.min_matches * 2, 0.0)),
            matches_used=len(poss_values),
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_pressing(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Defensive intensity - quality adjusted."""
        total_actions = 0.0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for tms, quality_ctx in matches_with_quality:
            tackles = tms.stats.get('tackles', 0) or 0
            interceptions = tms.stats.get('interceptions', 0) or 0
            actions = tackles + interceptions
            
            if quality_ctx:
                adjusted = adjust_stat_for_quality(actions, quality_ctx.quality_multiplier)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted = actions
            
            total_actions += adjusted
            tiers_used.add(tms.data_tier)
        
        num_matches = len(matches_with_quality)
        avg = safe_divide(total_actions, num_matches, 25.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg - 25) * 0.4))
        
        avg_quality = safe_divide(total_quality, quality_count, REFERENCE_RATING) if quality_count > 0 else REFERENCE_RATING
        quality_factor = min(1.0, avg_quality / REFERENCE_RATING)
        base_confidence = min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0))
        confidence = base_confidence * (0.8 + 0.2 * quality_factor)
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_physicality(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Duel dominance - quality adjusted."""
        total_duels = 0.0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for tms, quality_ctx in matches_with_quality:
            duels = (tms.stats.get('duels_won', 0) or 0) + (tms.stats.get('aerial_duels_won', 0) or 0)
            
            if quality_ctx:
                adjusted = adjust_stat_for_quality(duels, quality_ctx.quality_multiplier)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted = duels
            
            total_duels += adjusted
            tiers_used.add(tms.data_tier)
        
        num_matches = len(matches_with_quality)
        avg = safe_divide(total_duels, num_matches, 30.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg - 30) * 0.3))
        
        avg_quality = safe_divide(total_quality, quality_count, REFERENCE_RATING) if quality_count > 0 else REFERENCE_RATING
        quality_factor = min(1.0, avg_quality / REFERENCE_RATING)
        base_confidence = min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0))
        confidence = base_confidence * (0.8 + 0.2 * quality_factor)
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_discipline(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Card and foul rate - no quality adjustment."""
        total_cost = 0.0
        tiers_used = set()
        
        for tms, _ in matches_with_quality:
            cost = (
                (tms.stats.get('yellow_cards', 0) or 0) + 
                (tms.stats.get('red_cards', 0) or 0) * 3 + 
                (tms.stats.get('fouls', 0) or 0) * 0.1
            )
            total_cost += cost
            tiers_used.add(tms.data_tier)
        
        num_matches = len(matches_with_quality)
        avg_cost = safe_divide(total_cost, num_matches, 3.0)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (3 - avg_cost) * 3))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0)),
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_set_piece_attack(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Attacking set piece threat - quality adjusted."""
        total_xg = 0.0
        valid_count = 0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for tms, quality_ctx in matches_with_quality:
            xg_sp = tms.stats.get('xg_set_play')
            if xg_sp is not None:
                if quality_ctx:
                    adjusted = adjust_stat_for_quality(xg_sp, quality_ctx.quality_multiplier)
                    total_quality += quality_ctx.combined_quality
                    quality_count += 1
                else:
                    adjusted = xg_sp
                total_xg += adjusted
                valid_count += 1
                tiers_used.add(tms.data_tier)
        
        if valid_count < definition.min_matches:
            return None
        
        avg = safe_divide(total_xg, valid_count, 0.3)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (avg - 0.3) * 20))
        
        avg_quality = safe_divide(total_quality, quality_count, REFERENCE_RATING) if quality_count > 0 else REFERENCE_RATING
        quality_factor = min(1.0, avg_quality / REFERENCE_RATING)
        base_confidence = min(1.0, safe_divide(valid_count, definition.min_matches * 2, 0.0))
        confidence = base_confidence * (0.8 + 0.2 * quality_factor)
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=valid_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_set_piece_defense(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Defending set pieces - inverse quality adjusted."""
        total_xg = 0.0
        valid_count = 0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for tms, quality_ctx in matches_with_quality:
            xg_sp = tms.stats.get('xg_set_play_conceded')
            if xg_sp is not None:
                if quality_ctx:
                    inverse_mult = safe_divide(REFERENCE_RATING, quality_ctx.combined_quality, 1.0)
                    adjusted = adjust_stat_for_quality(xg_sp, inverse_mult)
                    total_quality += quality_ctx.combined_quality
                    quality_count += 1
                else:
                    adjusted = xg_sp
                total_xg += adjusted
                valid_count += 1
                tiers_used.add(tms.data_tier)
        
        if valid_count < definition.min_matches:
            return None
        
        avg = safe_divide(total_xg, valid_count, 0.3)
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (0.3 - avg) * 20))
        
        avg_quality = safe_divide(total_quality, quality_count, REFERENCE_RATING) if quality_count > 0 else REFERENCE_RATING
        quality_factor = min(1.0, avg_quality / REFERENCE_RATING)
        base_confidence = min(1.0, safe_divide(valid_count, definition.min_matches * 2, 0.0))
        confidence = base_confidence * (0.8 + 0.2 * quality_factor)
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=valid_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_consistency(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Result variance - no quality adjustment (measures consistency itself)."""
        points = []
        tiers_used = set()
        
        for tms, _ in matches_with_quality:
            result = tms.stats.get('result', 'D')
            pts = {'W': 3, 'D': 1, 'L': 0}.get(result, 1)
            points.append(pts)
            tiers_used.add(tms.data_tier)
        
        if len(points) < definition.min_matches:
            return None
        
        avg = safe_avg(points, 1.0)
        variance = safe_divide(sum((p - avg) ** 2 for p in points), len(points), 0.0)
        std_dev = math.sqrt(variance)
        
        value = max(ATTR_MIN, min(ATTR_MAX, 10 + (1.0 - std_dev) * 6))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(len(points), definition.min_matches * 2, 0.0)),
            matches_used=len(points),
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_big_game(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Results vs higher-rated opponents - inherently quality-aware."""
        big_game_results = []
        regular_results = []
        tiers_used = set()
        
        for tms, quality_ctx in matches_with_quality:
            result = tms.stats.get('result', 'D')
            points = {'W': 3, 'D': 1, 'L': 0}.get(result, 1)
            tiers_used.add(tms.data_tier)
            
            if quality_ctx and quality_ctx.opponent_rating > quality_ctx.own_rating + 50:
                big_game_results.append(points)
            else:
                regular_results.append(points)
        
        num_matches = len(matches_with_quality)
        
        if len(big_game_results) < 5:
            all_points = big_game_results + regular_results
            ppg = safe_avg(all_points, 1.0)
            value = 10 + (ppg - 1.0) * 5
            confidence = 0.5
        else:
            big_game_ppg = safe_avg(big_game_results, 1.0)
            regular_ppg = safe_avg(regular_results, 1.0) if regular_results else big_game_ppg
            improvement = big_game_ppg - regular_ppg
            value = 10 + (big_game_ppg - 1.0) * 5 + improvement * 2
            confidence = min(1.0, safe_divide(len(big_game_results), 15, 0.0))
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )
    
    def _calc_squad_depth(
        self, 
        matches_with_quality: List[Tuple[TeamMatchStats, Optional[MatchQualityContext]]],
        definition: AttributeDefinition
    ) -> Optional[AttributeScore]:
        """Performance with rotation - no quality adjustment."""
        if len(matches_with_quality) < 10:
            return None
        
        all_players = set()
        all_starters = set()
        player_appearances: Dict[int, int] = defaultdict(int)
        tiers_used = set()
        
        for tms, _ in matches_with_quality:
            all_players.update(tms.lineup_player_ids)
            all_starters.update(tms.starter_player_ids)
            for pid in tms.lineup_player_ids:
                player_appearances[pid] += 1
            tiers_used.add(tms.data_tier)
        
        num_matches = len(matches_with_quality)
        regulars = sum(1 for pid, apps in player_appearances.items() if apps >= num_matches * 0.2)
        
        value = 8 + (regulars - 11) * 1.5
        if len(all_starters) > 15:
            value += (len(all_starters) - 15) * 0.3
        
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        return AttributeScore(
            value=value,
            confidence=min(1.0, safe_divide(num_matches, definition.min_matches * 2, 0.0)),
            matches_used=num_matches,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class Per90Calculator(BaseAttributeCalculator):
    """Calculate per-90-minute statistics with quality adjustment."""
    
    def calculate(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup, 
        current_match_count: int,
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        total_stat = 0.0
        total_minutes = 0
        total_quality = 0.0
        quality_count = 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            if ms.minutes_played <= 0:
                continue
            
            stat_value = self.get_stat_value(ms, definition.primary_stats[0].stat_key, 0.0)
            if stat_value is None:
                continue
            
            # Get quality context
            if league_ratings:
                quality_ctx = self.get_quality_context(ms, league_ratings)
                adjusted_stat = adjust_stat_for_quality(stat_value, quality_ctx.quality_multiplier)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted_stat = stat_value
            
            total_stat += adjusted_stat
            total_minutes += ms.minutes_played
            tiers_used.add(ms.data_tier)
        
        if total_minutes == 0:
            return None
        
        per_90_value = (total_stat / total_minutes) * 90
        
        # Get percentile with quality awareness
        avg_quality_ctx = None
        if quality_count > 0:
            avg_quality = total_quality / quality_count
            avg_quality_ctx = MatchQualityContext(
                match_id=0,
                opponent_rating=avg_quality,
                league_rating=avg_quality,
                own_rating=REFERENCE_RATING
            )
        
        percentile = self.percentile_calc.get_percentile(
            definition.primary_stats[0].stat_key,
            per_90_value,
            position_group,
            avg_quality_ctx
        )
        
        match_count = len([ms for ms in match_stats if ms.data_tier.value >= definition.min_tier.value])
        
        return AttributeScore(
            value=self.percentile_calc.percentile_to_attribute(percentile),
            confidence=min(1.0, safe_divide(match_count, definition.min_matches * 2, 0.0)),
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class PercentageCalculator(BaseAttributeCalculator):
    """Calculate success rate percentages with quality adjustment."""
    
    def calculate(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup, 
        current_match_count: int,
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        total_success = 0.0
        total_attempts = 0.0
        total_quality = 0.0
        quality_count = 0
        match_count = 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            success = self.get_stat_value(ms, definition.primary_stats[0].stat_key, 0.0)
            attempts = success  # Default: success count is the attempts
            
            if len(definition.primary_stats) > 1:
                attempts_val = self.get_stat_value(ms, definition.primary_stats[1].stat_key)
                if attempts_val and attempts_val > 0:
                    attempts = attempts_val
            
            if success is None or attempts <= 0:
                continue
            
            # Quality adjustment - success rate against better teams is more impressive
            if league_ratings:
                quality_ctx = self.get_quality_context(ms, league_ratings)
                # Boost success value based on opponent quality
                adjusted_success = adjust_stat_for_quality(success, quality_ctx.quality_multiplier)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted_success = success
            
            total_success += adjusted_success
            total_attempts += attempts
            match_count += 1
            tiers_used.add(ms.data_tier)
        
        if total_attempts == 0:
            return None
        
        success_rate = total_success / total_attempts
        
        # Get percentile with quality awareness
        avg_quality_ctx = None
        if quality_count > 0:
            avg_quality = total_quality / quality_count
            avg_quality_ctx = MatchQualityContext(
                match_id=0,
                opponent_rating=avg_quality,
                league_rating=avg_quality,
                own_rating=REFERENCE_RATING
            )
        
        percentile = self.percentile_calc.get_percentile(
            f"{definition.code}_rate",
            success_rate,
            position_group,
            avg_quality_ctx
        )
        
        return AttributeScore(
            value=self.percentile_calc.percentile_to_attribute(percentile),
            confidence=min(1.0, safe_divide(match_count, definition.min_matches * 2, 0.0)),
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )

class CompositeCalculator(BaseAttributeCalculator):
    """Calculate attributes from multiple weighted stats - with quality adjustment."""
    
    def calculate(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup, 
        current_match_count: int,
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Optional['AttributeScore']:
        
        if not self.has_required_data(definition, match_stats):
            return None
        
        # Accumulate stats with quality context
        stat_totals: Dict[str, float] = defaultdict(float)
        stat_counts: Dict[str, int] = defaultdict(int)
        quality_sum = 0.0
        quality_count = 0
        total_minutes = 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            if ms.minutes_played <= 0:
                continue
            
            # Get quality context using LIVE ratings
            quality_ctx = self.get_quality_context(ms, league_ratings)
            quality_sum += quality_ctx.combined_quality
            quality_count += 1
            
            tiers_used.add(ms.data_tier)
            total_minutes += ms.minutes_played
            
            # Process each stat
            for stat_req in definition.primary_stats + definition.secondary_stats:
                raw_value = self.get_stat_value(ms, stat_req.stat_key)
                if raw_value is None:
                    continue
                
                # Convert to per-90
                per_90 = (raw_value / ms.minutes_played) * 90
                
                # Apply quality adjustment
                adjusted = adjust_stat_for_quality(
                    per_90, 
                    quality_ctx.quality_multiplier
                )
                
                stat_totals[stat_req.stat_key] += adjusted
                stat_counts[stat_req.stat_key] += 1
        
        if quality_count == 0:
            return None
        
        # Calculate average quality faced
        avg_quality = quality_sum / quality_count
        avg_quality_ctx = MatchQualityContext(
            match_id=0,
            opponent_rating=avg_quality * 0.6 / 0.6,  # Reverse the weighting
            league_rating=avg_quality * 0.4 / 0.4,
            own_rating=REFERENCE_RATING
        )
        
        # Calculate percentile for each stat component
        component_scores = []
        total_weight = 0.0
        
        for stat_req in definition.primary_stats:
            if stat_counts[stat_req.stat_key] == 0:
                if not stat_req.is_optional:
                    return None
                continue
            
            avg_value = stat_totals[stat_req.stat_key] / stat_counts[stat_req.stat_key]
            
            # Get percentile with quality awareness
            percentile = self.percentile_calc.get_percentile(
                stat_req.stat_key,
                avg_value,
                position_group,
                avg_quality_ctx
            )
            
            weight = definition.stat_weights.get(stat_req.stat_key, stat_req.weight)
            component_scores.append((percentile, weight))
            total_weight += weight
        
        if not component_scores or total_weight == 0:
            return None
        
        # Combine percentiles
        final_percentile = sum(p * w for p, w in component_scores) / total_weight
        
        if not definition.higher_is_better:
            final_percentile = 100 - final_percentile
        
        # Convert to attribute score
        value = self.percentile_calc.percentile_to_attribute(final_percentile)
        value = max(ATTR_MIN, min(ATTR_MAX, value))
        
        # Confidence factors in quality of data
        match_count = max(stat_counts.values()) if stat_counts else 0
        base_confidence = min(1.0, match_count / (definition.min_matches * 2))
        
        # Higher quality opposition = slightly higher confidence
        quality_confidence = min(1.0, avg_quality / REFERENCE_RATING)
        confidence = base_confidence * (0.8 + 0.2 * quality_confidence)
        
        from models.entities import AttributeScore
        return AttributeScore(
            value=value,
            confidence=confidence,
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class DifferentialCalculator(BaseAttributeCalculator):
    """Calculate based on expected vs actual with quality adjustment."""
    
    def calculate(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup, 
        current_match_count: int,
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        if len(definition.primary_stats) < 2:
            return None
        
        total_actual = 0.0
        total_expected = 0.0
        total_quality = 0.0
        quality_count = 0
        match_count = 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            actual = self.get_stat_value(ms, definition.primary_stats[0].stat_key, 0.0)
            expected = self.get_stat_value(ms, definition.primary_stats[1].stat_key)
            
            if expected is None:
                continue
            
            # Quality adjustment - overperforming xG vs top teams is more impressive
            if league_ratings:
                quality_ctx = self.get_quality_context(ms, league_ratings)
                # Adjust both actual and expected, but actual gets quality boost
                adjusted_actual = adjust_stat_for_quality(actual, quality_ctx.quality_multiplier)
                # Expected stays as-is (it's already quality-adjusted by the xG model)
                total_quality += quality_ctx.combined_quality
                quality_count += 1
            else:
                adjusted_actual = actual
            
            total_actual += adjusted_actual
            total_expected += expected
            match_count += 1
            tiers_used.add(ms.data_tier)
        
        if match_count == 0 or total_expected == 0:
            return None
        
        # Calculate differential as percentage over/under expected
        differential = safe_divide(total_actual - total_expected, total_expected, 0.0)
        
        # Get percentile with quality awareness
        avg_quality_ctx = None
        if quality_count > 0:
            avg_quality = total_quality / quality_count
            avg_quality_ctx = MatchQualityContext(
                match_id=0,
                opponent_rating=avg_quality,
                league_rating=avg_quality,
                own_rating=REFERENCE_RATING
            )
        
        percentile = self.percentile_calc.get_percentile(
            f"{definition.code}_diff",
            differential,
            position_group,
            avg_quality_ctx
        )
        
        return AttributeScore(
            value=self.percentile_calc.percentile_to_attribute(percentile),
            confidence=min(1.0, safe_divide(match_count, definition.min_matches * 2, 0.0)),
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class AggregateCalculator(BaseAttributeCalculator):
    """Calculate aggregate statistics (no quality adjustment needed)."""
    
    def calculate(
        self, 
        definition: AttributeDefinition, 
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup, 
        current_match_count: int,
        league_ratings: Optional[Dict[int, float]] = None  # Accept but don't use
    ) -> Optional[AttributeScore]:
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
        player_registry: Dict[int, 'Player'],
        league_ratings: Optional[Dict[int, float]] = None
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
                
                # Get quality context if league ratings available
                quality_ctx = None
                if league_ratings:
                    quality_ctx = MatchQualityContext.from_player_match_stats(ms, league_ratings)
                
                for stat_key, value in ms.stats.items():
                    if value is not None and value > 0:
                        per_90 = (value / ms.minutes_played) * 90 if ms.minutes_played > 0 else value
                        self.percentile_calc.record_stat(
                            stat_key, 
                            per_90, 
                            ms.position_group or player.position_group,
                            quality_ctx  # Pass quality context
                        )
        
        self._bootstrapped = True
        print(f"Bootstrapped percentiles with {len(historical_stats)} players")

    def determine_primary_league(
        self,  # ← FIX: Added 'self' to make it a method
        match_stats_history: List[PlayerMatchStats],
        recency_weight: bool = True,
        lookback_matches: int = 50
    ) -> Optional[int]:
        """
        Determine player's primary league with recency weighting.
        
        Args:
            match_stats_history: Player's match history (should be sorted by date desc)
            recency_weight: If True, weight recent matches more heavily
            lookback_matches: Only consider this many recent matches
        
        Returns:
            league_id of the primary league, or None
        """
        if not match_stats_history:
            return None
        
        # Sort by date (most recent first) if we have dates
        sorted_history = sorted(
            match_stats_history, 
            key=lambda ms: ms.match_date or datetime.min,
            reverse=True
        )
        
        # Only consider recent matches
        recent_matches = sorted_history[:lookback_matches]
        
        league_scores: Dict[int, float] = {}
        
        for i, ms in enumerate(recent_matches):
            if not ms.league_id:
                continue
            
            if recency_weight:
                # Exponential decay: most recent match = 1.0, older = less
                weight = math.exp(-0.05 * i)  # ~0.6 at match 10, ~0.37 at match 20
            else:
                weight = 1.0
            
            league_scores[ms.league_id] = league_scores.get(ms.league_id, 0) + weight
        
        if not league_scores:
            return None
        
        return max(league_scores, key=league_scores.get)

    def calculate_player_attributes(
        self, 
        player: 'Player', 
        match_stats_history: List[PlayerMatchStats],
        league_ratings: Optional[Dict[int, float]] = None
    ) -> Dict[str, AttributeScore]:
        """Calculate all attributes for a player with quality adjustment."""
        results = {}
        
        if player.is_goalkeeper:
            from config.attribute_definitions import GOALKEEPER_ATTRIBUTES
            attributes = GOALKEEPER_ATTRIBUTES
        else:
            attributes = ALL_PLAYER_ATTRIBUTES
        
        # Use recency-weighted league determination
        primary_league_id = self.determine_primary_league(  # ← Now works correctly
            match_stats_history,
            recency_weight=True,
            lookback_matches=50
        )
        
        for attr_code, definition in attributes.items():
            if definition.applicable_positions and player.position_group not in definition.applicable_positions:
                continue
            if definition.exclude_positions and player.position_group in definition.exclude_positions:
                continue
            
            calculator = self.calculators.get(definition.calculation_method)
            if calculator:
                # Pass league_ratings to calculator for quality adjustment
                score = calculator.calculate(
                    definition, 
                    match_stats_history, 
                    player.position_group, 
                    player.matches_played,
                    league_ratings=league_ratings  # ← Pass to calculator
                )
                if score:
                    # Apply league normalization if available
                    if league_ratings and primary_league_id:
                        score.value = self.normalize_by_league_strength(
                            score.value,
                            primary_league_id,
                            league_ratings
                        )
                        score.value = max(ATTR_MIN, min(ATTR_MAX, score.value))
                    
                    results[attr_code] = score
        
        return results
    
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
        
        # Soft adjustment: ~10% bonus/penalty per 200 rating points difference
        adjustment = (multiplier - 1) * 0.5
        return raw_value * (1 + adjustment)
    
    def calculate_coach_attributes(
        self, 
        coach: Any, 
        match_stats_history: List[CoachMatchStats],
        league_ratings: Optional[Dict[int, float]] = None  # ← NEW
    ) -> Dict[str, 'AttributeScore']:
        return self.coach_calculator.calculate_all_attributes(
            coach, 
            match_stats_history,
            league_ratings  # ← Pass through
        )
    
    def calculate_team_attributes(
        self, 
        team: Any, 
        match_stats_history: List[TeamMatchStats],
        league_ratings: Optional[Dict[int, float]] = None  # ← NEW
    ) -> Dict[str, 'AttributeScore']:
        return self.team_calculator.calculate_all_attributes(
            team, 
            match_stats_history,
            league_ratings  # ← Pass through
        )
    
    def update_population_stats(
        self, 
        all_players: Dict[int, Player], 
        all_match_stats: Dict[int, List[PlayerMatchStats]]
    ):
        for player_id, player in all_players.items():
            if player_id not in all_match_stats:
                continue
            
            for ms in all_match_stats[player_id]:
                for stat_key, value in ms.stats.items():
                    if value is not None:
                        per_90 = safe_divide(value * 90, ms.minutes_played, value) if ms.minutes_played > 0 else value
                        self.percentile_calc.record_stat(stat_key, per_90, player.position_group)