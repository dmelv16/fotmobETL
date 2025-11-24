"""
Attribute calculation engine for converting raw stats to 0-20 scale.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import math
from collections import defaultdict

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


@dataclass
class StatAccumulator:
    """Accumulates stats across matches with decay weighting."""
    values: List[Tuple[float, float, int]] = field(default_factory=list)  # (value, weight, match_id)
    
    def add(self, value: float, weight: float, match_id: int):
        self.values.append((value, weight, match_id))
    
    @property
    def weighted_sum(self) -> float:
        if not self.values:
            return 0.0
        return sum(v * w for v, w, _ in self.values)
    
    @property
    def total_weight(self) -> float:
        return sum(w for _, w, _ in self.values)
    
    @property
    def weighted_average(self) -> Optional[float]:
        if self.total_weight == 0:
            return None
        return self.weighted_sum / self.total_weight
    
    @property
    def match_count(self) -> int:
        return len(self.values)
    
    @property
    def raw_values(self) -> List[float]:
        return [v for v, _, _ in self.values]


@dataclass
class PlayerMatchStats:
    """Stats for a single player in a single match."""
    match_id: int
    player_id: int
    team_id: int
    data_tier: DataTier
    minutes_played: int = 0
    
    # All stats as a flat dict
    stats: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    position_id: Optional[int] = None
    is_starter: bool = False
    rating: Optional[float] = None


class PercentileCalculator:
    """Calculates percentiles within a population for normalization."""
    
    def __init__(self):
        # stat_key -> position_group -> list of values
        self.stat_distributions: Dict[str, Dict[PositionGroup, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
    
    def record_stat(self, stat_key: str, value: float, position_group: PositionGroup):
        """Record a stat value for percentile calculation."""
        self.stat_distributions[stat_key][position_group].append(value)
    
    def get_percentile(
        self, 
        stat_key: str, 
        value: float, 
        position_group: PositionGroup
    ) -> float:
        """Get percentile rank (0-100) for a value within its position group."""
        values = self.stat_distributions[stat_key].get(position_group, [])
        
        if not values:
            return 50.0  # Default to median if no data
        
        # Count values below this one
        below = sum(1 for v in values if v < value)
        equal = sum(1 for v in values if v == value)
        
        # Percentile = (below + 0.5 * equal) / total * 100
        percentile = (below + 0.5 * equal) / len(values) * 100
        
        return percentile
    
    def percentile_to_attribute(self, percentile: float) -> float:
        """Convert percentile to 0-20 attribute scale."""
        # Find the two closest percentile keys
        percentiles = sorted(PERCENTILE_TO_ATTR.keys())
        
        for i, p in enumerate(percentiles):
            if percentile <= p:
                if i == 0:
                    return float(PERCENTILE_TO_ATTR[p])
                
                # Interpolate between adjacent values
                lower_p = percentiles[i - 1]
                upper_p = p
                lower_attr = PERCENTILE_TO_ATTR[lower_p]
                upper_attr = PERCENTILE_TO_ATTR[upper_p]
                
                ratio = (percentile - lower_p) / (upper_p - lower_p)
                return lower_attr + ratio * (upper_attr - lower_attr)
        
        return float(ATTR_MAX)


class DecayCalculator:
    """Calculates time-based decay weights for historical data."""
    
    @staticmethod
    def get_match_decay(matches_ago: int) -> float:
        """Get decay weight based on how many matches ago."""
        for threshold, weight in sorted(MATCH_DECAY_WEIGHTS.items()):
            if matches_ago <= threshold:
                return weight
        return 0.1  # Minimum weight for very old matches
    
    @staticmethod
    def get_recency_weight(
        match_date: datetime, 
        current_date: datetime,
        half_life_days: int = 180
    ) -> float:
        """Exponential decay based on calendar time."""
        days_ago = (current_date - match_date).days
        return math.exp(-0.693 * days_ago / half_life_days)


# =============================================================================
# BASE CALCULATOR
# =============================================================================

class BaseAttributeCalculator(ABC):
    """Base class for attribute calculators."""
    
    def __init__(self, percentile_calc: PercentileCalculator):
        self.percentile_calc = percentile_calc
    
    @abstractmethod
    def calculate(
        self,
        definition: AttributeDefinition,
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup,
        current_match_count: int
    ) -> Optional[AttributeScore]:
        """Calculate attribute value from match stats."""
        pass
    
    def has_required_data(
        self,
        definition: AttributeDefinition,
        match_stats: List[PlayerMatchStats]
    ) -> bool:
        """Check if we have minimum required data for this attribute."""
        # Check minimum matches
        if len(match_stats) < definition.min_matches:
            return False
        
        # Check minimum data tier
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
        """Safely get a stat value from match stats."""
        return match_stat.stats.get(stat_key, default)


# =============================================================================
# SPECIFIC CALCULATORS
# =============================================================================

class Per90Calculator(BaseAttributeCalculator):
    """Calculate per-90-minute statistics."""
    
    def calculate(
        self,
        definition: AttributeDefinition,
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup,
        current_match_count: int
    ) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        accumulator = StatAccumulator()
        tiers_used = set()
        
        for i, ms in enumerate(match_stats):
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            # Get primary stat
            primary_stat = definition.primary_stats[0]
            value = self.get_stat_value(ms, primary_stat.stat_key, 0.0)
            
            if value is None:
                continue
            
            # Convert to per 90
            minutes = max(ms.minutes_played, 1)
            per_90 = (value / minutes) * 90
            
            # Apply decay
            matches_ago = current_match_count - i
            weight = DecayCalculator.get_match_decay(matches_ago)
            
            accumulator.add(per_90, weight, ms.match_id)
            tiers_used.add(ms.data_tier)
        
        if accumulator.match_count == 0:
            return None
        
        # Get weighted average
        avg_per_90 = accumulator.weighted_average
        
        # Convert to percentile then attribute
        percentile = self.percentile_calc.get_percentile(
            definition.primary_stats[0].stat_key,
            avg_per_90,
            position_group
        )
        attr_value = self.percentile_calc.percentile_to_attribute(percentile)
        
        # Calculate confidence based on data quality
        confidence = min(1.0, accumulator.match_count / (definition.min_matches * 2))
        
        return AttributeScore(
            value=attr_value,
            confidence=confidence,
            matches_used=accumulator.match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class PercentageCalculator(BaseAttributeCalculator):
    """Calculate success rate percentages."""
    
    def calculate(
        self,
        definition: AttributeDefinition,
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup,
        current_match_count: int
    ) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        total_success = 0.0
        total_attempts = 0.0
        match_count = 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            # Get success and total stats
            success_stat = definition.primary_stats[0]
            success = self.get_stat_value(ms, success_stat.stat_key, 0.0)
            
            # Try to get total (might be optional)
            attempts = success  # Default: assume success count is available
            if len(definition.primary_stats) > 1:
                total_stat = definition.primary_stats[1]
                attempts_val = self.get_stat_value(ms, total_stat.stat_key)
                if attempts_val is not None and attempts_val > 0:
                    attempts = attempts_val
            
            if success is not None and attempts > 0:
                total_success += success
                total_attempts += attempts
                match_count += 1
                tiers_used.add(ms.data_tier)
        
        if total_attempts == 0:
            return None
        
        success_rate = total_success / total_attempts
        
        # Convert to percentile
        percentile = self.percentile_calc.get_percentile(
            f"{definition.code}_rate",
            success_rate,
            position_group
        )
        attr_value = self.percentile_calc.percentile_to_attribute(percentile)
        
        confidence = min(1.0, match_count / (definition.min_matches * 2))
        
        return AttributeScore(
            value=attr_value,
            confidence=confidence,
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class CompositeCalculator(BaseAttributeCalculator):
    """Calculate attributes from multiple weighted stats."""
    
    def calculate(
        self,
        definition: AttributeDefinition,
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup,
        current_match_count: int
    ) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        # Accumulate each stat separately
        stat_accumulators: Dict[str, StatAccumulator] = {}
        for stat_req in definition.primary_stats + definition.secondary_stats:
            stat_accumulators[stat_req.stat_key] = StatAccumulator()
        
        tiers_used = set()
        
        for i, ms in enumerate(match_stats):
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            matches_ago = current_match_count - i
            weight = DecayCalculator.get_match_decay(matches_ago)
            
            for stat_req in definition.primary_stats + definition.secondary_stats:
                value = self.get_stat_value(ms, stat_req.stat_key)
                if value is not None:
                    # Normalize to per 90 if applicable
                    if ms.minutes_played > 0:
                        value = (value / ms.minutes_played) * 90
                    stat_accumulators[stat_req.stat_key].add(value, weight, ms.match_id)
            
            tiers_used.add(ms.data_tier)
        
        # Calculate component scores
        component_scores = []
        total_weight = 0.0
        
        for stat_req in definition.primary_stats:
            acc = stat_accumulators[stat_req.stat_key]
            if acc.match_count == 0:
                if not stat_req.is_optional:
                    return None  # Required stat missing
                continue
            
            avg = acc.weighted_average
            percentile = self.percentile_calc.get_percentile(
                stat_req.stat_key, avg, position_group
            )
            
            weight = definition.stat_weights.get(stat_req.stat_key, stat_req.weight)
            component_scores.append((percentile, weight))
            total_weight += weight
        
        if not component_scores or total_weight == 0:
            return None
        
        # Weighted average of percentiles
        final_percentile = sum(p * w for p, w in component_scores) / total_weight
        
        # Adjust for higher_is_better
        if not definition.higher_is_better:
            final_percentile = 100 - final_percentile
        
        attr_value = self.percentile_calc.percentile_to_attribute(final_percentile)
        
        match_count = max(acc.match_count for acc in stat_accumulators.values())
        confidence = min(1.0, match_count / (definition.min_matches * 2))
        
        return AttributeScore(
            value=attr_value,
            confidence=confidence,
            matches_used=match_count,
            last_updated=datetime.now(),
            data_tiers_used=list(tiers_used)
        )


class DifferentialCalculator(BaseAttributeCalculator):
    """Calculate based on expected vs actual (e.g., goals vs xG)."""
    
    def calculate(
        self,
        definition: AttributeDefinition,
        match_stats: List[PlayerMatchStats],
        position_group: PositionGroup,
        current_match_count: int
    ) -> Optional[AttributeScore]:
        if not self.has_required_data(definition, match_stats):
            return None
        
        total_actual = 0.0
        total_expected = 0.0
        match_count = 0
        tiers_used = set()
        
        for ms in match_stats:
            if ms.data_tier.value < definition.min_tier.value:
                continue
            
            # Get actual (first stat) and expected (second stat)
            actual = self.get_stat_value(ms, definition.primary_stats[0].stat_key, 0.0)
            expected_stat = definition.primary_stats[1] if len(definition.primary_stats) > 1 else None
            
            if expected_stat:
                expected = self.get_stat_value(ms, expected_stat.stat_key)
                if expected is None:
                    # If expected is missing, can't calculate differential
                    continue
            else:
                continue
            
            total_actual += actual
            total_expected += expected
            match_count += 1
            tiers_used.add(ms.data_tier)
        
        if match_count == 0 or total_expected == 0:
            return None
        
        # Calculate over/under performance
        differential = (total_actual - total_expected) / total_expected
        
        # Convert to percentile (differential around 0)
        percentile = self.percentile_calc.get_percentile(
            f"{definition.code}_diff",
            differential,
            position_group
        )
        attr_value = self.percentile_calc.percentile_to_attribute(percentile)
        
        confidence = min(1.0, match_count / (definition.min_matches * 2))
        
        return AttributeScore(
            value=attr_value,
            confidence=confidence,
            matches_used=match_count,
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
        
        # Initialize calculators
        self.calculators = {
            CalculationMethod.PER_90: Per90Calculator(self.percentile_calc),
            CalculationMethod.PERCENTAGE: PercentageCalculator(self.percentile_calc),
            CalculationMethod.COMPOSITE: CompositeCalculator(self.percentile_calc),
            CalculationMethod.DIFFERENTIAL: DifferentialCalculator(self.percentile_calc),
        }
    
    def calculate_player_attributes(
        self,
        player: Player,
        match_stats_history: List[PlayerMatchStats]
    ) -> Dict[str, AttributeScore]:
        """Calculate all applicable attributes for a player."""
        results = {}
        
        # Determine which attribute set to use
        if player.is_goalkeeper:
            from config.attribute_definitions import GOALKEEPER_ATTRIBUTES
            attributes = GOALKEEPER_ATTRIBUTES
        else:
            attributes = ALL_PLAYER_ATTRIBUTES
        
        for attr_code, definition in attributes.items():
            # Check position applicability
            if definition.applicable_positions:
                if player.position_group not in definition.applicable_positions:
                    continue
            if definition.exclude_positions:
                if player.position_group in definition.exclude_positions:
                    continue
            
            # Get appropriate calculator
            calculator = self.calculators.get(definition.calculation_method)
            if not calculator:
                continue
            
            # Calculate
            score = calculator.calculate(
                definition,
                match_stats_history,
                player.position_group,
                player.matches_played
            )
            
            if score:
                results[attr_code] = score
        
        return results
    
    def update_population_stats(
        self,
        all_players: Dict[int, Player],
        all_match_stats: Dict[int, List[PlayerMatchStats]]
    ):
        """
        Update percentile distributions from population.
        Should be called periodically to recalibrate.
        """
        for player_id, player in all_players.items():
            if player_id not in all_match_stats:
                continue
            
            for ms in all_match_stats[player_id]:
                for stat_key, value in ms.stats.items():
                    if value is not None:
                        # Normalize to per 90
                        if ms.minutes_played > 0:
                            per_90 = (value / ms.minutes_played) * 90
                        else:
                            per_90 = value
                        
                        self.percentile_calc.record_stat(
                            stat_key, 
                            per_90, 
                            player.position_group
                        )