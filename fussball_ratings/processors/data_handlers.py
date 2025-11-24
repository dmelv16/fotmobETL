"""
Data tier classification and null value handling.
Critical for managing data quality across different matches and sources.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set
from enum import Enum

from config.constants import (
    DataTier, TIER_CONFIGS, 
    ZERO_DEFAULT_STATS, RATE_STATS, CONTEXT_REQUIRED_STATS
)


# =============================================================================
# DATA TIER PROCESSOR
# =============================================================================

@dataclass
class MatchDataAvailability:
    """Tracks what data is available for a specific match."""
    match_id: int
    has_match_details: bool = False
    has_lineups: bool = False
    has_stats: bool = False
    has_momentum: bool = False
    has_player_stats: bool = False
    has_shotmap: bool = False
    has_events: bool = False
    
    # Additional tracking for stat granularity
    available_stat_keys: Set[str] = None  # Which player_stats keys exist
    has_team_xg: bool = False
    has_player_xg: bool = False
    
    def __post_init__(self):
        if self.available_stat_keys is None:
            self.available_stat_keys = set()


class DataTierProcessor:
    """Determines the data tier for a match based on extraction_status."""
    
    @staticmethod
    def classify_tier(availability: MatchDataAvailability) -> DataTier:
        """
        Classify match into a data tier.
        
        Tier Logic:
        - COMPLETE (5): has_player_stats AND has_shotmap
        - FULL (4): has_player_stats
        - TEAM_STATS (3): has_stats (team level)
        - BASIC (2): has_lineups
        - MINIMAL (1): has_match_details AND has_events only
        """
        if availability.has_player_stats and availability.has_shotmap:
            return DataTier.COMPLETE
        elif availability.has_player_stats:
            return DataTier.FULL
        elif availability.has_stats:
            return DataTier.TEAM_STATS
        elif availability.has_lineups:
            return DataTier.BASIC
        else:
            return DataTier.MINIMAL
    
    @staticmethod
    def from_extraction_status(row: Dict[str, Any]) -> MatchDataAvailability:
        """Create availability from extraction_status table row."""
        return MatchDataAvailability(
            match_id=row['match_id'],
            has_match_details=bool(row.get('has_match_details', 0)),
            has_lineups=bool(row.get('has_lineups', 0)),
            has_stats=bool(row.get('has_stats', 0)),
            has_momentum=bool(row.get('has_momentum', 0)),
            has_player_stats=bool(row.get('has_player_stats', 0)),
            has_shotmap=bool(row.get('has_shotmap', 0)),
            has_events=bool(row.get('has_events', 0)),
        )
    
    @staticmethod
    def get_tier_config(tier: DataTier):
        """Get configuration for a given tier."""
        return TIER_CONFIGS[tier]
    
    @staticmethod
    def can_calculate_attribute(tier: DataTier, required_tier: DataTier) -> bool:
        """Check if current tier meets minimum requirement."""
        return tier.value >= required_tier.value


# =============================================================================
# NULL VALUE HANDLER
# =============================================================================

class NullBehavior(Enum):
    """How to interpret a null value."""
    ZERO = "zero"           # NULL means the stat is 0
    EXCLUDE = "exclude"     # NULL means data is missing, exclude from calc
    DEFAULT = "default"     # Use a provided default value


@dataclass
class NullHandlingRule:
    """Rule for handling null values in a specific context."""
    behavior: NullBehavior
    default_value: Optional[float] = None
    

class NullHandler:
    """
    Handles null value interpretation based on table and column context.
    
    CRITICAL DISTINCTION:
    - player_stats: NULL or missing row = 0 (event didn't happen)
    - match_stats_summary: NULL = data not tracked (exclude from calc)
    - match_lineup_players: NULL rating = not available (exclude)
    """
    
    # Table-level default behaviors
    TABLE_DEFAULTS = {
        'player_stats': NullBehavior.ZERO,
        'match_events': NullBehavior.ZERO,
        'match_shotmap': NullBehavior.ZERO,
        'match_stats_summary': NullBehavior.EXCLUDE,
        'match_team_stats': NullBehavior.EXCLUDE,
        'match_lineup_players': NullBehavior.EXCLUDE,
        'match_facts': NullBehavior.EXCLUDE,
    }
    
    # Specific column overrides (table.column -> behavior)
    COLUMN_OVERRIDES = {
        # player_stats columns that should be excluded if null
        'player_stats.rating': NullBehavior.EXCLUDE,
        'player_stats.expected_goals': NullBehavior.EXCLUDE,
        'player_stats.expected_assists': NullBehavior.EXCLUDE,
        'player_stats.market_value': NullBehavior.EXCLUDE,
        'player_stats.fantasy_score': NullBehavior.EXCLUDE,
        
        # match_lineup_players
        'match_lineup_players.rating': NullBehavior.EXCLUDE,
        'match_lineup_players.market_value': NullBehavior.EXCLUDE,
        'match_lineup_players.fantasy_score': NullBehavior.EXCLUDE,
        
        # match_stats_summary - some have meaningful zeros
        # (but most should be excluded if null)
        
        # match_events - assist_player null = no assist
        'match_events.assist_player': NullBehavior.ZERO,
    }
    
    @classmethod
    def get_rule(cls, table: str, column: str) -> NullHandlingRule:
        """Get the null handling rule for a specific table.column."""
        
        # Check for specific column override
        key = f"{table}.{column}"
        if key in cls.COLUMN_OVERRIDES:
            return NullHandlingRule(behavior=cls.COLUMN_OVERRIDES[key])
        
        # Fall back to table default
        if table in cls.TABLE_DEFAULTS:
            return NullHandlingRule(behavior=cls.TABLE_DEFAULTS[table])
        
        # Ultimate fallback: exclude
        return NullHandlingRule(behavior=NullBehavior.EXCLUDE)
    
    @classmethod
    def process_value(
        cls, 
        value: Any, 
        table: str, 
        column: str,
        default_override: Optional[float] = None
    ) -> Optional[float]:
        """
        Process a potentially null value according to rules.
        
        Returns:
            float: The processed value
            None: If value should be excluded from calculations
        """
        rule = cls.get_rule(table, column)
        
        # If value is not null, return it
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        # Handle null based on behavior
        if rule.behavior == NullBehavior.ZERO:
            return 0.0
        elif rule.behavior == NullBehavior.DEFAULT:
            return default_override if default_override is not None else rule.default_value
        else:  # EXCLUDE
            return None
    
    @classmethod
    def process_stat_row(
        cls, 
        row: Dict[str, Any], 
        table: str
    ) -> Dict[str, Optional[float]]:
        """Process all numeric columns in a row."""
        processed = {}
        for column, value in row.items():
            processed[column] = cls.process_value(value, table, column)
        return processed


# =============================================================================
# PLAYER STATS AGGREGATOR
# =============================================================================

class PlayerStatsAggregator:
    """
    Aggregates player stats from the player_stats table.
    Handles the pivoting from long format (stat_key, stat_value) to wide format.
    """
    
    @staticmethod
    def aggregate_player_match_stats(
        stats_rows: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Convert player_stats rows for a single player/match to a dict.
        
        Input: List of rows like:
            {'stat_key': 'goals', 'stat_value': 2, 'stat_total': 2}
            {'stat_key': 'assists', 'stat_value': 1, 'stat_total': 1}
        
        Output: {'goals': 2.0, 'assists': 1.0, ...}
        """
        result = {}
        
        for row in stats_rows:
            stat_key = row.get('stat_key')
            if not stat_key:
                continue
            
            # Use stat_value, fall back to stat_total if needed
            value = row.get('stat_value')
            if value is None:
                value = row.get('stat_total')
            
            # Apply null handling
            processed_value = NullHandler.process_value(
                value, 'player_stats', stat_key
            )
            
            # For player_stats, missing stat_key = 0
            # But if the key exists with null value, we've already handled it
            if processed_value is not None:
                result[stat_key] = processed_value
        
        return result
    
    @staticmethod
    def fill_missing_stats(
        stats: Dict[str, float],
        required_keys: List[str]
    ) -> Dict[str, float]:
        """
        Fill in missing stat keys with 0 for player_stats.
        
        In player_stats, if a stat_key doesn't exist for a player,
        it means that stat is 0 (e.g., no goals = 0 goals).
        """
        filled = stats.copy()
        
        for key in required_keys:
            if key not in filled:
                # Only fill with 0 if it's a zero-default stat
                if key in ZERO_DEFAULT_STATS:
                    filled[key] = 0.0
                # Rate stats and context-required stats remain missing
        
        return filled
    
    @staticmethod
    def get_available_stat_keys(stats_rows: List[Dict[str, Any]]) -> Set[str]:
        """Get the set of stat_keys present in the data."""
        return {row.get('stat_key') for row in stats_rows if row.get('stat_key')}


# =============================================================================
# MATCH DATA VALIDATOR
# =============================================================================

class MatchDataValidator:
    """Validates match data completeness and quality."""
    
    @staticmethod
    def validate_match_details(details: Dict[str, Any]) -> List[str]:
        """Check for required fields in match_details."""
        issues = []
        
        required = ['match_id', 'home_team_id', 'away_team_id', 
                    'home_team_score', 'away_team_score']
        for field in required:
            if details.get(field) is None:
                issues.append(f"Missing required field: {field}")
        
        # Check for valid scores
        if details.get('finished') != 1:
            issues.append("Match not marked as finished")
        
        return issues
    
    @staticmethod
    def validate_lineups(
        lineups: List[Dict[str, Any]], 
        expected_home_id: int,
        expected_away_id: int
    ) -> List[str]:
        """Validate lineup data completeness."""
        issues = []
        
        home_starters = [p for p in lineups 
                        if p.get('team_id') == expected_home_id and p.get('is_starter')]
        away_starters = [p for p in lineups 
                        if p.get('team_id') == expected_away_id and p.get('is_starter')]
        
        if len(home_starters) < 11:
            issues.append(f"Home team has only {len(home_starters)} starters")
        if len(away_starters) < 11:
            issues.append(f"Away team has only {len(away_starters)} starters")
        
        return issues
    
    @staticmethod
    def validate_player_stats(
        stats: List[Dict[str, Any]],
        lineup_player_ids: Set[int]
    ) -> List[str]:
        """Validate player stats coverage."""
        issues = []
        
        stats_player_ids = {row.get('player_id') for row in stats if row.get('player_id')}
        
        missing = lineup_player_ids - stats_player_ids
        if missing:
            issues.append(f"{len(missing)} players in lineup missing from player_stats")
        
        return issues


# =============================================================================
# STAT KEY AVAILABILITY TRACKER
# =============================================================================

class StatKeyTracker:
    """
    Tracks which stat_keys are available for each match.
    Used to determine which attributes can be calculated.
    """
    
    def __init__(self):
        # match_id -> set of available stat_keys
        self.match_stat_keys: Dict[int, Set[str]] = {}
        
        # Global frequency of each stat_key (for analysis)
        self.stat_key_frequency: Dict[str, int] = {}
    
    def record_match_stats(self, match_id: int, stat_keys: Set[str]):
        """Record available stat_keys for a match."""
        self.match_stat_keys[match_id] = stat_keys
        for key in stat_keys:
            self.stat_key_frequency[key] = self.stat_key_frequency.get(key, 0) + 1
    
    def has_stat_for_match(self, match_id: int, stat_key: str) -> bool:
        """Check if a specific stat is available for a match."""
        if match_id not in self.match_stat_keys:
            return False
        return stat_key in self.match_stat_keys[match_id]
    
    def get_matches_with_stat(self, stat_key: str) -> Set[int]:
        """Get all matches that have a specific stat."""
        return {
            mid for mid, keys in self.match_stat_keys.items() 
            if stat_key in keys
        }
    
    def get_stat_coverage(self, stat_key: str) -> float:
        """Get percentage of matches with this stat."""
        if not self.match_stat_keys:
            return 0.0
        return self.stat_key_frequency.get(stat_key, 0) / len(self.match_stat_keys)
    
    def get_common_stats(self, min_coverage: float = 0.5) -> List[str]:
        """Get stat_keys available in at least min_coverage fraction of matches."""
        total_matches = len(self.match_stat_keys)
        if total_matches == 0:
            return []
        
        return [
            key for key, count in self.stat_key_frequency.items()
            if count / total_matches >= min_coverage
        ]