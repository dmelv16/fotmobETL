"""
Core constants for the rating and attribute system.
All magic numbers and thresholds defined here for easy tuning.
"""
from enum import IntEnum
from dataclasses import dataclass


# =============================================================================
# DATA TIER DEFINITIONS
# =============================================================================

class DataTier(IntEnum):
    """Data availability tiers based on extraction_status flags."""
    MINIMAL = 1      # match_details + events only
    BASIC = 2        # + lineups
    TEAM_STATS = 3   # + stats (team level)
    FULL = 4         # + player_stats
    COMPLETE = 5     # + shotmap


@dataclass(frozen=True)
class TierConfig:
    """Configuration for each data tier."""
    k_factor_multiplier: float
    can_update_individual_attrs: bool
    can_update_advanced_attrs: bool
    can_update_shot_attrs: bool


TIER_CONFIGS = {
    DataTier.MINIMAL: TierConfig(0.4, False, False, False),
    DataTier.BASIC: TierConfig(0.6, False, False, False),
    DataTier.TEAM_STATS: TierConfig(0.8, False, False, False),
    DataTier.FULL: TierConfig(1.0, True, True, False),
    DataTier.COMPLETE: TierConfig(1.0, True, True, True),
}


# =============================================================================
# RATING SYSTEM CONSTANTS
# =============================================================================

# Base starting ratings
BASE_RATING = 1000
RATING_FLOOR = 400
RATING_CEILING = 2200

# K-factors (base values before multipliers)
K_FACTORS = {
    'player': 24,
    'goalkeeper': 20,  # Slightly lower - fewer direct impacts
    'coach': 16,
    'team': 32,
    'league': 8,
}

# New entity initialization multipliers (relative to league rating)
INIT_MULTIPLIERS = {
    'player': 0.85,
    'goalkeeper': 0.85,
    'coach': 0.90,
    'team': 1.0,  # Teams start at base
}

# Rating change caps (per match)
MAX_RATING_CHANGE = 50
MIN_RATING_CHANGE = -50


# =============================================================================
# ATTRIBUTE SYSTEM CONSTANTS
# =============================================================================

# Scale: 0-20 (Football Manager style)
ATTR_MIN = 1
ATTR_MAX = 20
ATTR_DEFAULT = 10  # Starting point for unknown

# Minimum matches required for attribute calculation
MIN_MATCHES_FOR_ATTR = {
    'basic': 3,       # Goals, assists, cards
    'standard': 8,    # Most attributes
    'advanced': 15,   # Consistency, big game, etc.
    'meta': 25,       # Adaptability, versatility
}

# Percentile to attribute score mapping (for 0-20 scale)
PERCENTILE_TO_ATTR = {
    0: 1, 5: 3, 10: 5, 20: 7, 30: 8, 40: 9,
    50: 10, 60: 11, 70: 12, 80: 14, 90: 16,
    95: 17, 98: 18, 99: 19, 100: 20
}


# =============================================================================
# TIME DECAY CONSTANTS
# =============================================================================

# Match recency decay (matches ago -> weight)
MATCH_DECAY_WEIGHTS = {
    10: 1.0,    # Last 10 matches: 100%
    30: 0.85,   # 11-30 matches ago: 85%
    60: 0.60,   # 31-60 matches ago: 60%
    100: 0.35,  # 61-100 matches ago: 35%
    999: 0.15,  # 100+ matches ago: 15%
}

# Season decay (seasons ago -> weight)
SEASON_DECAY_WEIGHTS = {
    0: 1.0,     # Current season
    1: 0.85,    # Last season
    2: 0.65,    # 2 seasons ago
    3: 0.45,    # 3 seasons ago
    4: 0.25,    # 4+ seasons ago
    999: 0.10,
}


# =============================================================================
# POSITION CONSTANTS
# =============================================================================

class PositionGroup(IntEnum):
    """Broad position groupings for normalization."""
    GOALKEEPER = 0
    DEFENDER = 1
    MIDFIELDER = 2
    FORWARD = 3


# Granular position IDs (usual_position_id) to group mapping
# You'll need to populate these based on your actual data
POSITION_TO_GROUP = {
    # Goalkeepers
    1: PositionGroup.GOALKEEPER,
    
    # Defenders
    2: PositionGroup.DEFENDER,   # CB
    3: PositionGroup.DEFENDER,   # LB
    4: PositionGroup.DEFENDER,   # RB
    5: PositionGroup.DEFENDER,   # LWB
    6: PositionGroup.DEFENDER,   # RWB
    
    # Midfielders
    7: PositionGroup.MIDFIELDER,   # CDM
    8: PositionGroup.MIDFIELDER,   # CM
    9: PositionGroup.MIDFIELDER,   # CAM
    10: PositionGroup.MIDFIELDER,  # LM
    11: PositionGroup.MIDFIELDER,  # RM
    
    # Forwards
    12: PositionGroup.FORWARD,   # LW
    13: PositionGroup.FORWARD,   # RW
    14: PositionGroup.FORWARD,   # CF
    15: PositionGroup.FORWARD,   # ST
}

# Position names for display
POSITION_NAMES = {
    1: 'GK', 2: 'CB', 3: 'LB', 4: 'RB', 5: 'LWB', 6: 'RWB',
    7: 'CDM', 8: 'CM', 9: 'CAM', 10: 'LM', 11: 'RM',
    12: 'LW', 13: 'RW', 14: 'CF', 15: 'ST'
}


# =============================================================================
# COMPETITION WEIGHT CONSTANTS
# =============================================================================

# Base competition type multipliers
COMPETITION_MULTIPLIERS = {
    'Continental': 1.5,      # Base for continental
    'Cup': 1.15,             # Domestic cups
    'League': 1.0,           # Default for leagues
}

# Specific continental competition adjustments
CONTINENTAL_ADJUSTMENTS = {
    'Champions League': 1.8,
    'Europa League': 1.4,
    'Conference League': 1.2,
    'Copa Libertadores': 1.6,
    'AFC Champions League': 1.3,
}

# Division level multipliers (applied to League type)
DIVISION_MULTIPLIERS = {
    1: 1.0,
    2: 0.9,
    3: 0.8,
    4: 0.7,
    5: 0.6,
}

# Knockout stage bonus (applied on top of competition multiplier)
KNOCKOUT_BONUS = 1.2


# =============================================================================
# STAT KEY MAPPINGS
# =============================================================================

# player_stats stat_keys that should default to 0 if missing
ZERO_DEFAULT_STATS = {
    'goals', 'assists', 'yellow_cards', 'red_cards',
    'total_shots', 'shots_on_target', 'shots_off_target',
    'accurate_passes', 'key_passes', 'chances_created',
    'dribbles_succeeded', 'dribbles_attempted',
    'tackles', 'interceptions', 'clearances', 'blocks',
    'aerials_won', 'aerials_lost', 'duel_won', 'duel_lost',
    'fouls', 'was_fouled', 'offsides',
    'saves', 'punches', 'catches', 'goals_conceded',
    'error_led_to_goal', 'own_goals', 'penalty_saves',
    'big_chance_missed', 'big_chance_created',
}

# Stats that are percentages/rates (don't default to 0)
RATE_STATS = {
    'pass_accuracy', 'dribble_success_rate', 'aerial_win_rate',
    'tackle_success_rate', 'save_percentage',
}

# Stats that require context (NULL = unknown, not 0)
CONTEXT_REQUIRED_STATS = {
    'expected_goals', 'expected_assists', 'expected_goals_on_target',
    'rating', 'fantasy_score', 'market_value',
}