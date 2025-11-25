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


# Granular position IDs (position_id) to group mapping
# Derived from usual_position_id relationships in match_lineup_players
POSITION_TO_GROUP = {
    # Goalkeepers (usual_position_id = 0)
    11: PositionGroup.GOALKEEPER,
    112: PositionGroup.GOALKEEPER,
    113: PositionGroup.GOALKEEPER,
    117: PositionGroup.GOALKEEPER,
    118: PositionGroup.GOALKEEPER,
    
    # Defenders (usual_position_id = 1)
    1: PositionGroup.DEFENDER,
    24: PositionGroup.DEFENDER,
    27: PositionGroup.DEFENDER,
    29: PositionGroup.DEFENDER,
    31: PositionGroup.DEFENDER,
    32: PositionGroup.DEFENDER,
    34: PositionGroup.DEFENDER,
    35: PositionGroup.DEFENDER,
    36: PositionGroup.DEFENDER,
    37: PositionGroup.DEFENDER,
    38: PositionGroup.DEFENDER,
    39: PositionGroup.DEFENDER,
    41: PositionGroup.DEFENDER,
    49: PositionGroup.DEFENDER,
    51: PositionGroup.DEFENDER,
    59: PositionGroup.DEFENDER,
    64: PositionGroup.DEFENDER,
    67: PositionGroup.DEFENDER,
    71: PositionGroup.DEFENDER,
    73: PositionGroup.DEFENDER,
    81: PositionGroup.DEFENDER,
    86: PositionGroup.DEFENDER,
    88: PositionGroup.DEFENDER,
    105: PositionGroup.DEFENDER,
    107: PositionGroup.DEFENDER,
    
    # Midfielders (usual_position_id = 2)
    2: PositionGroup.MIDFIELDER,
    23: PositionGroup.MIDFIELDER,
    25: PositionGroup.MIDFIELDER,
    26: PositionGroup.MIDFIELDER,
    33: PositionGroup.MIDFIELDER,
    43: PositionGroup.MIDFIELDER,
    52: PositionGroup.MIDFIELDER,
    53: PositionGroup.MIDFIELDER,
    54: PositionGroup.MIDFIELDER,
    55: PositionGroup.MIDFIELDER,
    56: PositionGroup.MIDFIELDER,
    57: PositionGroup.MIDFIELDER,
    62: PositionGroup.MIDFIELDER,
    63: PositionGroup.MIDFIELDER,
    65: PositionGroup.MIDFIELDER,
    66: PositionGroup.MIDFIELDER,
    68: PositionGroup.MIDFIELDER,
    72: PositionGroup.MIDFIELDER,
    74: PositionGroup.MIDFIELDER,
    75: PositionGroup.MIDFIELDER,
    76: PositionGroup.MIDFIELDER,
    77: PositionGroup.MIDFIELDER,
    78: PositionGroup.MIDFIELDER,
    79: PositionGroup.MIDFIELDER,
    82: PositionGroup.MIDFIELDER,
    83: PositionGroup.MIDFIELDER,
    84: PositionGroup.MIDFIELDER,
    85: PositionGroup.MIDFIELDER,
    93: PositionGroup.MIDFIELDER,
    94: PositionGroup.MIDFIELDER,
    96: PositionGroup.MIDFIELDER,
    99: PositionGroup.MIDFIELDER,
    103: PositionGroup.MIDFIELDER,
    
    # Forwards (usual_position_id = 3)
    3: PositionGroup.FORWARD,
    4: PositionGroup.FORWARD,
    45: PositionGroup.FORWARD,
    47: PositionGroup.FORWARD,
    58: PositionGroup.FORWARD,
    87: PositionGroup.FORWARD,
    89: PositionGroup.FORWARD,
    91: PositionGroup.FORWARD,
    92: PositionGroup.FORWARD,
    95: PositionGroup.FORWARD,
    97: PositionGroup.FORWARD,
    98: PositionGroup.FORWARD,
    102: PositionGroup.FORWARD,
    104: PositionGroup.FORWARD,
    106: PositionGroup.FORWARD,
    108: PositionGroup.FORWARD,
    109: PositionGroup.FORWARD,
    114: PositionGroup.FORWARD,
    115: PositionGroup.FORWARD,
    116: PositionGroup.FORWARD,
    1000: PositionGroup.FORWARD,  # Substitute/special case
}


def get_position_group(position_id: int, usual_position_id: int = None) -> PositionGroup:
    """
    Get position group from position_id, with fallback to usual_position_id.
    
    Use this function instead of direct dict lookup to handle unknown position_ids.
    """
    # Try granular position_id first
    if position_id in POSITION_TO_GROUP:
        return POSITION_TO_GROUP[position_id]
    
    # Fallback to usual_position_id if provided
    if usual_position_id is not None and usual_position_id in (0, 1, 2, 3):
        return PositionGroup(usual_position_id)
    
    # Ultimate fallback: midfielder (most common)
    return PositionGroup.MIDFIELDER


# Position group names for display
POSITION_GROUP_NAMES = {
    PositionGroup.GOALKEEPER: 'GK',
    PositionGroup.DEFENDER: 'DEF',
    PositionGroup.MIDFIELDER: 'MID',
    PositionGroup.FORWARD: 'FWD',
}


# =============================================================================
# COMPETITION WEIGHT CONSTANTS
# =============================================================================

# Base competition type multipliers
COMPETITION_MULTIPLIERS = {
    'Continental': 2.0,      # Base for continental
    'Cup': 1.25,             # Domestic cups
    'League': 1.0,           # Default for leagues
}

# Specific continental competition adjustments
CONTINENTAL_ADJUSTMENTS = {
    'Champions League': 2.0,
    'Europa League': 1.6,
    'Conference League': 1.2,
    'Copa Libertadores': 1.6,
    'Copa Sudamericana': 1.3,
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