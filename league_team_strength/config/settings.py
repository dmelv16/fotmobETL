"""
System-wide configuration settings for strength calculation.
"""

# Update frequencies (in days)
LEAGUE_STRENGTH_UPDATE_INTERVAL = 30  # Recalculate monthly
TEAM_STRENGTH_UPDATE_INTERVAL = 7     # Recalculate weekly during season

# Elo rating parameters
ELO_K_FACTOR_LEAGUE = 20
ELO_K_FACTOR_CUP = 15
ELO_K_FACTOR_EUROPEAN = 25
ELO_HOME_ADVANTAGE = 100
ELO_BASE_RATING = 1500

# League strength component weights
LEAGUE_STRENGTH_WEIGHTS = {
    'transfer_matrix': 0.40,
    'european_results': 0.35,
    'network_inference': 0.15,
    'historical_consistency': 0.10
}

# Team strength component weights
TEAM_STRENGTH_WEIGHTS = {
    'elo_rating': 0.35,
    'xg_performance': 0.25,
    'squad_quality': 0.30,
    'coaching_effect': 0.10
}

# Transfer analysis parameters
MIN_TRANSFERS_FOR_LEAGUE_PAIR = 5  # Minimum transfers needed to estimate gap
TRANSFER_RECENCY_WEIGHT = 0.7      # Weight recent transfers more (exponential decay)
MAX_AGE_CHANGE_FOR_TRANSFER = 3    # Max years age change to include in analysis

# European competition weights
CHAMPIONS_LEAGUE_WEIGHT = 1.0
EUROPA_LEAGUE_WEIGHT = 0.8
CONFERENCE_LEAGUE_WEIGHT = 0.6

# Data quality thresholds
MIN_MATCHES_FOR_TEAM_RATING = 10   # Minimum matches before team rating is reliable
MIN_MATCHES_FOR_SQUAD_QUALITY = 5   # Minimum matches for squad quality estimate

# Historical tracking
SEASONS_TO_TRACK = 12  # Track back to 2012/13
TEMPORAL_DECAY_RATE = 0.15  # How much to discount older seasons

# Output settings
STRENGTH_SCALE_MIN = 0
STRENGTH_SCALE_MAX = 100
STRENGTH_DECIMALS = 1