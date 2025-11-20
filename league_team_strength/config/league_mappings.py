"""
League classifications and hierarchies.
This maps your FotMob league_ids to tier systems.
"""

# UEFA Top 5 Leagues
TIER_1_LEAGUES = {
    47: {'name': 'Premier League', 'country': 'England', 'base_strength': 95},
    87: {'name': 'La Liga', 'country': 'Spain', 'base_strength': 93},
    54: {'name': 'Bundesliga', 'country': 'Germany', 'base_strength': 90},
    55: {'name': 'Serie A', 'country': 'Italy', 'base_strength': 88},
    53: {'name': 'Ligue 1', 'country': 'France', 'base_strength': 84}
}

# Major European Leagues (Tier 2)
TIER_2_LEAGUES = {
    57: {'name': 'Eredivisie', 'country': 'Netherlands', 'base_strength': 68},
    63: {'name': 'Primeira Liga', 'country': 'Portugal', 'base_strength': 66},
    59: {'name': 'Belgian Pro League', 'country': 'Belgium', 'base_strength': 61},
    60: {'name': 'Austrian Bundesliga', 'country': 'Austria', 'base_strength': 56},
    56: {'name': 'Scottish Premiership', 'country': 'Scotland', 'base_strength': 49}
}

# Second divisions of top leagues (Tier 2.5)
TIER_2_5_LEAGUES = {
    48: {'name': 'Championship', 'country': 'England', 'base_strength': 60},
    89: {'name': 'La Liga 2', 'country': 'Spain', 'base_strength': 58},
    76: {'name': 'Bundesliga 2', 'country': 'Germany', 'base_strength': 57},
    56: {'name': 'Serie B', 'country': 'Italy', 'base_strength': 52},
    96: {'name': 'Ligue 2', 'country': 'France', 'base_strength': 50}
}

# Third divisions and lower (Tier 3)
TIER_3_LEAGUES = {
    # Add your Serie C, League One, etc. with league_ids
    # Format: league_id: {'name': 'League Name', 'country': 'Country', 'base_strength': XX}
}

# European Competitions
EUROPEAN_COMPETITIONS = {
    42: {'name': 'Champions League', 'type': 'european', 'weight': 1.0},
    73: {'name': 'Europa League', 'type': 'european', 'weight': 0.8},
    848: {'name': 'Conference League', 'type': 'european', 'weight': 0.6}
}

# Domestic Cups (lower weight for strength calculation)
DOMESTIC_CUPS = {
    # Add your domestic cup league_ids
    # These are weighted less heavily in team strength calculations
}

def get_league_tier(league_id: int) -> int:
    """Get the tier classification for a league."""
    if league_id in TIER_1_LEAGUES:
        return 1
    elif league_id in TIER_2_LEAGUES:
        return 2
    elif league_id in TIER_2_5_LEAGUES:
        return 2.5
    elif league_id in TIER_3_LEAGUES:
        return 3
    elif league_id in EUROPEAN_COMPETITIONS:
        return 0  # Special tier for European competitions
    else:
        return 4  # Unknown/lower leagues

def get_league_info(league_id: int) -> dict:
    """Get league information."""
    for tier_dict in [TIER_1_LEAGUES, TIER_2_LEAGUES, TIER_2_5_LEAGUES, 
                      TIER_3_LEAGUES, EUROPEAN_COMPETITIONS]:
        if league_id in tier_dict:
            return {**tier_dict[league_id], 'tier': get_league_tier(league_id)}
    return {'name': 'Unknown', 'country': 'Unknown', 'tier': 4, 'base_strength': 30}

def is_european_competition(league_id: int) -> bool:
    """Check if league is a European competition."""
    return league_id in EUROPEAN_COMPETITIONS