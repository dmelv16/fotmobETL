"""
Coach, Team, and League attribute definitions.
Updated to use side-neutral stat keys - home/away resolution happens in calculators.
"""
from dataclasses import dataclass, field
from typing import List, Dict
from .attribute_definitions import (
    AttributeDefinition, AttributeCategory, 
    CalculationMethod, StatRequirement
)
from .constants import DataTier


# =============================================================================
# COACH ATTRIBUTES
# =============================================================================

COACH_ATTRIBUTES: Dict[str, AttributeDefinition] = {
    
    'tactical_attack': AttributeDefinition(
        name="Attacking Tactics",
        code="ATT",
        category=AttributeCategory.COACH,
        description="Offensive output vs league average",
        min_tier=DataTier.TEAM_STATS,
        min_matches=15,
        calculation_method=CalculationMethod.DIFFERENTIAL,
        primary_stats=[
            # Side-neutral keys - calculator resolves based on is_home_team
            StatRequirement('xg', 'match_stats_summary'),
            StatRequirement('team_score', 'match_details'),
        ],
        secondary_stats=[
            StatRequirement('total_shots', 'match_stats_summary', is_optional=True),
            StatRequirement('big_chances', 'match_stats_summary', is_optional=True),
        ],
    ),
    
    'tactical_defense': AttributeDefinition(
        name="Defensive Tactics",
        code="DEF",
        category=AttributeCategory.COACH,
        description="Defensive solidity vs league average",
        min_tier=DataTier.TEAM_STATS,
        min_matches=15,
        calculation_method=CalculationMethod.DIFFERENTIAL,
        primary_stats=[
            StatRequirement('xg_conceded', 'match_stats_summary'),  # Opponent's xG
            StatRequirement('team_score_conceded', 'match_details'),
        ],
        higher_is_better=False,  # Lower goals/xG conceded = better
    ),
    
    'adaptability': AttributeDefinition(
        name="Adaptability",
        code="ADP",
        category=AttributeCategory.COACH,
        description="Points gained when trailing",
        min_tier=DataTier.MINIMAL,
        min_matches=20,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('time_minute', 'match_events'),
            StatRequirement('score_before', 'match_events'),  # Resolved by side
            StatRequirement('score_after', 'match_events'),
            StatRequirement('team_score', 'match_details'),
            StatRequirement('team_score_conceded', 'match_details'),
        ],
    ),
    
    'game_management': AttributeDefinition(
        name="Game Management",
        code="GMG",
        category=AttributeCategory.COACH,
        description="Late game results (75+ min)",
        min_tier=DataTier.MINIMAL,
        min_matches=20,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('time_minute', 'match_events'),
            StatRequirement('event_type', 'match_events'),
            StatRequirement('is_home_team', 'match_events'),  # To filter events
        ],
    ),
    
    'player_development': AttributeDefinition(
        name="Player Development",
        code="PDV",
        category=AttributeCategory.COACH,
        description="Player rating improvements under management",
        min_tier=DataTier.FULL,
        min_matches=30,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('rating', 'player_stats'),
            StatRequirement('player_id', 'match_lineup_players'),
        ],
    ),
    
    'youth_development': AttributeDefinition(
        name="Youth Development",
        code="YDV",
        category=AttributeCategory.COACH,
        description="U23 player progression",
        min_tier=DataTier.FULL,
        min_matches=30,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('age', 'match_lineup_players'),
            StatRequirement('rating', 'player_stats'),
        ],
    ),
    
    'discipline': AttributeDefinition(
        name="Discipline",
        code="DIS",
        category=AttributeCategory.COACH,
        description="Card and foul management",
        min_tier=DataTier.TEAM_STATS,
        min_matches=15,
        calculation_method=CalculationMethod.DIFFERENTIAL,
        primary_stats=[
            StatRequirement('yellow_cards', 'match_stats_summary'),
            StatRequirement('red_cards', 'match_stats_summary'),
            StatRequirement('fouls', 'match_stats_summary'),
        ],
        higher_is_better=False,
    ),
    
    'squad_rotation': AttributeDefinition(
        name="Squad Rotation",
        code="ROT",
        category=AttributeCategory.COACH,
        description="Lineup variation management",
        min_tier=DataTier.BASIC,
        min_matches=20,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('player_id', 'match_lineup_players'),
            StatRequirement('is_starter', 'match_lineup_players'),
        ],
    ),
    
    'substitution_impact': AttributeDefinition(
        name="Substitution Impact",
        code="SUB",
        category=AttributeCategory.COACH,
        description="Effect of substitutions on results",
        min_tier=DataTier.MINIMAL,
        min_matches=25,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('substitution_time', 'match_substitutions'),
            StatRequirement('time_minute', 'match_events'),
            StatRequirement('event_type', 'match_events'),
        ],
    ),
    
    'formation_mastery': AttributeDefinition(
        name="Formation Mastery",
        code="FOR",
        category=AttributeCategory.COACH,
        description="Success rate by formation",
        min_tier=DataTier.BASIC,
        min_matches=20,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('formation', 'match_team_stats'),
            StatRequirement('team_score', 'match_details'),
            StatRequirement('team_score_conceded', 'match_details'),
        ],
    ),
}


# =============================================================================
# TEAM ATTRIBUTES
# =============================================================================

TEAM_ATTRIBUTES: Dict[str, AttributeDefinition] = {
    
    'team_attack': AttributeDefinition(
        name="Attack",
        code="ATK",
        category=AttributeCategory.TEAM,
        description="Goals and chances created",
        min_tier=DataTier.TEAM_STATS,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('xg', 'match_stats_summary'),
            StatRequirement('team_score', 'match_details'),
            StatRequirement('total_shots', 'match_stats_summary', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('big_chances', 'match_stats_summary', is_optional=True),
        ],
    ),
    
    'team_defense': AttributeDefinition(
        name="Defense",
        code="DEF",
        category=AttributeCategory.TEAM,
        description="Goals and chances conceded",
        min_tier=DataTier.TEAM_STATS,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('xg_conceded', 'match_stats_summary'),
            StatRequirement('team_score_conceded', 'match_details'),
        ],
        higher_is_better=False,
    ),
    
    'team_possession': AttributeDefinition(
        name="Possession",
        code="POS",
        category=AttributeCategory.TEAM,
        description="Ball retention style",
        min_tier=DataTier.TEAM_STATS,
        min_matches=10,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[
            StatRequirement('possession', 'match_stats_summary'),
            StatRequirement('accurate_passes', 'match_stats_summary', is_optional=True),
        ],
    ),
    
    'team_pressing': AttributeDefinition(
        name="Pressing",
        code="PRS",
        category=AttributeCategory.TEAM,
        description="Defensive intensity",
        min_tier=DataTier.TEAM_STATS,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('tackles', 'match_stats_summary'),
            StatRequirement('interceptions', 'match_stats_summary', is_optional=True),
        ],
    ),
    
    'team_physicality': AttributeDefinition(
        name="Physicality",
        code="PHY",
        category=AttributeCategory.TEAM,
        description="Duel dominance",
        min_tier=DataTier.TEAM_STATS,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('duels_won', 'match_stats_summary'),
            StatRequirement('aerial_duels_won', 'match_stats_summary', is_optional=True),
        ],
    ),
    
    'team_discipline': AttributeDefinition(
        name="Discipline",
        code="DIS",
        category=AttributeCategory.TEAM,
        description="Card and foul rate",
        min_tier=DataTier.TEAM_STATS,
        min_matches=10,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[
            StatRequirement('yellow_cards', 'match_stats_summary'),
            StatRequirement('red_cards', 'match_stats_summary'),
            StatRequirement('fouls', 'match_stats_summary'),
        ],
        higher_is_better=False,
    ),
    
    'team_set_piece_attack': AttributeDefinition(
        name="Set Pieces (Attack)",
        code="SPA",
        category=AttributeCategory.TEAM,
        description="Attacking set piece threat",
        min_tier=DataTier.TEAM_STATS,
        min_matches=15,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('xg_set_play', 'match_stats_summary'),
            StatRequirement('corners', 'match_stats_summary', is_optional=True),
        ],
    ),
    
    'team_set_piece_defense': AttributeDefinition(
        name="Set Pieces (Defense)",
        code="SPD",
        category=AttributeCategory.TEAM,
        description="Defending set pieces",
        min_tier=DataTier.TEAM_STATS,
        min_matches=15,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('xg_set_play_conceded', 'match_stats_summary'),
        ],
        higher_is_better=False,
    ),
    
    'team_big_game': AttributeDefinition(
        name="Big Game Performance",
        code="BIG",
        category=AttributeCategory.TEAM,
        description="Results vs higher-rated opponents",
        min_tier=DataTier.TEAM_STATS,
        min_matches=15,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('rating', 'match_team_stats'),
        ],
    ),
    
    'team_consistency': AttributeDefinition(
        name="Consistency",
        code="CON",
        category=AttributeCategory.TEAM,
        description="Result variance",
        min_tier=DataTier.MINIMAL,
        min_matches=20,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('team_score', 'match_details'),
            StatRequirement('team_score_conceded', 'match_details'),
        ],
    ),
    
    'team_depth': AttributeDefinition(
        name="Squad Depth",
        code="DEP",
        category=AttributeCategory.TEAM,
        description="Performance with rotation",
        min_tier=DataTier.BASIC,
        min_matches=20,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('player_id', 'match_lineup_players'),
            StatRequirement('is_starter', 'match_lineup_players'),
        ],
    ),
}


# =============================================================================
# LEAGUE ATTRIBUTES (Unchanged - leagues aggregate both sides)
# =============================================================================

LEAGUE_ATTRIBUTES: Dict[str, AttributeDefinition] = {
    
    'league_strength': AttributeDefinition(
        name="Overall Strength",
        code="STR",
        category=AttributeCategory.LEAGUE,
        description="Competitive quality",
        min_tier=DataTier.MINIMAL,
        min_matches=100,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[],
    ),
    
    'league_depth': AttributeDefinition(
        name="Competitive Depth",
        code="DEP",
        category=AttributeCategory.LEAGUE,
        description="Parity across teams",
        min_tier=DataTier.MINIMAL,
        min_matches=100,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[],
    ),
    
    'league_attacking': AttributeDefinition(
        name="Attacking Style",
        code="ATK",
        category=AttributeCategory.LEAGUE,
        description="Goals per match",
        min_tier=DataTier.TEAM_STATS,
        min_matches=100,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[
            # Leagues use both sides (total goals in league)
            StatRequirement('home_team_score', 'match_details'),
            StatRequirement('away_team_score', 'match_details'),
            StatRequirement('home_xg', 'match_stats_summary', is_optional=True),
            StatRequirement('away_xg', 'match_stats_summary', is_optional=True),
        ],
    ),
    
    'league_physicality': AttributeDefinition(
        name="Physicality",
        code="PHY",
        category=AttributeCategory.LEAGUE,
        description="Average physical intensity",
        min_tier=DataTier.TEAM_STATS,
        min_matches=100,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[
            StatRequirement('home_fouls', 'match_stats_summary'),
            StatRequirement('away_fouls', 'match_stats_summary'),
            StatRequirement('home_duels_won', 'match_stats_summary', is_optional=True),
            StatRequirement('away_duels_won', 'match_stats_summary', is_optional=True),
        ],
    ),
    
    'league_technical': AttributeDefinition(
        name="Technical Quality",
        code="TEC",
        category=AttributeCategory.LEAGUE,
        description="Passing and possession style",
        min_tier=DataTier.TEAM_STATS,
        min_matches=100,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[
            StatRequirement('home_pass_accuracy_pct', 'match_stats_summary'),
            StatRequirement('away_pass_accuracy_pct', 'match_stats_summary'),
            StatRequirement('home_possession', 'match_stats_summary'),
        ],
    ),
    
    'league_youth': AttributeDefinition(
        name="Youth Integration",
        code="YTH",
        category=AttributeCategory.LEAGUE,
        description="Minutes for U23 players",
        min_tier=DataTier.BASIC,
        min_matches=100,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[
            StatRequirement('age', 'match_lineup_players'),
            StatRequirement('is_starter', 'match_lineup_players'),
        ],
    ),
    
    'league_competitiveness': AttributeDefinition(
        name="Competitiveness",
        code="CMP",
        category=AttributeCategory.LEAGUE,
        description="Frequency of close matches",
        min_tier=DataTier.MINIMAL,
        min_matches=100,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[
            StatRequirement('home_team_score', 'match_details'),
            StatRequirement('away_team_score', 'match_details'),
        ],
    ),
}