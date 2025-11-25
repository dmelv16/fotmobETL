"""
Attribute definitions including required stats, calculation methods, and tier requirements.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
from .constants import DataTier, PositionGroup


class AttributeCategory(Enum):
    """Categories of attributes."""
    TECHNICAL = "technical"
    MENTAL = "mental"
    PHYSICAL = "physical"
    GOALKEEPING = "goalkeeping"
    META = "meta"
    COACH = "coach"
    TEAM = "team"
    LEAGUE = "league"


class CalculationMethod(Enum):
    """How the attribute value is derived."""
    PER_90 = "per_90"                    # Stat per 90 minutes
    PERCENTAGE = "percentage"             # Success rate
    INVERSE = "inverse"                   # Lower is better (errors, etc)
    COMPOSITE = "composite"               # Multiple stats combined
    DIFFERENTIAL = "differential"         # Compared to expected
    CONTEXTUAL = "contextual"             # Based on match context
    AGGREGATE = "aggregate"               # Sum/average over period


@dataclass
class StatRequirement:
    """A single stat required for an attribute calculation."""
    stat_key: str
    table: str = "player_stats"          # Which table to pull from
    weight: float = 1.0                   # Weight in composite calcs
    is_optional: bool = False            # Can calculate without it
    default_if_missing: Optional[float] = 0.0  # None = exclude match


@dataclass
class AttributeDefinition:
    """Complete definition of an attribute."""
    name: str
    code: str                             # Short code (e.g., 'FIN' for Finishing)
    category: AttributeCategory
    description: str
    
    # Calculation configuration
    min_tier: DataTier                    # Minimum data tier required
    min_matches: int                      # Minimum matches to calculate
    calculation_method: CalculationMethod
    
    # Required stats
    primary_stats: List[StatRequirement]
    secondary_stats: List[StatRequirement] = field(default_factory=list)
    
    # Position applicability (empty = all positions)
    applicable_positions: List[PositionGroup] = field(default_factory=list)
    exclude_positions: List[PositionGroup] = field(default_factory=list)
    
    # Normalization
    normalize_by_position: bool = True
    higher_is_better: bool = True
    
    # Weights for composite calculations
    stat_weights: Dict[str, float] = field(default_factory=dict)
    
    # Extended description for complex calculations
    description_extended: Optional[str] = None


# =============================================================================
# OUTFIELD PLAYER ATTRIBUTES
# =============================================================================

PLAYER_ATTRIBUTES: Dict[str, AttributeDefinition] = {
    
    # -------------------------------------------------------------------------
    # TECHNICAL ATTRIBUTES
    # -------------------------------------------------------------------------
    
    'finishing': AttributeDefinition(
        name="Finishing",
        code="FIN",
        category=AttributeCategory.TECHNICAL,
        description="Ability to convert chances into goals",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.DIFFERENTIAL,
        primary_stats=[
            StatRequirement('goals', 'player_stats'),
            StatRequirement('expected_goals', 'player_stats', is_optional=True),
            StatRequirement('total_shots', 'player_stats'),
        ],
        secondary_stats=[
            StatRequirement('big_chance_missed_title', 'player_stats', is_optional=True),
            StatRequirement('ShotsOnTarget', 'player_stats', is_optional=True),
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
        stat_weights={'goals_vs_xg': 0.5, 'conversion_rate': 0.3, 'on_target_rate': 0.2}
    ),
    
    'long_shots': AttributeDefinition(
        name="Long Shots",
        code="LSH",
        category=AttributeCategory.TECHNICAL,
        description="Shooting ability from outside the box",
        min_tier=DataTier.COMPLETE,  # Needs shotmap
        min_matches=15,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('expected_goals', 'match_shotmap'),
            StatRequirement('is_from_inside_box', 'match_shotmap'),
            StatRequirement('event_type', 'match_shotmap'),  # Goal/Miss/Saved
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'free_kicks': AttributeDefinition(
        name="Free Kick Taking",
        code="FRK",
        category=AttributeCategory.TECHNICAL,
        description="Dead ball striking ability",
        min_tier=DataTier.COMPLETE,
        min_matches=10,  # Need enough FK attempts
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('situation', 'match_shotmap'),  # Filter: 'FreeKick'
            StatRequirement('expected_goals', 'match_shotmap'),
            StatRequirement('event_type', 'match_shotmap'),
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'crossing': AttributeDefinition(
        name="Crossing",
        code="CRO",
        category=AttributeCategory.TECHNICAL,
        description="Delivery from wide areas",
        min_tier=DataTier.FULL,
        min_matches=8,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('accurate_crosses', 'player_stats'),
            StatRequirement('crosses', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('chances_created', 'player_stats', is_optional=True),
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'dribbling': AttributeDefinition(
        name="Dribbling",
        code="DRI",
        category=AttributeCategory.TECHNICAL,
        description="Ball carrying and beating defenders",
        min_tier=DataTier.FULL,
        min_matches=8,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('dribbles_succeeded', 'player_stats'),
            StatRequirement('dribbles_attempted', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('dispossessed', 'player_stats', is_optional=True),
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
        stat_weights={'success_rate': 0.6, 'volume': 0.4}
    ),
    
    'first_touch': AttributeDefinition(
        name="First Touch",
        code="FTO",
        category=AttributeCategory.TECHNICAL,
        description="Ball control on reception",
        min_tier=DataTier.FULL,
        min_matches=8,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('touches', 'player_stats'),
            StatRequirement('dispossessed', 'player_stats'),
        ],
        secondary_stats=[
            StatRequirement('big_chance_missed_title', 'player_stats', is_optional=True),
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
        higher_is_better=True,  # Composite considers dispossessed inversely
    ),
    
    'passing': AttributeDefinition(
        name="Passing",
        code="PAS",
        category=AttributeCategory.TECHNICAL,
        description="Short and medium range distribution",
        min_tier=DataTier.FULL,
        min_matches=5,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('accurate_passes', 'player_stats'),
            StatRequirement('total_passes', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('pass_accuracy', 'player_stats', is_optional=True),
            StatRequirement('key_passes', 'player_stats', is_optional=True),
        ],
        stat_weights={'accuracy': 0.5, 'volume': 0.3, 'key_passes': 0.2}
    ),
    
    'technique': AttributeDefinition(
        name="Technique",
        code="TEC",
        category=AttributeCategory.TECHNICAL,
        description="Overall technical quality",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('accurate_passes', 'player_stats'),
            StatRequirement('dribbles_succeeded', 'player_stats'),
            StatRequirement('accurate_crosses', 'player_stats', is_optional=True),
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
        stat_weights={'passing': 0.4, 'dribbling': 0.4, 'crossing': 0.2}
    ),
    
    'heading': AttributeDefinition(
        name="Heading",
        code="HEA",
        category=AttributeCategory.TECHNICAL,
        description="Aerial offensive ability",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('aerials_won', 'player_stats'),
            StatRequirement('aerials_lost', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('headed_clearance', 'player_stats', is_optional=True),
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
    ),
    
    # -------------------------------------------------------------------------
    # MENTAL ATTRIBUTES
    # -------------------------------------------------------------------------
    
    'aggression': AttributeDefinition(
        name="Aggression",
        code="AGG",
        category=AttributeCategory.MENTAL,
        description="Physical engagement intensity",
        min_tier=DataTier.FULL,
        min_matches=8,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('fouls', 'player_stats'),
            StatRequirement('duel_won', 'player_stats', is_optional=True),
            StatRequirement('tackles', 'player_stats', is_optional=True),
        ],
        normalize_by_position=True,
    ),
    
    'anticipation': AttributeDefinition(
        name="Anticipation",
        code="ANT",
        category=AttributeCategory.MENTAL,
        description="Reading the game",
        min_tier=DataTier.FULL,
        min_matches=8,
        calculation_method=CalculationMethod.PER_90,
        primary_stats=[
            StatRequirement('interceptions', 'player_stats'),
        ],
        secondary_stats=[
            StatRequirement('recoveries', 'player_stats', is_optional=True),
        ],
    ),
    
    'bravery': AttributeDefinition(
        name="Bravery",
        code="BRA",
        category=AttributeCategory.MENTAL,
        description="Willingness to engage physically",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('duel_won', 'player_stats'),
            StatRequirement('duel_lost', 'player_stats'),
            StatRequirement('tackles', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('shot_blocks', 'player_stats', is_optional=True),
            StatRequirement('clearances', 'player_stats', is_optional=True),
        ],
        stat_weights={'duel_engagement': 0.5, 'tackles': 0.3, 'shot_blocks': 0.2}
    ),
    
    'composure': AttributeDefinition(
        name="Composure",
        code="COM",
        category=AttributeCategory.MENTAL,
        description="Performance under pressure",
        min_tier=DataTier.FULL,
        min_matches=12,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('big_chance_missed_title', 'player_stats'),
            StatRequirement('dispossessed', 'player_stats'),
        ],
        secondary_stats=[
            StatRequirement('error_led_to_goal', 'player_stats', is_optional=True),
            StatRequirement('accurate_passes', 'player_stats', is_optional=True),
        ],
        higher_is_better=False,  # Inverse - fewer errors = better
        exclude_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'concentration': AttributeDefinition(
        name="Concentration",
        code="CON",
        category=AttributeCategory.MENTAL,
        description="Consistency and error avoidance",
        min_tier=DataTier.FULL,
        min_matches=15,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('rating', 'player_stats'),  # Variance analysis
        ],
        secondary_stats=[
            StatRequirement('error_led_to_goal', 'player_stats', is_optional=True),
        ],
        description_extended="Based on rating variance - lower variance = higher concentration"
    ),
    
    'decisions': AttributeDefinition(
        name="Decisions",
        code="DEC",
        category=AttributeCategory.MENTAL,
        description="Quality of choices on the ball",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('accurate_passes', 'player_stats'),
            StatRequirement('dispossessed', 'player_stats'),
            StatRequirement('ShotsOnTarget', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('key_passes', 'player_stats', is_optional=True),
            StatRequirement('passes_into_final_third', 'player_stats', is_optional=True),
        ],
    ),
    
    'flair': AttributeDefinition(
        name="Flair",
        code="FLA",
        category=AttributeCategory.MENTAL,
        description="Creativity and unpredictability",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('dribbles_succeeded', 'player_stats'),
            StatRequirement('was_fouled', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('long_balls_accurate', 'player_stats', is_optional=True),
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'leadership': AttributeDefinition(
        name="Leadership",
        code="LEA",
        category=AttributeCategory.MENTAL,
        description="Team performance impact when playing",
        min_tier=DataTier.BASIC,  # Only needs lineup + results
        min_matches=20,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('is_starter', 'match_lineup_players'),
            StatRequirement('is_captain', 'match_lineup_players'),
        ],
        description_extended="Compares team results when player starts vs doesn't"
    ),
    
    'off_the_ball': AttributeDefinition(
        name="Off The Ball",
        code="OTB",
        category=AttributeCategory.MENTAL,
        description="Movement without possession",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('touches_opp_box', 'player_stats', is_optional=True),
            StatRequirement('chances_created', 'player_stats', is_optional=True),
            StatRequirement('rating', 'player_stats', is_optional=True)
        ],
        secondary_stats=[
            StatRequirement('expected_goals', 'player_stats', is_optional=True),
        ],
        applicable_positions=[PositionGroup.MIDFIELDER, PositionGroup.FORWARD],
    ),
    
    'positioning': AttributeDefinition(
        name="Positioning",
        code="POS",
        category=AttributeCategory.MENTAL,
        description="Defensive positioning",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('interceptions', 'player_stats'),
            StatRequirement('shot_blocks', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('clearances', 'player_stats', is_optional=True),
        ],
        applicable_positions=[PositionGroup.DEFENDER, PositionGroup.MIDFIELDER],
    ),
    
    'teamwork': AttributeDefinition(
        name="Teamwork",
        code="TEA",
        category=AttributeCategory.MENTAL,
        description="Unselfish play",
        min_tier=DataTier.FULL,
        min_matches=8,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('assists', 'player_stats'),
            StatRequirement('chances_created', 'player_stats', is_optional=True),
            StatRequirement('accurate_passes', 'player_stats'),
        ],
        secondary_stats=[
            StatRequirement('defensive_actions', 'player_stats', is_optional=True),
        ],
    ),
    
    'vision': AttributeDefinition(
        name="Vision",
        code="VIS",
        category=AttributeCategory.MENTAL,
        description="Ability to spot key passes",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('assists', 'player_stats'),
            StatRequirement('expected_assists', 'player_stats', is_optional=True),
            StatRequirement('chances_created', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('passes_into_final_third', 'player_stats', is_optional=True),
            StatRequirement('long_balls_accurate', 'player_stats', is_optional=True),
        ],
    ),
    
    'work_rate': AttributeDefinition(
        name="Work Rate",
        code="WOR",
        category=AttributeCategory.MENTAL,
        description="Overall effort and activity",
        min_tier=DataTier.FULL,
        min_matches=8,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('minutes_played', 'player_stats'),
            StatRequirement('rating', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('recoveries', 'player_stats', is_optional=True),
            StatRequirement('defensive_actions', 'player_stats', is_optional=True),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # PHYSICAL ATTRIBUTES
    # -------------------------------------------------------------------------
    
    'stamina': AttributeDefinition(
        name="Stamina",
        code="STA",
        category=AttributeCategory.PHYSICAL,
        description="Minutes played sustainability",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.AGGREGATE,
        primary_stats=[
            StatRequirement('minutes_played', 'player_stats'),
        ],
    ),
    
    'jumping': AttributeDefinition(
        name="Jumping Reach",
        code="JUM",
        category=AttributeCategory.PHYSICAL,
        description="Aerial duel ability",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('aerials_won', 'player_stats'),
            StatRequirement('aerials_lost', 'player_stats', is_optional=True),
        ],
    ),
    
    'natural_fitness': AttributeDefinition(
        name="Natural Fitness",
        code="NFI",
        category=AttributeCategory.PHYSICAL,
        description="Injury resistance",
        min_tier=DataTier.BASIC,  # Uses unavailable_players table
        min_matches=20,
        calculation_method=CalculationMethod.INVERSE,
        primary_stats=[
            StatRequirement('injury_id', 'match_unavailable_players'),
        ],
        description_extended="Based on frequency of injury appearances in unavailable_players"
    ),
    
    'strength': AttributeDefinition(
        name="Strength",
        code="STR",
        category=AttributeCategory.PHYSICAL,
        description="Physical duel dominance",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('duel_won', 'player_stats'),
            StatRequirement('duel_lost', 'player_stats'),
        ],
        secondary_stats=[
            StatRequirement('ground_duels_won', 'player_stats', is_optional=True),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # META ATTRIBUTES
    # -------------------------------------------------------------------------
    
    'consistency': AttributeDefinition(
        name="Consistency",
        code="CNS",
        category=AttributeCategory.META,
        description="Rating stability match to match",
        min_tier=DataTier.FULL,
        min_matches=20,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('rating', 'player_stats'),
        ],
        higher_is_better=False,  # Lower variance = higher consistency
    ),
    
    'versatility': AttributeDefinition(
        name="Versatility",
        code="VER",
        category=AttributeCategory.META,
        description="Performance across positions",
        min_tier=DataTier.BASIC,
        min_matches=25,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('position_id', 'match_lineup_players'),
            StatRequirement('rating', 'match_lineup_players', is_optional=True),
        ],
    ),
    
    'adaptability': AttributeDefinition(
        name="Adaptability",
        code="ADA",
        category=AttributeCategory.META,
        description="Performance across leagues/countries",
        min_tier=DataTier.BASIC,
        min_matches=30,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('league_id', 'match_details'),
            StatRequirement('rating', 'player_stats', is_optional=True),
        ],
    ),
    
    'big_game': AttributeDefinition(
        name="Important Matches",
        code="BIG",
        category=AttributeCategory.META,
        description="Performance vs higher-rated opposition",
        min_tier=DataTier.TEAM_STATS,
        min_matches=15,
        calculation_method=CalculationMethod.CONTEXTUAL,
        primary_stats=[
            StatRequirement('rating', 'match_team_stats'),  # Team ratings comparison
            StatRequirement('rating', 'player_stats', is_optional=True),
        ],
    ),

    'marking': AttributeDefinition(
        name="Marking",
        code="MAR",
        category=AttributeCategory.TECHNICAL,
        description="Defensive positioning and man-marking ability",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('interceptions', 'player_stats'),
            StatRequirement('clearances', 'player_stats'),
            StatRequirement('shot_blocks', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('dribbled_past', 'player_stats', is_optional=True),  # Inverse
        ],
        applicable_positions=[PositionGroup.DEFENDER],
        higher_is_better=True,  # dribbled_past handled inversely in composite
    ),

    'penalties': AttributeDefinition(
        name="Penalty Taking",
        code="PEN",
        category=AttributeCategory.TECHNICAL,
        description="Penalty kick conversion",
        min_tier=DataTier.COMPLETE,
        min_matches=5,  # Low because penalties are rare
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('situation', 'match_shotmap'),  # Filter: 'Penalty'
            StatRequirement('event_type', 'match_shotmap'),  # Goal/Miss/Saved
        ],
        exclude_positions=[PositionGroup.GOALKEEPER],
    ),

    'long_passing': AttributeDefinition(
        name="Long Passing",
        code="LPA",
        category=AttributeCategory.TECHNICAL,
        description="Accuracy and effectiveness of long-range distribution",
        min_tier=DataTier.FULL,
        min_matches=8,
        calculation_method=CalculationMethod.PERCENTAGE,
        primary_stats=[
            StatRequirement('accurate_long_balls', 'player_stats'),
            StatRequirement('long_balls', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('passes_into_final_third', 'player_stats', is_optional=True),
        ],
    ),
}


# =============================================================================
# GOALKEEPER ATTRIBUTES
# =============================================================================

GOALKEEPER_ATTRIBUTES: Dict[str, AttributeDefinition] = {
    
    'reflexes': AttributeDefinition(
        name="Reflexes",
        code="REF",
        category=AttributeCategory.GOALKEEPING,
        description="Reaction saves",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('saves', 'player_stats'),
            StatRequirement('saves_inside_box', 'player_stats', is_optional=True),
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'one_on_ones': AttributeDefinition(
        name="One On Ones",
        code="OOO",
        category=AttributeCategory.GOALKEEPING,
        description="Stopping breakaways",
        min_tier=DataTier.FULL,
        min_matches=15,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('goals_prevented', 'player_stats', is_optional=True),
            StatRequirement('goals_conceded', 'player_stats'),
            StatRequirement('saves', 'player_stats'),
            StatRequirement('expected_goals_on_target_faced', 'player_stats', is_optional=True)
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'shot_stopping': AttributeDefinition(
        name="Shot Stopping",
        code="STP",
        category=AttributeCategory.GOALKEEPING,
        description="General save ability",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.DIFFERENTIAL,
        primary_stats=[
            StatRequirement('saves', 'player_stats'),
            StatRequirement('goals_conceded', 'player_stats'),
            StatRequirement('expected_goals_on_target_faced', 'player_stats', is_optional=True),
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'kicking': AttributeDefinition(
        name="Kicking",
        code="KIC",
        category=AttributeCategory.GOALKEEPING,
        description="Long distribution",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('long_balls_accurate', 'player_stats'),
            StatRequirement('accurate_passes', 'player_stats', is_optional=True),
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'throwing': AttributeDefinition(
        name="Throwing",
        code="THR",
        category=AttributeCategory.GOALKEEPING,
        description="Short distribution",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('punches', 'player_stats'),
            StatRequirement('player_throws', 'player_stats'),
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'aerial_reach': AttributeDefinition(
        name="Aerial Reach",
        code="AER",
        category=AttributeCategory.GOALKEEPING,
        description="Claiming crosses",
        min_tier=DataTier.FULL,
        min_matches=10,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('punches', 'player_stats'),
            StatRequirement('keeper_high_claim', 'player_stats'),
            StatRequirement('catches', 'player_stats', is_optional=True),
        ],
        secondary_stats=[
            StatRequirement('aerials_won', 'player_stats', is_optional=True),
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'command_of_area': AttributeDefinition(
        name="Command of Area",
        code="CMD",
        category=AttributeCategory.GOALKEEPING,
        description="Box dominance",
        min_tier=DataTier.FULL,
        min_matches=12,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('punches', 'player_stats'),
            StatRequirement('catches', 'player_stats', is_optional=True),
            StatRequirement('clearances', 'player_stats', is_optional=True),
            StatRequirement('keeper_high_claim', 'player_stats'),
            StatRequirement('keeper_sweeper', 'player_stats', is_optional=True),
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'rushing_out': AttributeDefinition(
        name="Rushing Out",
        code="RUS",
        category=AttributeCategory.GOALKEEPING,
        description="Sweeper keeper ability",
        min_tier=DataTier.FULL,
        min_matches=15,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('clearances', 'player_stats'),
            StatRequirement('interceptions', 'player_stats', is_optional=True),
            StatRequirement('keeper_sweeper', 'player_stats', is_optional=True),
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'gk_positioning': AttributeDefinition(
        name="Positioning",
        code="GPO",
        category=AttributeCategory.GOALKEEPING,
        description="Shot positioning",
        min_tier=DataTier.COMPLETE,  # Benefits from shotmap
        min_matches=10,
        calculation_method=CalculationMethod.DIFFERENTIAL,
        primary_stats=[
            StatRequirement('expected_goals_faced', 'player_stats', is_optional=True),
            StatRequirement('goals_conceded', 'player_stats'),
        ],
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
    
    'gk_composure': AttributeDefinition(
        name="Composure",
        code="GCM",
        category=AttributeCategory.GOALKEEPING,
        description="Performance under pressure",
        min_tier=DataTier.FULL,
        min_matches=15,
        calculation_method=CalculationMethod.COMPOSITE,
        primary_stats=[
            StatRequirement('errors_led_to_goal', 'player_stats'),
            StatRequirement('saved_penalties', 'player_stats', is_optional=True),
        ],
        higher_is_better=False,  # Fewer errors = better
        applicable_positions=[PositionGroup.GOALKEEPER],
    ),
}


# Combine all player attributes
ALL_PLAYER_ATTRIBUTES = {**PLAYER_ATTRIBUTES, **GOALKEEPER_ATTRIBUTES}