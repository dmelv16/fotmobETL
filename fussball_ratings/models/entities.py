"""
Core entity models for players, coaches, teams, and leagues.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from config.constants import (
    BASE_RATING, ATTR_DEFAULT, PositionGroup,
    POSITION_TO_GROUP, DataTier
)


class EntityType(Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    COACH = "coach"
    TEAM = "team"
    LEAGUE = "league"


@dataclass
class RatingSnapshot:
    """A point-in-time rating record."""
    rating: float
    match_id: int
    timestamp: datetime
    change: float = 0.0
    k_factor_used: float = 0.0
    expected_result: float = 0.0
    actual_result: float = 0.0
    data_tier: DataTier = DataTier.MINIMAL
    competition_multiplier: float = 1.0


@dataclass
class AttributeScore:
    """A single attribute value with metadata."""
    value: float
    confidence: float  # 0-1, based on data quality/quantity
    matches_used: int
    last_updated: datetime
    data_tiers_used: List[DataTier] = field(default_factory=list)
    
    @property
    def display_value(self) -> int:
        """Round to integer for display (FM style 1-20)."""
        return max(1, min(20, round(self.value)))


@dataclass
class BaseEntity:
    """Base class for all rated entities."""
    id: int
    name: str
    entity_type: EntityType
    
    # Current state
    current_rating: float = BASE_RATING
    attributes: Dict[str, AttributeScore] = field(default_factory=dict)
    
    # Historical tracking
    rating_history: List[RatingSnapshot] = field(default_factory=list)
    matches_played: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_match_date: Optional[datetime] = None
    
    def update_rating(self, new_rating: float, snapshot: RatingSnapshot):
        """Update rating and record history."""
        snapshot.rating = new_rating
        snapshot.change = new_rating - self.current_rating
        self.current_rating = new_rating
        self.rating_history.append(snapshot)
        self.matches_played += 1
        self.last_match_date = snapshot.timestamp
    
    def get_rating_at(self, match_id: int) -> Optional[float]:
        """Get rating as of a specific match."""
        for snapshot in self.rating_history:
            if snapshot.match_id == match_id:
                return snapshot.rating
        return None
    
    def get_recent_form(self, n_matches: int = 5) -> float:
        """Get average rating change over last n matches."""
        recent = self.rating_history[-n_matches:]
        if not recent:
            return 0.0
        return sum(s.change for s in recent) / len(recent)


@dataclass
class Player(BaseEntity):
    """Player entity with position and attribute tracking."""
    
    # Player-specific fields
    primary_position_id: Optional[int] = None
    position_group: PositionGroup = PositionGroup.MIDFIELDER
    country_code: Optional[str] = None
    birth_date: Optional[datetime] = None
    
    # Team history
    current_team_id: Optional[int] = None
    team_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # League/country history for adaptability
    leagues_played: List[int] = field(default_factory=list)
    countries_played: List[str] = field(default_factory=list)
    
    # Position versatility tracking
    positions_played: Dict[int, int] = field(default_factory=dict)  # pos_id: match_count
    
    def __post_init__(self):
        self.entity_type = EntityType.PLAYER
        if self.primary_position_id:
            self.position_group = POSITION_TO_GROUP.get(
                self.primary_position_id, 
                PositionGroup.MIDFIELDER
            )
            # Check if goalkeeper
            if self.position_group == PositionGroup.GOALKEEPER:
                self.entity_type = EntityType.GOALKEEPER
    
    def record_position(self, position_id: int):
        """Track a position played."""
        self.positions_played[position_id] = self.positions_played.get(position_id, 0) + 1
    
    def record_team(self, team_id: int, league_id: int, country_code: str, match_date: datetime):
        """Track team/league/country changes."""
        if team_id != self.current_team_id:
            if self.current_team_id:
                self.team_history.append({
                    'team_id': self.current_team_id,
                    'left_date': match_date
                })
            self.current_team_id = team_id
        
        if league_id not in self.leagues_played:
            self.leagues_played.append(league_id)
        if country_code and country_code not in self.countries_played:
            self.countries_played.append(country_code)
    
    @property
    def is_goalkeeper(self) -> bool:
        return self.position_group == PositionGroup.GOALKEEPER
    
    @property
    def versatility_score(self) -> float:
        """Calculate raw versatility (positions played effectively)."""
        if len(self.positions_played) <= 1:
            return 0.0
        # More positions with significant time = higher versatility
        significant_positions = sum(1 for count in self.positions_played.values() if count >= 3)
        return min(1.0, significant_positions / 5.0)  # Cap at 5 positions


@dataclass
class Coach(BaseEntity):
    """Coach entity."""
    
    current_team_id: Optional[int] = None
    country_code: Optional[str] = None
    
    # Coaching history
    team_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Formation preferences
    formations_used: Dict[str, int] = field(default_factory=dict)  # formation: count
    
    # Win/draw/loss record
    wins: int = 0
    draws: int = 0
    losses: int = 0
    
    def __post_init__(self):
        self.entity_type = EntityType.COACH
    
    def record_result(self, result: str, formation: Optional[str] = None):
        """Track match result and formation."""
        if result == 'W':
            self.wins += 1
        elif result == 'D':
            self.draws += 1
        elif result == 'L':
            self.losses += 1
        
        if formation:
            self.formations_used[formation] = self.formations_used.get(formation, 0) + 1
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.draws + self.losses
        return self.wins / total if total > 0 else 0.0
    
    @property
    def points_per_game(self) -> float:
        total = self.wins + self.draws + self.losses
        if total == 0:
            return 0.0
        return (self.wins * 3 + self.draws) / total


@dataclass 
class Team(BaseEntity):
    """Team entity."""
    
    league_id: Optional[int] = None
    country_code: Optional[str] = None
    division_level: int = 1
    
    # Current squad (player_ids)
    current_squad: List[int] = field(default_factory=list)
    
    # Coach history
    current_coach_id: Optional[int] = None
    coach_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Season tracking
    seasons_in_league: Dict[int, List[str]] = field(default_factory=dict)  # league_id: [seasons]
    
    def __post_init__(self):
        self.entity_type = EntityType.TEAM
    
    def get_squad_average_rating(self, player_registry: Dict[int, Player]) -> float:
        """Calculate average rating of current squad."""
        if not self.current_squad:
            return self.current_rating
        
        ratings = []
        for pid in self.current_squad:
            if pid in player_registry:
                ratings.append(player_registry[pid].current_rating)
        
        return sum(ratings) / len(ratings) if ratings else self.current_rating


@dataclass
class League(BaseEntity):
    """League entity."""
    
    country_code: Optional[str] = None
    division_level: int = 1
    competition_type: str = "League"  # League, Cup, Continental
    parent_league_id: Optional[int] = None
    
    # Teams in league
    team_ids: List[int] = field(default_factory=list)
    
    # Season stats
    seasons_data: Dict[str, Dict] = field(default_factory=dict)
    
    def __post_init__(self):
        self.entity_type = EntityType.LEAGUE
    
    def get_average_team_rating(self, team_registry: Dict[int, Team]) -> float:
        """Calculate average rating of teams in league."""
        if not self.team_ids:
            return self.current_rating
        
        ratings = []
        for tid in self.team_ids:
            if tid in team_registry:
                ratings.append(team_registry[tid].current_rating)
        
        return sum(ratings) / len(ratings) if ratings else self.current_rating
    
    def get_rating_spread(self, team_registry: Dict[int, Team]) -> float:
        """Calculate std deviation of team ratings (league depth indicator)."""
        if len(self.team_ids) < 2:
            return 0.0
        
        ratings = [team_registry[tid].current_rating 
                   for tid in self.team_ids if tid in team_registry]
        
        if len(ratings) < 2:
            return 0.0
        
        mean = sum(ratings) / len(ratings)
        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
        return variance ** 0.5


# =============================================================================
# ENTITY REGISTRY
# =============================================================================

@dataclass
class EntityRegistry:
    """Central registry for all entities."""
    
    players: Dict[int, Player] = field(default_factory=dict)
    coaches: Dict[int, Coach] = field(default_factory=dict)
    teams: Dict[int, Team] = field(default_factory=dict)
    leagues: Dict[int, League] = field(default_factory=dict)
    
    def get_or_create_player(self, player_id: int, name: str, **kwargs) -> Player:
        """Get existing player or create new one."""
        if player_id not in self.players:
            self.players[player_id] = Player(
                id=player_id, 
                name=name,
                entity_type=EntityType.PLAYER,
                **kwargs
            )
        return self.players[player_id]
    
    def get_or_create_coach(self, coach_id: int, name: str, **kwargs) -> Coach:
        if coach_id not in self.coaches:
            self.coaches[coach_id] = Coach(
                id=coach_id,
                name=name,
                entity_type=EntityType.COACH,
                **kwargs
            )
        return self.coaches[coach_id]
    
    def get_or_create_team(self, team_id: int, name: str, **kwargs) -> Team:
        if team_id not in self.teams:
            self.teams[team_id] = Team(
                id=team_id,
                name=name,
                entity_type=EntityType.TEAM,
                **kwargs
            )
        return self.teams[team_id]
    
    def get_or_create_league(self, league_id: int, name: str, **kwargs) -> League:
        if league_id not in self.leagues:
            self.leagues[league_id] = League(
                id=league_id,
                name=name,
                entity_type=EntityType.LEAGUE,
                **kwargs
            )
        return self.leagues[league_id]
    
    def initialize_player_rating(self, player: Player, league_rating: float):
        """Initialize a new player's rating based on league context."""
        from config.constants import INIT_MULTIPLIERS
        multiplier = INIT_MULTIPLIERS['goalkeeper' if player.is_goalkeeper else 'player']
        player.current_rating = league_rating * multiplier