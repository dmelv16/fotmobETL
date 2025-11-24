"""
Core rating engine implementing Elo-style rating calculations.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import math

from config.constants import (
    BASE_RATING, RATING_FLOOR, RATING_CEILING,
    MAX_RATING_CHANGE, MIN_RATING_CHANGE,
    K_FACTORS, DataTier, TIER_CONFIGS,
    COMPETITION_MULTIPLIERS, CONTINENTAL_ADJUSTMENTS,
    DIVISION_MULTIPLIERS, KNOCKOUT_BONUS
)
from models.entities import (
    BaseEntity, Player, Coach, Team, League,
    EntityType, RatingSnapshot
)


@dataclass
class MatchContext:
    """Context information for a match used in rating calculations."""
    match_id: int
    match_date: datetime
    
    # Teams
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    
    # Competition context
    league_id: int
    competition_type: str  # 'League', 'Cup', 'Continental'
    division_level: int = 1
    is_knockout: bool = False
    competition_name: Optional[str] = None
    
    # Data quality
    data_tier: DataTier = DataTier.MINIMAL
    
    # Pre-calculated ratings (optional, for efficiency)
    home_team_rating: Optional[float] = None
    away_team_rating: Optional[float] = None


@dataclass
class RatingUpdate:
    """Result of a rating calculation."""
    entity_id: int
    old_rating: float
    new_rating: float
    change: float
    expected_result: float
    actual_result: float
    k_factor_used: float


class RatingEngine:
    """
    Calculates rating updates using an Elo-style system.
    
    Key features:
    - Variable K-factors by entity type
    - Competition weighting (continental > domestic)
    - Data tier confidence multipliers
    - Rating floor and ceiling
    """
    
    def __init__(self):
        self.rating_scale = 400  # Standard Elo scale factor
    
    # =========================================================================
    # CORE ELO CALCULATIONS
    # =========================================================================
    
    def expected_result(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected result (0-1) for entity A vs entity B.
        Uses standard Elo formula.
        
        Returns probability of A winning (1.0 = certain win, 0.5 = even, 0.0 = certain loss)
        """
        exponent = (rating_b - rating_a) / self.rating_scale
        return 1.0 / (1.0 + math.pow(10, exponent))
    
    def actual_result(
        self, 
        score_for: int, 
        score_against: int,
        use_goal_difference: bool = False
    ) -> float:
        """
        Convert match score to actual result value (0-1).
        
        Basic: Win=1, Draw=0.5, Loss=0
        With goal difference: Scaled based on margin
        """
        if score_for > score_against:
            base = 1.0
        elif score_for < score_against:
            base = 0.0
        else:
            base = 0.5
        
        if use_goal_difference and base != 0.5:
            # Add bonus/penalty for goal difference (capped)
            gd = abs(score_for - score_against)
            gd_factor = min(0.1 * gd, 0.3)  # Max 0.3 bonus for 3+ goal difference
            
            if base == 1.0:
                return min(1.0, base + gd_factor * 0.5)  # Reduce bonus for already winning
            else:
                return max(0.0, base - gd_factor * 0.5)
        
        return base
    
    def calculate_rating_change(
        self,
        current_rating: float,
        expected: float,
        actual: float,
        k_factor: float
    ) -> float:
        """
        Calculate rating change.
        
        Formula: change = K * (actual - expected)
        """
        change = k_factor * (actual - expected)
        
        # Apply caps
        change = max(MIN_RATING_CHANGE, min(MAX_RATING_CHANGE, change))
        
        return change
    
    def apply_rating_bounds(self, rating: float) -> float:
        """Ensure rating stays within bounds."""
        return max(RATING_FLOOR, min(RATING_CEILING, rating))
    
    # =========================================================================
    # K-FACTOR CALCULATIONS
    # =========================================================================
    
    def get_base_k_factor(self, entity_type: EntityType) -> float:
        """Get base K-factor for entity type."""
        type_map = {
            EntityType.PLAYER: K_FACTORS['player'],
            EntityType.GOALKEEPER: K_FACTORS['goalkeeper'],
            EntityType.COACH: K_FACTORS['coach'],
            EntityType.TEAM: K_FACTORS['team'],
            EntityType.LEAGUE: K_FACTORS['league'],
        }
        return type_map.get(entity_type, K_FACTORS['player'])
    
    def get_competition_multiplier(self, context: MatchContext) -> float:
        """
        Calculate competition weight multiplier.
        
        Continental competitions weighted highest for cross-league calibration.
        """
        # Start with competition type base
        base = COMPETITION_MULTIPLIERS.get(context.competition_type, 1.0)
        
        # Check for specific continental competition
        if context.competition_type == 'Continental' and context.competition_name:
            for comp_name, mult in CONTINENTAL_ADJUSTMENTS.items():
                if comp_name.lower() in context.competition_name.lower():
                    base = mult
                    break
        
        # Apply division level multiplier (for leagues)
        if context.competition_type == 'League':
            div_mult = DIVISION_MULTIPLIERS.get(context.division_level, 0.6)
            base *= div_mult
        
        # Knockout bonus
        if context.is_knockout:
            base *= KNOCKOUT_BONUS
        
        return base
    
    def get_tier_multiplier(self, tier: DataTier) -> float:
        """Get confidence multiplier based on data tier."""
        return TIER_CONFIGS[tier].k_factor_multiplier
    
    def get_experience_multiplier(self, matches_played: int) -> float:
        """
        Adjust K-factor based on experience.
        New entities are more volatile (higher K).
        """
        if matches_played < 10:
            return 1.5  # New entity, ratings should move faster
        elif matches_played < 30:
            return 1.2
        elif matches_played < 100:
            return 1.0
        else:
            return 0.85  # Established entity, more stable
    
    def calculate_effective_k_factor(
        self,
        entity_type: EntityType,
        context: MatchContext,
        matches_played: int
    ) -> float:
        """Calculate final K-factor with all multipliers."""
        base = self.get_base_k_factor(entity_type)
        competition_mult = self.get_competition_multiplier(context)
        tier_mult = self.get_tier_multiplier(context.data_tier)
        experience_mult = self.get_experience_multiplier(matches_played)
        
        return base * competition_mult * tier_mult * experience_mult
    
    # =========================================================================
    # ENTITY-SPECIFIC RATING UPDATES
    # =========================================================================
    
    def update_team_rating(
        self,
        team: Team,
        opponent_rating: float,
        context: MatchContext,
        is_home: bool
    ) -> RatingUpdate:
        """
        Update a team's rating based on match result.
        """
        # Determine scores from team's perspective
        if is_home:
            score_for = context.home_score
            score_against = context.away_score
        else:
            score_for = context.away_score
            score_against = context.home_score
        
        # Home advantage adjustment (add ~50 rating points equivalent)
        adjusted_opponent_rating = opponent_rating
        if not is_home:
            adjusted_opponent_rating += 50  # Away team faces "harder" opponent
        
        # Calculate expected and actual results
        expected = self.expected_result(team.current_rating, adjusted_opponent_rating)
        actual = self.actual_result(score_for, score_against, use_goal_difference=True)
        
        # Calculate K-factor
        k_factor = self.calculate_effective_k_factor(
            EntityType.TEAM, context, team.matches_played
        )
        
        # Calculate change
        change = self.calculate_rating_change(
            team.current_rating, expected, actual, k_factor
        )
        
        new_rating = self.apply_rating_bounds(team.current_rating + change)
        
        return RatingUpdate(
            entity_id=team.id,
            old_rating=team.current_rating,
            new_rating=new_rating,
            change=change,
            expected_result=expected,
            actual_result=actual,
            k_factor_used=k_factor
        )
    
    def update_player_rating(
        self,
        player: Player,
        team_rating_change: float,
        individual_performance: Optional[float],
        context: MatchContext,
        minutes_played: int = 90
    ) -> RatingUpdate:
        """
        Update a player's rating based on team result and individual performance.
        
        Player rating is influenced by:
        1. Team result (shared credit/blame)
        2. Individual performance (if available)
        3. Minutes played (partial credit)
        """
        # Calculate K-factor
        k_factor = self.calculate_effective_k_factor(
            player.entity_type, context, player.matches_played
        )
        
        # Minutes factor (players who played less get proportionally less change)
        minutes_factor = min(1.0, minutes_played / 90.0)
        
        # Base change from team result (reduced for individual)
        team_contribution = team_rating_change * 0.4 * minutes_factor
        
        # Individual performance contribution
        if individual_performance is not None:
            # individual_performance should be normalized around 0
            # Positive = above average, negative = below average
            individual_contribution = k_factor * individual_performance * 0.6 * minutes_factor
        else:
            # If no individual stats, rely more on team result
            individual_contribution = team_rating_change * 0.3 * minutes_factor
        
        total_change = team_contribution + individual_contribution
        
        # Apply caps
        total_change = max(MIN_RATING_CHANGE * minutes_factor, 
                          min(MAX_RATING_CHANGE * minutes_factor, total_change))
        
        new_rating = self.apply_rating_bounds(player.current_rating + total_change)
        
        return RatingUpdate(
            entity_id=player.id,
            old_rating=player.current_rating,
            new_rating=new_rating,
            change=total_change,
            expected_result=0.0,  # Not directly applicable for players
            actual_result=individual_performance or 0.0,
            k_factor_used=k_factor
        )
    
    def update_coach_rating(
        self,
        coach: Coach,
        team_rating_change: float,
        context: MatchContext,
        tactical_performance: Optional[float] = None
    ) -> RatingUpdate:
        """
        Update coach rating based on team performance.
        
        Coach is credited/blamed for:
        1. Team result
        2. Expected vs actual performance
        3. Tactical indicators (if available)
        """
        k_factor = self.calculate_effective_k_factor(
            EntityType.COACH, context, coach.matches_played
        )
        
        # Base change from team result
        base_change = team_rating_change * 0.6
        
        # Tactical bonus/penalty
        if tactical_performance is not None:
            tactical_change = k_factor * tactical_performance * 0.4
        else:
            tactical_change = 0.0
        
        total_change = base_change + tactical_change
        
        new_rating = self.apply_rating_bounds(coach.current_rating + total_change)
        
        return RatingUpdate(
            entity_id=coach.id,
            old_rating=coach.current_rating,
            new_rating=new_rating,
            change=total_change,
            expected_result=0.0,
            actual_result=tactical_performance or 0.0,
            k_factor_used=k_factor
        )
    
    def update_league_rating(
        self,
        league: League,
        team_ratings: List[float],
        continental_results: Optional[Dict] = None
    ) -> RatingUpdate:
        """
        Update league rating based on:
        1. Average team ratings in the league
        2. Performance in continental competition (if available)
        """
        old_rating = league.current_rating
        
        # Base: average of team ratings
        if team_ratings:
            avg_team_rating = sum(team_ratings) / len(team_ratings)
        else:
            avg_team_rating = old_rating
        
        # Slow adjustment toward average
        base_change = (avg_team_rating - old_rating) * 0.05
        
        # Continental performance bonus
        if continental_results:
            # Expected: teams should perform at their rating
            # If they overperform, league rating goes up
            wins = continental_results.get('wins', 0)
            losses = continental_results.get('losses', 0)
            expected_wins = continental_results.get('expected_wins', wins)
            
            if wins + losses > 0:
                performance_diff = (wins - expected_wins) / (wins + losses)
                continental_bonus = performance_diff * K_FACTORS['league']
                base_change += continental_bonus
        
        new_rating = self.apply_rating_bounds(old_rating + base_change)
        
        return RatingUpdate(
            entity_id=league.id,
            old_rating=old_rating,
            new_rating=new_rating,
            change=new_rating - old_rating,
            expected_result=0.0,
            actual_result=0.0,
            k_factor_used=K_FACTORS['league']
        )


# =============================================================================
# MATCH RATING PROCESSOR
# =============================================================================

class MatchRatingProcessor:
    """
    Orchestrates rating updates for all entities involved in a match.
    """
    
    def __init__(self, rating_engine: RatingEngine):
        self.engine = rating_engine
    
    def process_match(
        self,
        context: MatchContext,
        home_team: Team,
        away_team: Team,
        home_players: List[Tuple[Player, int, Optional[float]]],  # (player, minutes, performance)
        away_players: List[Tuple[Player, int, Optional[float]]],
        home_coach: Optional[Coach] = None,
        away_coach: Optional[Coach] = None
    ) -> Dict[str, List[RatingUpdate]]:
        """
        Process all rating updates for a match.
        
        Returns dict with keys: 'teams', 'players', 'coaches'
        """
        updates = {
            'teams': [],
            'players': [],
            'coaches': []
        }
        
        # Update team ratings
        home_update = self.engine.update_team_rating(
            home_team, away_team.current_rating, context, is_home=True
        )
        away_update = self.engine.update_team_rating(
            away_team, home_team.current_rating, context, is_home=False
        )
        updates['teams'].extend([home_update, away_update])
        
        # Update player ratings
        for player, minutes, performance in home_players:
            player_update = self.engine.update_player_rating(
                player, home_update.change, performance, context, minutes
            )
            updates['players'].append(player_update)
        
        for player, minutes, performance in away_players:
            player_update = self.engine.update_player_rating(
                player, away_update.change, performance, context, minutes
            )
            updates['players'].append(player_update)
        
        # Update coach ratings
        if home_coach:
            coach_update = self.engine.update_coach_rating(
                home_coach, home_update.change, context
            )
            updates['coaches'].append(coach_update)
        
        if away_coach:
            coach_update = self.engine.update_coach_rating(
                away_coach, away_update.change, context
            )
            updates['coaches'].append(coach_update)
        
        return updates
    
    def apply_updates(
        self,
        updates: Dict[str, List[RatingUpdate]],
        home_team: Team,
        away_team: Team,
        home_players: List[Player],
        away_players: List[Player],
        home_coach: Optional[Coach],
        away_coach: Optional[Coach],
        context: MatchContext
    ):
        """Apply rating updates to entities."""
        timestamp = context.match_date
        
        # Apply team updates
        for update in updates['teams']:
            team = home_team if update.entity_id == home_team.id else away_team
            snapshot = RatingSnapshot(
                rating=update.new_rating,
                match_id=context.match_id,
                timestamp=timestamp,
                change=update.change,
                k_factor_used=update.k_factor_used,
                expected_result=update.expected_result,
                actual_result=update.actual_result,
                data_tier=context.data_tier,
                competition_multiplier=self.engine.get_competition_multiplier(context)
            )
            team.update_rating(update.new_rating, snapshot)
        
        # Apply player updates
        all_players = {p.id: p for p in home_players + away_players}
        for update in updates['players']:
            if update.entity_id in all_players:
                player = all_players[update.entity_id]
                snapshot = RatingSnapshot(
                    rating=update.new_rating,
                    match_id=context.match_id,
                    timestamp=timestamp,
                    change=update.change,
                    k_factor_used=update.k_factor_used,
                    data_tier=context.data_tier
                )
                player.update_rating(update.new_rating, snapshot)
        
        # Apply coach updates
        coaches = [c for c in [home_coach, away_coach] if c]
        for update in updates['coaches']:
            for coach in coaches:
                if coach and coach.id == update.entity_id:
                    snapshot = RatingSnapshot(
                        rating=update.new_rating,
                        match_id=context.match_id,
                        timestamp=timestamp,
                        change=update.change,
                        k_factor_used=update.k_factor_used,
                        data_tier=context.data_tier
                    )
                    coach.update_rating(update.new_rating, snapshot)