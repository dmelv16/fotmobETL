"""
Match processor - handles loading and processing individual matches.
Updated to properly handle home/away context.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from collections import defaultdict
from config.constants import DataTier, PositionGroup, get_position_group, BASE_RATING
from models.entities import (
    Player, Coach, Team, League, EntityRegistry
)
from processors.data_handlers import (
    DataTierProcessor, MatchDataAvailability, NullHandler,
    PlayerStatsAggregator, StatKeyTracker
)
from processors.stat_resolver import StatResolver, TeamMatchStatsBuilder, SideContext
from database.queries.match_queries import MatchQueries, LeagueQueries
from database.queries.player_queries import PlayerQueries, CoachQueries
from engines.rating_engine import MatchContext, RatingEngine, MatchRatingProcessor

logger = logging.getLogger(__name__)


@dataclass
class LoadedMatchData:
    """Container for all loaded data for a single match."""
    match_id: int
    data_tier: DataTier
    availability: MatchDataAvailability
    
    # Core match info
    match_details: Dict[str, Any] = field(default_factory=dict)
    match_facts: Optional[Dict[str, Any]] = None
    
    # Team data
    team_stats_summary: Optional[Dict[str, Any]] = None
    home_team_stats: Optional[Dict[str, Any]] = None
    away_team_stats: Optional[Dict[str, Any]] = None
    
    # Player data
    lineups: List[Dict[str, Any]] = field(default_factory=list)
    player_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)
    shotmap: List[Dict[str, Any]] = field(default_factory=list)
    substitutions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Coaches
    coaches: List[Dict[str, Any]] = field(default_factory=list)
    
    # Unavailable players (for injury tracking)
    unavailable_players: List[Dict[str, Any]] = field(default_factory=list)
    
    # Competition context
    league_info: Optional[Dict[str, Any]] = None
    competition_type: str = "League"


@dataclass
class TeamMatchContext:
    """Full context for a team in a match, with resolved stats."""
    team_id: int
    team_name: str
    is_home_team: bool
    
    # Resolved stats (no home_/away_ prefix)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Players for this team
    lineup: List[Dict[str, Any]] = field(default_factory=list)
    player_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Coach
    coach_id: Optional[int] = None
    coach_name: Optional[str] = None

@dataclass
class PlayerMatchStats:
    """Stats for a single player in a single match."""
    match_id: int
    player_id: int
    team_id: int
    data_tier: DataTier
    minutes_played: int = 0
    
    stats: Dict[str, float] = field(default_factory=dict)
    
    position_id: Optional[int] = None
    position_group: PositionGroup = PositionGroup.MIDFIELDER  # NEW: Added!
    usual_position_id: Optional[int] = None
    is_starter: bool = False
    is_captain: bool = False  # NEW: Added!
    rating: Optional[float] = None
    is_home_team: bool = False
    
    league_id: Optional[int] = None
    league_name: Optional[str] = None
    country_code: Optional[str] = None
    
    own_team_rating: Optional[float] = None
    opponent_team_rating: Optional[float] = None
    
    team_score: Optional[int] = None
    opponent_score: Optional[int] = None
    match_result: Optional[str] = None
    
    was_injured: bool = False
    injury_id: Optional[int] = None
    
    match_date: Optional[datetime] = None  # NEW: Added for time-based calcs

@dataclass 
class CoachMatchStats:
    """Stats collected for a coach from a single match."""
    coach_id: int
    match_id: int
    team_id: int
    is_home_team: bool
    data_tier: DataTier
    match_date: Optional[datetime] = None
    
    # Resolved team stats (neutral keys)
    team_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Additional coach-specific data
    formation: Optional[str] = None
    lineup_player_ids: List[int] = field(default_factory=list)
    substitutions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Player ratings for development tracking
    player_ratings: Dict[int, float] = field(default_factory=dict)
    player_ages: Dict[int, int] = field(default_factory=dict)
    
    # Goal events with timing for adaptability
    goal_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TeamMatchStats:
    """Stats collected for a team from a single match."""
    team_id: int
    match_id: int
    is_home_team: bool
    data_tier: DataTier
    
    # Resolved stats (neutral keys)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Additional context
    opponent_team_id: Optional[int] = None
    opponent_rating: Optional[float] = None
    
    # Lineup data for squad depth
    lineup_player_ids: List[int] = field(default_factory=list)
    starter_player_ids: List[int] = field(default_factory=list)
    sub_player_ids: List[int] = field(default_factory=list)


class MatchDataLoader:
    """Loads all data for a match from the database."""
    
    def __init__(
        self,
        match_queries: MatchQueries,
        player_queries: PlayerQueries,
        coach_queries: CoachQueries,
        league_queries: LeagueQueries
    ):
        self.match_q = match_queries
        self.player_q = player_queries
        self.coach_q = coach_queries
        self.league_q = league_queries
        self.stats_aggregator = PlayerStatsAggregator()
    
    def load_match(self, match_id: int) -> Optional[LoadedMatchData]:
        """Load all available data for a match."""
        
        # Get extraction status first
        status = self.match_q.get_extraction_status(match_id)
        if not status:
            logger.warning(f"No extraction status for match {match_id}")
            return None
        
        availability = DataTierProcessor.from_extraction_status(status)
        data_tier = DataTierProcessor.classify_tier(availability)
        
        # Get match details (required)
        details = self.match_q.get_match_details(match_id)
        if not details:
            logger.warning(f"No match details for match {match_id}")
            return None
        
        # Initialize loaded data
        loaded = LoadedMatchData(
            match_id=match_id,
            data_tier=data_tier,
            availability=availability,
            match_details=details
        )
        
        # Get league info
        league_id = details.get('league_id')
        if league_id:
            loaded.league_info = self.league_q.get_league_info(league_id)
            loaded.competition_type = self.league_q.get_competition_type(league_id)
        
        # Load based on availability
        if availability.has_events:
            loaded.events = self.match_q.get_match_events(match_id)
        
        if availability.has_lineups:
            loaded.lineups = self.player_q.get_match_lineup(match_id)
            loaded.coaches = self.coach_q.get_match_coaches(match_id)
            # Load unavailable players for injury tracking
            loaded.unavailable_players = self.match_q.get_unavailable_players(match_id)
        
        if availability.has_stats:
            loaded.team_stats_summary = self.match_q.get_match_stats_summary(match_id)
            team_stats = self.match_q.get_match_team_stats(match_id)
            for ts in team_stats:
                if ts.get('is_home_team'):
                    loaded.home_team_stats = ts
                else:
                    loaded.away_team_stats = ts
        
        if availability.has_player_stats:
            raw_stats = self.player_q.get_all_player_stats_for_match(match_id)
            loaded.player_stats = self._aggregate_player_stats(raw_stats)
        
        if availability.has_shotmap:
            loaded.shotmap = self.match_q.get_match_shotmap(match_id)
        
        if availability.has_events:
            loaded.substitutions = self.match_q.get_match_substitutions(match_id)
        
        loaded.match_facts = self.match_q.get_match_facts(match_id)
        
        return loaded
    
    def _aggregate_player_stats(
        self, 
        raw_stats: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, float]]:
        """Aggregate raw player stats by player_id."""
        by_player: Dict[int, List[Dict]] = {}
        
        for row in raw_stats:
            pid = row.get('player_id')
            if pid:
                if pid not in by_player:
                    by_player[pid] = []
                by_player[pid].append(row)
        
        result = {}
        for pid, rows in by_player.items():
            result[pid] = self.stats_aggregator.aggregate_player_match_stats(rows)
        
        return result


class MatchProcessor:
    """
    Processes a single match - updates ratings and collects stats for attributes.
    Updated to properly resolve home/away stats.
    """
    
    def __init__(
        self,
        entity_registry: EntityRegistry,
        rating_engine: RatingEngine,
        stat_key_tracker: StatKeyTracker
    ):
        self.registry = entity_registry
        self.rating_engine = rating_engine
        self.rating_processor = MatchRatingProcessor(rating_engine)
        self.stat_tracker = stat_key_tracker
        self.stats_builder = TeamMatchStatsBuilder()
    
    def process_match(
        self, 
        loaded: LoadedMatchData
    ) -> Dict[str, Any]:
        """
        Process a loaded match.
        Returns dict with processing results and collected stats.
        """
        result = {
            'match_id': loaded.match_id,
            'data_tier': loaded.data_tier,
            'success': False,
            'errors': [],
            'entities_updated': {
                'teams': 0,
                'players': 0,
                'coaches': 0
            }
        }
        
        try:
            # Build match context
            context = self._build_match_context(loaded)
            
            # Build team contexts with resolved stats
            home_context = self._build_team_context(loaded, is_home=True)
            away_context = self._build_team_context(loaded, is_home=False)
            
            # Ensure entities exist
            home_team = self._ensure_team(loaded, is_home=True)
            away_team = self._ensure_team(loaded, is_home=False)
            
            # Process players with proper team context
            home_players = self._process_players(loaded, home_context)
            away_players = self._process_players(loaded, away_context)
            
            # Get coaches with proper context
            home_coach = self._get_coach(loaded, is_home=True)
            away_coach = self._get_coach(loaded, is_home=False)
            
            # Calculate rating updates
            updates = self.rating_processor.process_match(
                context=context,
                home_team=home_team,
                away_team=away_team,
                home_players=home_players,
                away_players=away_players,
                home_coach=home_coach,
                away_coach=away_coach
            )
            
            # Apply updates
            self.rating_processor.apply_updates(
                updates=updates,
                home_team=home_team,
                away_team=away_team,
                home_players=[p for p, _, _ in home_players],
                away_players=[p for p, _, _ in away_players],
                home_coach=home_coach,
                away_coach=away_coach,
                context=context
            )
            
            # Track stats availability
            if loaded.player_stats:
                all_stat_keys = set()
                for stats in loaded.player_stats.values():
                    all_stat_keys.update(stats.keys())
                self.stat_tracker.record_match_stats(loaded.match_id, all_stat_keys)
            
            # Collect stats for attribute calculation - pass both contexts
            player_match_stats = self._collect_player_match_stats(
                loaded, context, home_context, away_context
            )
            coach_match_stats = self._collect_coach_match_stats(
                loaded, home_context, away_context
            )
            team_match_stats = self._collect_team_match_stats(
                loaded, home_context, away_context
            )
            
            # Collect shotmap stats
            shotmap_stats = self._collect_shotmap_stats(loaded)
            
            result['success'] = True
            result['entities_updated']['teams'] = 2
            result['entities_updated']['players'] = len(home_players) + len(away_players)
            result['entities_updated']['coaches'] = (1 if home_coach else 0) + (1 if away_coach else 0)
            result['player_match_stats'] = player_match_stats
            result['coach_match_stats'] = coach_match_stats
            result['team_match_stats'] = team_match_stats
            result['shotmap_stats'] = shotmap_stats
            result['rating_updates'] = updates
            
        except Exception as e:
            logger.error(f"Error processing match {loaded.match_id}: {e}")
            result['errors'].append(str(e))
        
        return result
    
    def _build_team_context(
        self, 
        loaded: LoadedMatchData, 
        is_home: bool
    ) -> TeamMatchContext:
        """Build complete context for a team with resolved stats."""
        details = loaded.match_details
        
        if is_home:
            team_id = details['home_team_id']
            team_name = details.get('home_team_name', '')
            team_stats_raw = loaded.home_team_stats
        else:
            team_id = details['away_team_id']
            team_name = details.get('away_team_name', '')
            team_stats_raw = loaded.away_team_stats
        
        # Build resolved stats
        stats = self.stats_builder.build_stats(
            match_details=details,
            match_stats_summary=loaded.team_stats_summary,
            match_team_stats=team_stats_raw,
            is_home_team=is_home
        )
        
        # Get lineup for this team
        lineup = [p for p in loaded.lineups if p.get('team_id') == team_id]
        
        # Get player stats for this team's players
        lineup_player_ids = {p.get('player_id') for p in lineup if p.get('player_id')}
        player_stats = {
            pid: pstats for pid, pstats in loaded.player_stats.items()
            if pid in lineup_player_ids
        }
        
        # Get coach
        coach_id = None
        coach_name = None
        for coach in loaded.coaches:
            if coach.get('is_home_team') == is_home:
                coach_id = coach.get('coach_id')
                coach_name = coach.get('coach_name')
                break
        
        return TeamMatchContext(
            team_id=team_id,
            team_name=team_name,
            is_home_team=is_home,
            stats=stats,
            lineup=lineup,
            player_stats=player_stats,
            coach_id=coach_id,
            coach_name=coach_name
        )
    
    def _build_match_context(self, loaded: LoadedMatchData) -> MatchContext:
        """Build match context from loaded data."""
        details = loaded.match_details
        
        division_level = 1
        if loaded.league_info:
            div = loaded.league_info.get('DivisionLevel')
            if div and isinstance(div, int):
                division_level = div
        
        is_knockout = loaded.competition_type in ('Cup', 'Continental')
        
        return MatchContext(
            match_id=loaded.match_id,
            match_date=details.get('match_time_utc', datetime.now()),
            home_team_id=details['home_team_id'],
            away_team_id=details['away_team_id'],
            home_score=details.get('home_team_score', 0),
            away_score=details.get('away_team_score', 0),
            league_id=details.get('league_id', 0),
            competition_type=loaded.competition_type,
            division_level=division_level,
            is_knockout=is_knockout,
            competition_name=details.get('league_name'),
            data_tier=loaded.data_tier
        )
    
    def _ensure_team(self, loaded: LoadedMatchData, is_home: bool) -> Team:
        """Get or create team entity."""
        details = loaded.match_details
        
        if is_home:
            team_id = details['home_team_id']
            team_name = details.get('home_team_name', f'Team {team_id}')
        else:
            team_id = details['away_team_id']
            team_name = details.get('away_team_name', f'Team {team_id}')
        
        team = self.registry.get_or_create_team(team_id, team_name)
        
        if loaded.league_info:
            team.league_id = loaded.league_info.get('LeagueID')
            team.country_code = loaded.league_info.get('CountryName')
            div = loaded.league_info.get('DivisionLevel')
            team.division_level = div if isinstance(div, int) else 1
        
        return team
    
    def _process_players(
        self, 
        loaded: LoadedMatchData,
        team_context: TeamMatchContext
    ) -> List[Tuple[Player, int, Optional[float]]]:
        """Process players for a team using team context."""
        players = []
        
        for lineup_row in team_context.lineup:
            player_id = lineup_row.get('player_id')
            if not player_id:
                continue
            
            player_name = lineup_row.get('player_name', f'Player {player_id}')
            
            # Get or create player
            player = self.registry.get_or_create_player(player_id, player_name)
            
            # ⭐ INITIALIZE IF NEW PLAYER (rating == BASE_RATING means uninitialized)
            if player.current_rating == BASE_RATING and player.matches_played == 0:
                self._initialize_player_rating(
                    player,
                    league_id=loaded.match_details.get('league_id'),
                    team_rating=team_context.stats.get('team_rating'),
                    team_id=team_context.team_id
                )
            
            # Update metadata
            position_id = lineup_row.get('position_id')
            usual_position_id = lineup_row.get('usual_position_id')
            player.primary_position_id = usual_position_id or position_id
            player.position_group = get_position_group(
                position_id or 0, 
                usual_position_id
            )
            player.country_code = lineup_row.get('country_code')
            
            if position_id:
                player.record_position(position_id)
            
            minutes = self._calculate_minutes_played(
                player_id, loaded, lineup_row.get('is_starter', False)
            )
            
            performance = self._calculate_individual_performance(
                player_id, team_context, lineup_row
            )
            
            players.append((player, minutes, performance))
        
        return players
    
    def _initialize_player_rating(
        self,
        player: Player,
        league_id: Optional[int],
        team_rating: Optional[float],
        team_id: int
    ) -> None:
        """
        Initialize a new player's rating based on context.
        Called the first time we encounter a player in any match.
        """
        from config.constants import BASE_RATING, INIT_MULTIPLIERS
        
        # Priority 1: Use team rating (most accurate context)
        if team_rating and team_rating > 0:
            base = team_rating
        
        # Priority 2: Use league rating
        elif league_id and league_id in self.registry.leagues:
            base = self.registry.leagues[league_id].current_rating
        
        # Priority 3: Fall back to base rating
        else:
            base = BASE_RATING
        
        # Apply position-based multiplier
        # Players typically start slightly below their team/league rating
        multiplier = INIT_MULTIPLIERS.get(
            'goalkeeper' if player.is_goalkeeper else 'player',
            0.85  # 85% of team/league rating
        )
        
        player.current_rating = base * multiplier
        
        # Log for debugging
        logger.debug(
            f"Initialized player {player.id} ({player.name}) at rating {player.current_rating:.0f} "
            f"(base: {base:.0f}, multiplier: {multiplier})"
        )
    
    def _calculate_minutes_played(
        self,
        player_id: int,
        loaded: LoadedMatchData,
        is_starter: bool
    ) -> int:
        """Calculate minutes played by a player."""
        if not loaded.substitutions:
            return 90 if is_starter else 0
        
        subbed_on = None
        subbed_off = None
        
        for sub in loaded.substitutions:
            if sub.get('player_id') == player_id:
                sub_time = sub.get('substitution_time', 0)
                sub_type = sub.get('substitution_type', '').lower()
                
                if 'in' in sub_type or 'on' in sub_type:
                    subbed_on = sub_time
                elif 'out' in sub_type or 'off' in sub_type:
                    subbed_off = sub_time
        
        if is_starter:
            minutes = subbed_off if subbed_off else 90
        else:
            minutes = 90 - subbed_on if subbed_on else 0
        
        return max(0, min(120, minutes))
    
    def _calculate_individual_performance(
        self,
        player_id: int,
        team_context: TeamMatchContext,
        lineup_row: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate individual performance score using team context."""
        rating = lineup_row.get('rating')
        if rating is not None:
            return (rating - 7.0) / 3.0
        
        if player_id in team_context.player_stats:
            stats = team_context.player_stats[player_id]
            return self._performance_from_stats(stats, lineup_row)
        
        return None
    
    def _performance_from_stats(
        self,
        stats: Dict[str, float],
        lineup_row: Dict[str, Any]
    ) -> float:
        """Calculate performance from stats."""
        score = 0.0
        
        score += stats.get('goals', 0) * 0.3
        score += stats.get('assists', 0) * 0.2
        score += stats.get('chances_created', stats.get('key_passes', 0)) * 0.05
        score += (stats.get('tackles', 0) + stats.get('interceptions', 0)) * 0.02
        score -= stats.get('error_led_to_goal', 0) * 0.4
        score -= stats.get('yellow_cards', 0) * 0.1 + stats.get('red_cards', 0) * 0.3
        
        return max(-1.0, min(1.0, score))
    
    def _get_coach(
        self, 
        loaded: LoadedMatchData, 
        is_home: bool
    ) -> Optional[Coach]:
        """Get coach entity if available."""
        for coach_row in loaded.coaches:
            if coach_row.get('is_home_team') == is_home:
                coach_id = coach_row.get('coach_id')
                if coach_id:
                    coach_name = coach_row.get('coach_name', f'Coach {coach_id}')
                    return self.registry.get_or_create_coach(coach_id, coach_name)
        return None
    
    def _check_player_injured(
        self, 
        player_id: int, 
        loaded: LoadedMatchData
    ) -> bool:
        """Check if player was marked as injured/unavailable for this match."""
        for unavailable in loaded.unavailable_players:
            if unavailable.get('player_id') == player_id:
                # Has injury_id means injured (not suspended or other)
                if unavailable.get('injury_id') is not None:
                    return True
        return False
    
    def _get_player_injury_id(
        self, 
        player_id: int, 
        loaded: LoadedMatchData
    ) -> Optional[int]:
        """Get injury_id if player was injured for this match."""
        for unavailable in loaded.unavailable_players:
            if unavailable.get('player_id') == player_id:
                return unavailable.get('injury_id')
        return None
    
    def _collect_player_match_stats(
        self,
        loaded: LoadedMatchData,
        context: MatchContext,
        home_context: TeamMatchContext,
        away_context: TeamMatchContext
    ) -> List[PlayerMatchStats]:
        """Collect player match stats with ALL required fields populated."""
        from config.constants import get_position_group
        
        stats_list = []
        
        for lineup_row in loaded.lineups:
            player_id = lineup_row.get('player_id')
            if not player_id:
                continue
            
            is_home = lineup_row.get('is_home_team', False)
            player_stats = loaded.player_stats.get(player_id, {})
            
            # Determine team context
            if is_home:
                own_context = home_context
                opp_context = away_context
                team_score = loaded.match_details.get('home_team_score', 0)
                opp_score = loaded.match_details.get('away_team_score', 0)
            else:
                own_context = away_context
                opp_context = home_context
                team_score = loaded.match_details.get('away_team_score', 0)
                opp_score = loaded.match_details.get('home_team_score', 0)
            
            # Calculate match result
            if team_score > opp_score:
                match_result = 'W'
            elif team_score < opp_score:
                match_result = 'L'
            else:
                match_result = 'D'
            
            # Get position group
            position_id = lineup_row.get('position_id')
            usual_position_id = lineup_row.get('usual_position_id')
            position_group = get_position_group(position_id or 0, usual_position_id)
            
            minutes = self._calculate_minutes_played(
                player_id, loaded, lineup_row.get('is_starter', False)
            )
            
            # Add is_captain to stats dict for leadership calculation
            if lineup_row.get('is_captain'):
                player_stats['is_captain'] = True
            
            pms = PlayerMatchStats(
                match_id=loaded.match_id,
                player_id=player_id,
                team_id=lineup_row.get('team_id', 0),
                data_tier=loaded.data_tier,
                minutes_played=minutes,
                stats=player_stats,
                position_id=position_id,
                position_group=position_group,  # NOW POPULATED!
                usual_position_id=usual_position_id,
                is_starter=lineup_row.get('is_starter', False),
                is_captain=lineup_row.get('is_captain', False),
                rating=lineup_row.get('rating'),
                is_home_team=is_home,
                league_id=loaded.match_details.get('league_id'),
                league_name=loaded.match_details.get('league_name'),
                country_code=loaded.league_info.get('CountryName') if loaded.league_info else None,
                own_team_rating=own_context.stats.get('team_rating'),
                opponent_team_rating=opp_context.stats.get('team_rating'),
                team_score=team_score,
                opponent_score=opp_score,
                match_result=match_result,
                was_injured=self._check_player_injured(player_id, loaded),
                injury_id=self._get_player_injury_id(player_id, loaded),
                match_date=loaded.match_details.get('match_time_utc'),
            )
            stats_list.append(pms)
        
        return stats_list
    
    def _collect_coach_match_stats(
        self,
        loaded: LoadedMatchData,
        home_context: TeamMatchContext,
        away_context: TeamMatchContext
    ) -> List[CoachMatchStats]:
        """Collect coach stats with resolved home/away data."""
        stats_list = []
        
        # Extract goal events for adaptability calculation
        goal_events = [
            {
                'minute': e.get('time_minute', 0),
                'is_home_goal': e.get('is_home_team', False),
                'home_score_before': e.get('home_score_before', 0),
                'away_score_before': e.get('away_score_before', 0),
                'home_score_after': e.get('home_score_after', 0),
                'away_score_after': e.get('away_score_after', 0),
                'is_own_goal': e.get('own_goal', False),
            }
            for e in loaded.events
            if e.get('event_type', '').lower() == 'goal'
        ]
        
        for team_context in [home_context, away_context]:
            if team_context.coach_id:
                # Build player ratings and ages from lineup
                player_ratings = {}
                player_ages = {}
                for p in team_context.lineup:
                    pid = p.get('player_id')
                    if pid:
                        if p.get('rating'):
                            player_ratings[pid] = p.get('rating')
                        if p.get('age'):
                            player_ages[pid] = p.get('age')
                
                # Build substitution data with context
                subs_for_team = []
                for s in loaded.substitutions:
                    if s.get('team_id') == team_context.team_id:
                        sub_minute = s.get('substitution_time', 0)
                        # Find score at substitution time
                        score_at_sub = self._get_score_at_minute(
                            goal_events, sub_minute, team_context.is_home_team
                        )
                        subs_for_team.append({
                            'player_in_id': s.get('player_id') if 'in' in s.get('substitution_type', '').lower() else None,
                            'minute': sub_minute,
                            'score_before': score_at_sub,
                        })
                
                cms = CoachMatchStats(
                    coach_id=team_context.coach_id,
                    match_id=loaded.match_id,
                    team_id=team_context.team_id,
                    is_home_team=team_context.is_home_team,
                    data_tier=loaded.data_tier,
                    match_date=loaded.match_details.get('match_time_utc'),
                    team_stats=team_context.stats,
                    formation=team_context.stats.get('formation'),
                    lineup_player_ids=[p.get('player_id') for p in team_context.lineup if p.get('player_id')],
                    substitutions=subs_for_team,
                    player_ratings=player_ratings,
                    player_ages=player_ages,
                    goal_events=goal_events,
                )
                stats_list.append(cms)
        
        return stats_list
    
    def _get_score_at_minute(
        self, 
        goal_events: List[Dict], 
        minute: int,
        is_home_team: bool
    ) -> Tuple[int, int]:
        """Get (own_score, opponent_score) at a given minute."""
        own_score = 0
        opp_score = 0
        
        for event in goal_events:
            if event['minute'] < minute:
                if is_home_team:
                    own_score = event['home_score_after']
                    opp_score = event['away_score_after']
                else:
                    own_score = event['away_score_after']
                    opp_score = event['home_score_after']
        
        return (own_score, opp_score)
    
    def _collect_team_match_stats(
        self,
        loaded: LoadedMatchData,
        home_context: TeamMatchContext,
        away_context: TeamMatchContext
    ) -> List[TeamMatchStats]:
        """Collect team stats with resolved home/away data."""
        result = []
        
        for team_context, opp_context in [
            (home_context, away_context), 
            (away_context, home_context)
        ]:
            # Get lineup breakdown
            all_player_ids = [p.get('player_id') for p in team_context.lineup if p.get('player_id')]
            starter_ids = [p.get('player_id') for p in team_context.lineup 
                          if p.get('player_id') and p.get('is_starter')]
            sub_ids = [p.get('player_id') for p in team_context.lineup 
                      if p.get('player_id') and not p.get('is_starter')]
            
            tms = TeamMatchStats(
                team_id=team_context.team_id,
                match_id=loaded.match_id,
                is_home_team=team_context.is_home_team,
                data_tier=loaded.data_tier,
                stats=team_context.stats,
                opponent_team_id=opp_context.team_id,
                opponent_rating=opp_context.stats.get('team_rating'),
                lineup_player_ids=all_player_ids,
                starter_player_ids=starter_ids,
                sub_player_ids=sub_ids,
            )
            result.append(tms)
        
        return result

    def _collect_shotmap_stats(self, loaded: LoadedMatchData) -> Dict[int, List[Dict]]:
        """Group shotmap data by player_id for attribute calculation."""
        by_player = defaultdict(list)
        for shot in loaded.shotmap:
            pid = shot.get('player_id')
            if pid:
                by_player[pid].append({
                    'match_id': loaded.match_id,
                    'x': shot.get('x'),
                    'y': shot.get('y'),
                    'xg': shot.get('expected_goals'),
                    'is_goal': shot.get('event_type') == 'Goal',
                    'is_from_inside_box': shot.get('is_from_inside_box'),
                    'situation': shot.get('situation'),  # 'FreeKick', 'Penalty', 'OpenPlay', etc.
                    'is_on_target': shot.get('is_on_target'),
                    'is_blocked': shot.get('is_blocked'),
                    'shot_type': shot.get('shot_type'),
                    'minute': shot.get('minute'),
                })
        return dict(by_player)