"""
Match processor - handles loading and processing individual matches.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import logging

from config.constants import DataTier, PositionGroup, POSITION_TO_GROUP
from models.entities import (
    Player, Coach, Team, League, EntityRegistry
)
from processors.data_handlers import (
    DataTierProcessor, MatchDataAvailability, NullHandler,
    PlayerStatsAggregator, StatKeyTracker
)
from database.queries.match_queries import MatchQueries, LeagueQueries
from database.queries.player_queries import PlayerQueries, CoachQueries
from engines.rating_engine import MatchContext, RatingEngine, MatchRatingProcessor
from engines.attribute_engine import PlayerMatchStats

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
    player_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)  # player_id -> stats
    
    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)
    shotmap: List[Dict[str, Any]] = field(default_factory=list)
    substitutions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Coaches
    coaches: List[Dict[str, Any]] = field(default_factory=list)
    
    # Competition context
    league_info: Optional[Dict[str, Any]] = None
    competition_type: str = "League"


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
        
        # Always try to get substitutions if events exist
        if availability.has_events:
            loaded.substitutions = self.match_q.get_match_substitutions(match_id)
        
        # Get match facts if available
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
    
    def process_match(
        self, 
        loaded: LoadedMatchData
    ) -> Dict[str, Any]:
        """
        Process a loaded match.
        
        Returns dict with processing results and any collected stats.
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
            
            # Ensure teams exist in registry
            home_team = self._ensure_team(loaded, is_home=True)
            away_team = self._ensure_team(loaded, is_home=False)
            
            # Process players
            home_players = self._process_players(loaded, home_team.id)
            away_players = self._process_players(loaded, away_team.id)
            
            # Get coaches
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
            
            # Collect player match stats for attribute calculation
            player_match_stats = self._collect_player_match_stats(loaded, context)
            
            result['success'] = True
            result['entities_updated']['teams'] = 2
            result['entities_updated']['players'] = len(home_players) + len(away_players)
            result['entities_updated']['coaches'] = (1 if home_coach else 0) + (1 if away_coach else 0)
            result['player_match_stats'] = player_match_stats
            result['rating_updates'] = updates
            
        except Exception as e:
            logger.error(f"Error processing match {loaded.match_id}: {e}")
            result['errors'].append(str(e))
        
        return result
    
    def _build_match_context(self, loaded: LoadedMatchData) -> MatchContext:
        """Build match context from loaded data."""
        details = loaded.match_details
        
        # Determine division level
        division_level = 1
        if loaded.league_info:
            div = loaded.league_info.get('DivisionLevel')
            if div and isinstance(div, int):
                division_level = div
        
        # Check if knockout
        is_knockout = False
        if loaded.competition_type in ('Cup', 'Continental'):
            # Could enhance this with more specific detection
            is_knockout = True
        
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
        
        # Update league info
        if loaded.league_info:
            team.league_id = loaded.league_info.get('LeagueID')
            team.country_code = loaded.league_info.get('CountryName')
            div = loaded.league_info.get('DivisionLevel')
            team.division_level = div if isinstance(div, int) else 1
        
        return team
    
    def _process_players(
        self, 
        loaded: LoadedMatchData,
        team_id: int
    ) -> List[Tuple[Player, int, Optional[float]]]:
        """
        Process players for a team.
        
        Returns list of (player, minutes_played, individual_performance).
        """
        players = []
        
        # Get players from lineup
        team_lineup = [p for p in loaded.lineups if p.get('team_id') == team_id]
        
        for lineup_row in team_lineup:
            player_id = lineup_row.get('player_id')
            if not player_id:
                continue
            
            # Get or create player
            player_name = lineup_row.get('player_name', f'Player {player_id}')
            player = self.registry.get_or_create_player(player_id, player_name)
            
            # Update player metadata
            player.primary_position_id = lineup_row.get('usual_position_id')
            if player.primary_position_id:
                player.position_group = POSITION_TO_GROUP.get(
                    player.primary_position_id, 
                    PositionGroup.MIDFIELDER
                )
            player.country_code = lineup_row.get('country_code')
            
            # Track position played
            position_played = lineup_row.get('position_id', player.primary_position_id)
            if position_played:
                player.record_position(position_played)
            
            # Calculate minutes played
            minutes = self._calculate_minutes_played(
                player_id, loaded, lineup_row.get('is_starter', False)
            )
            
            # Get individual performance score
            performance = self._calculate_individual_performance(
                player_id, loaded, lineup_row
            )
            
            players.append((player, minutes, performance))
        
        return players
    
    def _calculate_minutes_played(
        self,
        player_id: int,
        loaded: LoadedMatchData,
        is_starter: bool
    ) -> int:
        """Calculate minutes played by a player."""
        if not loaded.substitutions:
            # No sub data - assume full match for starters, 0 for subs
            return 90 if is_starter else 0
        
        minutes = 0
        
        # Find substitution events involving this player
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
            if subbed_off:
                minutes = subbed_off
            else:
                minutes = 90
        else:
            if subbed_on:
                minutes = 90 - subbed_on
            else:
                minutes = 0
        
        return max(0, min(120, minutes))  # Cap at 120 for extra time
    
    def _calculate_individual_performance(
        self,
        player_id: int,
        loaded: LoadedMatchData,
        lineup_row: Dict[str, Any]
    ) -> Optional[float]:
        """
        Calculate individual performance score.
        
        Returns value around 0 (negative = below average, positive = above).
        """
        # If we have a rating, use it
        rating = lineup_row.get('rating')
        if rating is not None:
            # Normalize: 7.0 is average, range usually 5-10
            return (rating - 7.0) / 3.0  # Normalized to roughly -0.67 to +1.0
        
        # If we have player stats, calculate from those
        if player_id in loaded.player_stats:
            stats = loaded.player_stats[player_id]
            return self._performance_from_stats(stats, lineup_row)
        
        # No individual data available
        return None
    
    def _performance_from_stats(
        self,
        stats: Dict[str, float],
        lineup_row: Dict[str, Any]
    ) -> float:
        """Calculate performance from stats."""
        score = 0.0
        
        # Goals are very positive
        goals = stats.get('goals', 0)
        score += goals * 0.3
        
        # Assists are positive
        assists = stats.get('assists', 0)
        score += assists * 0.2
        
        # Key passes / chances created
        chances = stats.get('chances_created', stats.get('key_passes', 0))
        score += chances * 0.05
        
        # Defensive contributions (for defenders/midfielders)
        tackles = stats.get('tackles', 0)
        interceptions = stats.get('interceptions', 0)
        score += (tackles + interceptions) * 0.02
        
        # Negative: errors, cards
        errors = stats.get('error_led_to_goal', 0)
        score -= errors * 0.4
        
        yellow = stats.get('yellow_cards', 0)
        red = stats.get('red_cards', 0)
        score -= yellow * 0.1 + red * 0.3
        
        return max(-1.0, min(1.0, score))  # Clamp to reasonable range
    
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
    
    def _collect_player_match_stats(
        self,
        loaded: LoadedMatchData,
        context: MatchContext
    ) -> List[PlayerMatchStats]:
        """Collect player match stats for attribute engine."""
        stats_list = []
        
        for lineup_row in loaded.lineups:
            player_id = lineup_row.get('player_id')
            if not player_id:
                continue
            
            # Get stats dict for this player
            stats = loaded.player_stats.get(player_id, {})
            
            # Calculate minutes
            minutes = self._calculate_minutes_played(
                player_id, loaded, lineup_row.get('is_starter', False)
            )
            
            pms = PlayerMatchStats(
                match_id=loaded.match_id,
                player_id=player_id,
                team_id=lineup_row.get('team_id', 0),
                data_tier=loaded.data_tier,
                minutes_played=minutes,
                stats=stats,
                position_id=lineup_row.get('position_id'),
                is_starter=lineup_row.get('is_starter', False),
                rating=lineup_row.get('rating')
            )
            stats_list.append(pms)
        
        return stats_list