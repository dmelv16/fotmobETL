"""
Main processing pipeline for batch processing and orchestration.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import logging
import time

from config.constants import DataTier
from models.entities import EntityRegistry, Player
from processors.data_handlers import StatKeyTracker
from processors.match_processor import MatchDataLoader, MatchProcessor, LoadedMatchData
from engines.rating_engine import RatingEngine
from engines.attribute_engine import (
    AttributeEngine, PlayerMatchStats, CoachMatchStats, TeamMatchStats
)
from database.connection import ConnectionManager, DatabaseConfig
from database.queries.match_queries import MatchQueries, LeagueQueries
from database.queries.player_queries import PlayerQueries, CoachQueries
from database.repositories.rating_repository import (
    RatingRepository, AttributeRepository, ProcessingStateRepository
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""
    batch_size: int = 100
    save_interval: int = 50  # Save to DB every N matches
    log_interval: int = 10   # Log progress every N matches
    
    # What to process
    process_ratings: bool = True
    process_attributes: bool = True
    
    # Data tier filters
    min_tier_for_ratings: DataTier = DataTier.MINIMAL
    min_tier_for_attributes: DataTier = DataTier.FULL
    
    # Attribute calculation settings
    recalculate_attributes_every: int = 100  # Recalc attributes every N matches
    
    # Resume settings
    resume_from_last: bool = True


@dataclass
class PipelineStats:
    """Statistics for a pipeline run."""
    matches_processed: int = 0
    matches_skipped: int = 0
    matches_failed: int = 0
    
    teams_updated: int = 0
    players_updated: int = 0
    coaches_updated: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def matches_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.matches_processed / self.duration_seconds
        return 0.0


class ProcessingPipeline:
    """
    Main orchestration pipeline for processing matches.
    
    Handles:
    - Loading matches in chronological order
    - Processing ratings
    - Collecting stats for attributes
    - Periodic attribute recalculation
    - Persisting results to database
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        db_config: DatabaseConfig
    ):
        self.config = config
        self.db_config = db_config
        
        # Initialize connection
        self.conn_manager = ConnectionManager()
        self.conn_manager.initialize(db_config)
        executor = self.conn_manager.executor
        
        # Initialize queries
        self.match_queries = MatchQueries(executor)
        self.player_queries = PlayerQueries(executor)
        self.coach_queries = CoachQueries(executor)
        self.league_queries = LeagueQueries(executor)
        
        # Initialize repositories
        self.rating_repo = RatingRepository(executor)
        self.attribute_repo = AttributeRepository(executor)
        self.state_repo = ProcessingStateRepository(executor)
        
        # Initialize engines
        self.entity_registry = EntityRegistry()
        self.rating_engine = RatingEngine()
        self.attribute_engine = AttributeEngine()
        self.stat_tracker = StatKeyTracker()
        
        # Initialize loader and processor
        self.data_loader = MatchDataLoader(
            self.match_queries,
            self.player_queries,
            self.coach_queries,
            self.league_queries
        )
        self.match_processor = MatchProcessor(
            self.entity_registry,
            self.rating_engine,
            self.stat_tracker
        )
        
        # Stats collection for attribute calculation
        self.player_stats_history: Dict[int, List[PlayerMatchStats]] = defaultdict(list)
        self.coach_stats_history: Dict[int, List[CoachMatchStats]] = defaultdict(list)
        self.team_stats_history: Dict[int, List[TeamMatchStats]] = defaultdict(list)
        
        # Pipeline stats
        self.stats = PipelineStats()
    
    def run(
        self,
        match_ids: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> PipelineStats:
        """Run the processing pipeline with proper initialization."""
        self.stats = PipelineStats()
        self.stats.start_time = datetime.now()
        
        try:
            
            # Get matches to process
            if match_ids is None:
                match_ids = self._get_matches_to_process()
            
            total_matches = len(match_ids)
            logger.info(f"Starting pipeline with {total_matches} matches")

            if self.config.resume_from_last:
                self._initialize_existing_players()

            # STEP 1: Bootstrap phase - collect stats without calculating attributes
            if self.config.process_attributes and not self.attribute_engine._bootstrapped:
                logger.info("Bootstrapping percentile distributions...")
                bootstrap_count = min(1000, total_matches // 10)  # Use 10% or 1000 matches
                
                for match_id in match_ids[:bootstrap_count]:
                    loaded = self.data_loader.load_match(match_id)
                    if loaded and loaded.data_tier.value >= DataTier.FULL.value:
                        # Just collect stats, don't process ratings yet
                        context = self.match_processor._build_match_context(loaded)
                        home_ctx = self.match_processor._build_team_context(loaded, True)
                        away_ctx = self.match_processor._build_team_context(loaded, False)
                        
                        for pms in self.match_processor._collect_player_match_stats(
                            loaded, context, home_ctx, away_ctx
                        ):
                            self.player_stats_history[pms.player_id].append(pms)
                
                # Now bootstrap
                self.attribute_engine.bootstrap_percentiles(
                    self.player_stats_history,
                    self.entity_registry.players
                )
            
            # STEP 2: Process all matches
            for i, match_id in enumerate(match_ids):
                try:
                    self._process_single_match(match_id)
                    
                    # Periodic league rating updates
                    if self.stats.matches_processed > 0 and self.stats.matches_processed % 100 == 0:
                        self._update_league_ratings()
                    
                    # Periodic attribute recalculation
                    if (self.config.process_attributes and 
                        self.stats.matches_processed > 0 and
                        self.stats.matches_processed % self.config.recalculate_attributes_every == 0):
                        self._recalculate_attributes()
                    
                    # Periodic save
                    if self.stats.matches_processed % self.config.save_interval == 0:
                        self._save_state()
                    
                    # Progress
                    if self.stats.matches_processed % self.config.log_interval == 0:
                        self._log_progress(i + 1, total_matches)
                    
                    if progress_callback:
                        progress_callback(i + 1, total_matches)
                        
                except Exception as e:
                    logger.error(f"Error processing match {match_id}: {e}")
                    self.stats.matches_failed += 1
                    self.stats.errors.append(f"Match {match_id}: {str(e)}")
            
            # STEP 3: Final updates
            self._update_league_ratings()
            
            if self.config.process_attributes:
                self._recalculate_attributes()
            
            self._save_state()
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats.errors.append(f"Pipeline: {str(e)}")
            
        finally:
            self.stats.end_time = datetime.now()
            self._log_final_stats()
        
        return self.stats

    def _initialize_existing_players(self) -> None:
        """
        Initialize ratings for players that already have data.
        Call this when resuming processing or loading from database.
        """
        logger.info("Initializing existing player ratings...")
        
        # Get all unique players from database
        unique_players = self.player_queries.get_all_unique_players()
        
        for player_data in unique_players:
            player_id = player_data['player_id']
            
            # Check if we already have a rating
            existing_rating = self.rating_repo.get_player_current_rating(player_id)
            
            if existing_rating:
                # Load existing rating
                player = self.entity_registry.get_or_create_player(
                    player_id,
                    player_data['player_name']
                )
                player.current_rating = existing_rating
                
                # Get match count
                history = self.player_queries.get_player_match_history(player_id, limit=1)
                if history:
                    all_history = self.player_queries.get_player_match_history(player_id)
                    player.matches_played = len(all_history)
            
            else:
                # Player exists in DB but has no rating yet
                # Will be initialized when first encountered in _process_players
                pass
        
        logger.info(f"Initialized {len(unique_players)} existing players")

    def _get_matches_to_process(self) -> List[int]:
        """Get list of matches to process in chronological order."""
        if self.config.resume_from_last:
            # Get unprocessed matches
            unprocessed = self.state_repo.get_unprocessed_matches('ratings')
            logger.info(f"Found {len(unprocessed)} unprocessed matches")
            return unprocessed
        else:
            # Get all matches chronologically
            all_matches = self.match_queries.get_all_matches_chronological()
            return [m['match_id'] for m in all_matches]
    
    def _process_single_match(self, match_id: int):
        """Process a single match."""
        # Load match data
        loaded = self.data_loader.load_match(match_id)
        
        if not loaded:
            logger.warning(f"Could not load match {match_id}")
            self.stats.matches_skipped += 1
            return
        
        # Check tier requirements
        if loaded.data_tier.value < self.config.min_tier_for_ratings.value:
            logger.debug(f"Match {match_id} below min tier, skipping")
            self.stats.matches_skipped += 1
            return
        
        # Process match
        result = self.match_processor.process_match(loaded)
        
        if result['success']:
            self.stats.matches_processed += 1
            self.stats.teams_updated += result['entities_updated']['teams']
            self.stats.players_updated += result['entities_updated']['players']
            self.stats.coaches_updated += result['entities_updated']['coaches']
            
            # Collect player stats for attributes
            if self.config.process_attributes and 'player_match_stats' in result:
                for pms in result['player_match_stats']:
                    self.player_stats_history[pms.player_id].append(pms)
            
            # Collect coach stats for attributes
            if self.config.process_attributes and 'coach_match_stats' in result:
                for cms in result['coach_match_stats']:
                    self.coach_stats_history[cms.coach_id].append(cms)
            
            # Collect team stats for attributes
            if self.config.process_attributes and 'team_match_stats' in result:
                for tms in result['team_match_stats']:
                    self.team_stats_history[tms.team_id].append(tms)
            
            # Mark as processed
            self.state_repo.mark_match_processed(match_id, 'ratings', True)
        else:
            self.stats.matches_failed += 1
            for error in result.get('errors', []):
                self.stats.errors.append(f"Match {match_id}: {error}")
            self.state_repo.mark_match_processed(match_id, 'ratings', False, 
                                                  '; '.join(result.get('errors', [])))

    def _update_league_ratings(self) -> None:
        """Update all league ratings based on their teams."""
        from collections import defaultdict
        
        # Group teams by league
        teams_by_league: Dict[int, List[float]] = defaultdict(list)
        
        for team_id, team in self.entity_registry.teams.items():
            if team.league_id:
                teams_by_league[team.league_id].append(team.current_rating)
        
        # Update each league
        for league_id, team_ratings in teams_by_league.items():
            if not team_ratings:
                continue
            
            league = self.entity_registry.get_or_create_league(
                league_id, 
                f"League {league_id}"
            )
            
            update = self.rating_engine.update_league_rating(
                league,
                team_ratings,
                continental_results=None  # Add if you track continental performance
            )
            
            # Apply update
            league.current_rating = update.new_rating

    def _recalculate_attributes(self) -> None:
        """Recalculate attributes with league normalization."""
        logger.info("Recalculating attributes...")
        
        # Get current league ratings
        league_ratings = {
            league_id: league.current_rating 
            for league_id, league in self.entity_registry.leagues.items()
        }
        
        players_updated = 0
        coaches_updated = 0
        teams_updated = 0
        
        # Player attributes with normalization
        for player_id, stats_history in self.player_stats_history.items():
            if player_id not in self.entity_registry.players:
                continue
            
            player = self.entity_registry.players[player_id]
            
            new_attributes = self.attribute_engine.calculate_player_attributes(
                player, 
                stats_history,
                league_ratings=league_ratings  # Pass league ratings!
            )
            
            if new_attributes:
                player.attributes = new_attributes
                players_updated += 1
        
        # Coach attributes
        for coach_id, stats_history in self.coach_stats_history.items():
            if coach_id not in self.entity_registry.coaches:
                continue
            
            coach = self.entity_registry.coaches[coach_id]
            new_attributes = self.attribute_engine.calculate_coach_attributes(
                coach, stats_history
            )
            
            if new_attributes:
                coach.attributes = new_attributes
                coaches_updated += 1
        
        # Team attributes
        for team_id, stats_history in self.team_stats_history.items():
            if team_id not in self.entity_registry.teams:
                continue
            
            team = self.entity_registry.teams[team_id]
            new_attributes = self.attribute_engine.calculate_team_attributes(
                team, stats_history
            )
            
            if new_attributes:
                team.attributes = new_attributes
                teams_updated += 1
        
        logger.info(f"Updated: {players_updated} players, {coaches_updated} coaches, {teams_updated} teams")
    
    def _save_state(self):
        """Save current state to database."""
        logger.info("Saving state to database...")
        
        # Save player ratings
        player_ratings = []
        for player_id, player in self.entity_registry.players.items():
            if player.rating_history:
                latest = player.rating_history[-1]
                player_ratings.append({
                    'player_id': player_id,
                    'match_id': latest.match_id,
                    'rating': player.current_rating,
                    'rating_change': latest.change,
                    'k_factor': latest.k_factor_used,
                    'data_tier': latest.data_tier.value,
                    'competition_multiplier': latest.competition_multiplier
                })
        
        if player_ratings:
            self.rating_repo.save_player_ratings_batch(player_ratings)
        
        # Save team ratings
        for team_id, team in self.entity_registry.teams.items():
            if team.rating_history:
                latest = team.rating_history[-1]
                self.rating_repo.save_team_rating(
                    team_id=team_id,
                    match_id=latest.match_id,
                    rating=team.current_rating,
                    rating_change=latest.change,
                    expected_result=latest.expected_result,
                    actual_result=latest.actual_result,
                    k_factor=latest.k_factor_used,
                    data_tier=latest.data_tier.value
                )
        
        # Save coach ratings
        for coach_id, coach in self.entity_registry.coaches.items():
            if coach.rating_history:
                latest = coach.rating_history[-1]
                self.rating_repo.save_coach_rating(
                    coach_id=coach_id,
                    match_id=latest.match_id,
                    rating=coach.current_rating,
                    rating_change=latest.change,
                    k_factor=latest.k_factor_used,
                    data_tier=latest.data_tier.value
                )
        
        # Save player attributes
        for player_id, player in self.entity_registry.players.items():
            if player.attributes:
                attrs_dict = {
                    code: {
                        'value': score.value,
                        'confidence': score.confidence,
                        'matches_used': score.matches_used,
                        'data_tiers_used': [t.value for t in score.data_tiers_used]
                    }
                    for code, score in player.attributes.items()
                }
                self.attribute_repo.save_player_attributes_batch(player_id, attrs_dict)
        
        # Save coach attributes
        for coach_id, coach in self.entity_registry.coaches.items():
            if coach.attributes:
                for code, score in coach.attributes.items():
                    self.attribute_repo.save_coach_attribute(
                        coach_id=coach_id,
                        attribute_code=code,
                        value=score.value,
                        matches_used=score.matches_used
                    )
        
        # Save team attributes
        for team_id, team in self.entity_registry.teams.items():
            if team.attributes:
                for code, score in team.attributes.items():
                    self.attribute_repo.save_team_attribute(
                        team_id=team_id,
                        attribute_code=code,
                        value=score.value,
                        matches_used=score.matches_used
                    )
        
        logger.info("State saved")
    
    def _log_progress(self, current: int, total: int):
        """Log processing progress."""
        elapsed = (datetime.now() - self.stats.start_time).total_seconds()
        rate = self.stats.matches_processed / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"Progress: {current}/{total} matches "
            f"({self.stats.matches_processed} processed, "
            f"{self.stats.matches_failed} failed, "
            f"{rate:.1f} matches/sec)"
        )
    
    def _log_final_stats(self):
        """Log final statistics."""
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Duration: {self.stats.duration_seconds:.1f} seconds")
        logger.info(f"Matches processed: {self.stats.matches_processed}")
        logger.info(f"Matches skipped: {self.stats.matches_skipped}")
        logger.info(f"Matches failed: {self.stats.matches_failed}")
        logger.info(f"Rate: {self.stats.matches_per_second:.1f} matches/sec")
        logger.info(f"Teams updated: {self.stats.teams_updated}")
        logger.info(f"Players updated: {self.stats.players_updated}")
        logger.info(f"Coaches updated: {self.stats.coaches_updated}")
        
        if self.stats.errors:
            logger.warning(f"Errors ({len(self.stats.errors)}):")
            for error in self.stats.errors[:10]:  # Show first 10
                logger.warning(f"  - {error}")
        
        logger.info("=" * 60)
    
    def close(self):
        """Clean up resources."""
        self.conn_manager.close()


class IncrementalPipeline(ProcessingPipeline):
    """
    Pipeline for incremental updates (new matches only).
    """
    
    def run_incremental(
        self,
        since_date: Optional[datetime] = None
    ) -> PipelineStats:
        """Run pipeline for matches since a given date."""
        if since_date is None:
            since_date = self.state_repo.get_last_processed_date('ratings')
        
        if since_date:
            logger.info(f"Processing matches since {since_date}")
            matches = self.match_queries.get_matches_in_date_range(
                start_date=since_date,
                end_date=datetime.now()
            )
            match_ids = [m['match_id'] for m in matches]
        else:
            match_ids = None  # Process all
        
        return self.run(match_ids)