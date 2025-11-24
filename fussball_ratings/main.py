"""
Main entry point for the Fussball Rating System.

Usage:
    python main.py --full           # Full historical processing
    python main.py --incremental    # Process new matches only
    python main.py --recalculate    # Recalculate all attributes
    python main.py --player 12345   # Show player profile
    python main.py --team 678       # Show team profile
"""
import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

from config.constants import DataTier
from database.connection import DatabaseConfig, ConnectionManager
from processors.pipeline import ProcessingPipeline, IncrementalPipeline, PipelineConfig
from database.repositories.rating_repository import RatingRepository, AttributeRepository
from database.queries.player_queries import PlayerQueries
from database.queries.match_queries import MatchQueries


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fussball_ratings.log')
    ]
)
logger = logging.getLogger(__name__)


def get_db_config() -> DatabaseConfig:
    """Get database configuration."""
    # In production, load from environment or config file
    return DatabaseConfig(
        server="localhost",  # Update with your server
        database="fussballDB",
        trusted_connection=True
    )


def run_full_processing(reset: bool = False):
    """Run full historical processing."""
    logger.info("Starting full historical processing...")
    
    config = PipelineConfig(
        batch_size=100,
        save_interval=50,
        log_interval=25,
        process_ratings=True,
        process_attributes=True,
        min_tier_for_ratings=DataTier.MINIMAL,
        min_tier_for_attributes=DataTier.FULL,
        recalculate_attributes_every=500,
        resume_from_last=not reset
    )
    
    db_config = get_db_config()
    pipeline = ProcessingPipeline(config, db_config)
    
    try:
        if reset:
            logger.warning("Resetting processing state...")
            pipeline.state_repo.reset_processing_state()
        
        stats = pipeline.run(
            progress_callback=lambda cur, tot: None  # Could hook up progress bar
        )
        
        logger.info(f"Processing complete: {stats.matches_processed} matches in {stats.duration_seconds:.1f}s")
        
    finally:
        pipeline.close()


def run_incremental_processing(since_date: Optional[datetime] = None):
    """Run incremental processing for new matches."""
    logger.info("Starting incremental processing...")
    
    config = PipelineConfig(
        batch_size=50,
        save_interval=25,
        log_interval=10,
        process_ratings=True,
        process_attributes=True,
        resume_from_last=True
    )
    
    db_config = get_db_config()
    pipeline = IncrementalPipeline(config, db_config)
    
    try:
        stats = pipeline.run_incremental(since_date)
        logger.info(f"Incremental processing complete: {stats.matches_processed} new matches")
        
    finally:
        pipeline.close()


def show_player_profile(player_id: int):
    """Display a player's rating and attributes."""
    db_config = get_db_config()
    conn_manager = ConnectionManager()
    conn_manager.initialize(db_config)
    
    try:
        executor = conn_manager.executor
        rating_repo = RatingRepository(executor)
        attr_repo = AttributeRepository(executor)
        player_queries = PlayerQueries(executor)
        
        # Get player info
        history = player_queries.get_player_match_history(player_id, limit=1)
        if not history:
            print(f"Player {player_id} not found")
            return
        
        # Get current rating
        rating = rating_repo.get_player_current_rating(player_id)
        
        # Get attributes
        attributes = attr_repo.get_player_attributes(player_id)
        
        # Display
        print("\n" + "=" * 60)
        print(f"PLAYER PROFILE: {player_id}")
        print("=" * 60)
        print(f"Current Rating: {rating:.0f}" if rating else "Rating: Not calculated")
        print(f"Matches in DB: {len(player_queries.get_player_match_history(player_id))}")
        
        if attributes:
            print("\nAttributes (0-20 scale):")
            print("-" * 40)
            
            # Sort by category
            sorted_attrs = sorted(attributes.items(), key=lambda x: x[0])
            for code, data in sorted_attrs:
                value = data['value']
                confidence = data['confidence']
                matches = data['matches_used']
                conf_str = "★" * int(confidence * 5)
                print(f"  {code:12} {value:5.1f}  [{conf_str:5}] ({matches} matches)")
        else:
            print("\nNo attributes calculated yet")
        
        print("=" * 60 + "\n")
        
    finally:
        conn_manager.close()


def show_team_profile(team_id: int):
    """Display a team's rating and attributes."""
    db_config = get_db_config()
    conn_manager = ConnectionManager()
    conn_manager.initialize(db_config)
    
    try:
        executor = conn_manager.executor
        rating_repo = RatingRepository(executor)
        attr_repo = AttributeRepository(executor)
        
        # Get current rating
        rating = rating_repo.get_team_current_rating(team_id)
        
        # Get attributes
        attributes = attr_repo.get_team_attributes(team_id)
        
        # Display
        print("\n" + "=" * 60)
        print(f"TEAM PROFILE: {team_id}")
        print("=" * 60)
        print(f"Current Rating: {rating:.0f}" if rating else "Rating: Not calculated")
        
        if attributes:
            print("\nAttributes (0-20 scale):")
            print("-" * 40)
            for code, data in sorted(attributes.items()):
                value = data['value']
                matches = data['matches_used']
                print(f"  {code:15} {value:5.1f}  ({matches} matches)")
        else:
            print("\nNo attributes calculated yet")
        
        print("=" * 60 + "\n")
        
    finally:
        conn_manager.close()


def show_leaderboard(attribute: str, limit: int = 20):
    """Show top players for an attribute."""
    db_config = get_db_config()
    conn_manager = ConnectionManager()
    conn_manager.initialize(db_config)
    
    try:
        executor = conn_manager.executor
        attr_repo = AttributeRepository(executor)
        
        leaders = attr_repo.get_attribute_leaderboard(attribute, limit=limit)
        
        print("\n" + "=" * 60)
        print(f"TOP {limit} PLAYERS: {attribute.upper()}")
        print("=" * 60)
        
        for i, player in enumerate(leaders, 1):
            name = player.get('player_name', f"Player {player['player_id']}")
            value = player['value']
            confidence = player['confidence']
            print(f"{i:3}. {name:30} {value:5.1f}  (conf: {confidence:.2f})")
        
        print("=" * 60 + "\n")
        
    finally:
        conn_manager.close()


def show_stats():
    """Show processing statistics."""
    db_config = get_db_config()
    conn_manager = ConnectionManager()
    conn_manager.initialize(db_config)
    
    try:
        executor = conn_manager.executor
        
        # Count processed matches
        processed = executor.execute_scalar(
            "SELECT COUNT(DISTINCT match_id) FROM processing_log WHERE success = 1"
        )
        
        # Count entities
        players = executor.execute_scalar(
            "SELECT COUNT(DISTINCT player_id) FROM player_ratings"
        )
        teams = executor.execute_scalar(
            "SELECT COUNT(DISTINCT team_id) FROM team_ratings"
        )
        coaches = executor.execute_scalar(
            "SELECT COUNT(DISTINCT coach_id) FROM coach_ratings"
        )
        
        # Total matches in DB
        total_matches = executor.execute_scalar(
            "SELECT COUNT(*) FROM match_details WHERE finished = 1"
        )
        
        print("\n" + "=" * 60)
        print("PROCESSING STATISTICS")
        print("=" * 60)
        print(f"Total matches in database: {total_matches:,}")
        print(f"Matches processed:         {processed:,}")
        print(f"Processing coverage:       {100*processed/total_matches:.1f}%")
        print("-" * 40)
        print(f"Players with ratings:      {players:,}")
        print(f"Teams with ratings:        {teams:,}")
        print(f"Coaches with ratings:      {coaches:,}")
        print("=" * 60 + "\n")
        
    finally:
        conn_manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fussball Rating System - Process matches and calculate ratings/attributes"
    )
    
    # Processing modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', 
                       help='Run full historical processing')
    group.add_argument('--incremental', action='store_true',
                       help='Process new matches only')
    group.add_argument('--reset', action='store_true',
                       help='Reset and reprocess everything')
    
    # Query modes
    parser.add_argument('--player', type=int, metavar='ID',
                       help='Show player profile')
    parser.add_argument('--team', type=int, metavar='ID',
                       help='Show team profile')
    parser.add_argument('--leaderboard', type=str, metavar='ATTR',
                       help='Show leaderboard for attribute')
    parser.add_argument('--stats', action='store_true',
                       help='Show processing statistics')
    
    # Options
    parser.add_argument('--since', type=str, metavar='DATE',
                       help='Process matches since date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=20,
                       help='Limit for leaderboard display')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute requested action
    if args.full:
        run_full_processing(reset=False)
    elif args.reset:
        response = input("This will reset all ratings. Are you sure? (yes/no): ")
        if response.lower() == 'yes':
            run_full_processing(reset=True)
        else:
            print("Cancelled")
    elif args.incremental:
        since = datetime.strptime(args.since, '%Y-%m-%d') if args.since else None
        run_incremental_processing(since)
    elif args.player:
        show_player_profile(args.player)
    elif args.team:
        show_team_profile(args.team)
    elif args.leaderboard:
        show_leaderboard(args.leaderboard, args.limit)
    elif args.stats:
        show_stats()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()