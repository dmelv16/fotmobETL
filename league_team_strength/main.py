"""
Main orchestration script for league and team strength calculation system.
Run this to process seasons and update all strength metrics.
"""

import logging
from datetime import datetime
from typing import List, Optional
import sys

from config.database import DatabaseConfig
from config.league_mappings import TIER_1_LEAGUES, TIER_2_LEAGUES, TIER_2_5_LEAGUES
from core.database_manager import StrengthDatabaseManager
from core.data_loader import FotMobDataLoader
from models.elo_rating import EloRatingSystem
from models.transfer_analyzer import TransferPerformanceAnalyzer
from models.european_results import EuropeanResultsAnalyzer
from models.xg_performance import XGPerformanceRater
from models.squad_quality import SquadQualityEvaluator
from models.style_profiler import StyleProfiler
from calculators.league_strength import LeagueStrengthCalculator
from calculators.team_strength import TeamStrengthCalculator
from calculators.cross_league_network import CrossLeagueNetworkAnalyzer
from calculators.temporal_tracker import TemporalStrengthTracker
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

class StrengthCalculationOrchestrator:
    """
    Main orchestrator for the entire strength calculation pipeline.
    """
    
    def __init__(self, connection_string: str):
        self.db_config = DatabaseConfig(connection_string)
        self.conn = None
        
        # Initialize components (will be set up in initialize())
        self.data_loader = None
        self.elo_system = None
        self.transfer_analyzer = None
        self.european_analyzer = None
        self.xg_rater = None
        self.squad_evaluator = None
        self.style_profiler = None
        self.league_calculator = None
        self.team_calculator = None
        self.network_analyzer = None
        self.temporal_tracker = None
    
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing strength calculation system...")
        
        # Test database connection
        if not self.db_config.test_connection():
            raise Exception("Failed to connect to database")
        
        self.conn = self.db_config.get_connection()
        
        # Create tables
        db_manager = StrengthDatabaseManager(self.conn)
        db_manager.create_all_tables()
        
        # Initialize data loader
        self.data_loader = FotMobDataLoader(self.conn)
        
        # Initialize models
        self.elo_system = EloRatingSystem(self.conn, self.data_loader)
        self.transfer_analyzer = TransferPerformanceAnalyzer(self.conn, self.data_loader)
        self.european_analyzer = EuropeanResultsAnalyzer(self.conn, self.data_loader)
        self.xg_rater = XGPerformanceRater(self.conn, self.data_loader)
        self.squad_evaluator = SquadQualityEvaluator(self.conn, self.data_loader)
        self.style_profiler = StyleProfiler(self.conn, self.data_loader)
        
        # Initialize calculators
        self.league_calculator = LeagueStrengthCalculator(
            self.conn, self.transfer_analyzer, self.european_analyzer, self.data_loader
        )
        self.team_calculator = TeamStrengthCalculator(
            self.conn, self.elo_system, self.xg_rater, self.squad_evaluator, self.data_loader
        )
        self.network_analyzer = CrossLeagueNetworkAnalyzer(self.conn)
        self.temporal_tracker = TemporalStrengthTracker(self.conn)
        
        logger.info("System initialized successfully")
    
    def process_season(self, season_year: int, league_ids: Optional[List[int]] = None,
                      skip_existing: bool = True):
        """
        Process a complete season: calculate all strength metrics.
        
        Pipeline:
        1. Process Elo ratings (match-by-match)
        2. Analyze transfers between leagues
        3. Process European competition results
        4. Calculate league strengths
        5. Calculate team strengths
        6. Generate style profiles
        7. Fill network gaps
        """
        logger.info(f"=" * 80)
        logger.info(f"Processing season {season_year}")
        logger.info(f"=" * 80)
        
        if league_ids is None:
            # Default to all configured leagues
            league_ids = (list(TIER_1_LEAGUES.keys()) + 
                         list(TIER_2_LEAGUES.keys()) + 
                         list(TIER_2_5_LEAGUES.keys()))
        
        try:
            # Step 1: Calculate Elo ratings for all leagues
            logger.info("Step 1: Calculating Elo ratings...")
            for league_id in league_ids:
                try:
                    self.elo_system.process_season(season_year, league_id, save_to_db=True)
                except Exception as e:
                    logger.error(f"Error processing Elo for league {league_id}: {e}")
            
            # Step 2: Analyze transfer performance
            logger.info("Step 2: Analyzing transfer performance...")
            transfer_matrix = self.transfer_analyzer.build_transfer_matrix(
                season_year, league_ids, lookback_seasons=3
            )
            if len(transfer_matrix) > 0:
                self.transfer_analyzer.save_to_database(transfer_matrix, season_year)
                logger.info(f"Saved {len(transfer_matrix)} transfer relationships")
            else:
                logger.warning("No transfer relationships found")
            
            # Step 3: Process European competition results
            logger.info("Step 3: Processing European competition results...")
            self.european_analyzer.process_season(season_year, save_to_db=True)
            
            # Step 4: Calculate league strengths
            logger.info("Step 4: Calculating league strengths...")
            league_results = self.league_calculator.calculate_all_leagues(season_year, league_ids)
            if len(league_results) > 0:
                logger.info(f"Calculated strength for {len(league_results)} leagues")
                logger.info("\nTop 10 Leagues:")
                for i, row in league_results.head(10).iterrows():
                    logger.info(f"  {i+1}. {row['league_name']}: {row['overall_strength']:.1f}")
            else:
                logger.warning("No league strengths calculated")
            
            # Step 5: Calculate team strengths for each league
            logger.info("Step 5: Calculating team strengths...")
            total_teams = 0
            for league_id in league_ids:
                try:
                    team_results = self.team_calculator.calculate_league_teams(league_id, season_year)
                    total_teams += len(team_results)
                    if len(team_results) > 0:
                        logger.info(f"  League {league_id}: {len(team_results)} teams processed")
                except Exception as e:
                    logger.error(f"Error processing teams for league {league_id}: {e}")
            
            logger.info(f"Calculated strength for {total_teams} teams total")
            
            # Step 6: Generate style profiles
            logger.info("Step 6: Generating style profiles...")
            
            # League profiles
            league_profiles = []
            for league_id in league_ids:
                try:
                    profile = self.style_profiler.calculate_league_style_profile(league_id, season_year)
                    if profile:
                        league_profiles.append(profile)
                except Exception as e:
                    logger.error(f"Error generating style profile for league {league_id}: {e}")
            
            if league_profiles:
                self.style_profiler.save_profiles_to_database(league_profiles)
                logger.info(f"Saved {len(league_profiles)} league style profiles")
            
            # Team profiles with clustering
            team_profiles = []
            for league_id in league_ids:
                try:
                    team_ids = self.data_loader.get_league_teams_for_season(league_id, season_year)
                    for team_id in team_ids:
                        profile = self.style_profiler.calculate_team_style_profile(team_id, season_year)
                        if profile:
                            team_profiles.append(profile)
                except Exception as e:
                    logger.error(f"Error generating team profiles for league {league_id}: {e}")
            
            if len(team_profiles) > 5:  # Need minimum for clustering
                try:
                    team_profiles_df = self.style_profiler.cluster_teams_by_style(team_profiles)
                    team_profiles_with_clusters = team_profiles_df.to_dict('records')
                    self.style_profiler.save_profiles_to_database(team_profiles_with_clusters)
                    logger.info(f"Saved {len(team_profiles_with_clusters)} team style profiles with clusters")
                except Exception as e:
                    logger.error(f"Error clustering team profiles: {e}")
            
            # Step 7: Fill network gaps using inference
            logger.info("Step 7: Filling network gaps...")
            inferred_edges = self.network_analyzer.fill_missing_edges(season_year, max_path_length=3)
            logger.info(f"Inferred {len(inferred_edges)} missing league relationships")
            
            logger.info(f"Season {season_year} processing complete!")
            
        except Exception as e:
            logger.error(f"Error processing season {season_year}: {e}", exc_info=True)
            raise
    
    def process_multiple_seasons(self, start_season: int, end_season: int,
                                league_ids: Optional[List[int]] = None):
        """
        Process multiple seasons in chronological order.
        """
        for season_year in range(start_season, end_season + 1):
            try:
                self.process_season(season_year, league_ids)
            except Exception as e:
                logger.error(f"Failed to process season {season_year}: {e}")
                # Continue with next season
                continue
    
    def generate_reports(self, season_year: int):
        """
        Generate summary reports for a season.
        """
        from outputs.report_generator import ReportGenerator
        
        report_gen = ReportGenerator(self.conn, self.temporal_tracker)
        
        # League strength report
        logger.info("Generating league strength report...")
        league_report = report_gen.generate_league_strength_report(season_year)
        print("\n" + league_report)
        
        # Top teams report
        logger.info("Generating top teams report...")
        teams_report = report_gen.generate_top_teams_report(season_year, top_n=50)
        print("\n" + teams_report)
    
    def close(self):
        """Clean up resources."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Main entry point."""
    
    # Setup logging
    setup_logging()
    
    # Connection string - update with your credentials
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=fussballDB;"
        "Trusted_Connection=yes;"
    )
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print("\nCommands:")
        print("  process-season <year>              Process a single season")
        print("  process-range <start> <end>        Process range of seasons")
        print("  report <year>                      Generate reports for a season")
        print("\nExamples:")
        print("  python main.py process-season 2023")
        print("  python main.py process-range 2020 2024")
        print("  python main.py report 2023")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Initialize orchestrator
    orchestrator = StrengthCalculationOrchestrator(connection_string)
    
    try:
        orchestrator.initialize()
        
        if command == "process-season":
            if len(sys.argv) < 3:
                print("Error: season year required")
                sys.exit(1)
            
            season_year = int(sys.argv[2])
            orchestrator.process_season(season_year)
            orchestrator.generate_reports(season_year)
        
        elif command == "process-range":
            if len(sys.argv) < 4:
                print("Error: start and end years required")
                sys.exit(1)
            
            start_season = int(sys.argv[2])
            end_season = int(sys.argv[3])
            orchestrator.process_multiple_seasons(start_season, end_season)
        
        elif command == "report":
            if len(sys.argv) < 3:
                print("Error: season year required")
                sys.exit(1)
            
            season_year = int(sys.argv[2])
            orchestrator.generate_reports(season_year)
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()