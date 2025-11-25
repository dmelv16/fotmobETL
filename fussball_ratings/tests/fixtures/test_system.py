"""
VERIFICATION & TEST SCRIPT
==========================

Run this to verify everything is set up correctly before running the full pipeline.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all imports work correctly."""
    print("\n" + "="*60)
    print("TEST 1: IMPORTS")
    print("="*60)
    
    try:
        # Test models
        from processors.match_processor import PlayerMatchStats, CoachMatchStats, TeamMatchStats
        logger.info("✓ models.match_stats imports successfully")
        
        # Test that PlayerMatchStats has new fields
        pms = PlayerMatchStats(
            match_id=1, player_id=1, team_id=1, 
            data_tier=5, position_group=2
        )
        assert hasattr(pms, 'position_group'), "Missing position_group field"
        assert hasattr(pms, 'is_captain'), "Missing is_captain field"
        assert hasattr(pms, 'match_date'), "Missing match_date field"
        logger.info("✓ PlayerMatchStats has all required fields")
        
        # Test config
        from config.constants import DataTier, PositionGroup, get_position_group
        logger.info("✓ config.constants imports successfully")
        
        # Test entities
        from models.entities import Player, Coach, Team, League, EntityRegistry
        logger.info("✓ models.entities imports successfully")
        
        # Test database
        from database.connection import DatabaseConfig, ConnectionManager
        logger.info("✓ database.connection imports successfully")
        
        # Test queries
        from database.queries.match_queries import MatchQueries, LeagueQueries
        from database.queries.player_queries import PlayerQueries, CoachQueries
        logger.info("✓ database.queries imports successfully")
        
        # Test engines - check for circular imports
        from engines.rating_engine import RatingEngine, MatchContext
        logger.info("✓ engines.rating_engine imports successfully")
        
        from engines.attribute_engine import AttributeEngine
        logger.info("✓ engines.attribute_engine imports successfully")
        
        # Test processors
        from processors.data_handlers import DataTierProcessor, NullHandler
        logger.info("✓ processors.data_handlers imports successfully")
        
        from processors.match_processor import MatchDataLoader, MatchProcessor
        logger.info("✓ processors.match_processor imports successfully")
        
        from processors.pipeline import ProcessingPipeline, PipelineConfig
        logger.info("✓ processors.pipeline imports successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except AssertionError as e:
        logger.error(f"❌ Field validation failed: {e}")
        return False


def test_database_connection():
    """Test database connection and required tables."""
    print("\n" + "="*60)
    print("TEST 2: DATABASE CONNECTION")
    print("="*60)
    
    try:
        from database.connection import DatabaseConfig, ConnectionManager
        
        config = DatabaseConfig(
            server="DESKTOP-J9IV3OH",
            database="fussballDB",
            trusted_connection=True
        )
        
        conn_manager = ConnectionManager()
        conn_manager.initialize(config)
        
        executor = conn_manager.executor
        
        # Test basic query
        result = executor.execute_scalar("SELECT 1")
        assert result == 1, "Basic query failed"
        logger.info("✓ Database connection successful")
        
        # Test extraction_status exists and has data
        count = executor.execute_scalar(
            "SELECT COUNT(*) FROM extraction_status"
        )
        logger.info(f"✓ extraction_status table exists with {count:,} rows")
        
        if count == 0:
            logger.warning("⚠️  extraction_status is empty - you may need to populate it")
        
        # Test LeagueDivisions exists and has data
        count = executor.execute_scalar(
            "SELECT COUNT(*) FROM LeagueDivisions"
        )
        logger.info(f"✓ LeagueDivisions table exists with {count:,} rows")
        
        if count == 0:
            logger.warning("⚠️  LeagueDivisions is empty - you may need to populate it")
        
        # Test match_details has data
        count = executor.execute_scalar(
            "SELECT COUNT(*) FROM match_details WHERE finished = 1"
        )
        logger.info(f"✓ Found {count:,} finished matches")
        
        # Test rating tables exist
        tables = [
            'player_ratings', 'team_ratings', 'coach_ratings', 'league_ratings',
            'player_attributes', 'team_attributes', 'coach_attributes',
            'processing_log'
        ]
        
        for table in tables:
            result = executor.execute_scalar(
                f"SELECT OBJECT_ID('{table}', 'U')"
            )
            if result:
                logger.info(f"✓ Table {table} exists")
            else:
                logger.error(f"❌ Table {table} missing!")
                return False
        
        conn_manager.close()
        print("\n✅ Database connection and schema validated!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False


def test_safe_divide():
    """Test that safe_divide helper functions exist."""
    print("\n" + "="*60)
    print("TEST 3: SAFE DIVIDE FUNCTIONS")
    print("="*60)
    
    try:
        from engines.attribute_engine import AttributeEngine
        
        # Check if module has safe_divide
        import engines.attribute_engine as ae_module
        
        if hasattr(ae_module, 'safe_divide'):
            logger.info("✓ safe_divide() function exists")
            
            # Test it
            result = ae_module.safe_divide(10, 2, 0)
            assert result == 5.0, "safe_divide basic case failed"
            
            result = ae_module.safe_divide(10, 0, 99)
            assert result == 99, "safe_divide zero case failed"
            
            logger.info("✓ safe_divide() works correctly")
        else:
            logger.error("❌ safe_divide() not found - division by zero will crash!")
            return False
        
        if hasattr(ae_module, 'safe_avg'):
            logger.info("✓ safe_avg() function exists")
            
            # Test it
            result = ae_module.safe_avg([1, 2, 3], 0)
            assert result == 2.0, "safe_avg basic case failed"
            
            result = ae_module.safe_avg([], 99)
            assert result == 99, "safe_avg empty case failed"
            
            logger.info("✓ safe_avg() works correctly")
        else:
            logger.error("❌ safe_avg() not found - empty list averaging will crash!")
            return False
        
        print("\n✅ Safe divide functions validated!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Safe divide test failed: {e}")
        return False


def test_initialization_method():
    """Test that _initialize_player_rating exists."""
    print("\n" + "="*60)
    print("TEST 4: PLAYER INITIALIZATION")
    print("="*60)
    
    try:
        from processors.match_processor import MatchProcessor
        from models.entities import EntityRegistry
        from engines.rating_engine import RatingEngine
        from processors.data_handlers import StatKeyTracker
        
        # Create processor
        registry = EntityRegistry()
        rating_engine = RatingEngine()
        stat_tracker = StatKeyTracker()
        
        processor = MatchProcessor(registry, rating_engine, stat_tracker)
        
        # Check if method exists
        if hasattr(processor, '_initialize_player_rating'):
            logger.info("✓ _initialize_player_rating() method exists")
        else:
            logger.error("❌ _initialize_player_rating() not found!")
            logger.error("   Players will all start at BASE_RATING (1000)")
            return False
        
        print("\n✅ Player initialization method exists!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return False


def test_bootstrap_method():
    """Test that bootstrap_percentiles exists."""
    print("\n" + "="*60)
    print("TEST 5: PERCENTILE BOOTSTRAPPING")
    print("="*60)
    
    try:
        from engines.attribute_engine import AttributeEngine
        
        engine = AttributeEngine()
        
        # Check if method exists
        if hasattr(engine, 'bootstrap_percentiles'):
            logger.info("✓ bootstrap_percentiles() method exists")
        else:
            logger.error("❌ bootstrap_percentiles() not found!")
            logger.error("   First players will get inaccurate attributes")
            return False
        
        # Check if _bootstrapped flag exists
        if hasattr(engine, '_bootstrapped'):
            logger.info("✓ _bootstrapped flag exists")
        else:
            logger.warning("⚠️  _bootstrapped flag not found")
        
        print("\n✅ Bootstrap method exists!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bootstrap test failed: {e}")
        return False


def test_simple_match_processing():
    """Test loading and processing a single match."""
    print("\n" + "="*60)
    print("TEST 6: SIMPLE MATCH PROCESSING")
    print("="*60)
    
    try:
        from database.connection import DatabaseConfig, ConnectionManager
        from database.queries.match_queries import MatchQueries, LeagueQueries
        from database.queries.player_queries import PlayerQueries, CoachQueries
        from processors.match_processor import MatchDataLoader, MatchProcessor
        from models.entities import EntityRegistry
        from engines.rating_engine import RatingEngine
        from processors.data_handlers import StatKeyTracker
        
        # Setup
        config = DatabaseConfig(
            server="DESKTOP-J9IV3OH",
            database="fussballDB",
            trusted_connection=True
        )
        
        conn_manager = ConnectionManager()
        conn_manager.initialize(config)
        executor = conn_manager.executor
        
        # Get a single match ID
        # match_id = executor.execute_scalar(
        #     """SELECT TOP 1 md.match_id 
        #        FROM match_details md
        #        JOIN extraction_status es ON md.match_id = es.match_id
        #        WHERE md.finished = 1 
        #          AND es.has_match_details = 1
        #          AND es.has_lineups = 1
        #        ORDER BY md.match_time_utc"""
        # )
        
        match_id = 4506597

        if not match_id:
            logger.error("❌ No suitable match found for testing")
            return False
        
        logger.info(f"Testing with match_id: {match_id}")
        
        # Create components
        match_queries = MatchQueries(executor)
        player_queries = PlayerQueries(executor)
        coach_queries = CoachQueries(executor)
        league_queries = LeagueQueries(executor)
        
        loader = MatchDataLoader(
            match_queries, player_queries, coach_queries, league_queries
        )
        
        # Test loading
        loaded = loader.load_match(match_id)
        
        if not loaded:
            logger.error(f"❌ Failed to load match {match_id}")
            return False
        
        logger.info(f"✓ Loaded match {match_id}")
        logger.info(f"  Data tier: {loaded.data_tier.name}")
        logger.info(f"  Lineups: {len(loaded.lineups)} players")
        logger.info(f"  Events: {len(loaded.events)} events")
        
        # Test processing
        registry = EntityRegistry()
        rating_engine = RatingEngine()
        stat_tracker = StatKeyTracker()
        
        processor = MatchProcessor(registry, rating_engine, stat_tracker)
        
        result = processor.process_match(loaded)
        
        if result['success']:
            logger.info(f"✓ Successfully processed match {match_id}")
            logger.info(f"  Teams updated: {result['entities_updated']['teams']}")
            logger.info(f"  Players updated: {result['entities_updated']['players']}")
            logger.info(f"  Coaches updated: {result['entities_updated']['coaches']}")
        else:
            logger.error(f"❌ Failed to process match {match_id}")
            logger.error(f"  Errors: {result.get('errors', [])}")
            return False
        
        conn_manager.close()
        print("\n✅ Match processing works!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Match processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print(" FUSSBALL RATINGS SYSTEM - VERIFICATION TESTS")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Database", test_database_connection),
        ("Safe Divide", test_safe_divide),
        ("Player Init", test_initialization_method),
        ("Bootstrap", test_bootstrap_method),
        ("Match Processing", test_simple_match_processing),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("\n" + "-"*70)
    print(f"Results: {total_passed}/{total_tests} tests passed")
    print("="*70)
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! System is ready to run.")
        print("\nNext steps:")
        print("1. python main.py --stats  # Check system status")
        print("2. python main.py --full   # Run full processing")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED - Fix issues before running pipeline")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)