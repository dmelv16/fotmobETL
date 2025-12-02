"""
Main script to run walk-forward validation on soccer match data.

This properly trains and evaluates models across 2012-2025:
- Uses only historical data for each prediction (no data leakage)
- Retrains models periodically as new data becomes available
- Generates predictions for every finished match
"""
from datetime import datetime
from config import PipelineConfig, DatabaseConfig, FeatureConfig, ModelConfig
from walk_forward import WalkForwardValidator, WalkForwardConfig, WalkForwardStrategy
from tiered_models import TierConfig


def run_expanding_window():
    """
    EXPANDING WINDOW STRATEGY (Recommended)
    
    - Start with 2 seasons of training data (2012-2014)
    - Predict next year of matches
    - Add those matches to training set
    - Repeat until 2025
    
    Training set grows over time, using all available historical data.
    """
    
    # Pipeline configuration
    pipeline_config = PipelineConfig(
        db=DatabaseConfig(
            server="DESKTOP-J9IV3OH",  # <-- UPDATE THIS
            database="fussballDB"
        ),
        features=FeatureConfig(
            include_bench_players=True,
            bench_weight=0.3,
            fill_missing_with="median",
            min_player_coverage=0.0  # Accept matches without player data
        ),
        model=ModelConfig(
            train_score_model=True,
            train_total_goals_model=True,
            train_result_model=True,
            xgb_params={
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        )
    )
    
    # Walk-forward configuration
    wf_config = WalkForwardConfig(
        strategy=WalkForwardStrategy.EXPANDING,
        
        # Initial training period: 2 full seasons
        initial_train_start=datetime(2012, 7, 1),
        initial_train_end=datetime(2014, 6, 30),
        
        # Start predicting from 2014/15 season onwards
        prediction_start=datetime(2014, 7, 1),
        prediction_end=datetime(2025, 6, 30),
        
        # Retrain every year
        retrain_frequency_days=365,
        
        # Minimum matches to train
        min_training_matches=500,
        
        # Output
        output_dir="./walk_forward_results/expanding",
        save_intermediate_models=False,
        
        # Single model (not tiered)
        use_tiered_models=False
    )
    
    # Run
    validator = WalkForwardValidator(pipeline_config, wf_config)
    results = validator.run()
    
    return results


def run_tiered_expanding():
    """
    TIERED EXPANDING WINDOW STRATEGY
    
    Same as expanding, but trains separate models for each coverage level:
    - Tier 1 (xG): Has xG data - richest features
    - Tier 2 (Ratings): Has player/team ratings but no xG
    - Tier 3 (Lower): Basic match data only
    """
    
    pipeline_config = PipelineConfig(
        db=DatabaseConfig(
            server="DESKTOP-J9IV3OH",  # <-- UPDATE THIS
            database="fussballDB"
        ),
        features=FeatureConfig(
            include_bench_players=True,
            bench_weight=0.3,
            fill_missing_with="median",
            min_player_coverage=0.0
        ),
        model=ModelConfig(
            train_score_model=True,
            train_total_goals_model=True,
            train_result_model=True,
            xgb_params={
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        )
    )
    
    # Define your tiers based on coverage_level values in your database
    tier_configs = [
        TierConfig(
            name="xg",
            coverage_levels=["xG"],  # String value from your DB
            description="Full xG data available"
        ),
        TierConfig(
            name="ratings", 
            coverage_levels=["ratings"],  # String value from your DB
            description="Player/team ratings, no xG"
        ),
        TierConfig(
            name="lower",
            coverage_levels=["lower"],  # String value from your DB
            description="Basic match data only"
        ),
    ]
    
    wf_config = WalkForwardConfig(
        strategy=WalkForwardStrategy.EXPANDING,
        
        initial_train_start=datetime(2012, 7, 1),
        initial_train_end=datetime(2014, 6, 30),
        
        prediction_start=datetime(2014, 7, 1),
        prediction_end=datetime(2025, 6, 30),
        
        retrain_frequency_days=365,
        min_training_matches=500,
        
        output_dir="./walk_forward_results/tiered_expanding",
        save_intermediate_models=False,
        
        # Enable tiered models
        use_tiered_models=True,
        tier_configs=tier_configs
    )
    
    validator = WalkForwardValidator(pipeline_config, wf_config)
    results = validator.run()
    
    return results


def run_rolling_window():
    """
    ROLLING WINDOW STRATEGY
    
    Uses a fixed 2-year training window that slides forward.
    Older matches are dropped from training.
    
    Pros: Adapts to recent patterns, less compute
    Cons: Loses historical knowledge
    """
    
    pipeline_config = PipelineConfig(
        db=DatabaseConfig(
            server="DESKTOP-J9IV3OH",
            database="fussballDB"
        ),
        features=FeatureConfig(
            include_bench_players=True,
            fill_missing_with="median"
        ),
        model=ModelConfig(
            train_score_model=True,
            train_total_goals_model=True,
            train_result_model=True,
            xgb_params={
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_jobs': -1
            }
        )
    )
    
    wf_config = WalkForwardConfig(
        strategy=WalkForwardStrategy.ROLLING,
        
        initial_train_start=datetime(2012, 7, 1),
        initial_train_end=datetime(2014, 6, 30),
        
        prediction_start=datetime(2014, 7, 1),
        prediction_end=datetime(2025, 6, 30),
        
        # Rolling window: always use last 2 years
        rolling_window_days=730,
        
        retrain_frequency_days=365,  # Yearly
        min_training_matches=500,
        output_dir="./walk_forward_results/rolling"
    )
    
    validator = WalkForwardValidator(pipeline_config, wf_config)
    return validator.run()


def run_seasonal():
    """
    SEASONAL STRATEGY
    
    Retrains once at the start of each season.
    Uses all completed seasons for training.
    
    Pros: Conceptually clean (season boundaries), less retraining
    Cons: Model is stale by end of season
    """
    
    pipeline_config = PipelineConfig(
        db=DatabaseConfig(
            server="DESKTOP-J9IV3OH",
            database="fussballDB"
        ),
        features=FeatureConfig(
            include_bench_players=True,
            fill_missing_with="median"
        ),
        model=ModelConfig(
            train_score_model=True,
            train_total_goals_model=True,
            train_result_model=True
        )
    )
    
    wf_config = WalkForwardConfig(
        strategy=WalkForwardStrategy.SEASONAL,
        
        initial_train_start=datetime(2012, 7, 1),
        initial_train_end=datetime(2014, 6, 30),
        
        prediction_start=datetime(2014, 7, 1),
        prediction_end=datetime(2025, 6, 30),
        
        # Large windows for seasonal approach
        retrain_frequency_days=365,  # Yearly
        
        min_training_matches=500,
        output_dir="./walk_forward_results/seasonal"
    )
    
    validator = WalkForwardValidator(pipeline_config, wf_config)
    return validator.run()


def run_quick_test():
    """
    Quick test on a smaller date range to verify everything works.
    """
    pipeline_config = PipelineConfig(
        db=DatabaseConfig(
            server="DESKTOP-J9IV3OH",
            database="fussballDB"
        ),
        features=FeatureConfig(
            include_bench_players=False,  # Faster
            fill_missing_with="median"
        ),
        model=ModelConfig(
            train_score_model=True,
            train_total_goals_model=True,
            train_result_model=True,
            xgb_params={
                'n_estimators': 50,  # Fewer trees for speed
                'max_depth': 4,
                'n_jobs': -1
            }
        )
    )
    
    wf_config = WalkForwardConfig(
        strategy=WalkForwardStrategy.EXPANDING,
        
        # Just test on one season
        initial_train_start=datetime(2020, 7, 1),
        initial_train_end=datetime(2022, 6, 30),
        
        prediction_start=datetime(2022, 7, 1),
        prediction_end=datetime(2023, 6, 30),
        
        retrain_frequency_days=60,  # Less frequent
        min_training_matches=200,
        output_dir="./walk_forward_results/test"
    )
    
    validator = WalkForwardValidator(pipeline_config, wf_config)
    return validator.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument(
        "--strategy", 
        choices=["expanding", "tiered", "rolling", "seasonal", "test"],
        default="expanding",
        help="Walk-forward strategy to use"
    )
    
    args = parser.parse_args()
    
    if args.strategy == "expanding":
        results = run_expanding_window()
    elif args.strategy == "tiered":
        results = run_tiered_expanding()
    elif args.strategy == "rolling":
        results = run_rolling_window()
    elif args.strategy == "seasonal":
        results = run_seasonal()
    else:
        results = run_quick_test()
    
    print(f"\n✅ Complete! {len(results)} predictions generated.")
    print(f"Results saved to ./walk_forward_results/{args.strategy}/")