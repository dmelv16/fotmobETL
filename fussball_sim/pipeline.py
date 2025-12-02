"""
Main training pipeline - orchestrates the full ML workflow
"""
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import json

from config import PipelineConfig, DatabaseConfig, FeatureConfig, ModelConfig
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import ScorePredictor, TotalGoalsPredictor, ResultPredictor
from prediction_service import PredictionService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SoccerPredictionPipeline:
    """
    End-to-end pipeline for training and deploying soccer match prediction models.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.loader = DataLoader(self.config)
        self.engineer = FeatureEngineer(self.config, self.loader)
        
        self.score_model = None
        self.goals_model = None
        self.result_model = None
        
        self.training_results = {}
        
    def run_training(self,
                     min_date: Optional[datetime] = None,
                     max_date: Optional[datetime] = None,
                     league_ids: Optional[List[int]] = None) -> Dict:
        """
        Run the complete training pipeline.
        
        Args:
            min_date: Filter matches after this date
            max_date: Filter matches before this date
            league_ids: Only include these leagues
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Soccer Match Prediction Training Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load matches
        logger.info("\n[Step 1/4] Loading match data...")
        matches = self.loader.get_finished_matches(min_date, max_date, league_ids)
        logger.info(f"Loaded {len(matches)} finished matches")
        
        if len(matches) == 0:
            raise ValueError("No matches found with the given filters")
        
        # Step 2: Feature engineering
        logger.info("\n[Step 2/4] Building feature matrix...")
        cache_path = None
        if self.config.use_cache:
            cache_path = f"{self.config.feature_cache_dir}/features.pkl"
            
        X, y = self.engineer.build_feature_matrix(matches, cache_path)
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features with data: {X.notna().sum().sum() / X.size:.1%}")
        
        # Data quality summary
        logger.info(f"Matches with player data: {X['has_player_data'].sum()} ({X['has_player_data'].mean():.1%})")
        logger.info(f"Matches with coach data: {X['has_coach_data'].sum()} ({X['has_coach_data'].mean():.1%})")
        logger.info(f"Matches with team data: {X['has_team_data'].sum()} ({X['has_team_data'].mean():.1%})")
        
        results = {
            'n_matches': len(matches),
            'n_features': X.shape[1],
            'feature_coverage': float(X.notna().sum().sum() / X.size),
            'models': {}
        }
        
        # Step 3: Train models
        logger.info("\n[Step 3/4] Training models...")
        
        if self.config.model.train_score_model:
            logger.info("\n--- Training Score Prediction Model ---")
            self.score_model = ScorePredictor(self.config)
            score_results = self.score_model.train(X, y)
            results['models']['score'] = score_results.metrics
            self.training_results['score'] = score_results
            
        if self.config.model.train_total_goals_model:
            logger.info("\n--- Training Total Goals Model ---")
            self.goals_model = TotalGoalsPredictor(self.config)
            goals_results = self.goals_model.train(X, y)
            results['models']['total_goals'] = goals_results.metrics
            self.training_results['total_goals'] = goals_results
            
        if self.config.model.train_result_model:
            logger.info("\n--- Training Result Prediction Model ---")
            self.result_model = ResultPredictor(self.config)
            result_results = self.result_model.train(X, y)
            results['models']['result'] = result_results.metrics
            self.training_results['result'] = result_results
        
        # Step 4: Save models
        logger.info("\n[Step 4/4] Saving models...")
        self.save_models()
        
        # Save training summary
        summary_path = Path(self.config.model_output_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print a formatted summary of training results."""
        print("\n📊 TRAINING SUMMARY")
        print("-" * 40)
        print(f"Total matches processed: {results['n_matches']}")
        print(f"Total features: {results['n_features']}")
        print(f"Feature coverage: {results['feature_coverage']:.1%}")
        
        if 'score' in results['models']:
            m = results['models']['score']
            print(f"\n🎯 Score Prediction:")
            print(f"   Home MAE: {m['home_mae']:.2f} | Away MAE: {m['away_mae']:.2f}")
            print(f"   Exact Score Accuracy: {m['exact_score_accuracy']:.1%}")
            print(f"   Result from Score Accuracy: {m['result_accuracy']:.1%}")
            
        if 'total_goals' in results['models']:
            m = results['models']['total_goals']
            print(f"\n⚽ Total Goals Prediction:")
            print(f"   MAE: {m['mae']:.2f} | R²: {m['r2']:.3f}")
            print(f"   Within ±1 Goal: {m['within_1_accuracy']:.1%}")
            
        if 'result' in results['models']:
            m = results['models']['result']
            print(f"\n🏆 Result Prediction (1X2):")
            print(f"   Accuracy: {m['accuracy']:.1%}")
            print(f"   Home Win Precision: {m.get('home_win_precision', 0):.1%}")
            print(f"   Draw Precision: {m.get('draw_precision', 0):.1%}")
            print(f"   Away Win Precision: {m.get('away_win_precision', 0):.1%}")
    
    def save_models(self):
        """Save all trained models to disk."""
        model_dir = Path(self.config.model_output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if self.score_model:
            self.score_model.save(str(model_dir / 'score_model.joblib'))
            logger.info(f"Saved score model to {model_dir / 'score_model.joblib'}")
            
        if self.goals_model:
            self.goals_model.save(str(model_dir / 'goals_model.joblib'))
            logger.info(f"Saved goals model to {model_dir / 'goals_model.joblib'}")
            
        if self.result_model:
            self.result_model.save(str(model_dir / 'result_model.joblib'))
            logger.info(f"Saved result model to {model_dir / 'result_model.joblib'}")
    
    def get_feature_importance(self, model_name: str = 'result', top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances from a trained model."""
        if model_name not in self.training_results:
            raise ValueError(f"Model '{model_name}' not trained. Available: {list(self.training_results.keys())}")
        
        return self.training_results[model_name].feature_importance.head(top_n)
    
    def create_prediction_service(self) -> PredictionService:
        """Create a prediction service with loaded models."""
        service = PredictionService(self.config, self.loader, self.engineer)
        service.score_model = self.score_model
        service.goals_model = self.goals_model
        service.result_model = self.result_model
        return service


def main():
    """Example usage of the pipeline."""
    
    # Configure the pipeline
    config = PipelineConfig(
        db=DatabaseConfig(
            server="your_server_name",
            database="fussballDB"
        ),
        features=FeatureConfig(
            include_bench_players=True,
            bench_weight=0.3,
            fill_missing_with="median",
            min_player_coverage=0.0  # Accept matches with no player data
        ),
        model=ModelConfig(
            train_score_model=True,
            train_total_goals_model=True,
            train_result_model=True,
            test_size=0.2,
            cv_folds=5
        ),
        model_output_dir="./models",
        feature_cache_dir="./cache",
        use_cache=True
    )
    
    # Initialize and run pipeline
    pipeline = SoccerPredictionPipeline(config)
    
    # Train on matches from the last 2 years
    from datetime import timedelta
    results = pipeline.run_training(
        min_date=datetime.now() - timedelta(days=730),
        max_date=datetime.now() - timedelta(days=1)  # Exclude today
    )
    
    # Show top features
    print("\n📈 Top 15 Most Important Features (Result Model):")
    print(pipeline.get_feature_importance('result', top_n=15))
    
    # Create prediction service for new matches
    predictor = pipeline.create_prediction_service()
    
    # Example: Predict upcoming matches
    # predictions = predictor.predict_upcoming(limit=10)
    # for pred in predictions:
    #     print(pred.to_dict())


if __name__ == "__main__":
    main()