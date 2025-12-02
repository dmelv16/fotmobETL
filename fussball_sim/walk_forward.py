"""
Walk-Forward Validation Module (Updated)

Changes:
- Integrated comprehensive metrics (Over/Under, Double Chance, Draw)
- Better tier validation and logging
- Performance optimizations
"""
import pandas as pd
import numpy as np
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import warnings

from config import PipelineConfig
from data_loader import BatchDataLoader
from feature_engineering import OptimizedFeatureEngineer
from models import ScorePredictor, TotalGoalsPredictor, ResultPredictor
from tiered_models import TieredModelSystem, TierConfig
from metrics import (
    calculate_all_metrics, 
    calculate_metrics_by_tier,
    print_comprehensive_report,
    metrics_to_dataframe
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardStrategy(Enum):
    EXPANDING = "expanding"
    ROLLING = "rolling"
    SEASONAL = "seasonal"


@dataclass
class WalkForwardConfig:
    strategy: WalkForwardStrategy = WalkForwardStrategy.EXPANDING
    initial_train_start: datetime = datetime(2012, 7, 1)
    initial_train_end: datetime = datetime(2014, 6, 30)
    prediction_start: datetime = datetime(2014, 7, 1)
    prediction_end: datetime = datetime(2025, 6, 30)
    retrain_frequency_days: int = 30
    rolling_window_days: int = 730
    min_training_matches: int = 500
    save_intermediate_models: bool = False
    output_dir: str = "./walk_forward_results"
    use_tiered_models: bool = False
    tier_configs: Optional[List[TierConfig]] = None


@dataclass
class WindowResult:
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train_matches: int
    n_test_matches: int
    predictions: List[Dict]
    metrics: Dict[str, float]
    tier_breakdown: Optional[Dict[str, int]] = None  # NEW: track tier distribution


class WalkForwardValidator:
    
    def __init__(self, 
                 pipeline_config: PipelineConfig,
                 wf_config: WalkForwardConfig):
        self.pipeline_config = pipeline_config
        self.wf_config = wf_config
        
        self.loader = BatchDataLoader(pipeline_config)
        self.engineer = OptimizedFeatureEngineer(pipeline_config, self.loader)
        
        self.all_results: List[WindowResult] = []
        self.all_predictions: List[Dict] = []
        self._feature_names: List[str] = []
        
        self.tiered_system: Optional[TieredModelSystem] = None
        if wf_config.use_tiered_models:
            self.tiered_system = TieredModelSystem(
                pipeline_config, 
                tiers=wf_config.tier_configs
            )
    
    def generate_windows(self) -> Generator[Tuple[datetime, datetime, datetime, datetime], None, None]:
        wf = self.wf_config
        current_test_start = wf.prediction_start
        
        while current_test_start < wf.prediction_end:
            test_end = min(
                current_test_start + timedelta(days=wf.retrain_frequency_days),
                wf.prediction_end
            )
            
            if wf.strategy == WalkForwardStrategy.EXPANDING:
                train_start = wf.initial_train_start
                train_end = current_test_start - timedelta(days=1)
            elif wf.strategy == WalkForwardStrategy.ROLLING:
                train_end = current_test_start - timedelta(days=1)
                train_start = train_end - timedelta(days=wf.rolling_window_days)
            elif wf.strategy == WalkForwardStrategy.SEASONAL:
                train_start = wf.initial_train_start
                if current_test_start.month >= 7:
                    season_start = datetime(current_test_start.year, 7, 1)
                else:
                    season_start = datetime(current_test_start.year - 1, 7, 1)
                train_end = season_start - timedelta(days=1)
            
            yield (train_start, train_end, current_test_start, test_end)
            current_test_start = test_end
    
    def _validate_tier_split(self, coverage_levels: pd.Series):
        """Log the tier distribution for this window."""
        if not self.wf_config.use_tiered_models:
            return
        
        # Map coverage levels to our tier names
        tier_counts = {}
        for tier in self.wf_config.tier_configs or []:
            mask = coverage_levels.isin(tier.coverage_levels)
            tier_counts[tier.name] = mask.sum()
        
        logger.info("\nTier Distribution:")
        for tier_name, count in tier_counts.items():
            tier_config = next((t for t in self.wf_config.tier_configs if t.name == tier_name), None)
            min_req = tier_config.min_training_samples if tier_config else 100
            status = "✓" if count >= min_req else "✗"
            logger.info(f"  {status} {tier_name}: {count:,} matches (need {min_req})")
    
    def _train_models(self, train_matches: pd.DataFrame) -> Tuple:
        """Train models on the given training data."""
        
        logger.info(f"Building features for {len(train_matches)} matches in batches...")
        X_train, y_train = self._build_features_in_batches(train_matches, batch_size=10000)
        
        if len(X_train) < self.wf_config.min_training_matches:
            logger.warning(f"Insufficient training data: {len(X_train)} matches")
            return None, None, None, None, None, None
        
        # Get coverage levels for tiered training
        coverage_levels = None
        tier_breakdown = None
        
        if self.wf_config.use_tiered_models:
            coverage_df = train_matches[['match_id', 'coverage_level']].drop_duplicates()
            X_with_coverage = X_train.merge(coverage_df, on='match_id', how='left')
            coverage_levels = X_with_coverage['coverage_level'].fillna('lower')
            
            # Log tier distribution
            self._validate_tier_split(coverage_levels)
            
            # Store breakdown
            tier_breakdown = {}
            for tier in self.wf_config.tier_configs or []:
                tier_breakdown[tier.name] = coverage_levels.isin(tier.coverage_levels).sum()
        
        self._feature_names = [c for c in X_train.columns if c not in 
                              ['match_id', 'has_player_data', 'has_coach_data', 'has_team_data', 'player_coverage']]
        
        logger.info(f"Training on {len(X_train)} matches with {len(self._feature_names)} features...")
        
        if self.wf_config.use_tiered_models and self.tiered_system is not None:
            tier_stats = self.tiered_system.train(X_train, y_train, coverage_levels)
            return None, None, None, X_train, coverage_levels, tier_breakdown
        else:
            score_model = None
            goals_model = None
            result_model = None
            
            if self.pipeline_config.model.train_score_model:
                score_model = ScorePredictor(self.pipeline_config)
                score_model._train_full(X_train, y_train)
                
            if self.pipeline_config.model.train_total_goals_model:
                goals_model = TotalGoalsPredictor(self.pipeline_config)
                goals_model._train_full(X_train, y_train)
                
            if self.pipeline_config.model.train_result_model:
                result_model = ResultPredictor(self.pipeline_config)
                result_model._train_full(X_train, y_train)
            
            return score_model, goals_model, result_model, X_train, None, None
    
    def _build_features_in_batches(self, matches: pd.DataFrame, batch_size: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        matches = matches.copy()
        matches['match_time_utc'] = pd.to_datetime(matches['match_time_utc'])
        matches = matches.sort_values('match_time_utc').reset_index(drop=True)
        
        all_X = []
        all_y = []
        n_batches = (len(matches) + batch_size - 1) // batch_size
        
        for i in range(0, len(matches), batch_size):
            batch_num = i // batch_size + 1
            batch_matches = matches.iloc[i:i + batch_size]
            
            logger.info(f"  Processing batch {batch_num}/{n_batches} ({len(batch_matches)} matches)...")
            
            try:
                X_batch, y_batch = self.engineer.build_feature_matrix(batch_matches)
                all_X.append(X_batch)
                all_y.append(y_batch)
            except Exception as e:
                logger.warning(f"  Batch {batch_num} failed: {e}")
                continue
            
            gc.collect()
        
        if not all_X:
            return pd.DataFrame(), pd.DataFrame()
        
        X = pd.concat(all_X, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)
        
        all_cols = set()
        for df in all_X:
            all_cols.update(df.columns)
        for col in all_cols:
            if col not in X.columns:
                X[col] = np.nan
        
        return X, y
    
    def _predict_window(self,
                        test_matches: pd.DataFrame,
                        score_model: Optional[ScorePredictor],
                        goals_model: Optional[TotalGoalsPredictor],
                        result_model: Optional[ResultPredictor]) -> List[Dict]:
        
        X_test, y_test = self._build_features_in_batches(test_matches, batch_size=10000)
        
        if X_test.empty:
            return []
        
        coverage_levels = None
        if self.wf_config.use_tiered_models:
            coverage_df = test_matches[['match_id', 'coverage_level']].drop_duplicates()
            X_with_coverage = X_test.merge(coverage_df, on='match_id', how='left')
            coverage_levels = X_with_coverage['coverage_level'].fillna('lower')
        
        # Make predictions
        if self.wf_config.use_tiered_models and self.tiered_system is not None:
            pred_home_scores, pred_away_scores, pred_total_goals, pred_results, pred_probs = \
                self.tiered_system.predict(X_test, coverage_levels)
        else:
            pred_home_scores = np.zeros(len(X_test))
            pred_away_scores = np.zeros(len(X_test))
            pred_total_goals = np.zeros(len(X_test))
            pred_results = np.ones(len(X_test), dtype=int)
            pred_probs = np.full((len(X_test), 3), 0.33)
            
            try:
                if score_model:
                    scores = score_model.predict(X_test)
                    pred_home_scores = scores[:, 0]
                    pred_away_scores = scores[:, 1]
                if goals_model:
                    pred_total_goals = goals_model.predict(X_test)
                else:
                    pred_total_goals = pred_home_scores + pred_away_scores
                if result_model:
                    results, probs = result_model.predict(X_test)
                    pred_results = results
                    pred_probs = probs
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
        
        # Build predictions list
        predictions = []
        test_matches = test_matches.copy()
        test_matches['match_time_utc'] = pd.to_datetime(test_matches['match_time_utc'])
        match_info = test_matches.set_index('match_id')
        
        for i in range(len(X_test)):
            match_id = int(X_test.iloc[i]['match_id'])
            
            if match_id not in match_info.index:
                continue
                
            match_row = match_info.loc[match_id]
            actual_total = int(match_row['home_team_score'] + match_row['away_team_score'])
            
            pred_dict = {
                'match_id': match_id,
                'match_time_utc': match_row['match_time_utc'],
                'home_team_id': match_row['home_team_id'],
                'home_team_name': match_row['home_team_name'],
                'away_team_id': match_row['away_team_id'],
                'away_team_name': match_row['away_team_name'],
                'league_id': match_row.get('league_id'),
                'league_name': match_row.get('league_name'),
                'coverage_level': match_row.get('coverage_level'),
                
                # Predictions
                'pred_home_score': int(round(pred_home_scores[i])),
                'pred_away_score': int(round(pred_away_scores[i])),
                'pred_total_goals': int(round(pred_total_goals[i])),
                'pred_result': ['Away Win', 'Draw', 'Home Win'][int(pred_results[i])],
                'prob_away_win': float(pred_probs[i, 0]),
                'prob_draw': float(pred_probs[i, 1]),
                'prob_home_win': float(pred_probs[i, 2]),
                
                # Actuals
                'actual_home_score': int(match_row['home_team_score']),
                'actual_away_score': int(match_row['away_team_score']),
                'actual_total_goals': actual_total,
                'actual_result': (
                    'Home Win' if match_row['home_team_score'] > match_row['away_team_score']
                    else ('Away Win' if match_row['home_team_score'] < match_row['away_team_score'] 
                          else 'Draw')
                ),
                
                # Data quality
                'has_player_data': bool(X_test.iloc[i].get('has_player_data', False)),
                'has_coach_data': bool(X_test.iloc[i].get('has_coach_data', False)),
                'has_team_data': bool(X_test.iloc[i].get('has_team_data', False)),
                'player_coverage': float(X_test.iloc[i].get('player_coverage', 0))
            }
            
            predictions.append(pred_dict)
        
        return predictions
    
    def _calculate_window_metrics(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive metrics for a prediction window."""
        if not predictions:
            return {}
        
        df = pd.DataFrame(predictions)
        metrics = calculate_all_metrics(df)
        
        # Flatten for storage
        flat_metrics = {
            'n_predictions': metrics['n_matches'],
            'result_accuracy': metrics['result_accuracy'],
            'exact_score_rate': metrics['exact_score_rate'],
            'home_score_mae': metrics['home_score_mae'],
            'away_score_mae': metrics['away_score_mae'],
            'total_goals_mae': metrics['total_goals_mae'],
            
            # Over/Under
            'ou_1.5_accuracy': metrics['over_under'][1.5].accuracy,
            'ou_2.5_accuracy': metrics['over_under'][2.5].accuracy,
            'ou_3.5_accuracy': metrics['over_under'][3.5].accuracy,
            'ou_4.5_accuracy': metrics['over_under'][4.5].accuracy,
            
            # Double Chance
            'dc_home_or_draw': metrics['double_chance'].home_or_draw_accuracy,
            'dc_away_or_draw': metrics['double_chance'].away_or_draw_accuracy,
            'dc_home_or_away': metrics['double_chance'].home_or_away_accuracy,
            
            # Draw
            'draw_accuracy': metrics['draw'].draw_accuracy,
            'draw_recall': metrics['draw'].draw_recall,
            'draw_brier': metrics['draw'].brier_score,
        }
        
        return flat_metrics
    
    def run(self) -> pd.DataFrame:
        logger.info("=" * 70)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info(f"Strategy: {self.wf_config.strategy.value}")
        logger.info(f"Tiered Models: {self.wf_config.use_tiered_models}")
        if self.wf_config.use_tiered_models and self.wf_config.tier_configs:
            for tier in self.wf_config.tier_configs:
                logger.info(f"  - {tier.name}: {tier.coverage_levels} (min: {tier.min_training_samples})")
        logger.info("=" * 70)
        
        # Load all matches
        logger.info("\nLoading all matches...")
        all_matches = self.loader.get_finished_matches(
            min_date=self.wf_config.initial_train_start,
            max_date=self.wf_config.prediction_end
        )
        logger.info(f"Total matches available: {len(all_matches)}")
        
        # Log coverage distribution
        if 'coverage_level' in all_matches.columns:
            logger.info("\nCoverage Level Distribution:")
            for level, count in all_matches['coverage_level'].value_counts().items():
                logger.info(f"  {level}: {count:,} ({count/len(all_matches)*100:.1f}%)")
        
        all_matches['match_time_utc'] = pd.to_datetime(all_matches['match_time_utc'])
        
        windows = list(self.generate_windows())
        logger.info(f"\nTotal windows to process: {len(windows)}")
        
        for window_id, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"\n{'='*70}")
            logger.info(f"WINDOW {window_id + 1}/{len(windows)}")
            logger.info(f"Training: {train_start.date()} to {train_end.date()}")
            logger.info(f"Testing:  {test_start.date()} to {test_end.date()}")
            logger.info("="*70)
            
            train_matches = all_matches[
                (all_matches['match_time_utc'] >= train_start) &
                (all_matches['match_time_utc'] <= train_end)
            ].copy()
            
            test_matches = all_matches[
                (all_matches['match_time_utc'] >= test_start) &
                (all_matches['match_time_utc'] < test_end)
            ].copy()
            
            logger.info(f"Training matches: {len(train_matches)}, Test matches: {len(test_matches)}")
            
            if len(train_matches) < self.wf_config.min_training_matches:
                logger.warning(f"Skipping window: insufficient training data")
                continue
            if len(test_matches) == 0:
                logger.warning(f"Skipping window: no test matches")
                continue
            
            # Train models
            logger.info("Training models...")
            score_model, goals_model, result_model, X_train, coverage_levels, tier_breakdown = \
                self._train_models(train_matches)
            
            if not self.wf_config.use_tiered_models:
                if score_model is None and goals_model is None and result_model is None:
                    logger.warning("No models trained, skipping window")
                    continue
            
            # Generate predictions
            logger.info("Generating predictions...")
            predictions = self._predict_window(test_matches, score_model, goals_model, result_model)
            
            # Calculate metrics
            metrics = self._calculate_window_metrics(predictions)
            
            logger.info(f"\nWindow Results:")
            logger.info(f"  Predictions: {metrics.get('n_predictions', 0)}")
            logger.info(f"  Result Acc: {metrics.get('result_accuracy', 0):.1%}")
            logger.info(f"  O/U 2.5 Acc: {metrics.get('ou_2.5_accuracy', 0):.1%}")
            logger.info(f"  DC 1X Acc: {metrics.get('dc_home_or_draw', 0):.1%}")
            logger.info(f"  Draw Acc: {metrics.get('draw_accuracy', 0):.1%}")
            
            window_result = WindowResult(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train_matches=len(train_matches),
                n_test_matches=len(test_matches),
                predictions=predictions,
                metrics=metrics,
                tier_breakdown=tier_breakdown
            )
            
            self.all_results.append(window_result)
            self.all_predictions.extend(predictions)
            
            if self.wf_config.save_intermediate_models:
                self._save_window_results(window_result)
            
            # Cleanup
            gc.collect()
        
        results_df = pd.DataFrame(self.all_predictions)
        self._save_final_results(results_df)
        self._print_comprehensive_summary(results_df)
        
        return results_df
    
    def _save_window_results(self, result: WindowResult):
        output_dir = Path(self.wf_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        window_file = output_dir / f"window_{result.window_id:04d}.json"
        with open(window_file, 'w') as f:
            json.dump({
                'window_id': result.window_id,
                'train_start': str(result.train_start),
                'train_end': str(result.train_end),
                'test_start': str(result.test_start),
                'test_end': str(result.test_end),
                'n_train': result.n_train_matches,
                'n_test': result.n_test_matches,
                'tier_breakdown': result.tier_breakdown,
                'metrics': result.metrics
            }, f, indent=2)
    
    def _save_final_results(self, results_df: pd.DataFrame):
        output_dir = Path(self.wf_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_dir / 'all_predictions.csv', index=False)
        results_df.to_parquet(output_dir / 'all_predictions.parquet', index=False)
        
        # Save metrics summary
        metrics_df = metrics_to_dataframe(results_df)
        metrics_df.to_csv(output_dir / 'metrics_summary.csv', index=False)
        
        # Window summaries
        window_summary = []
        for r in self.all_results:
            row = {
                'window_id': r.window_id,
                'train_start': r.train_start,
                'train_end': r.train_end,
                'test_start': r.test_start,
                'test_end': r.test_end,
                'n_train': r.n_train_matches,
                'n_test': r.n_test_matches,
            }
            if r.tier_breakdown:
                row.update({f'tier_{k}': v for k, v in r.tier_breakdown.items()})
            row.update(r.metrics)
            window_summary.append(row)
        
        pd.DataFrame(window_summary).to_csv(output_dir / 'window_summary.csv', index=False)
        
        logger.info(f"\nResults saved to: {output_dir}")
    
    def _print_comprehensive_summary(self, results_df: pd.DataFrame):
        """Print comprehensive summary using new metrics module."""
        print_comprehensive_report(results_df, by_tier=self.wf_config.use_tiered_models)
        
        # Additional tier evolution summary for tiered models
        if self.wf_config.use_tiered_models and self.all_results:
            print("\n" + "="*70)
            print("TIER TRAINING EVOLUTION BY WINDOW")
            print("="*70)
            
            for result in self.all_results:
                if result.tier_breakdown:
                    breakdown_str = ", ".join([f"{k}: {v:,}" for k, v in result.tier_breakdown.items()])
                    print(f"Window {result.window_id + 1} ({result.train_end.year}): {breakdown_str}")