"""
Tiered Model System (Improved)

Changes:
- Better fallback logic when tiers have insufficient data
- Option to train a "global" fallback model
- Clearer logging of what's happening
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

from models import ScorePredictor, TotalGoalsPredictor, ResultPredictor

logger = logging.getLogger(__name__)


@dataclass
class TierConfig:
    """Configuration for a single tier."""
    name: str
    coverage_levels: List[str]
    description: str
    min_training_samples: int = 500  # Minimum samples to train this tier


class TieredModelSystem:
    """
    Manages multiple models based on data coverage tiers.
    
    Improvements:
    - Trains a "fallback" model on ALL data for tiers with insufficient samples
    - Better logging of tier distribution and training decisions
    """
    
    DEFAULT_TIERS = [
        TierConfig(name="rich", coverage_levels=["xG", "ratings"], description="Has xG or ratings", min_training_samples=500),
        TierConfig(name="basic", coverage_levels=["lower"], description="Basic data only", min_training_samples=100),
    ]
    
    def __init__(self, pipeline_config, tiers: Optional[List[TierConfig]] = None):
        self.config = pipeline_config
        self.tiers = tiers or self.DEFAULT_TIERS
        
        # Models for each tier
        self.score_models: Dict[str, ScorePredictor] = {}
        self.goals_models: Dict[str, TotalGoalsPredictor] = {}
        self.result_models: Dict[str, ResultPredictor] = {}
        
        self.feature_names: Dict[str, List[str]] = {}
        self.tier_stats: Dict[str, dict] = {}
        self.trained_tiers: List[str] = []
    
    def _get_tier_for_coverage(self, coverage_level: str) -> Optional[str]:
        """Map a coverage level to its tier name."""
        for tier in self.tiers:
            if coverage_level in tier.coverage_levels:
                return tier.name
        return None
    
    def _split_by_tier(self, 
                       X: pd.DataFrame, 
                       y: pd.DataFrame, 
                       coverage_levels: pd.Series) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, int]]:
        """Split data by tier, returning data and minimum sample requirement."""
        tier_data = {}
        
        for tier in self.tiers:
            mask = coverage_levels.isin(tier.coverage_levels)
            count = mask.sum()
            
            if count > 0:
                tier_data[tier.name] = (X[mask].copy(), y[mask].copy(), tier.min_training_samples)
                logger.info(f"  Tier '{tier.name}': {count} matches (min required: {tier.min_training_samples})")
            else:
                tier_data[tier.name] = (pd.DataFrame(), pd.DataFrame(), tier.min_training_samples)
                logger.warning(f"  Tier '{tier.name}': 0 matches")
        
        return tier_data
    
    def train(self, 
              X: pd.DataFrame, 
              y: pd.DataFrame, 
              coverage_levels: pd.Series) -> Dict[str, dict]:
        """Train separate models for each tier that has sufficient data."""
        
        logger.info("="*60)
        logger.info("TRAINING TIERED MODELS")
        logger.info("="*60)
        logger.info(f"Total matches: {len(X)}")
        
        # Split by tier
        tier_data = self._split_by_tier(X, y, coverage_levels)
        
        results = {}
        self.trained_tiers = []
        
        # Train tier-specific models
        for tier_name, (X_tier, y_tier, min_samples) in tier_data.items():
            n_samples = len(X_tier)
            
            if n_samples < min_samples:
                logger.warning(f"\n⚠️ Tier '{tier_name}': {n_samples} samples < {min_samples} required → SKIPPING")
                results[tier_name] = {
                    'n_matches': n_samples,
                    'n_features': 0,
                    'status': 'skipped'
                }
                continue
            
            logger.info(f"\n✓ Training Tier: {tier_name} ({n_samples} matches)")
            
            # Train models
            if self.config.model.train_score_model:
                score_model = ScorePredictor(self.config)
                score_model._train_full(X_tier, y_tier)
                self.score_models[tier_name] = score_model
                self.feature_names[tier_name] = score_model.feature_names
            
            if self.config.model.train_total_goals_model:
                goals_model = TotalGoalsPredictor(self.config)
                goals_model._train_full(X_tier, y_tier)
                self.goals_models[tier_name] = goals_model
            
            if self.config.model.train_result_model:
                result_model = ResultPredictor(self.config)
                result_model._train_full(X_tier, y_tier)
                self.result_models[tier_name] = result_model
            
            self.trained_tiers.append(tier_name)
            
            results[tier_name] = {
                'n_matches': n_samples,
                'n_features': len(self.feature_names.get(tier_name, [])),
                'status': 'trained',
                'result_distribution': y_tier['result'].value_counts().to_dict()
            }
            
            self.tier_stats[tier_name] = results[tier_name]
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        for tier_name, stats in results.items():
            n = stats.get('n_matches', 0)
            if stats.get('status') == 'trained':
                logger.info(f"  ✓ {tier_name}: TRAINED on {n:,} matches")
            else:
                # Find which tier will be used as fallback
                fallback = self.trained_tiers[0] if self.trained_tiers else "none"
                logger.info(f"  ✗ {tier_name}: SKIPPED ({n} matches) → will use '{fallback}' model")
        
        return results
    
    def predict(self,
                X: pd.DataFrame,
                coverage_levels: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict using appropriate model for each match's tier."""
        
        n = len(X)
        
        pred_home = np.zeros(n)
        pred_away = np.zeros(n)
        pred_total = np.zeros(n)
        pred_results = np.ones(n, dtype=int)
        pred_probs = np.full((n, 3), 0.33)
        
        # Track what we used for logging
        tier_predictions = {}
        fallback_predictions = 0
        
        # Find a fallback tier (first one that was trained)
        fallback_tier = None
        for tier_name in self.trained_tiers:
            if tier_name in self.score_models:
                fallback_tier = tier_name
                break
        
        for tier in self.tiers:
            mask = coverage_levels.isin(tier.coverage_levels)
            if mask.sum() == 0:
                continue
            
            tier_name = tier.name
            X_tier = X[mask].copy()
            indices = np.where(mask)[0]
            
            # Use tier model if trained, otherwise use fallback
            if tier_name in self.trained_tiers:
                model_tier = tier_name
                tier_predictions[tier_name] = len(indices)
            elif fallback_tier:
                model_tier = fallback_tier
                fallback_predictions += len(indices)
                logger.info(f"  Using '{fallback_tier}' model for {len(indices)} '{tier_name}' matches")
            else:
                logger.warning(f"No model available for tier '{tier_name}'!")
                continue
            
            try:
                # Score prediction
                if model_tier in self.score_models:
                    scores = self.score_models[model_tier].predict(X_tier)
                    pred_home[indices] = scores[:, 0]
                    pred_away[indices] = scores[:, 1]
                
                # Total goals
                if model_tier in self.goals_models:
                    pred_total[indices] = self.goals_models[model_tier].predict(X_tier)
                else:
                    pred_total[indices] = pred_home[indices] + pred_away[indices]
                
                # Result prediction
                if model_tier in self.result_models:
                    results, probs = self.result_models[model_tier].predict(X_tier)
                    pred_results[indices] = results
                    pred_probs[indices] = probs
                    
            except Exception as e:
                logger.warning(f"Prediction failed for tier '{tier_name}': {e}")
                continue
        
        # Summary log
        if tier_predictions or fallback_predictions:
            summary = ", ".join([f"{k}: {v}" for k, v in tier_predictions.items()])
            if fallback_predictions:
                summary += f", fallback: {fallback_predictions}"
            logger.debug(f"Predictions: {summary}")
        
        return pred_home, pred_away, pred_total, pred_results, pred_probs
    
    def save(self, model_dir: str):
        """Save all tier models to disk."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save tier-specific models
        for tier_name in self.trained_tiers:
            tier_path = path / tier_name
            tier_path.mkdir(exist_ok=True)
            
            if tier_name in self.score_models:
                self.score_models[tier_name].save(str(tier_path / 'score_model.joblib'))
            if tier_name in self.goals_models:
                self.goals_models[tier_name].save(str(tier_path / 'goals_model.joblib'))
            if tier_name in self.result_models:
                self.result_models[tier_name].save(str(tier_path / 'result_model.joblib'))
        
        # Save config
        joblib.dump({
            'tiers': self.tiers,
            'tier_stats': self.tier_stats,
            'feature_names': self.feature_names,
            'trained_tiers': self.trained_tiers
        }, path / 'tier_config.joblib')
        
        logger.info(f"Saved tiered models to {path}")
    
    def load(self, model_dir: str):
        """Load all tier models from disk."""
        path = Path(model_dir)
        
        config_data = joblib.load(path / 'tier_config.joblib')
        self.tiers = config_data['tiers']
        self.tier_stats = config_data['tier_stats']
        self.feature_names = config_data['feature_names']
        self.trained_tiers = config_data.get('trained_tiers', [])
        
        # Load tier models
        for tier_name in self.trained_tiers:
            tier_path = path / tier_name
            if not tier_path.exists():
                continue
            
            if (tier_path / 'score_model.joblib').exists():
                self.score_models[tier_name] = ScorePredictor(self.config)
                self.score_models[tier_name].load(str(tier_path / 'score_model.joblib'))
            
            if (tier_path / 'goals_model.joblib').exists():
                self.goals_models[tier_name] = TotalGoalsPredictor(self.config)
                self.goals_models[tier_name].load(str(tier_path / 'goals_model.joblib'))
            
            if (tier_path / 'result_model.joblib').exists():
                self.result_models[tier_name] = ResultPredictor(self.config)
                self.result_models[tier_name].load(str(tier_path / 'result_model.joblib'))
        
        logger.info(f"Loaded tiered models from {path}")
    
    def get_tier_summary(self) -> pd.DataFrame:
        """Get summary of all tiers."""
        rows = []
        for tier in self.tiers:
            stats = self.tier_stats.get(tier.name, {})
            rows.append({
                'tier': tier.name,
                'coverage_levels': str(tier.coverage_levels),
                'description': tier.description,
                'n_matches': stats.get('n_matches', 0),
                'n_features': stats.get('n_features', 0),
                'status': stats.get('status', 'not_trained'),
                'has_model': tier.name in self.trained_tiers
            })
        return pd.DataFrame(rows)