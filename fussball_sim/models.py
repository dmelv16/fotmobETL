"""
Model training and evaluation module
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, log_loss
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.multioutput import MultiOutputRegressor
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    predictions: Optional[np.ndarray] = None
    cv_scores: Optional[np.ndarray] = None


class ScorePredictor:
    """Predicts exact home and away scores."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
    def _prepare_features(self, X: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """Prepare feature matrix, handling missing values."""
        X = X.copy()
        
        # Drop non-feature columns
        drop_cols = ['match_id', 'has_player_data', 'has_coach_data', 
                     'has_team_data', 'player_coverage']
        feature_cols = [c for c in X.columns if c not in drop_cols]
        
        if is_training:
            # During training, learn the feature names
            self.feature_names = feature_cols
        else:
            # During prediction, align to training features
            for feat in self.feature_names:
                if feat not in X.columns:
                    X[feat] = np.nan
            feature_cols = self.feature_names
        
        X_features = X[feature_cols].copy()
        
        # Impute missing values
        if self.imputer is None:
            strategy = self.config.features.fill_missing_with
            self.imputer = SimpleImputer(
                strategy=strategy if strategy != 'zero' else 'constant', 
                fill_value=0 if strategy == 'zero' else None
            )
            X_imputed = self.imputer.fit_transform(X_features)
        else:
            X_imputed = self.imputer.transform(X_features)
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)
            
        return X_scaled
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> ModelResults:
        """Train score prediction model."""
        logger.info("Training score prediction model...")
        
        X_prepared = self._prepare_features(X, is_training=True)
        y_scores = y[['home_score', 'away_score']].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y_scores, 
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state
        )
        
        # Multi-output regressor for both scores
        base_model = XGBRegressor(**self.config.model.xgb_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_rounded = np.round(y_pred).clip(0, None).astype(int)
        
        # Metrics
        metrics = {
            'home_mae': mean_absolute_error(y_test[:, 0], y_pred_rounded[:, 0]),
            'away_mae': mean_absolute_error(y_test[:, 1], y_pred_rounded[:, 1]),
            'home_rmse': np.sqrt(mean_squared_error(y_test[:, 0], y_pred_rounded[:, 0])),
            'away_rmse': np.sqrt(mean_squared_error(y_test[:, 1], y_pred_rounded[:, 1])),
            'exact_score_accuracy': np.mean((y_pred_rounded == y_test).all(axis=1)),
            'result_accuracy': self._result_accuracy(y_test, y_pred_rounded)
        }
        
        # Feature importance
        importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Score Model - Home MAE: {metrics['home_mae']:.3f}, Away MAE: {metrics['away_mae']:.3f}")
        
        return ModelResults(
            model_name='score_predictor',
            metrics=metrics,
            feature_importance=feature_importance,
            predictions=y_pred_rounded
        )
    
    def _result_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy of implied match result."""
        true_results = np.sign(y_true[:, 0] - y_true[:, 1])
        pred_results = np.sign(y_pred[:, 0] - y_pred[:, 1])
        return np.mean(true_results == pred_results)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict scores for new matches."""
        X_prepared = self._prepare_features(X, is_training=False)
        predictions = self.model.predict(X_prepared)
        return np.round(predictions).clip(0, None).astype(int)
    
    def _train_full(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train on full dataset (for walk-forward validation)."""
        X_prepared = self._prepare_features(X, is_training=True)
        y_scores = y[['home_score', 'away_score']].values
        
        base_model = XGBRegressor(**self.config.model.xgb_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_prepared, y_scores)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        
    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.imputer = data['imputer']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']


class TotalGoalsPredictor:
    """Predicts total goals in a match."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
    def _prepare_features(self, X: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        X = X.copy()
        
        drop_cols = ['match_id', 'has_player_data', 'has_coach_data', 
                     'has_team_data', 'player_coverage']
        feature_cols = [c for c in X.columns if c not in drop_cols]
        
        if is_training:
            self.feature_names = feature_cols
        else:
            for feat in self.feature_names:
                if feat not in X.columns:
                    X[feat] = np.nan
            feature_cols = self.feature_names
        
        X_features = X[feature_cols].copy()
        
        if self.imputer is None:
            strategy = self.config.features.fill_missing_with
            self.imputer = SimpleImputer(
                strategy=strategy if strategy != 'zero' else 'constant',
                fill_value=0 if strategy == 'zero' else None
            )
            X_imputed = self.imputer.fit_transform(X_features)
        else:
            X_imputed = self.imputer.transform(X_features)
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)
            
        return X_scaled
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> ModelResults:
        """Train total goals prediction model."""
        logger.info("Training total goals model...")
        
        X_prepared = self._prepare_features(X, is_training=True)
        y_goals = y['total_goals'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y_goals,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state
        )
        
        self.model = XGBRegressor(**self.config.model.xgb_params)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_rounded = np.round(y_pred).clip(0, None).astype(int)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred_rounded),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rounded)),
            'r2': r2_score(y_test, y_pred),
            'exact_accuracy': np.mean(y_pred_rounded == y_test),
            'within_1_accuracy': np.mean(np.abs(y_pred_rounded - y_test) <= 1)
        }
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        cv_scores = cross_val_score(
            XGBRegressor(**self.config.model.xgb_params),
            X_prepared, y_goals, cv=self.config.model.cv_folds,
            scoring='neg_mean_absolute_error'
        )
        
        logger.info(f"Total Goals Model - MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
        
        return ModelResults(
            model_name='total_goals_predictor',
            metrics=metrics,
            feature_importance=feature_importance,
            predictions=y_pred_rounded,
            cv_scores=-cv_scores
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._prepare_features(X, is_training=False)
        predictions = self.model.predict(X_prepared)
        return np.round(predictions).clip(0, None).astype(int)
    
    def _train_full(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train on full dataset (for walk-forward validation)."""
        X_prepared = self._prepare_features(X, is_training=True)
        y_goals = y['total_goals'].values
        
        self.model = XGBRegressor(**self.config.model.xgb_params)
        self.model.fit(X_prepared, y_goals)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        
    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.imputer = data['imputer']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']


class ResultPredictor:
    """Predicts match result: Home Win, Draw, or Away Win."""
    
    RESULT_LABELS = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
    def _prepare_features(self, X: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        X = X.copy()
        
        drop_cols = ['match_id', 'has_player_data', 'has_coach_data',
                     'has_team_data', 'player_coverage']
        feature_cols = [c for c in X.columns if c not in drop_cols]
        
        if is_training:
            self.feature_names = feature_cols
        else:
            for feat in self.feature_names:
                if feat not in X.columns:
                    X[feat] = np.nan
            feature_cols = self.feature_names
        
        X_features = X[feature_cols].copy()
        
        if self.imputer is None:
            strategy = self.config.features.fill_missing_with
            self.imputer = SimpleImputer(
                strategy=strategy if strategy != 'zero' else 'constant',
                fill_value=0 if strategy == 'zero' else None
            )
            X_imputed = self.imputer.fit_transform(X_features)
        else:
            X_imputed = self.imputer.transform(X_features)
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)
            
        return X_scaled
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> ModelResults:
        """Train match result classifier."""
        logger.info("Training result prediction model...")
        
        X_prepared = self._prepare_features(X, is_training=True)
        y_result = y['result'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y_result,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y_result
        )
        
        xgb_params = self.config.model.xgb_params.copy()
        xgb_params['objective'] = 'multi:softprob'
        xgb_params['num_class'] = 3
        
        self.model = XGBClassifier(**xgb_params)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba),
        }
        
        for i, label in self.RESULT_LABELS.items():
            y_binary_true = (y_test == i).astype(int)
            y_binary_pred = (y_pred == i).astype(int)
            tp = np.sum((y_binary_pred == 1) & (y_binary_true == 1))
            metrics[f'{label.lower().replace(" ", "_")}_precision'] = tp / max(np.sum(y_binary_pred == 1), 1)
            metrics[f'{label.lower().replace(" ", "_")}_recall'] = tp / max(np.sum(y_binary_true == 1), 1)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        cv = StratifiedKFold(n_splits=self.config.model.cv_folds, shuffle=True, 
                            random_state=self.config.model.random_state)
        cv_scores = cross_val_score(
            XGBClassifier(**xgb_params),
            X_prepared, y_result, cv=cv, scoring='accuracy'
        )
        
        logger.info(f"Result Model - Accuracy: {metrics['accuracy']:.3%}")
        
        return ModelResults(
            model_name='result_predictor',
            metrics=metrics,
            feature_importance=feature_importance,
            predictions=y_pred,
            cv_scores=cv_scores
        )
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return predictions and probabilities."""
        X_prepared = self._prepare_features(X, is_training=False)
        predictions = self.model.predict(X_prepared)
        probabilities = self.model.predict_proba(X_prepared)
        return predictions, probabilities
    
    def _train_full(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train on full dataset (for walk-forward validation)."""
        X_prepared = self._prepare_features(X, is_training=True)
        y_result = y['result'].values
        
        xgb_params = self.config.model.xgb_params.copy()
        xgb_params['objective'] = 'multi:softprob'
        xgb_params['num_class'] = 3
        
        self.model = XGBClassifier(**xgb_params)
        self.model.fit(X_prepared, y_result)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        
    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.imputer = data['imputer']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']