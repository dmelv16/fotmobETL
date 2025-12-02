"""
Prediction service - unified interface for making predictions on new matches
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class MatchPrediction:
    """Complete prediction for a single match."""
    match_id: int
    home_team: str
    away_team: str
    
    # Score prediction
    predicted_home_score: int
    predicted_away_score: int
    
    # Total goals
    predicted_total_goals: int
    
    # Result prediction with probabilities
    predicted_result: str  # "Home Win", "Draw", "Away Win"
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    
    # Confidence indicators
    has_player_data: bool
    has_coach_data: bool
    has_team_data: bool
    player_coverage: float
    
    def to_dict(self) -> Dict:
        return {
            'match_id': self.match_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'predicted_score': f"{self.predicted_home_score}-{self.predicted_away_score}",
            'predicted_home_score': self.predicted_home_score,
            'predicted_away_score': self.predicted_away_score,
            'predicted_total_goals': self.predicted_total_goals,
            'predicted_result': self.predicted_result,
            'home_win_prob': round(self.home_win_prob, 3),
            'draw_prob': round(self.draw_prob, 3),
            'away_win_prob': round(self.away_win_prob, 3),
            'data_quality': {
                'has_player_data': self.has_player_data,
                'has_coach_data': self.has_coach_data,
                'has_team_data': self.has_team_data,
                'player_coverage': round(self.player_coverage, 2)
            }
        }


class PredictionService:
    """
    Unified prediction service that combines all models.
    """
    RESULT_LABELS = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    
    def __init__(self, config, data_loader, feature_engineer):
        self.config = config
        self.loader = data_loader
        self.engineer = feature_engineer
        
        self.score_model = None
        self.goals_model = None
        self.result_model = None
        
    def load_models(self, model_dir: str):
        """Load all trained models from directory."""
        from models import ScorePredictor, TotalGoalsPredictor, ResultPredictor
        
        model_path = Path(model_dir)
        
        if (model_path / 'score_model.joblib').exists():
            self.score_model = ScorePredictor(self.config)
            self.score_model.load(str(model_path / 'score_model.joblib'))
            logger.info("Loaded score prediction model")
            
        if (model_path / 'goals_model.joblib').exists():
            self.goals_model = TotalGoalsPredictor(self.config)
            self.goals_model.load(str(model_path / 'goals_model.joblib'))
            logger.info("Loaded total goals model")
            
        if (model_path / 'result_model.joblib').exists():
            self.result_model = ResultPredictor(self.config)
            self.result_model.load(str(model_path / 'result_model.joblib'))
            logger.info("Loaded result prediction model")
    
    def predict_match(self, match_row: pd.Series) -> MatchPrediction:
        """
        Generate predictions for a single match.
        
        Args:
            match_row: Row from match_details table
            
        Returns:
            MatchPrediction object with all predictions
        """
        # Extract features for this match
        match_features = self.engineer.extract_match_features(match_row)
        
        # Convert to DataFrame for model input
        X = pd.DataFrame([{'match_id': match_features.match_id, **match_features.features}])
        X['has_player_data'] = match_features.has_player_data
        X['has_coach_data'] = match_features.has_coach_data
        X['has_team_data'] = match_features.has_team_data
        X['player_coverage'] = match_features.player_coverage
        
        # Ensure all expected features exist (fill with NaN if missing)
        if self.score_model and self.score_model.feature_names:
            for feat in self.score_model.feature_names:
                if feat not in X.columns:
                    X[feat] = np.nan
        
        # Make predictions
        pred_home_score, pred_away_score = 0, 0
        pred_total_goals = 0
        pred_result = 1  # Default to draw
        probs = [0.33, 0.34, 0.33]
        
        if self.score_model:
            scores = self.score_model.predict(X)
            pred_home_score, pred_away_score = scores[0]
            
        if self.goals_model:
            pred_total_goals = self.goals_model.predict(X)[0]
        else:
            pred_total_goals = pred_home_score + pred_away_score
            
        if self.result_model:
            results, probs_arr = self.result_model.predict(X)
            pred_result = results[0]
            probs = probs_arr[0]
        else:
            # Infer from score prediction
            if pred_home_score > pred_away_score:
                pred_result = 2
            elif pred_home_score < pred_away_score:
                pred_result = 0
            else:
                pred_result = 1
        
        return MatchPrediction(
            match_id=match_row['match_id'],
            home_team=match_row['home_team_name'],
            away_team=match_row['away_team_name'],
            predicted_home_score=int(pred_home_score),
            predicted_away_score=int(pred_away_score),
            predicted_total_goals=int(pred_total_goals),
            predicted_result=self.RESULT_LABELS[pred_result],
            home_win_prob=float(probs[2]),
            draw_prob=float(probs[1]),
            away_win_prob=float(probs[0]),
            has_player_data=match_features.has_player_data,
            has_coach_data=match_features.has_coach_data,
            has_team_data=match_features.has_team_data,
            player_coverage=match_features.player_coverage
        )
    
    def predict_matches(self, matches: pd.DataFrame) -> List[MatchPrediction]:
        """Generate predictions for multiple matches."""
        predictions = []
        for _, row in matches.iterrows():
            try:
                pred = self.predict_match(row)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict match {row['match_id']}: {e}")
        return predictions
    
    def predict_upcoming(self, 
                         league_ids: Optional[List[int]] = None,
                         limit: int = 50) -> List[MatchPrediction]:
        """
        Predict upcoming matches (not yet finished).
        Note: Requires modifying data_loader to fetch unfinished matches.
        """
        # This would query matches where finished = 0
        query = """
        SELECT TOP (:limit)
            id, match_id, match_name, league_id, league_name,
            match_time_utc, home_team_id, home_team_name,
            away_team_id, away_team_name
        FROM match_details
        WHERE finished = 0 AND cancelled = 0
        """
        if league_ids:
            query += f" AND league_id IN ({','.join(map(str, league_ids))})"
        query += " ORDER BY match_time_utc"
        
        from sqlalchemy import text
        matches = pd.read_sql(
            text(query), 
            self.loader.engine, 
            params={'limit': limit}
        )
        
        return self.predict_matches(matches)
    
    def evaluate_predictions(self, 
                            predictions: List[MatchPrediction],
                            actuals: pd.DataFrame) -> Dict:
        """
        Evaluate prediction quality against actual results.
        
        Args:
            predictions: List of MatchPrediction objects
            actuals: DataFrame with match_id, home_team_score, away_team_score
        """
        pred_df = pd.DataFrame([p.to_dict() for p in predictions])
        merged = pred_df.merge(
            actuals[['match_id', 'home_team_score', 'away_team_score']], 
            on='match_id'
        )
        
        # Calculate actual results
        merged['actual_result'] = merged.apply(
            lambda r: 'Home Win' if r['home_team_score'] > r['away_team_score']
                      else ('Away Win' if r['home_team_score'] < r['away_team_score'] else 'Draw'),
            axis=1
        )
        merged['actual_total'] = merged['home_team_score'] + merged['away_team_score']
        
        metrics = {
            'n_predictions': len(merged),
            'result_accuracy': (merged['predicted_result'] == merged['actual_result']).mean(),
            'exact_score_accuracy': (
                (merged['predicted_home_score'] == merged['home_team_score']) &
                (merged['predicted_away_score'] == merged['away_team_score'])
            ).mean(),
            'total_goals_mae': np.abs(
                merged['predicted_total_goals'] - merged['actual_total']
            ).mean(),
            'home_score_mae': np.abs(
                merged['predicted_home_score'] - merged['home_team_score']
            ).mean(),
            'away_score_mae': np.abs(
                merged['predicted_away_score'] - merged['away_team_score']
            ).mean()
        }
        
        # Breakdown by data quality
        for has_data in [True, False]:
            subset = merged[merged['data_quality'].apply(lambda x: x['has_player_data']) == has_data]
            if len(subset) > 0:
                label = 'with_player_data' if has_data else 'without_player_data'
                metrics[f'{label}_result_accuracy'] = (
                    subset['predicted_result'] == subset['actual_result']
                ).mean()
                metrics[f'{label}_count'] = len(subset)
        
        return metrics