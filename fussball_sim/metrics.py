"""
Comprehensive Metrics Module for Soccer Prediction

Calculates:
- Over/Under thresholds (1.5, 2.5, 3.5, 4.5)
- Double Chance (Home/Draw, Away/Draw, Home/Away)
- Draw probability calibration
- Result prediction accuracy
- Score prediction accuracy
- Brier scores for probability calibration
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OverUnderMetrics:
    """Metrics for Over/Under predictions at a threshold."""
    threshold: float
    accuracy: float
    precision_over: float
    recall_over: float
    precision_under: float
    recall_under: float
    n_actual_over: int
    n_actual_under: int
    n_pred_over: int
    n_pred_under: int


@dataclass
class DoubleChanceMetrics:
    """Metrics for Double Chance predictions."""
    home_or_draw_accuracy: float
    away_or_draw_accuracy: float
    home_or_away_accuracy: float  # No draw
    n_home_or_draw: int
    n_away_or_draw: int
    n_home_or_away: int


@dataclass 
class DrawMetrics:
    """Metrics for Draw prediction quality."""
    draw_accuracy: float  # When we predict draw, how often correct
    draw_recall: float    # Of actual draws, how many did we catch
    draw_calibration: Dict[str, float]  # Binned calibration
    brier_score: float
    n_actual_draws: int
    n_predicted_draws: int


@dataclass
class TierMetrics:
    """Metrics broken down by tier."""
    tier_name: str
    n_matches: int
    result_accuracy: float
    over_under: Dict[float, OverUnderMetrics]
    double_chance: DoubleChanceMetrics
    draw_metrics: DrawMetrics
    score_mae: Tuple[float, float]  # home, away
    exact_score_rate: float


def calculate_over_under_metrics(
    pred_total: np.ndarray,
    actual_total: np.ndarray,
    threshold: float
) -> OverUnderMetrics:
    """Calculate Over/Under metrics for a specific threshold."""
    
    # Ensure numpy arrays
    pred_total = np.asarray(pred_total)
    actual_total = np.asarray(actual_total)
    
    pred_over = pred_total > threshold
    actual_over = actual_total > threshold
    
    # Accuracy
    accuracy = float((pred_over == actual_over).mean())
    
    # Precision/Recall for Over
    tp_over = int((pred_over & actual_over).sum())
    fp_over = int((pred_over & ~actual_over).sum())
    fn_over = int((~pred_over & actual_over).sum())
    
    precision_over = tp_over / max(tp_over + fp_over, 1)
    recall_over = tp_over / max(tp_over + fn_over, 1)
    
    # Precision/Recall for Under
    tp_under = int((~pred_over & ~actual_over).sum())
    fp_under = int((~pred_over & actual_over).sum())
    fn_under = int((pred_over & ~actual_over).sum())
    
    precision_under = tp_under / max(tp_under + fp_under, 1)
    recall_under = tp_under / max(tp_under + fn_under, 1)
    
    return OverUnderMetrics(
        threshold=threshold,
        accuracy=accuracy,
        precision_over=precision_over,
        recall_over=recall_over,
        precision_under=precision_under,
        recall_under=recall_under,
        n_actual_over=int(actual_over.sum()),
        n_actual_under=int((~actual_over).sum()),
        n_pred_over=int(pred_over.sum()),
        n_pred_under=int((~pred_over).sum())
    )


def calculate_double_chance_metrics(
    pred_result: np.ndarray,  # 0=Away, 1=Draw, 2=Home
    actual_result: np.ndarray,
    prob_home: Optional[np.ndarray] = None,
    prob_draw: Optional[np.ndarray] = None,
    prob_away: Optional[np.ndarray] = None
) -> DoubleChanceMetrics:
    """Calculate Double Chance accuracy metrics."""
    
    # Ensure numpy arrays
    pred_result = np.asarray(pred_result)
    actual_result = np.asarray(actual_result)
    
    # Actual outcomes
    actual_home = actual_result == 2
    actual_draw = actual_result == 1
    actual_away = actual_result == 0
    
    actual_home_or_draw = actual_home | actual_draw
    actual_away_or_draw = actual_away | actual_draw
    actual_home_or_away = actual_home | actual_away
    
    # If we have probabilities, use them for double chance predictions
    if prob_home is not None and prob_draw is not None and prob_away is not None:
        prob_home = np.asarray(prob_home)
        prob_draw = np.asarray(prob_draw)
        prob_away = np.asarray(prob_away)
        pred_home_or_draw = (prob_home + prob_draw) > 0.5
        pred_away_or_draw = (prob_away + prob_draw) > 0.5
        pred_home_or_away = (prob_home + prob_away) > 0.5
    else:
        # Fall back to result-based
        pred_home_or_draw = pred_result != 0
        pred_away_or_draw = pred_result != 2
        pred_home_or_away = pred_result != 1
    
    return DoubleChanceMetrics(
        home_or_draw_accuracy=float((pred_home_or_draw == actual_home_or_draw).mean()),
        away_or_draw_accuracy=float((pred_away_or_draw == actual_away_or_draw).mean()),
        home_or_away_accuracy=float((pred_home_or_away == actual_home_or_away).mean()),
        n_home_or_draw=int(actual_home_or_draw.sum()),
        n_away_or_draw=int(actual_away_or_draw.sum()),
        n_home_or_away=int(actual_home_or_away.sum())
    )


def calculate_draw_metrics(
    pred_result: np.ndarray,
    actual_result: np.ndarray,
    prob_draw: np.ndarray
) -> DrawMetrics:
    """Calculate Draw prediction quality metrics."""
    
    # Ensure numpy arrays
    pred_result = np.asarray(pred_result)
    actual_result = np.asarray(actual_result)
    prob_draw = np.asarray(prob_draw)
    
    actual_draw = actual_result == 1
    pred_draw = pred_result == 1
    
    # Accuracy (when we predict draw)
    if pred_draw.sum() > 0:
        draw_accuracy = float(actual_draw[pred_draw].mean())
    else:
        draw_accuracy = 0.0
    
    # Recall (of actual draws, how many caught)
    if actual_draw.sum() > 0:
        draw_recall = float(pred_draw[actual_draw].mean())
    else:
        draw_recall = 0.0
    
    # Calibration: bin draw probabilities and check actual draw rate
    calibration = {}
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
    
    for low, high in bins:
        mask = (prob_draw >= low) & (prob_draw < high)
        if mask.sum() > 0:
            actual_rate = float(actual_draw[mask].mean())
            expected_rate = float(prob_draw[mask].mean())
            calibration[f"{low:.1f}-{high:.1f}"] = {
                'n': int(mask.sum()),
                'predicted_prob': expected_rate,
                'actual_rate': actual_rate,
                'calibration_error': abs(expected_rate - actual_rate)
            }
    
    # Brier score for draw probability
    brier_score = float(((prob_draw - actual_draw.astype(float)) ** 2).mean())
    
    return DrawMetrics(
        draw_accuracy=draw_accuracy,
        draw_recall=draw_recall,
        draw_calibration=calibration,
        brier_score=brier_score,
        n_actual_draws=int(actual_draw.sum()),
        n_predicted_draws=int(pred_draw.sum())
    )


def calculate_all_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive metrics from a predictions DataFrame.
    
    Expected columns:
    - pred_home_score, pred_away_score, pred_total_goals
    - actual_home_score, actual_away_score, actual_total_goals
    - pred_result (string: 'Home Win', 'Draw', 'Away Win')
    - actual_result (string)
    - prob_home_win, prob_draw, prob_away_win
    """
    
    # Convert results to numeric
    result_map = {'Away Win': 0, 'Draw': 1, 'Home Win': 2}
    pred_result = df['pred_result'].map(result_map).values.astype(int)
    actual_result = df['actual_result'].map(result_map).values.astype(int)
    
    pred_total = df['pred_total_goals'].values.astype(float)
    actual_total = df['actual_total_goals'].values.astype(float)
    
    prob_home = df['prob_home_win'].values.astype(float)
    prob_draw = df['prob_draw'].values.astype(float)
    prob_away = df['prob_away_win'].values.astype(float)
    
    metrics = {
        'n_matches': len(df),
        
        # Basic result accuracy
        'result_accuracy': float((pred_result == actual_result).mean()),
        
        # Score metrics
        'home_score_mae': float(np.abs(df['pred_home_score'].values - df['actual_home_score'].values).mean()),
        'away_score_mae': float(np.abs(df['pred_away_score'].values - df['actual_away_score'].values).mean()),
        'total_goals_mae': float(np.abs(pred_total - actual_total).mean()),
        'exact_score_rate': float((
            (df['pred_home_score'] == df['actual_home_score']) &
            (df['pred_away_score'] == df['actual_away_score'])
        ).mean()),
        
        # Over/Under for each threshold
        'over_under': {},
        
        # Double Chance
        'double_chance': None,
        
        # Draw specific
        'draw': None,
        
        # Brier scores for all outcomes
        'brier_home': float(((prob_home - (actual_result == 2).astype(float)) ** 2).mean()),
        'brier_draw': float(((prob_draw - (actual_result == 1).astype(float)) ** 2).mean()),
        'brier_away': float(((prob_away - (actual_result == 0).astype(float)) ** 2).mean()),
    }
    
    # Over/Under for each threshold
    for threshold in [1.5, 2.5, 3.5, 4.5]:
        metrics['over_under'][threshold] = calculate_over_under_metrics(
            pred_total, actual_total, threshold
        )
    
    # Double Chance
    metrics['double_chance'] = calculate_double_chance_metrics(
        pred_result, actual_result, prob_home, prob_draw, prob_away
    )
    
    # Draw metrics
    metrics['draw'] = calculate_draw_metrics(pred_result, actual_result, prob_draw)
    
    return metrics


def calculate_metrics_by_tier(df: pd.DataFrame) -> Dict[str, TierMetrics]:
    """Calculate metrics broken down by coverage tier."""
    
    tier_metrics = {}
    
    for tier_name, tier_df in df.groupby('coverage_level'):
        if len(tier_df) < 10:
            continue
        
        all_metrics = calculate_all_metrics(tier_df)
        
        tier_metrics[tier_name] = TierMetrics(
            tier_name=tier_name,
            n_matches=len(tier_df),
            result_accuracy=all_metrics['result_accuracy'],
            over_under=all_metrics['over_under'],
            double_chance=all_metrics['double_chance'],
            draw_metrics=all_metrics['draw'],
            score_mae=(all_metrics['home_score_mae'], all_metrics['away_score_mae']),
            exact_score_rate=all_metrics['exact_score_rate']
        )
    
    return tier_metrics


def print_comprehensive_report(df: pd.DataFrame, by_tier: bool = True):
    """Print a comprehensive metrics report."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE PREDICTION METRICS REPORT")
    print("="*70)
    
    # Overall metrics
    metrics = calculate_all_metrics(df)
    
    print(f"\n📊 OVERALL ({metrics['n_matches']} matches)")
    print("-"*50)
    
    # Result accuracy
    print(f"\n🎯 Result Prediction (1X2):")
    print(f"   Accuracy: {metrics['result_accuracy']:.1%}")
    print(f"   Brier Scores - Home: {metrics['brier_home']:.4f}, "
          f"Draw: {metrics['brier_draw']:.4f}, Away: {metrics['brier_away']:.4f}")
    
    # Over/Under
    print(f"\n⚽ Over/Under Goals:")
    for threshold, ou in metrics['over_under'].items():
        print(f"   {threshold}: Acc={ou.accuracy:.1%} | "
              f"Over P/R={ou.precision_over:.1%}/{ou.recall_over:.1%} | "
              f"Under P/R={ou.precision_under:.1%}/{ou.recall_under:.1%}")
    
    # Double Chance
    dc = metrics['double_chance']
    print(f"\n🎲 Double Chance:")
    print(f"   Home or Draw (1X): {dc.home_or_draw_accuracy:.1%} (n={dc.n_home_or_draw})")
    print(f"   Away or Draw (X2): {dc.away_or_draw_accuracy:.1%} (n={dc.n_away_or_draw})")
    print(f"   Home or Away (12): {dc.home_or_away_accuracy:.1%} (n={dc.n_home_or_away})")
    
    # Draw metrics
    dm = metrics['draw']
    print(f"\n🤝 Draw Prediction:")
    print(f"   Accuracy (when predicting draw): {dm.draw_accuracy:.1%}")
    print(f"   Recall (of actual draws): {dm.draw_recall:.1%}")
    print(f"   Brier Score: {dm.brier_score:.4f}")
    print(f"   Actual draws: {dm.n_actual_draws}, Predicted: {dm.n_predicted_draws}")
    print(f"\n   Draw Probability Calibration:")
    for bin_name, cal_data in dm.draw_calibration.items():
        print(f"     {bin_name}: Pred={cal_data['predicted_prob']:.1%} "
              f"Actual={cal_data['actual_rate']:.1%} (n={cal_data['n']})")
    
    # Score prediction
    print(f"\n📈 Score Prediction:")
    print(f"   Home MAE: {metrics['home_score_mae']:.3f}")
    print(f"   Away MAE: {metrics['away_score_mae']:.3f}")
    print(f"   Total MAE: {metrics['total_goals_mae']:.3f}")
    print(f"   Exact Score: {metrics['exact_score_rate']:.1%}")
    
    # By tier if requested
    if by_tier and 'coverage_level' in df.columns:
        print("\n" + "="*70)
        print("METRICS BY TIER")
        print("="*70)
        
        tier_metrics = calculate_metrics_by_tier(df)
        
        for tier_name, tm in tier_metrics.items():
            print(f"\n📦 Tier: {tier_name} ({tm.n_matches} matches)")
            print("-"*40)
            print(f"   Result Accuracy: {tm.result_accuracy:.1%}")
            print(f"   Score MAE: Home={tm.score_mae[0]:.3f}, Away={tm.score_mae[1]:.3f}")
            print(f"   Exact Score: {tm.exact_score_rate:.1%}")
            
            # Over/Under summary
            print(f"   Over/Under Accuracy:")
            for threshold, ou in tm.over_under.items():
                print(f"     {threshold}: {ou.accuracy:.1%}")
            
            # Double Chance summary
            print(f"   Double Chance: 1X={tm.double_chance.home_or_draw_accuracy:.1%}, "
                  f"X2={tm.double_chance.away_or_draw_accuracy:.1%}")
            
            # Draw
            print(f"   Draw: Acc={tm.draw_metrics.draw_accuracy:.1%}, "
                  f"Recall={tm.draw_metrics.draw_recall:.1%}")


def metrics_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert metrics to a flat DataFrame for easy export."""
    
    rows = []
    
    # Overall
    metrics = calculate_all_metrics(df)
    row = {
        'tier': 'ALL',
        'n_matches': metrics['n_matches'],
        'result_accuracy': metrics['result_accuracy'],
        'home_score_mae': metrics['home_score_mae'],
        'away_score_mae': metrics['away_score_mae'],
        'total_goals_mae': metrics['total_goals_mae'],
        'exact_score_rate': metrics['exact_score_rate'],
        'brier_home': metrics['brier_home'],
        'brier_draw': metrics['brier_draw'],
        'brier_away': metrics['brier_away'],
    }
    
    # Over/Under
    for threshold, ou in metrics['over_under'].items():
        row[f'ou_{threshold}_accuracy'] = ou.accuracy
        row[f'ou_{threshold}_over_precision'] = ou.precision_over
        row[f'ou_{threshold}_over_recall'] = ou.recall_over
    
    # Double Chance
    dc = metrics['double_chance']
    row['dc_home_or_draw'] = dc.home_or_draw_accuracy
    row['dc_away_or_draw'] = dc.away_or_draw_accuracy
    row['dc_home_or_away'] = dc.home_or_away_accuracy
    
    # Draw
    dm = metrics['draw']
    row['draw_accuracy'] = dm.draw_accuracy
    row['draw_recall'] = dm.draw_recall
    row['draw_brier'] = dm.brier_score
    
    rows.append(row)
    
    # By tier
    if 'coverage_level' in df.columns:
        for tier_name, tier_df in df.groupby('coverage_level'):
            if len(tier_df) < 10:
                continue
            
            tier_metrics = calculate_all_metrics(tier_df)
            tier_row = {
                'tier': tier_name,
                'n_matches': tier_metrics['n_matches'],
                'result_accuracy': tier_metrics['result_accuracy'],
                'home_score_mae': tier_metrics['home_score_mae'],
                'away_score_mae': tier_metrics['away_score_mae'],
                'total_goals_mae': tier_metrics['total_goals_mae'],
                'exact_score_rate': tier_metrics['exact_score_rate'],
                'brier_home': tier_metrics['brier_home'],
                'brier_draw': tier_metrics['brier_draw'],
                'brier_away': tier_metrics['brier_away'],
            }
            
            for threshold, ou in tier_metrics['over_under'].items():
                tier_row[f'ou_{threshold}_accuracy'] = ou.accuracy
                tier_row[f'ou_{threshold}_over_precision'] = ou.precision_over
                tier_row[f'ou_{threshold}_over_recall'] = ou.recall_over
            
            dc = tier_metrics['double_chance']
            tier_row['dc_home_or_draw'] = dc.home_or_draw_accuracy
            tier_row['dc_away_or_draw'] = dc.away_or_draw_accuracy
            tier_row['dc_home_or_away'] = dc.home_or_away_accuracy
            
            dm = tier_metrics['draw']
            tier_row['draw_accuracy'] = dm.draw_accuracy
            tier_row['draw_recall'] = dm.draw_recall
            tier_row['draw_brier'] = dm.brier_score
            
            rows.append(tier_row)
    
    return pd.DataFrame(rows)