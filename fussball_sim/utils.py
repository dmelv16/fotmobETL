"""
Utility functions and helpers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path


def plot_feature_importance(feature_importance: pd.DataFrame, 
                           top_n: int = 20,
                           title: str = "Feature Importance",
                           figsize: tuple = (12, 8)) -> plt.Figure:
    """Plot feature importance as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = feature_importance.head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         labels: List[str] = ['Away Win', 'Draw', 'Home Win'],
                         figsize: tuple = (8, 6)) -> plt.Figure:
    """Plot confusion matrix for result predictions."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Match Result Confusion Matrix')
    
    plt.tight_layout()
    return fig


def plot_prediction_distribution(predictions: List, 
                                 actuals: Optional[pd.DataFrame] = None,
                                 figsize: tuple = (15, 5)) -> plt.Figure:
    """Plot distribution of predictions vs actuals."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    pred_df = pd.DataFrame([p.to_dict() for p in predictions])
    
    # Result probabilities
    ax = axes[0]
    prob_data = pd.DataFrame({
        'Home Win': pred_df['home_win_prob'],
        'Draw': pred_df['draw_prob'],
        'Away Win': pred_df['away_win_prob']
    })
    prob_data.boxplot(ax=ax)
    ax.set_title('Result Probability Distribution')
    ax.set_ylabel('Probability')
    
    # Score distribution
    ax = axes[1]
    ax.hist(pred_df['predicted_home_score'], bins=range(0, 8), alpha=0.5, label='Home')
    ax.hist(pred_df['predicted_away_score'], bins=range(0, 8), alpha=0.5, label='Away')
    ax.set_xlabel('Goals')
    ax.set_ylabel('Frequency')
    ax.set_title('Predicted Score Distribution')
    ax.legend()
    
    # Total goals
    ax = axes[2]
    ax.hist(pred_df['predicted_total_goals'], bins=range(0, 10), alpha=0.7)
    if actuals is not None:
        merged = pred_df.merge(actuals[['match_id', 'home_team_score', 'away_team_score']], on='match_id')
        merged['actual_total'] = merged['home_team_score'] + merged['away_team_score']
        ax.hist(merged['actual_total'], bins=range(0, 10), alpha=0.5, label='Actual')
        ax.legend()
    ax.set_xlabel('Total Goals')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Goals Distribution')
    
    plt.tight_layout()
    return fig


def calculate_betting_metrics(predictions: List,
                             actuals: pd.DataFrame,
                             stake: float = 1.0) -> Dict:
    """
    Calculate betting performance metrics.
    Assumes decimal odds and flat staking.
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
    
    # Simple betting simulation: bet on most likely outcome
    merged['bet_won'] = merged['predicted_result'] == merged['actual_result']
    
    # Calculate implied odds from probabilities
    merged['implied_odds'] = merged.apply(
        lambda r: 1 / max(r['home_win_prob'], r['draw_prob'], r['away_win_prob'], 0.01),
        axis=1
    )
    
    # ROI calculation (simplified)
    total_staked = len(merged) * stake
    returns = merged[merged['bet_won']]['implied_odds'].sum() * stake
    
    return {
        'total_bets': len(merged),
        'winning_bets': merged['bet_won'].sum(),
        'win_rate': merged['bet_won'].mean(),
        'total_staked': total_staked,
        'total_returns': returns,
        'profit': returns - total_staked,
        'roi': (returns - total_staked) / total_staked if total_staked > 0 else 0
    }


def export_predictions_to_csv(predictions: List, 
                             output_path: str,
                             include_probabilities: bool = True):
    """Export predictions to CSV file."""
    records = []
    for p in predictions:
        record = {
            'match_id': p.match_id,
            'home_team': p.home_team,
            'away_team': p.away_team,
            'predicted_home_score': p.predicted_home_score,
            'predicted_away_score': p.predicted_away_score,
            'predicted_total_goals': p.predicted_total_goals,
            'predicted_result': p.predicted_result
        }
        if include_probabilities:
            record['home_win_prob'] = p.home_win_prob
            record['draw_prob'] = p.draw_prob
            record['away_win_prob'] = p.away_win_prob
        record['has_player_data'] = p.has_player_data
        record['player_coverage'] = p.player_coverage
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} predictions to {output_path}")


def analyze_by_league(predictions: List,
                      actuals: pd.DataFrame,
                      matches: pd.DataFrame) -> pd.DataFrame:
    """Analyze prediction performance by league."""
    pred_df = pd.DataFrame([p.to_dict() for p in predictions])
    
    merged = pred_df.merge(
        actuals[['match_id', 'home_team_score', 'away_team_score']], 
        on='match_id'
    ).merge(
        matches[['match_id', 'league_name']], 
        on='match_id'
    )
    
    merged['actual_result'] = merged.apply(
        lambda r: 'Home Win' if r['home_team_score'] > r['away_team_score']
                  else ('Away Win' if r['home_team_score'] < r['away_team_score'] else 'Draw'),
        axis=1
    )
    merged['correct_result'] = merged['predicted_result'] == merged['actual_result']
    merged['score_error'] = (
        abs(merged['predicted_home_score'] - merged['home_team_score']) +
        abs(merged['predicted_away_score'] - merged['away_team_score'])
    )
    
    return merged.groupby('league_name').agg({
        'match_id': 'count',
        'correct_result': 'mean',
        'score_error': 'mean',
        'player_coverage': 'mean'
    }).rename(columns={
        'match_id': 'n_matches',
        'correct_result': 'result_accuracy',
        'score_error': 'avg_score_error',
        'player_coverage': 'avg_player_coverage'
    }).sort_values('n_matches', ascending=False)