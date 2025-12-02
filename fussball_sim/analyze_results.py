"""
Analyze walk-forward validation results.
Generates reports, visualizations, and insights.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class WalkForwardAnalyzer:
    """Analyze and visualize walk-forward validation results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.predictions = None
        self.window_summary = None
        self._load_results()
    
    def _load_results(self):
        """Load prediction results from disk."""
        pred_file = self.results_dir / 'all_predictions.parquet'
        if pred_file.exists():
            self.predictions = pd.read_parquet(pred_file)
        else:
            self.predictions = pd.read_csv(self.results_dir / 'all_predictions.csv')
        
        window_file = self.results_dir / 'window_summary.csv'
        if window_file.exists():
            self.window_summary = pd.read_csv(window_file)
        
        # Parse dates
        self.predictions['match_time_utc'] = pd.to_datetime(self.predictions['match_time_utc'])
        self.predictions['year'] = self.predictions['match_time_utc'].dt.year
        self.predictions['month'] = self.predictions['match_time_utc'].dt.to_period('M')
        self.predictions['season'] = self.predictions['match_time_utc'].apply(self._get_season)
        
        print(f"Loaded {len(self.predictions)} predictions")
    
    def _get_season(self, dt) -> str:
        """Convert date to season string (e.g., '2023/24')."""
        if dt.month >= 7:
            return f"{dt.year}/{str(dt.year + 1)[-2:]}"
        else:
            return f"{dt.year - 1}/{str(dt.year)[-2:]}"
    
    def overall_metrics(self) -> Dict:
        """Calculate overall performance metrics."""
        df = self.predictions
        
        return {
            'total_predictions': len(df),
            'date_range': f"{df['match_time_utc'].min().date()} to {df['match_time_utc'].max().date()}",
            
            # Result prediction
            'result_accuracy': (df['pred_result'] == df['actual_result']).mean(),
            'home_win_precision': self._precision(df, 'Home Win'),
            'home_win_recall': self._recall(df, 'Home Win'),
            'draw_precision': self._precision(df, 'Draw'),
            'draw_recall': self._recall(df, 'Draw'),
            'away_win_precision': self._precision(df, 'Away Win'),
            'away_win_recall': self._recall(df, 'Away Win'),
            
            # Score prediction
            'exact_score_accuracy': (
                (df['pred_home_score'] == df['actual_home_score']) &
                (df['pred_away_score'] == df['actual_away_score'])
            ).mean(),
            'home_score_mae': (df['pred_home_score'] - df['actual_home_score']).abs().mean(),
            'away_score_mae': (df['pred_away_score'] - df['actual_away_score']).abs().mean(),
            
            # Total goals
            'total_goals_mae': (df['pred_total_goals'] - df['actual_total_goals']).abs().mean(),
            'total_goals_within_1': ((df['pred_total_goals'] - df['actual_total_goals']).abs() <= 1).mean(),
            
            # Data quality
            'pct_with_player_data': df['has_player_data'].mean(),
            'avg_player_coverage': df['player_coverage'].mean()
        }
    
    def _precision(self, df, result_class):
        pred_mask = df['pred_result'] == result_class
        if pred_mask.sum() == 0:
            return 0.0
        return ((df['pred_result'] == result_class) & (df['actual_result'] == result_class)).sum() / pred_mask.sum()
    
    def _recall(self, df, result_class):
        actual_mask = df['actual_result'] == result_class
        if actual_mask.sum() == 0:
            return 0.0
        return ((df['pred_result'] == result_class) & (df['actual_result'] == result_class)).sum() / actual_mask.sum()
    
    def metrics_by_season(self) -> pd.DataFrame:
        """Performance breakdown by season."""
        return self.predictions.groupby('season').apply(
            lambda x: pd.Series({
                'n_matches': len(x),
                'result_accuracy': (x['pred_result'] == x['actual_result']).mean(),
                'exact_score': ((x['pred_home_score'] == x['actual_home_score']) & 
                               (x['pred_away_score'] == x['actual_away_score'])).mean(),
                'total_goals_mae': (x['pred_total_goals'] - x['actual_total_goals']).abs().mean(),
                'player_coverage': x['player_coverage'].mean()
            })
        ).reset_index()
    
    def metrics_by_league(self, top_n: int = 20) -> pd.DataFrame:
        """Performance breakdown by league."""
        df = self.predictions.groupby('league_name').apply(
            lambda x: pd.Series({
                'n_matches': len(x),
                'result_accuracy': (x['pred_result'] == x['actual_result']).mean(),
                'exact_score': ((x['pred_home_score'] == x['actual_home_score']) & 
                               (x['pred_away_score'] == x['actual_away_score'])).mean(),
                'home_score_mae': (x['pred_home_score'] - x['actual_home_score']).abs().mean(),
                'player_coverage': x['player_coverage'].mean()
            })
        ).reset_index()
        
        return df.sort_values('n_matches', ascending=False).head(top_n)
    
    def metrics_by_data_quality(self) -> pd.DataFrame:
        """Compare performance based on data availability."""
        results = []
        
        for has_player in [True, False]:
            subset = self.predictions[self.predictions['has_player_data'] == has_player]
            if len(subset) > 0:
                results.append({
                    'segment': f"Player data: {'Yes' if has_player else 'No'}",
                    'n_matches': len(subset),
                    'result_accuracy': (subset['pred_result'] == subset['actual_result']).mean(),
                    'exact_score': ((subset['pred_home_score'] == subset['actual_home_score']) & 
                                   (subset['pred_away_score'] == subset['actual_away_score'])).mean()
                })
        
        # By player coverage quartiles
        self.predictions['coverage_bin'] = pd.qcut(
            self.predictions['player_coverage'].clip(0.01, 1), 
            q=4, 
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
            duplicates='drop'
        )
        
        for bin_name in self.predictions['coverage_bin'].unique():
            subset = self.predictions[self.predictions['coverage_bin'] == bin_name]
            if len(subset) > 0:
                results.append({
                    'segment': f"Coverage {bin_name}",
                    'n_matches': len(subset),
                    'result_accuracy': (subset['pred_result'] == subset['actual_result']).mean(),
                    'exact_score': ((subset['pred_home_score'] == subset['actual_home_score']) & 
                                   (subset['pred_away_score'] == subset['actual_away_score'])).mean()
                })
        
        return pd.DataFrame(results)
    
    def plot_accuracy_over_time(self, window: str = 'M', figsize=(14, 6)):
        """Plot prediction accuracy over time."""
        df = self.predictions.copy()
        df['period'] = df['match_time_utc'].dt.to_period(window)
        
        monthly = df.groupby('period').apply(
            lambda x: pd.Series({
                'result_accuracy': (x['pred_result'] == x['actual_result']).mean(),
                'n_matches': len(x)
            })
        ).reset_index()
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Accuracy line
        ax1.plot(range(len(monthly)), monthly['result_accuracy'], 'b-', linewidth=2, label='Result Accuracy')
        ax1.axhline(y=monthly['result_accuracy'].mean(), color='b', linestyle='--', alpha=0.5, label=f'Mean: {monthly["result_accuracy"].mean():.1%}')
        ax1.set_ylabel('Result Accuracy', color='b')
        ax1.set_ylim(0.2, 0.7)
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Match count bars
        ax2 = ax1.twinx()
        ax2.bar(range(len(monthly)), monthly['n_matches'], alpha=0.3, color='gray', label='# Matches')
        ax2.set_ylabel('Number of Matches', color='gray')
        
        # X-axis labels (show every 12th month)
        tick_positions = list(range(0, len(monthly), 12))
        tick_labels = [str(monthly.iloc[i]['period']) for i in tick_positions if i < len(monthly)]
        ax1.set_xticks(tick_positions[:len(tick_labels)])
        ax1.set_xticklabels(tick_labels, rotation=45)
        
        ax1.set_title('Prediction Accuracy Over Time')
        ax1.legend(loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, figsize=(8, 6)):
        """Plot confusion matrix for result predictions."""
        from sklearn.metrics import confusion_matrix
        
        labels = ['Away Win', 'Draw', 'Home Win']
        cm = confusion_matrix(
            self.predictions['actual_result'],
            self.predictions['pred_result'],
            labels=labels
        )
        
        # Normalize
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Result Confusion Matrix (%)')
        
        plt.tight_layout()
        return fig
    
    def plot_score_distribution(self, figsize=(14, 5)):
        """Compare predicted vs actual score distributions."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Home scores
        axes[0].hist(self.predictions['actual_home_score'], bins=range(0, 8), 
                     alpha=0.5, label='Actual', density=True)
        axes[0].hist(self.predictions['pred_home_score'], bins=range(0, 8), 
                     alpha=0.5, label='Predicted', density=True)
        axes[0].set_xlabel('Goals')
        axes[0].set_title('Home Team Scores')
        axes[0].legend()
        
        # Away scores
        axes[1].hist(self.predictions['actual_away_score'], bins=range(0, 8), 
                     alpha=0.5, label='Actual', density=True)
        axes[1].hist(self.predictions['pred_away_score'], bins=range(0, 8), 
                     alpha=0.5, label='Predicted', density=True)
        axes[1].set_xlabel('Goals')
        axes[1].set_title('Away Team Scores')
        axes[1].legend()
        
        # Total goals
        axes[2].hist(self.predictions['actual_total_goals'], bins=range(0, 12), 
                     alpha=0.5, label='Actual', density=True)
        axes[2].hist(self.predictions['pred_total_goals'], bins=range(0, 12), 
                     alpha=0.5, label='Predicted', density=True)
        axes[2].set_xlabel('Goals')
        axes[2].set_title('Total Goals')
        axes[2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_calibration(self, figsize=(10, 6)):
        """Plot probability calibration for result predictions."""
        df = self.predictions.copy()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for i, (prob_col, result, ax) in enumerate([
            ('prob_home_win', 'Home Win', axes[0]),
            ('prob_draw', 'Draw', axes[1]),
            ('prob_away_win', 'Away Win', axes[2])
        ]):
            # Bin probabilities
            df['prob_bin'] = pd.cut(df[prob_col], bins=np.linspace(0, 1, 11))
            
            calibration = df.groupby('prob_bin').apply(
                lambda x: pd.Series({
                    'mean_prob': x[prob_col].mean(),
                    'actual_rate': (x['actual_result'] == result).mean(),
                    'count': len(x)
                })
            ).reset_index()
            
            # Filter bins with enough samples
            calibration = calibration[calibration['count'] >= 50]
            
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
            ax.scatter(calibration['mean_prob'], calibration['actual_rate'], 
                      s=calibration['count']/10, alpha=0.7)
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Actual Rate')
            ax.set_title(result)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.suptitle('Probability Calibration')
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_path: Optional[str] = None):
        """Generate a comprehensive text report."""
        metrics = self.overall_metrics()
        
        report = []
        report.append("=" * 70)
        report.append("WALK-FORWARD VALIDATION REPORT")
        report.append("=" * 70)
        report.append(f"\nDate Range: {metrics['date_range']}")
        report.append(f"Total Predictions: {metrics['total_predictions']:,}")
        report.append(f"Data with Player Info: {metrics['pct_with_player_data']:.1%}")
        
        report.append("\n" + "-" * 40)
        report.append("RESULT PREDICTION (1X2)")
        report.append("-" * 40)
        report.append(f"Overall Accuracy: {metrics['result_accuracy']:.1%}")
        report.append(f"\n{'Class':<15} {'Precision':>12} {'Recall':>12}")
        report.append("-" * 40)
        report.append(f"{'Home Win':<15} {metrics['home_win_precision']:>11.1%} {metrics['home_win_recall']:>11.1%}")
        report.append(f"{'Draw':<15} {metrics['draw_precision']:>11.1%} {metrics['draw_recall']:>11.1%}")
        report.append(f"{'Away Win':<15} {metrics['away_win_precision']:>11.1%} {metrics['away_win_recall']:>11.1%}")
        
        report.append("\n" + "-" * 40)
        report.append("SCORE PREDICTION")
        report.append("-" * 40)
        report.append(f"Exact Score Accuracy: {metrics['exact_score_accuracy']:.1%}")
        report.append(f"Home Score MAE: {metrics['home_score_mae']:.3f}")
        report.append(f"Away Score MAE: {metrics['away_score_mae']:.3f}")
        
        report.append("\n" + "-" * 40)
        report.append("TOTAL GOALS")
        report.append("-" * 40)
        report.append(f"MAE: {metrics['total_goals_mae']:.3f}")
        report.append(f"Within ±1 Goal: {metrics['total_goals_within_1']:.1%}")
        
        report.append("\n" + "-" * 40)
        report.append("PERFORMANCE BY SEASON")
        report.append("-" * 40)
        by_season = self.metrics_by_season()
        report.append(by_season.to_string(index=False))
        
        report.append("\n" + "-" * 40)
        report.append("PERFORMANCE BY DATA QUALITY")
        report.append("-" * 40)
        by_quality = self.metrics_by_data_quality()
        report.append(by_quality.to_string(index=False))
        
        full_report = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(full_report)
            print(f"Report saved to: {output_path}")
        
        print(full_report)
        return full_report


def main():
    """Analyze results from a walk-forward run."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze walk-forward results")
    parser.add_argument("--results-dir", type=str, default="./walk_forward_results/expanding",
                       help="Directory containing walk-forward results")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to disk")
    
    args = parser.parse_args()
    
    analyzer = WalkForwardAnalyzer(args.results_dir)
    
    # Generate report
    analyzer.generate_report(output_path=Path(args.results_dir) / "analysis_report.txt")
    
    # Generate plots
    if args.save_plots:
        plot_dir = Path(args.results_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        analyzer.plot_accuracy_over_time().savefig(plot_dir / "accuracy_over_time.png", dpi=150)
        analyzer.plot_confusion_matrix().savefig(plot_dir / "confusion_matrix.png", dpi=150)
        analyzer.plot_score_distribution().savefig(plot_dir / "score_distribution.png", dpi=150)
        analyzer.plot_calibration().savefig(plot_dir / "calibration.png", dpi=150)
        
        print(f"\nPlots saved to: {plot_dir}")
    else:
        plt.show()


if __name__ == "__main__":
    main()