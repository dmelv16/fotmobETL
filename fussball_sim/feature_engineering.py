"""
Optimized Feature Engineering Module v2

Key improvements:
1. Uses BatchDataLoader for efficient SQL queries
2. Vectorized operations - use pandas/numpy instead of row-by-row
3. Form/trend features - track attribute changes over 3 and 5 weeks
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class BatchFeatureConfig:
    """Configuration for batch feature extraction."""
    include_bench_players: bool = True
    bench_weight: float = 0.3
    min_player_coverage: float = 0.0
    
    # Form/trend windows (in days)
    form_windows: List[int] = field(default_factory=lambda: [21, 35])  # 3 and 5 weeks
    
    def __post_init__(self):
        if self.form_windows is None:
            self.form_windows = [21, 35]


class OptimizedFeatureEngineer:
    """
    Vectorized feature engineering using BatchDataLoader.
    
    Process:
    1. Load all data for matches via BatchDataLoader
    2. Merge and aggregate using vectorized pandas operations
    3. Compute form/trend features
    """
    
    def __init__(self, config, data_loader):
        self.config = config
        self.loader = data_loader  # Should be BatchDataLoader
        self.batch_config = BatchFeatureConfig()
    
    def build_feature_matrix(self, 
                             matches: pd.DataFrame,
                             cache_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build feature matrix for all matches using vectorized operations.
        
        Args:
            matches: DataFrame with match_id, match_time_utc, home_team_id, etc.
            cache_path: Optional path to cache results
            
        Returns:
            X: Feature DataFrame
            y: Target DataFrame
        """
        # Check cache
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        matches = matches.copy()
        matches['match_time_utc'] = pd.to_datetime(matches['match_time_utc'])
        
        logger.info(f"Building features for {len(matches)} matches...")
        
        # Step 1: Load all data via BatchDataLoader
        logger.info("Step 1/3: Loading all historical data...")
        max_form_window = max(self.batch_config.form_windows)
        data = self.loader.get_all_history_for_matches(matches, form_window_days=max_form_window)
        
        lineups = data['lineups']
        coaches = data['coaches']
        player_attrs = data['player_attrs']
        player_ratings = data['player_ratings']
        coach_attrs = data['coach_attrs']
        coach_ratings = data['coach_ratings']
        team_attrs = data['team_attrs']
        team_ratings = data['team_ratings']
        
        # Step 2: Add match times to lineups and coaches
        logger.info("Step 2/3: Preparing data...")
        if not lineups.empty:
            lineups = lineups.merge(matches[['match_id', 'match_time_utc']], on='match_id', how='left')
        if not coaches.empty:
            coaches = coaches.merge(matches[['match_id', 'match_time_utc']], on='match_id', how='left')
        
        # Convert dates
        for df in [player_attrs, player_ratings, coach_attrs, coach_ratings, team_attrs, team_ratings]:
            if not df.empty and 'last_date' in df.columns:
                df['last_date'] = pd.to_datetime(df['last_date'])
        
        # Step 3: Compute features vectorized
        logger.info("Step 3/3: Computing features...")
        X, y = self._compute_features_vectorized(
            matches, lineups, coaches,
            player_attrs, player_ratings,
            coach_attrs, coach_ratings,
            team_attrs, team_ratings
        )
        
        # Cache results
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump((X, y), f)
            logger.info(f"Cached features to: {cache_path}")
        
        logger.info(f"Feature matrix shape: {X.shape}")
        return X, y
    
    def _compute_features_vectorized(self,
                                     matches: pd.DataFrame,
                                     lineups: pd.DataFrame,
                                     coaches: pd.DataFrame,
                                     player_attrs: pd.DataFrame,
                                     player_ratings: pd.DataFrame,
                                     coach_attrs: pd.DataFrame,
                                     coach_ratings: pd.DataFrame,
                                     team_attrs: pd.DataFrame,
                                     team_ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute all features using vectorized operations."""
        
        # Compute player features
        logger.info("  Computing player features...")
        player_features = self._compute_player_features_vectorized(
            matches, lineups, player_attrs, player_ratings
        )
        
        # Compute coach features
        logger.info("  Computing coach features...")
        coach_features = self._compute_coach_features_vectorized(
            matches, coaches, coach_attrs, coach_ratings
        )
        
        # Compute team features
        logger.info("  Computing team features...")
        team_features = self._compute_team_features_vectorized(
            matches, team_attrs, team_ratings
        )
        
        # Merge all features
        X = matches[['match_id']].copy()
        
        if not player_features.empty:
            X = X.merge(player_features, on='match_id', how='left')
        if not coach_features.empty:
            X = X.merge(coach_features, on='match_id', how='left')
        if not team_features.empty:
            X = X.merge(team_features, on='match_id', how='left')
        
        # Add differential features
        X = self._add_differential_features_vectorized(X)
        
        # Add metadata
        player_cols = [c for c in X.columns if 'starter_' in c]
        coach_cols = [c for c in X.columns if 'coach_' in c]
        team_cols = [c for c in X.columns if 'team_' in c]
        
        X['has_player_data'] = X[player_cols].notna().any(axis=1) if player_cols else False
        X['has_coach_data'] = X[coach_cols].notna().any(axis=1) if coach_cols else False
        X['has_team_data'] = X[team_cols].notna().any(axis=1) if team_cols else False
        X['player_coverage'] = 0.0  # TODO: compute actual coverage
        
        # Build targets
        y = matches[['match_id', 'home_team_score', 'away_team_score']].copy()
        y.columns = ['match_id', 'home_score', 'away_score']
        y['total_goals'] = y['home_score'] + y['away_score']
        y['result'] = np.where(
            y['home_score'] > y['away_score'], 2,
            np.where(y['home_score'] < y['away_score'], 0, 1)
        )
        
        return X, y
    
    def _get_latest_before_match(self,
                                 entity_matches: pd.DataFrame,
                                 history: pd.DataFrame,
                                 entity_id_col: str,
                                 match_time_col: str) -> pd.DataFrame:
        """Get the latest historical record before each match for each entity."""
        if history.empty or entity_matches.empty:
            return pd.DataFrame()
        
        # Merge to get all combinations
        merged = entity_matches[[entity_id_col, 'match_id', match_time_col]].merge(
            history,
            on=entity_id_col,
            how='inner'
        )
        
        # Filter to only records before match
        merged = merged[merged['last_date'] < merged[match_time_col]]
        
        if merged.empty:
            return pd.DataFrame()
        
        # Get latest per entity per match
        if 'attribute_code' in merged.columns:
            idx = merged.groupby(['match_id', entity_id_col, 'attribute_code'])['last_date'].idxmax()
        else:
            idx = merged.groupby(['match_id', entity_id_col])['last_date'].idxmax()
        
        return merged.loc[idx]
    
    def _compute_player_features_vectorized(self,
                                            matches: pd.DataFrame,
                                            lineups: pd.DataFrame,
                                            attrs: pd.DataFrame,
                                            ratings: pd.DataFrame) -> pd.DataFrame:
        """Vectorized player feature computation."""
        if lineups.empty:
            return pd.DataFrame({'match_id': matches['match_id']})
        
        results = []
        
        for is_home in [True, False]:
            prefix = 'home' if is_home else 'away'
            team_lineups = lineups[lineups['is_home_team'] == is_home].copy()
            
            if team_lineups.empty:
                continue
            
            # Get latest attributes before each match for each player
            if not attrs.empty:
                player_attrs_latest = self._get_latest_before_match(
                    team_lineups, attrs, 'player_id', 'match_time_utc'
                )
                if not player_attrs_latest.empty:
                    attrs_wide = player_attrs_latest.pivot_table(
                        index=['match_id', 'player_id'],
                        columns='attribute_code',
                        values='value',
                        aggfunc='first'
                    ).reset_index()
                    team_lineups = team_lineups.merge(attrs_wide, on=['match_id', 'player_id'], how='left')
                
                # Compute form features for attributes
                form_features = self._compute_form_features(
                    team_lineups, attrs, 'player_id', prefix
                )
                if not form_features.empty:
                    results.append(form_features)
            
            # Get latest ratings
            if not ratings.empty:
                player_ratings_latest = self._get_latest_before_match(
                    team_lineups, ratings, 'player_id', 'match_time_utc'
                )
                if not player_ratings_latest.empty:
                    team_lineups = team_lineups.merge(
                        player_ratings_latest[['match_id', 'player_id', 'rating']].rename(
                            columns={'rating': 'hist_rating'}
                        ),
                        on=['match_id', 'player_id'],
                        how='left'
                    )
                
                # Rating form features
                rating_form = self._compute_rating_form_features(
                    team_lineups, ratings, 'player_id', prefix
                )
                if not rating_form.empty:
                    results.append(rating_form)
            
            # Aggregate to match level
            match_features = self._aggregate_lineup_features(team_lineups, prefix)
            results.append(match_features)
        
        if not results:
            return pd.DataFrame({'match_id': matches['match_id']})
        
        # Merge all results
        final = results[0]
        for df in results[1:]:
            final = final.merge(df, on='match_id', how='outer')
        
        return final
    
    def _compute_form_features(self,
                               entity_matches: pd.DataFrame,
                               attrs: pd.DataFrame,
                               entity_id_col: str,
                               prefix: str) -> pd.DataFrame:
        """Compute attribute trend features over form windows."""
        if attrs.empty or entity_matches.empty:
            return pd.DataFrame()
        
        form_features = []
        
        for window_days in self.batch_config.form_windows:
            window_name = f"{window_days}d"
            
            merged = entity_matches[['match_id', entity_id_col, 'match_time_utc', 'is_starter']].merge(
                attrs, on=entity_id_col, how='inner'
            )
            
            window_start = merged['match_time_utc'] - timedelta(days=window_days)
            merged = merged[
                (merged['last_date'] >= window_start) & 
                (merged['last_date'] < merged['match_time_utc'])
            ]
            
            if merged.empty:
                continue
            
            def calc_trend(group):
                if len(group) < 2:
                    return np.nan
                sorted_g = group.sort_values('last_date')
                return sorted_g['value'].iloc[-1] - sorted_g['value'].iloc[0]
            
            trends = merged.groupby(['match_id', entity_id_col, 'attribute_code']).apply(
                calc_trend
            ).reset_index(name='trend')
            
            if not trends.empty:
                trends = trends.merge(
                    entity_matches[['match_id', entity_id_col, 'is_starter']].drop_duplicates(),
                    on=['match_id', entity_id_col],
                    how='left'
                )
                
                starter_trends = trends[trends['is_starter'] == True].groupby(
                    ['match_id', 'attribute_code']
                )['trend'].mean().reset_index()
                
                if not starter_trends.empty:
                    pivoted = starter_trends.pivot(
                        index='match_id',
                        columns='attribute_code',
                        values='trend'
                    ).reset_index()
                    
                    pivoted.columns = ['match_id'] + [
                        f'{prefix}_form_{window_name}_{c}' for c in pivoted.columns[1:]
                    ]
                    form_features.append(pivoted)
        
        if not form_features:
            return pd.DataFrame()
        
        result = form_features[0]
        for df in form_features[1:]:
            result = result.merge(df, on='match_id', how='outer')
        
        return result
    
    def _compute_rating_form_features(self,
                                      entity_matches: pd.DataFrame,
                                      ratings: pd.DataFrame,
                                      entity_id_col: str,
                                      prefix: str) -> pd.DataFrame:
        """Compute rating trend features."""
        if ratings.empty or entity_matches.empty:
            return pd.DataFrame()
        
        form_features = []
        
        for window_days in self.batch_config.form_windows:
            window_name = f"{window_days}d"
            
            merged = entity_matches[['match_id', entity_id_col, 'match_time_utc', 'is_starter']].merge(
                ratings, on=entity_id_col, how='inner'
            )
            
            window_start = merged['match_time_utc'] - timedelta(days=window_days)
            merged = merged[
                (merged['last_date'] >= window_start) & 
                (merged['last_date'] < merged['match_time_utc'])
            ]
            
            if merged.empty:
                continue
            
            def calc_rating_trend(group):
                if len(group) < 2:
                    return pd.Series({'trend': np.nan, 'volatility': np.nan})
                sorted_g = group.sort_values('last_date')
                trend = sorted_g['rating'].iloc[-1] - sorted_g['rating'].iloc[0]
                volatility = sorted_g['rating'].std()
                return pd.Series({'trend': trend, 'volatility': volatility})
            
            player_trends = merged.groupby(['match_id', entity_id_col, 'is_starter']).apply(
                calc_rating_trend
            ).reset_index()
            
            starter_trends = player_trends[player_trends['is_starter'] == True].groupby('match_id').agg({
                'trend': 'mean',
                'volatility': 'mean'
            }).reset_index()
            
            starter_trends.columns = [
                'match_id',
                f'{prefix}_rating_form_{window_name}_trend',
                f'{prefix}_rating_form_{window_name}_volatility'
            ]
            
            form_features.append(starter_trends)
        
        if not form_features:
            return pd.DataFrame()
        
        result = form_features[0]
        for df in form_features[1:]:
            result = result.merge(df, on='match_id', how='outer')
        
        return result
    
    def _aggregate_lineup_features(self,
                                   team_lineups: pd.DataFrame,
                                   prefix: str) -> pd.DataFrame:
        """Aggregate player-level features to match level."""
        
        meta_cols = {'match_id', 'player_id', 'team_id', 'is_home_team', 'is_starter',
                     'position_id', 'age', 'rating', 'market_value', 'match_time_utc', 'hist_rating'}
        attr_cols = [c for c in team_lineups.columns if c not in meta_cols]
        
        results = []
        
        for match_id, group in team_lineups.groupby('match_id'):
            row = {'match_id': match_id}
            
            starter_group = group[group['is_starter'] == True]
            bench_group = group[group['is_starter'] == False]
            
            row[f'{prefix}_num_starters'] = len(starter_group)
            row[f'{prefix}_num_bench'] = len(bench_group)
            row[f'{prefix}_avg_age'] = starter_group['age'].mean() if 'age' in starter_group else np.nan
            row[f'{prefix}_avg_market_value'] = starter_group['market_value'].mean() if 'market_value' in starter_group else np.nan
            
            if 'rating' in starter_group.columns:
                row[f'{prefix}_avg_match_rating'] = starter_group['rating'].mean()
            
            if 'hist_rating' in starter_group.columns:
                hist_ratings = starter_group['hist_rating'].dropna()
                if len(hist_ratings) > 0:
                    row[f'{prefix}_starter_rating_mean'] = hist_ratings.mean()
                    row[f'{prefix}_starter_rating_std'] = hist_ratings.std() if len(hist_ratings) > 1 else 0
                    row[f'{prefix}_starter_rating_sum'] = hist_ratings.sum()
            
            for col in attr_cols:
                if col in starter_group.columns:
                    vals = starter_group[col].dropna()
                    if len(vals) > 0:
                        row[f'{prefix}_starter_{col}_mean'] = vals.mean()
                        row[f'{prefix}_starter_{col}_std'] = vals.std() if len(vals) > 1 else 0
                        row[f'{prefix}_starter_{col}_min'] = vals.min()
                        row[f'{prefix}_starter_{col}_max'] = vals.max()
                    
                    if self.batch_config.include_bench_players and col in bench_group.columns:
                        bench_vals = bench_group[col].dropna()
                        if len(bench_vals) > 0:
                            row[f'{prefix}_bench_{col}_mean'] = bench_vals.mean()
            
            results.append(row)
        
        return pd.DataFrame(results) if results else pd.DataFrame({'match_id': []})
    
    def _compute_coach_features_vectorized(self,
                                           matches: pd.DataFrame,
                                           coaches: pd.DataFrame,
                                           attrs: pd.DataFrame,
                                           ratings: pd.DataFrame) -> pd.DataFrame:
        """Vectorized coach feature computation."""
        if coaches.empty:
            return pd.DataFrame({'match_id': matches['match_id']})
        
        results = []
        
        for is_home in [True, False]:
            prefix = 'home' if is_home else 'away'
            team_coaches = coaches[coaches['is_home_team'] == is_home].copy()
            
            if team_coaches.empty:
                continue
            
            coach_basic = team_coaches.groupby('match_id').first().reset_index()
            coach_basic = coach_basic[['match_id', 'coach_id', 'age']].rename(
                columns={'age': f'{prefix}_coach_age'}
            )
            
            if not attrs.empty:
                coach_attrs_latest = self._get_latest_before_match(
                    team_coaches, attrs, 'coach_id', 'match_time_utc'
                )
                if not coach_attrs_latest.empty:
                    attrs_wide = coach_attrs_latest.pivot_table(
                        index='match_id',
                        columns='attribute_code',
                        values='value',
                        aggfunc='first'
                    ).reset_index()
                    attrs_wide.columns = ['match_id'] + [
                        f'{prefix}_coach_{c}' for c in attrs_wide.columns[1:]
                    ]
                    coach_basic = coach_basic.merge(attrs_wide, on='match_id', how='left')
            
            if not ratings.empty:
                coach_ratings_latest = self._get_latest_before_match(
                    team_coaches, ratings, 'coach_id', 'match_time_utc'
                )
                if not coach_ratings_latest.empty:
                    ratings_agg = coach_ratings_latest.groupby('match_id')['rating'].first().reset_index()
                    ratings_agg.columns = ['match_id', f'{prefix}_coach_rating']
                    coach_basic = coach_basic.merge(ratings_agg, on='match_id', how='left')
                
                coach_form = self._compute_coach_form_features(team_coaches, ratings, prefix)
                if not coach_form.empty:
                    coach_basic = coach_basic.merge(coach_form, on='match_id', how='left')
            
            results.append(coach_basic.drop(columns=['coach_id'], errors='ignore'))
        
        if not results:
            return pd.DataFrame({'match_id': matches['match_id']})
        
        final = results[0]
        for df in results[1:]:
            final = final.merge(df, on='match_id', how='outer')
        
        return final
    
    def _compute_coach_form_features(self,
                                     coaches: pd.DataFrame,
                                     ratings: pd.DataFrame,
                                     prefix: str) -> pd.DataFrame:
        """Compute coach form features."""
        form_features = []
        
        for window_days in self.batch_config.form_windows:
            window_name = f"{window_days}d"
            
            merged = coaches[['match_id', 'coach_id', 'match_time_utc']].merge(
                ratings, on='coach_id', how='inner'
            )
            
            window_start = merged['match_time_utc'] - timedelta(days=window_days)
            merged = merged[
                (merged['last_date'] >= window_start) & 
                (merged['last_date'] < merged['match_time_utc'])
            ]
            
            if not merged.empty:
                def calc_trend(g):
                    if len(g) < 2:
                        return np.nan
                    sorted_g = g.sort_values('last_date')
                    return sorted_g['rating'].iloc[-1] - sorted_g['rating'].iloc[0]
                
                trends = merged.groupby('match_id').apply(calc_trend).reset_index(name='trend')
                trends.columns = ['match_id', f'{prefix}_coach_rating_form_{window_name}']
                form_features.append(trends)
        
        if not form_features:
            return pd.DataFrame()
        
        result = form_features[0]
        for df in form_features[1:]:
            result = result.merge(df, on='match_id', how='outer')
        
        return result
    
    def _compute_team_features_vectorized(self,
                                          matches: pd.DataFrame,
                                          attrs: pd.DataFrame,
                                          ratings: pd.DataFrame) -> pd.DataFrame:
        """Vectorized team feature computation."""
        results = []
        
        for is_home, team_col in [(True, 'home_team_id'), (False, 'away_team_id')]:
            prefix = 'home' if is_home else 'away'
            
            team_matches = matches[['match_id', team_col, 'match_time_utc']].rename(
                columns={team_col: 'team_id'}
            )
            
            features = pd.DataFrame({'match_id': matches['match_id']})
            
            if not attrs.empty:
                attrs_latest = self._get_latest_before_match(
                    team_matches, attrs, 'team_id', 'match_time_utc'
                )
                if not attrs_latest.empty:
                    attrs_wide = attrs_latest.pivot_table(
                        index='match_id',
                        columns='attribute_code',
                        values='value',
                        aggfunc='first'
                    ).reset_index()
                    attrs_wide.columns = ['match_id'] + [
                        f'{prefix}_team_{c}' for c in attrs_wide.columns[1:]
                    ]
                    features = features.merge(attrs_wide, on='match_id', how='left')
            
            if not ratings.empty:
                ratings_latest = self._get_latest_before_match(
                    team_matches, ratings, 'team_id', 'match_time_utc'
                )
                if not ratings_latest.empty:
                    ratings_agg = ratings_latest.groupby('match_id')['rating'].first().reset_index()
                    ratings_agg.columns = ['match_id', f'{prefix}_team_rating']
                    features = features.merge(ratings_agg, on='match_id', how='left')
                
                team_form = self._compute_team_form_features(team_matches, ratings, prefix)
                if not team_form.empty:
                    features = features.merge(team_form, on='match_id', how='left')
            
            results.append(features)
        
        if not results:
            return pd.DataFrame({'match_id': matches['match_id']})
        
        final = results[0]
        for df in results[1:]:
            final = final.merge(df, on='match_id', how='outer')
        
        return final
    
    def _compute_team_form_features(self,
                                    team_matches: pd.DataFrame,
                                    ratings: pd.DataFrame,
                                    prefix: str) -> pd.DataFrame:
        """Compute team form features."""
        form_features = []
        
        for window_days in self.batch_config.form_windows:
            window_name = f"{window_days}d"
            
            merged = team_matches.merge(ratings, on='team_id', how='inner')
            
            window_start = merged['match_time_utc'] - timedelta(days=window_days)
            merged = merged[
                (merged['last_date'] >= window_start) & 
                (merged['last_date'] < merged['match_time_utc'])
            ]
            
            if not merged.empty:
                def calc_stats(g):
                    if len(g) < 2:
                        return pd.Series({'trend': np.nan, 'volatility': np.nan, 'momentum': np.nan})
                    sorted_g = g.sort_values('last_date')
                    trend = sorted_g['rating'].iloc[-1] - sorted_g['rating'].iloc[0]
                    volatility = sorted_g['rating'].std()
                    changes = sorted_g['rating'].diff().dropna()
                    momentum = changes.tail(3).mean() if len(changes) >= 3 else changes.mean()
                    return pd.Series({'trend': trend, 'volatility': volatility, 'momentum': momentum})
                
                stats = merged.groupby('match_id').apply(calc_stats).reset_index()
                stats.columns = ['match_id', 
                                f'{prefix}_team_rating_form_{window_name}_trend',
                                f'{prefix}_team_rating_form_{window_name}_volatility',
                                f'{prefix}_team_rating_form_{window_name}_momentum']
                form_features.append(stats)
        
        if not form_features:
            return pd.DataFrame()
        
        result = form_features[0]
        for df in form_features[1:]:
            result = result.merge(df, on='match_id', how='outer')
        
        return result
    
    def _add_differential_features_vectorized(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add home - away differential features."""
        home_cols = [c for c in X.columns if c.startswith('home_')]
        
        for home_col in home_cols:
            away_col = home_col.replace('home_', 'away_', 1)
            if away_col in X.columns:
                diff_col = home_col.replace('home_', 'diff_', 1)
                X[diff_col] = X[home_col] - X[away_col]
        
        return X