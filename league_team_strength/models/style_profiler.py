import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class StyleProfiler:
    """
    Analyzes tactical style profiles for leagues and teams.
    Used for understanding meta-game and making style-adjusted comparisons.
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader

    def _convert_to_python_types(self, profile: Dict) -> Dict:
        """Convert numpy types to native Python types for database insertion."""
        converted = {}
        for key, value in profile.items():
            if value is None:
                converted[key] = None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                converted[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                converted[key] = float(value)
            else:
                converted[key] = value
        return converted
    
    def calculate_team_style_profile(self, team_id: int, season_year: int) -> Optional[Dict]:
        """
        Calculate a team's tactical style profile.
        
        Returns: Dict with various style metrics (0-100 scales)
        """
        # Get team's match stats
        query = """
            SELECT 
                md.match_id,
                CASE WHEN md.home_team_id = ? THEN 1 ELSE 0 END as is_home,
                CASE WHEN md.home_team_id = ? THEN mss.home_possession ELSE mss.away_possession END as possession,
                CASE WHEN md.home_team_id = ? THEN mss.home_passes ELSE mss.away_passes END as passes,
                CASE WHEN md.home_team_id = ? THEN mss.home_accurate_long_balls ELSE mss.away_accurate_long_balls END as long_balls,
                CASE WHEN md.home_team_id = ? THEN mss.home_accurate_crosses ELSE mss.away_accurate_crosses END as crosses,
                CASE WHEN md.home_team_id = ? THEN mss.home_tackles ELSE mss.away_tackles END as tackles,
                CASE WHEN md.home_team_id = ? THEN mss.home_fouls ELSE mss.away_fouls END as fouls,
                CASE WHEN md.home_team_id = ? THEN mss.home_total_shots ELSE mss.away_total_shots END as shots,
                CASE WHEN md.home_team_id = ? THEN mss.home_corners ELSE mss.away_corners END as corners
            FROM [dbo].[match_details] md
            INNER JOIN [dbo].[match_stats_summary] mss ON md.match_id = mss.match_id
            WHERE (md.home_team_id = ? OR md.away_team_id = ?)
              AND YEAR(md.match_time_utc) = ?
              AND md.finished = 1
        """
        
        params = [team_id] * 11 + [season_year]
        
        df = pd.read_sql(query, self.conn, params=params)
        
        if len(df) < 5:
            return None
        
        # Calculate average metrics with safe handling
        def safe_mean(series):
            if len(series) == 0:
                return 0.0
            mean_val = series.mean()
            if pd.isna(mean_val) or np.isinf(mean_val):
                return 0.0
            return float(mean_val)
        
        avg_possession = safe_mean(df['possession'])
        avg_passes = safe_mean(df['passes'])
        avg_long_balls = safe_mean(df['long_balls'])
        avg_crosses = safe_mean(df['crosses'])
        avg_tackles = safe_mean(df['tackles'])
        avg_fouls = safe_mean(df['fouls'])
        avg_shots = safe_mean(df['shots'])
        avg_corners = safe_mean(df['corners'])
        
        # Derive style metrics (0-100 scales)
        
        # Possession focus (directly from possession %)
        possession_focus = avg_possession
        
        # Direct play index (long balls per total passes)
        if avg_passes > 0:
            direct_play_index = min((avg_long_balls / avg_passes) * 100, 100)
        else:
            direct_play_index = 50.0
        
        # Crossing frequency (normalized)
        crossing_frequency = min(avg_crosses * 5, 100)
        
        # Defensive intensity (tackles per game, normalized)
        defensive_intensity = min(avg_tackles * 5, 100)
        
        # Physical intensity (fouls per game, normalized)
        physical_intensity = min(avg_fouls * 5, 100)
        
        # Attacking volume (shots per game, normalized)
        attacking_volume = min(avg_shots * 5, 100)
        
        return {
            'entity_type': 'team',
            'entity_id': team_id,
            'season_year': season_year,
            'possession_focus': float(possession_focus),
            'direct_play_index': float(direct_play_index),
            'crossing_frequency': float(crossing_frequency),
            'defensive_intensity': float(defensive_intensity),
            'physical_intensity': float(physical_intensity),
            'attacking_volume': float(attacking_volume),
            'avg_possession_pct': float(avg_possession),
            'avg_passes_per_game': float(avg_passes),
            'avg_long_balls_per_game': float(avg_long_balls),
            'avg_crosses_per_game': float(avg_crosses),
            'avg_tackles_per_game': float(avg_tackles),
            'avg_fouls_per_game': float(avg_fouls),
            'matches_analyzed': int(len(df))
        }
    
    def calculate_league_style_profile(self, league_id: int, season_year: int) -> Optional[Dict]:
        """
        Calculate aggregate style profile for entire league.
        """
        # Get all teams in league
        team_ids = self.data_loader.get_league_teams_for_season(league_id, season_year)
        
        team_profiles = []
        for team_id in team_ids:
            profile = self.calculate_team_style_profile(team_id, season_year)
            if profile:
                team_profiles.append(profile)
        
        if not team_profiles:
            return None
        
        # Average across all teams
        df = pd.DataFrame(team_profiles)
        
        # Helper function to safely get mean, handling NaN/inf
        def safe_mean(series):
            if len(series) == 0:
                return None
            mean_val = series.mean()
            if pd.isna(mean_val) or np.isinf(mean_val):
                return None
            return float(mean_val)
        
        def safe_sum(series):
            if len(series) == 0:
                return 0
            sum_val = series.sum()
            if pd.isna(sum_val) or np.isinf(sum_val):
                return 0
            return int(sum_val)
        
        league_profile = {
            'entity_type': 'league',
            'entity_id': league_id,
            'season_year': season_year,
            'possession_focus': safe_mean(df['possession_focus']),
            'direct_play_index': safe_mean(df['direct_play_index']),
            'crossing_frequency': safe_mean(df['crossing_frequency']),
            'defensive_intensity': safe_mean(df['defensive_intensity']),
            'physical_intensity': safe_mean(df['physical_intensity']),
            'attacking_volume': safe_mean(df['attacking_volume']),
            'avg_possession_pct': safe_mean(df['avg_possession_pct']),
            'avg_passes_per_game': safe_mean(df['avg_passes_per_game']),
            'avg_long_balls_per_game': safe_mean(df['avg_long_balls_per_game']),
            'avg_crosses_per_game': safe_mean(df['avg_crosses_per_game']),
            'avg_tackles_per_game': safe_mean(df['avg_tackles_per_game']),
            'avg_fouls_per_game': safe_mean(df['avg_fouls_per_game']),
            'matches_analyzed': safe_sum(df['matches_analyzed'])
        }
        
        return league_profile
    
    def cluster_teams_by_style(self, team_profiles: List[Dict], n_clusters: int = 5) -> pd.DataFrame:
        """
        Cluster teams by tactical style using K-means.
        
        Returns: DataFrame with cluster assignments
        """
        if len(team_profiles) < n_clusters:
            n_clusters = max(2, len(team_profiles) // 2)
        
        df = pd.DataFrame(team_profiles)
        
        # Features for clustering
        features = [
            'possession_focus', 'direct_play_index', 'crossing_frequency',
            'defensive_intensity', 'physical_intensity', 'attacking_volume'
        ]
        
        X = df[features].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['style_cluster_id'] = kmeans.fit_predict(X_scaled)
        
        # Assign cluster names based on characteristics
        cluster_names = self._assign_cluster_names(df, features)
        df['style_cluster_name'] = df['style_cluster_id'].map(cluster_names)
        
        return df
    
    def _assign_cluster_names(self, df: pd.DataFrame, features: List[str]) -> Dict[int, str]:
        """
        Assign interpretable names to clusters based on their characteristics.
        """
        cluster_names = {}
        
        # Get overall means for comparison
        overall_means = df[features].mean()
        
        for cluster_id in df['style_cluster_id'].unique():
            cluster_df = df[df['style_cluster_id'] == cluster_id]
            cluster_means = cluster_df[features].mean()
            
            # Find top 2-3 distinguishing features (highest deviation from overall mean)
            deviations = {}
            for feature in features:
                # Calculate standardized deviation
                std = df[feature].std()
                if std > 0:
                    deviation = (cluster_means[feature] - overall_means[feature]) / std
                    deviations[feature] = deviation
            
            # Sort by absolute deviation
            top_features = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            
            # Build descriptive name
            descriptors = []
            for feature, deviation in top_features:
                if abs(deviation) > 0.5:  # Only include significant deviations
                    level = "High" if deviation > 0 else "Low"
                    # Clean up feature name
                    feature_name = feature.replace('_', ' ').title()
                    descriptors.append(f"{level} {feature_name}")
            
            if descriptors:
                cluster_names[cluster_id] = " | ".join(descriptors)
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}"
        
        return cluster_names
    
    def save_profiles_to_database(self, profiles: List[Dict]):
        """Save style profiles to database."""
        if not profiles:
            return
        
        cursor = self.conn.cursor()
        
        for profile in profiles:
            profile = self._convert_to_python_types(profile)
            # Check if exists
            cursor.execute("""
                SELECT id FROM [dbo].[style_profiles]
                WHERE entity_type = ? AND entity_id = ? AND season_year = ?
            """, profile['entity_type'], profile['entity_id'], profile['season_year'])
            
            existing = cursor.fetchone()
            
            if existing:
                # Update
                cursor.execute("""
                    UPDATE [dbo].[style_profiles]
                    SET possession_focus = ?,
                        direct_play_index = ?,
                        crossing_frequency = ?,
                        defensive_intensity = ?,
                        physical_intensity = ?,
                        avg_possession_pct = ?,
                        avg_passes_per_game = ?,
                        avg_long_balls_per_game = ?,
                        avg_crosses_per_game = ?,
                        avg_tackles_per_game = ?,
                        avg_fouls_per_game = ?,
                        style_cluster_id = ?,
                        style_cluster_name = ?,
                        matches_analyzed = ?,
                        last_updated = GETDATE()
                    WHERE entity_type = ? AND entity_id = ? AND season_year = ?
                """, profile.get('possession_focus'), profile.get('direct_play_index'),
                     profile.get('crossing_frequency'), profile.get('defensive_intensity'),
                     profile.get('physical_intensity'), profile.get('avg_possession_pct'),
                     profile.get('avg_passes_per_game'), profile.get('avg_long_balls_per_game'),
                     profile.get('avg_crosses_per_game'), profile.get('avg_tackles_per_game'),
                     profile.get('avg_fouls_per_game'), profile.get('style_cluster_id'),
                     profile.get('style_cluster_name'), profile.get('matches_analyzed'),
                     profile['entity_type'], profile['entity_id'], profile['season_year'])
            else:
                # Insert
                cursor.execute("""
                    INSERT INTO [dbo].[style_profiles] (
                        entity_type, entity_id, entity_name, season_year,
                        possession_focus, direct_play_index, crossing_frequency,
                        defensive_intensity, physical_intensity,
                        avg_possession_pct, avg_passes_per_game, avg_long_balls_per_game,
                        avg_crosses_per_game, avg_tackles_per_game, avg_fouls_per_game,
                        style_cluster_id, style_cluster_name, matches_analyzed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, profile['entity_type'], profile['entity_id'], profile.get('entity_name'),
                     profile['season_year'], profile.get('possession_focus'),
                     profile.get('direct_play_index'), profile.get('crossing_frequency'),
                     profile.get('defensive_intensity'), profile.get('physical_intensity'),
                     profile.get('avg_possession_pct'), profile.get('avg_passes_per_game'),
                     profile.get('avg_long_balls_per_game'), profile.get('avg_crosses_per_game'),
                     profile.get('avg_tackles_per_game'), profile.get('avg_fouls_per_game'),
                     profile.get('style_cluster_id'), profile.get('style_cluster_name'),
                     profile.get('matches_analyzed'))
        
        self.conn.commit()
        logger.info(f"Saved {len(profiles)} style profiles to database")