"""
Data quality validation functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_match_data(matches_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate match data quality.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    required_cols = ['match_id', 'home_team_id', 'away_team_id', 
                    'home_team_score', 'away_team_score', 'league_id']
    
    for col in required_cols:
        if col not in matches_df.columns:
            issues.append(f"Missing required column: {col}")
    
    if issues:
        return False, issues
    
    # Check for nulls in critical columns
    for col in required_cols:
        null_count = matches_df[col].isnull().sum()
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} null values")
    
    # Check for duplicate match_ids
    duplicates = matches_df['match_id'].duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate match_ids")
    
    # Check for invalid scores (negative)
    invalid_scores = (
        (matches_df['home_team_score'] < 0) | 
        (matches_df['away_team_score'] < 0)
    ).sum()
    
    if invalid_scores > 0:
        issues.append(f"Found {invalid_scores} matches with negative scores")
    
    # Check for unrealistic scores (>15 goals)
    unrealistic = (
        (matches_df['home_team_score'] > 15) | 
        (matches_df['away_team_score'] > 15)
    ).sum()
    
    if unrealistic > 0:
        logger.warning(f"Found {unrealistic} matches with unusually high scores (>15)")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def validate_player_performance_data(performance_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate player performance data.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    required_cols = ['player_id', 'season_year', 'matches_played']
    
    for col in required_cols:
        if col not in performance_df.columns:
            issues.append(f"Missing required column: {col}")
    
    if issues:
        return False, issues
    
    # Check for reasonable match counts (1-60 per season)
    invalid_matches = (
        (performance_df['matches_played'] < 1) | 
        (performance_df['matches_played'] > 60)
    ).sum()
    
    if invalid_matches > 0:
        issues.append(f"Found {invalid_matches} players with invalid match counts")
    
    # Check for negative stats
    stat_cols = [col for col in performance_df.columns if col.endswith('_per90') or col.endswith('_per_game')]
    
    for col in stat_cols:
        if col in performance_df.columns:
            negative_count = (performance_df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"Column '{col}' has {negative_count} negative values")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def check_data_completeness(conn, season_year: int) -> Dict[str, float]:
    """
    Check data completeness for a season.
    
    Returns:
        Dict with completeness percentages for different data types
    """
    import pyodbc
    cursor = conn.cursor()
    
    completeness = {}
    
    # Check match details
    cursor.execute("""
        SELECT COUNT(*) as total
        FROM [dbo].[match_details]
        WHERE YEAR(match_time_utc) = ?
    """, season_year)
    total_matches = cursor.fetchone()[0]
    
    # Check xG data availability
    cursor.execute("""
        SELECT COUNT(*) as with_xg
        FROM [dbo].[match_details] md
        INNER JOIN [dbo].[match_stats_summary] mss ON md.match_id = mss.match_id
        WHERE YEAR(md.match_time_utc) = ?
          AND mss.home_xg IS NOT NULL
    """, season_year)
    matches_with_xg = cursor.fetchone()[0]
    
    completeness['xg_data'] = (matches_with_xg / total_matches * 100) if total_matches > 0 else 0
    
    # Check lineup data
    cursor.execute("""
        SELECT COUNT(DISTINCT md.match_id) as with_lineups
        FROM [dbo].[match_details] md
        INNER JOIN [dbo].[match_lineup_players] mlp ON md.match_id = mlp.match_id
        WHERE YEAR(md.match_time_utc) = ?
    """, season_year)
    matches_with_lineups = cursor.fetchone()[0]
    
    completeness['lineup_data'] = (matches_with_lineups / total_matches * 100) if total_matches > 0 else 0
    
    # Check player stats
    cursor.execute("""
        SELECT COUNT(DISTINCT md.match_id) as with_player_stats
        FROM [dbo].[match_details] md
        INNER JOIN [dbo].[player_stats] ps ON md.match_id = ps.match_id
        WHERE YEAR(md.match_time_utc) = ?
    """, season_year)
    matches_with_player_stats = cursor.fetchone()[0]
    
    completeness['player_stats'] = (matches_with_player_stats / total_matches * 100) if total_matches > 0 else 0
    
    return completeness


def validate_league_strength_calculation(league_strength: Dict) -> Tuple[bool, List[str]]:
    """
    Validate calculated league strength.
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check if overall strength is within valid range
    strength = league_strength.get('overall_strength')
    
    if strength is None:
        return False, ["Overall strength is None"]
    
    if strength < 0 or strength > 100:
        warnings.append(f"Overall strength {strength:.1f} is outside valid range [0, 100]")
    
    # Check confidence
    confidence = league_strength.get('calculation_confidence', 0)
    
    if confidence < 0.2:
        warnings.append(f"Very low calculation confidence: {confidence:.2f}")
    
    # Check sample size
    sample_size = league_strength.get('sample_size', 0)
    
    if sample_size < 10:
        warnings.append(f"Small sample size: {sample_size}")
    
    # Check component consistency
    components = [
        league_strength.get('transfer_matrix_score'),
        league_strength.get('european_results_score'),
        league_strength.get('network_inference_score'),
        league_strength.get('historical_consistency_score')
    ]
    
    valid_components = [c for c in components if c is not None]
    
    if len(valid_components) >= 2:
        component_std = np.std(valid_components)
        if component_std > 20:
            warnings.append(f"High variance between components (std: {component_std:.1f})")
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings