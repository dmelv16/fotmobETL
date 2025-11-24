"""
Utility functions and helpers.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional, TypeVar, Callable
import math
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# STAT AGGREGATION HELPERS
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def weighted_average(values: List[float], weights: List[float]) -> Optional[float]:
    """Calculate weighted average."""
    if not values or not weights or len(values) != len(weights):
        return None
    
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def percentile(values: List[float], p: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


def standard_deviation(values: List[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def normalize_to_range(
    value: float, 
    old_min: float, 
    old_max: float,
    new_min: float = 0.0,
    new_max: float = 1.0
) -> float:
    """Normalize a value from one range to another."""
    if old_max == old_min:
        return (new_min + new_max) / 2
    
    normalized = (value - old_min) / (old_max - old_min)
    return new_min + normalized * (new_max - new_min)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


# =============================================================================
# PER-90 CALCULATIONS
# =============================================================================

def per_90(stat: float, minutes: int) -> float:
    """Convert a stat to per-90 minutes."""
    if minutes <= 0:
        return 0.0
    return (stat / minutes) * 90


def per_90_dict(stats: Dict[str, float], minutes: int) -> Dict[str, float]:
    """Convert all stats in a dict to per-90."""
    if minutes <= 0:
        return {k: 0.0 for k in stats}
    
    factor = 90 / minutes
    return {k: v * factor for k, v in stats.items()}


# =============================================================================
# SEASON/DATE HELPERS
# =============================================================================

def get_season_year(match_date: datetime) -> str:
    """
    Determine the season year for a match date.
    
    Assumes seasons run Aug-May (European style).
    A match in Jan 2024 is in "2023-24" season.
    A match in Sep 2024 is in "2024-25" season.
    """
    year = match_date.year
    month = match_date.month
    
    # If before August, it's the previous year's season
    if month < 8:
        return f"{year-1}-{str(year)[2:]}"
    else:
        return f"{year}-{str(year+1)[2:]}"


def parse_season_year(season_str: str) -> tuple:
    """Parse season string to start and end years."""
    parts = season_str.split('-')
    if len(parts) == 2:
        start_year = int(parts[0])
        end_suffix = parts[1]
        if len(end_suffix) == 2:
            end_year = int(f"{str(start_year)[:2]}{end_suffix}")
        else:
            end_year = int(end_suffix)
        return (start_year, end_year)
    return (int(parts[0]), int(parts[0]) + 1)


def days_between(date1: datetime, date2: datetime) -> int:
    """Get absolute days between two dates."""
    return abs((date2 - date1).days)


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_rating(rating: float) -> bool:
    """Check if a rating value is valid."""
    return 0 <= rating <= 3000


def validate_attribute(value: float) -> bool:
    """Check if an attribute value is valid (0-20 scale)."""
    return 0 <= value <= 20


def validate_percentage(value: float) -> bool:
    """Check if a percentage is valid."""
    return 0 <= value <= 100


def is_valid_match_result(home_score: int, away_score: int) -> bool:
    """Check if match scores are valid."""
    return (
        home_score is not None and 
        away_score is not None and
        home_score >= 0 and 
        away_score >= 0 and
        home_score < 50 and  # Sanity check
        away_score < 50
    )


# =============================================================================
# RESULT HELPERS
# =============================================================================

def get_result(home_score: int, away_score: int, is_home: bool) -> str:
    """Get result from team's perspective ('W', 'D', 'L')."""
    if is_home:
        if home_score > away_score:
            return 'W'
        elif home_score < away_score:
            return 'L'
    else:
        if away_score > home_score:
            return 'W'
        elif away_score < home_score:
            return 'L'
    return 'D'


def get_points(result: str) -> int:
    """Convert result to points."""
    return {'W': 3, 'D': 1, 'L': 0}.get(result, 0)


def goal_difference(home_score: int, away_score: int, is_home: bool) -> int:
    """Get goal difference from team's perspective."""
    if is_home:
        return home_score - away_score
    return away_score - home_score


# =============================================================================
# BATCH PROCESSING HELPERS
# =============================================================================

def batch_list(items: List[T], batch_size: int) -> List[List[T]]:
    """Split a list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Optional[T]:
    """Retry a function with exponential backoff."""
    import time
    
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= backoff_factor
    
    logger.error(f"All {max_retries} attempts failed")
    raise last_exception


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def format_rating(rating: float) -> str:
    """Format rating for display."""
    return f"{rating:.0f}"


def format_attribute(value: float) -> str:
    """Format attribute for display."""
    return f"{value:.1f}"


def format_percentage(value: float) -> str:
    """Format percentage for display."""
    return f"{value:.1f}%"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def rating_to_tier(rating: float) -> str:
    """Convert rating to a tier label."""
    if rating >= 1800:
        return "World Class"
    elif rating >= 1600:
        return "Elite"
    elif rating >= 1400:
        return "Very Good"
    elif rating >= 1200:
        return "Good"
    elif rating >= 1000:
        return "Average"
    elif rating >= 800:
        return "Below Average"
    else:
        return "Poor"


def attribute_to_description(value: float) -> str:
    """Convert attribute value to description."""
    if value >= 18:
        return "World Class"
    elif value >= 15:
        return "Excellent"
    elif value >= 12:
        return "Good"
    elif value >= 9:
        return "Average"
    elif value >= 6:
        return "Below Average"
    else:
        return "Poor"