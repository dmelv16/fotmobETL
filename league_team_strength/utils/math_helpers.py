"""
Mathematical helper functions for strength calculations.
"""

import numpy as np
from typing import List, Tuple, Optional

def normalize_to_scale(values: np.ndarray, min_val: float = 0, max_val: float = 100) -> np.ndarray:
    """
    Normalize values to a specified scale.
    
    Args:
        values: Array of values to normalize
        min_val: Minimum value of output scale
        max_val: Maximum value of output scale
    
    Returns:
        Normalized array
    """
    if len(values) == 0:
        return values
    
    val_min = values.min()
    val_max = values.max()
    
    if val_max == val_min:
        return np.full_like(values, (min_val + max_val) / 2)
    
    normalized = ((values - val_min) / (val_max - val_min)) * (max_val - min_val) + min_val
    
    return normalized


def weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted average, handling None values.
    
    Args:
        values: List of values (can contain None)
        weights: List of weights
    
    Returns:
        Weighted average
    """
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
    
    # Filter out None values
    valid_pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    
    if not valid_pairs:
        return 0.0
    
    values_array = np.array([v for v, w in valid_pairs])
    weights_array = np.array([w for v, w in valid_pairs])
    
    # Normalize weights
    weights_normalized = weights_array / weights_array.sum()
    
    return float(np.average(values_array, weights=weights_normalized))


def exponential_decay_weights(n: int, decay_rate: float = 0.15) -> np.ndarray:
    """
    Generate exponentially decaying weights (most recent = highest weight).
    
    Args:
        n: Number of weights to generate
        decay_rate: Decay rate (higher = faster decay)
    
    Returns:
        Array of weights (sum to 1)
    """
    indices = np.arange(n)
    weights = np.exp(-decay_rate * (n - 1 - indices))
    weights = weights / weights.sum()
    
    return weights


def calculate_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a set of values.
    
    Args:
        values: Array of values
        confidence: Confidence level (default 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (values[0], values[0]) if len(values) == 1 else (0.0, 0.0)
    
    mean = np.mean(values)
    std_error = np.std(values, ddof=1) / np.sqrt(len(values))
    
    # Use t-distribution for small samples
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    
    margin = t_value * std_error
    
    return (mean - margin, mean + margin)


def moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Calculate moving average.
    
    Args:
        values: Array of values
        window: Window size
    
    Returns:
        Smoothed array
    """
    if len(values) < window:
        return values
    
    cumsum = np.cumsum(np.insert(values, 0, 0))
    moving_avg = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad beginning with original values
    padding = values[:window-1]
    
    return np.concatenate([padding, moving_avg])


def sigmoid(x: float, center: float = 0, steepness: float = 1) -> float:
    """
    Sigmoid function for smooth transitions.
    
    Args:
        x: Input value
        center: Center point of sigmoid
        steepness: How steep the transition is
    
    Returns:
        Value between 0 and 1
    """
    return 1 / (1 + np.exp(-steepness * (x - center)))


def calculate_percentile_rank(value: float, distribution: np.ndarray) -> float:
    """
    Calculate percentile rank of a value within a distribution.
    
    Args:
        value: Value to rank
        distribution: Array of values representing the distribution
    
    Returns:
        Percentile rank (0-100)
    """
    if len(distribution) == 0:
        return 50.0
    
    rank = (distribution < value).sum() / len(distribution) * 100
    
    return float(rank)