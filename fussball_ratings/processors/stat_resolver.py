"""
Stat Resolver - Handles home/away resolution for team-level statistics.

This module resolves side-neutral stat keys (like 'xg', 'possession') to 
the correct home/away prefixed columns based on which team we're calculating for.
"""
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SideContext:
    """Context for resolving home/away stats."""
    is_home_team: bool
    team_id: int
    opponent_team_id: int
    match_id: int


class StatResolver:
    """
    Resolves side-neutral stat keys to actual column names based on context.
    
    Example:
        'xg' + is_home_team=True  → 'home_xg'
        'xg' + is_home_team=False → 'away_xg'
        'xg_conceded' + is_home_team=True  → 'away_xg' (opponent's xg)
        'xg_conceded' + is_home_team=False → 'home_xg' (opponent's xg)
    """
    
    # Stats that need home/away resolution from match_stats_summary
    SIDE_AWARE_STATS = {
        # Offensive stats (team's own)
        'xg': ('home_xg', 'away_xg'),
        'xg_open_play': ('home_xg_open_play', 'away_xg_open_play'),
        'xg_set_play': ('home_xg_set_play', 'away_xg_set_play'),
        'xgot': ('home_xgot', 'away_xgot'),
        'possession': ('home_possession', 'away_possession'),
        'total_shots': ('home_total_shots', 'away_total_shots'),
        'shots_on_target': ('home_shots_on_target', 'away_shots_on_target'),
        'shots_off_target': ('home_shots_off_target', 'away_shots_off_target'),
        'blocked_shots': ('home_blocked_shots', 'away_blocked_shots'),
        'shots_inside_box': ('home_shots_inside_box', 'away_shots_inside_box'),
        'shots_outside_box': ('home_shots_outside_box', 'away_shots_outside_box'),
        'big_chances': ('home_big_chances', 'away_big_chances'),
        'big_chances_missed': ('home_big_chances_missed', 'away_big_chances_missed'),
        'passes': ('home_passes', 'away_passes'),
        'accurate_passes': ('home_accurate_passes', 'away_accurate_passes'),
        'pass_accuracy_pct': ('home_pass_accuracy_pct', 'away_pass_accuracy_pct'),
        'own_half_passes': ('home_own_half_passes', 'away_own_half_passes'),
        'opposition_half_passes': ('home_opposition_half_passes', 'away_opposition_half_passes'),
        'accurate_long_balls': ('home_accurate_long_balls', 'away_accurate_long_balls'),
        'accurate_crosses': ('home_accurate_crosses', 'away_accurate_crosses'),
        'touches_opp_box': ('home_touches_opp_box', 'away_touches_opp_box'),
        'tackles': ('home_tackles', 'away_tackles'),
        'interceptions': ('home_interceptions', 'away_interceptions'),
        'blocks': ('home_blocks', 'away_blocks'),
        'clearances': ('home_clearances', 'away_clearances'),
        'keeper_saves': ('home_keeper_saves', 'away_keeper_saves'),
        'duels_won': ('home_duels_won', 'away_duels_won'),
        'ground_duels_won': ('home_ground_duels_won', 'away_ground_duels_won'),
        'aerial_duels_won': ('home_aerial_duels_won', 'away_aerial_duels_won'),
        'successful_dribbles': ('home_successful_dribbles', 'away_successful_dribbles'),
        'corners': ('home_corners', 'away_corners'),
        'offsides': ('home_offsides', 'away_offsides'),
        'fouls': ('home_fouls', 'away_fouls'),
        'yellow_cards': ('home_yellow_cards', 'away_yellow_cards'),
        'red_cards': ('home_red_cards', 'away_red_cards'),
    }
    
    # Stats where we want the OPPONENT's value (conceded stats)
    # Maps to the opposite side
    CONCEDED_STATS = {
        'xg_conceded': 'xg',
        'xg_open_play_conceded': 'xg_open_play',
        'xg_set_play_conceded': 'xg_set_play',
        'shots_conceded': 'total_shots',
        'shots_on_target_conceded': 'shots_on_target',
        'big_chances_conceded': 'big_chances',
        'corners_conceded': 'corners',
    }
    
    # Stats from match_details that need resolution
    MATCH_DETAIL_STATS = {
        'team_score': ('home_team_score', 'away_team_score'),
        'team_score_conceded': ('away_team_score', 'home_team_score'),  # Flipped!
    }
    
    # Stats from match_events that need filtering by is_home_team
    EVENT_STATS = {
        'score_before': ('home_score_before', 'away_score_before'),
        'score_after': ('home_score_after', 'away_score_after'),
    }
    
    @classmethod
    def resolve_stat_key(
        cls,
        stat_key: str,
        table: str,
        is_home_team: bool
    ) -> str:
        """
        Resolve a side-neutral stat key to the actual column name.
        
        Args:
            stat_key: The neutral stat key (e.g., 'xg', 'possession')
            table: The source table
            is_home_team: Whether we're calculating for the home team
            
        Returns:
            The actual column name to use
        """
        # Check if it's a conceded stat (need opponent's value)
        if stat_key in cls.CONCEDED_STATS:
            base_stat = cls.CONCEDED_STATS[stat_key]
            # Flip the side - if we're home, we want away (opponent's) value
            is_home_team = not is_home_team
            stat_key = base_stat
        
        # Resolve based on table
        if table == 'match_stats_summary':
            if stat_key in cls.SIDE_AWARE_STATS:
                home_col, away_col = cls.SIDE_AWARE_STATS[stat_key]
                return home_col if is_home_team else away_col
                
        elif table == 'match_details':
            if stat_key in cls.MATCH_DETAIL_STATS:
                home_col, away_col = cls.MATCH_DETAIL_STATS[stat_key]
                return home_col if is_home_team else away_col
                
        elif table == 'match_events':
            if stat_key in cls.EVENT_STATS:
                home_col, away_col = cls.EVENT_STATS[stat_key]
                return home_col if is_home_team else away_col
        
        # If not a side-aware stat, return as-is
        return stat_key
    
    @classmethod
    def resolve_stats_dict(
        cls,
        stats: Dict[str, Any],
        table: str,
        is_home_team: bool,
        requested_keys: list
    ) -> Dict[str, Any]:
        """
        Extract requested stats from a row, resolving home/away.
        
        Args:
            stats: The full row of stats (with home_/away_ prefixes)
            table: Source table name
            is_home_team: Whether we're the home team
            requested_keys: List of neutral stat keys we want
            
        Returns:
            Dict mapping neutral keys to their values
        """
        result = {}
        
        for key in requested_keys:
            resolved_key = cls.resolve_stat_key(key, table, is_home_team)
            if resolved_key in stats:
                result[key] = stats[resolved_key]
            else:
                result[key] = None
                
        return result
    
    @classmethod
    def get_team_stats_from_summary(
        cls,
        match_stats_summary: Dict[str, Any],
        is_home_team: bool
    ) -> Dict[str, Any]:
        """
        Extract all stats for one team from match_stats_summary.
        
        Returns a dict with neutral keys (no home_/away_ prefix).
        """
        result = {}
        
        for neutral_key in cls.SIDE_AWARE_STATS.keys():
            resolved = cls.resolve_stat_key(neutral_key, 'match_stats_summary', is_home_team)
            if resolved in match_stats_summary:
                result[neutral_key] = match_stats_summary[resolved]
        
        # Also add conceded stats
        for conceded_key in cls.CONCEDED_STATS.keys():
            resolved = cls.resolve_stat_key(conceded_key, 'match_stats_summary', is_home_team)
            if resolved in match_stats_summary:
                result[conceded_key] = match_stats_summary[resolved]
                
        return result
    
    @classmethod
    def get_team_scores(
        cls,
        match_details: Dict[str, Any],
        is_home_team: bool
    ) -> Tuple[int, int]:
        """
        Get (team_score, opponent_score) from match details.
        """
        if is_home_team:
            return (
                match_details.get('home_team_score', 0),
                match_details.get('away_team_score', 0)
            )
        else:
            return (
                match_details.get('away_team_score', 0),
                match_details.get('home_team_score', 0)
            )


class TeamMatchStatsBuilder:
    """
    Builds a complete stats dict for a team/coach from a match,
    handling all the home/away resolution.
    """
    
    def __init__(self, stat_resolver: StatResolver = None):
        self.resolver = stat_resolver or StatResolver()
    
    def build_stats(
        self,
        match_details: Dict[str, Any],
        match_stats_summary: Optional[Dict[str, Any]],
        match_team_stats: Optional[Dict[str, Any]],
        is_home_team: bool
    ) -> Dict[str, Any]:
        """
        Build complete stats dict for a team from all available sources.
        
        Args:
            match_details: Row from match_details table
            match_stats_summary: Row from match_stats_summary (may be None)
            match_team_stats: Row from match_team_stats for this team (may be None)
            is_home_team: Whether this is the home team
            
        Returns:
            Dict with all available stats, using neutral keys
        """
        stats = {}
        
        # From match_details
        team_score, opponent_score = StatResolver.get_team_scores(match_details, is_home_team)
        stats['team_score'] = team_score
        stats['team_score_conceded'] = opponent_score
        stats['goal_difference'] = team_score - opponent_score
        
        # Result
        if team_score > opponent_score:
            stats['result'] = 'W'
            stats['points'] = 3
        elif team_score < opponent_score:
            stats['result'] = 'L'
            stats['points'] = 0
        else:
            stats['result'] = 'D'
            stats['points'] = 1
        
        # From match_stats_summary
        if match_stats_summary:
            summary_stats = StatResolver.get_team_stats_from_summary(
                match_stats_summary, is_home_team
            )
            stats.update(summary_stats)
        
        # From match_team_stats (already team-specific, no resolution needed)
        if match_team_stats:
            stats['formation'] = match_team_stats.get('formation')
            stats['team_rating'] = match_team_stats.get('rating')
            stats['average_starter_age'] = match_team_stats.get('average_starter_age')
            stats['total_starter_market_value'] = match_team_stats.get('total_starter_market_value')
        
        # Add context
        stats['is_home_team'] = is_home_team
        stats['match_id'] = match_details.get('match_id')
        stats['league_id'] = match_details.get('league_id')
        
        return stats