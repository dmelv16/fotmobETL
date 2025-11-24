"""
SQL queries for match data retrieval.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

from database.connection import QueryExecutor


class MatchQueries:
    """Queries for match-related data."""
    
    def __init__(self, executor: QueryExecutor):
        self.executor = executor
    
    # =========================================================================
    # EXTRACTION STATUS & DATA AVAILABILITY
    # =========================================================================
    
    def get_extraction_status(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Get data availability for a single match."""
        query = """
            SELECT match_id, league_id, season_year,
                   has_match_details, has_lineups, has_stats,
                   has_momentum, has_player_stats, has_shotmap, has_events
            FROM extraction_status
            WHERE match_id = ?
        """
        results = self.executor.execute_query(query, (match_id,))
        return results[0] if results else None
    
    def get_all_extraction_status(self) -> List[Dict[str, Any]]:
        """Get data availability for all matches."""
        query = """
            SELECT match_id, league_id, season_year,
                   has_match_details, has_lineups, has_stats,
                   has_momentum, has_player_stats, has_shotmap, has_events
            FROM extraction_status
            ORDER BY match_id
        """
        return self.executor.execute_query(query)
    
    def get_matches_by_tier(self, min_tier: int) -> List[int]:
        """
        Get match IDs that meet minimum data tier.
        
        Tier mapping:
        1: has_match_details + has_events
        2: + has_lineups
        3: + has_stats
        4: + has_player_stats
        5: + has_shotmap
        """
        tier_conditions = {
            1: "has_match_details = 1 AND has_events = 1",
            2: "has_match_details = 1 AND has_events = 1 AND has_lineups = 1",
            3: "has_match_details = 1 AND has_events = 1 AND has_lineups = 1 AND has_stats = 1",
            4: "has_match_details = 1 AND has_events = 1 AND has_lineups = 1 AND has_stats = 1 AND has_player_stats = 1",
            5: "has_match_details = 1 AND has_events = 1 AND has_lineups = 1 AND has_stats = 1 AND has_player_stats = 1 AND has_shotmap = 1",
        }
        
        condition = tier_conditions.get(min_tier, tier_conditions[1])
        query = f"SELECT match_id FROM extraction_status WHERE {condition}"
        
        results = self.executor.execute_query(query)
        return [r['match_id'] for r in results]
    
    # =========================================================================
    # MATCH DETAILS
    # =========================================================================
    
    def get_match_details(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Get core match details."""
        query = """
            SELECT id, source_id, match_id, match_name, league_id, league_name,
                   parent_league_id, coverage_level, match_time_utc,
                   started, finished, cancelled, awarded,
                   home_team_id, home_team_name, home_team_score,
                   away_team_id, away_team_name, away_team_score,
                   score_str, status_reason_short, status_reason_long,
                   number_of_home_red_cards, number_of_away_red_cards,
                   first_half_started, first_half_ended,
                   second_half_started, second_half_ended,
                   who_lost_on_penalties, who_lost_on_aggregated
            FROM match_details
            WHERE match_id = ?
        """
        results = self.executor.execute_query(query, (match_id,))
        return results[0] if results else None
    
    def get_matches_in_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        league_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all matches within a date range."""
        query = """
            SELECT match_id, match_time_utc, league_id,
                   home_team_id, home_team_name, home_team_score,
                   away_team_id, away_team_name, away_team_score,
                   finished
            FROM match_details
            WHERE match_time_utc >= ? AND match_time_utc <= ?
                  AND finished = 1
        """
        params = [start_date, end_date]
        
        if league_id:
            query += " AND league_id = ?"
            params.append(league_id)
        
        query += " ORDER BY match_time_utc"
        
        return self.executor.execute_query(query, tuple(params))
    
    def get_all_matches_chronological(self) -> List[Dict[str, Any]]:
        """Get all finished matches in chronological order."""
        query = """
            SELECT md.match_id, md.match_time_utc, md.league_id, md.league_name,
                   md.home_team_id, md.home_team_name, md.home_team_score,
                   md.away_team_id, md.away_team_name, md.away_team_score,
                   es.has_match_details, es.has_lineups, es.has_stats,
                   es.has_player_stats, es.has_shotmap, es.has_events
            FROM match_details md
            LEFT JOIN extraction_status es ON md.match_id = es.match_id
            WHERE md.finished = 1
            ORDER BY md.match_time_utc
        """
        return self.executor.execute_query(query)
    
    # =========================================================================
    # MATCH FACTS
    # =========================================================================
    
    def get_match_facts(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Get match facts (stadium, attendance, referee)."""
        query = """
            SELECT match_id, stadium_name, stadium_country,
                   attendance, referee, tournament_id, tournament_name
            FROM match_facts
            WHERE match_id = ?
        """
        results = self.executor.execute_query(query, (match_id,))
        return results[0] if results else None
    
    # =========================================================================
    # MATCH STATS SUMMARY
    # =========================================================================
    
    def get_match_stats_summary(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Get team-level match statistics."""
        query = """
            SELECT *
            FROM match_stats_summary
            WHERE match_id = ?
        """
        results = self.executor.execute_query(query, (match_id,))
        return results[0] if results else None
    
    # =========================================================================
    # MATCH TEAM STATS
    # =========================================================================
    
    def get_match_team_stats(self, match_id: int) -> List[Dict[str, Any]]:
        """Get team stats for both teams in a match."""
        query = """
            SELECT match_id, team_id, team_name, formation, rating,
                   average_starter_age, total_starter_market_value, is_home_team
            FROM match_team_stats
            WHERE match_id = ?
        """
        return self.executor.execute_query(query, (match_id,))
    
    # =========================================================================
    # MATCH EVENTS
    # =========================================================================
    
    def get_match_events(self, match_id: int) -> List[Dict[str, Any]]:
        """Get all events (goals, cards, etc.) for a match."""
        query = """
            SELECT match_id, event_type, is_home_team, time_minute, time_str,
                   player_id, player_name, player_first_name, player_last_name,
                   assist_player, home_score_before, away_score_before,
                   home_score_after, away_score_after, own_goal,
                   penalty_shootout, goal_description
            FROM match_events
            WHERE match_id = ?
            ORDER BY time_minute
        """
        return self.executor.execute_query(query, (match_id,))
    
    def get_goals_for_match(self, match_id: int) -> List[Dict[str, Any]]:
        """Get only goal events for a match."""
        query = """
            SELECT match_id, is_home_team, time_minute,
                   player_id, player_name, assist_player,
                   own_goal, penalty_shootout
            FROM match_events
            WHERE match_id = ? AND event_type = 'goal'
            ORDER BY time_minute
        """
        return self.executor.execute_query(query, (match_id,))
    
    # =========================================================================
    # MATCH SHOTMAP
    # =========================================================================
    
    def get_match_shotmap(self, match_id: int) -> List[Dict[str, Any]]:
        """Get detailed shot data for a match."""
        query = """
            SELECT match_id, shot_id, event_type, team_id,
                   player_id, player_name, first_name, last_name, full_name,
                   x, y, minute, minute_added,
                   is_blocked, is_on_target, blocked_x, blocked_y,
                   goal_crossed_y, goal_crossed_z,
                   expected_goals, expected_goals_on_target,
                   shot_type, situation, period,
                   is_own_goal, is_saved_off_line, is_from_inside_box
            FROM match_shotmap
            WHERE match_id = ?
            ORDER BY minute, minute_added
        """
        return self.executor.execute_query(query, (match_id,))
    
    def get_player_shots(self, match_id: int, player_id: int) -> List[Dict[str, Any]]:
        """Get shots for a specific player in a match."""
        query = """
            SELECT shot_id, event_type, x, y, minute,
                   is_blocked, is_on_target, expected_goals,
                   shot_type, situation, is_from_inside_box
            FROM match_shotmap
            WHERE match_id = ? AND player_id = ?
            ORDER BY minute
        """
        return self.executor.execute_query(query, (match_id, player_id))
    
    # =========================================================================
    # SUBSTITUTIONS
    # =========================================================================
    
    def get_match_substitutions(self, match_id: int) -> List[Dict[str, Any]]:
        """Get substitution data for a match."""
        query = """
            SELECT match_id, player_id, player_name, team_id,
                   substitution_time, substitution_type, substitution_reason
            FROM match_substitutions
            WHERE match_id = ?
            ORDER BY substitution_time
        """
        return self.executor.execute_query(query, (match_id,))
    
    # =========================================================================
    # UNAVAILABLE PLAYERS
    # =========================================================================
    
    def get_unavailable_players(self, match_id: int) -> List[Dict[str, Any]]:
        """Get players unavailable for a match (injuries, suspensions)."""
        query = """
            SELECT match_id, team_id, player_id, player_name,
                   first_name, last_name, age, country_name, country_code,
                   market_value, is_home_team, injury_id,
                   unavailability_type, expected_return
            FROM match_unavailable_players
            WHERE match_id = ?
        """
        return self.executor.execute_query(query, (match_id,))


class LeagueQueries:
    """Queries for league/competition data."""
    
    def __init__(self, executor: QueryExecutor):
        self.executor = executor
    
    def get_league_divisions(self) -> List[Dict[str, Any]]:
        """Get all league division mappings."""
        query = """
            SELECT ID, CountryID, CountryName, LeagueName, 
                   LeagueID, DivisionLevel
            FROM LeagueDivisions
        """
        return self.executor.execute_query(query)
    
    def get_league_info(self, league_id: int) -> Optional[Dict[str, Any]]:
        """Get info for a specific league."""
        query = """
            SELECT ID, CountryID, CountryName, LeagueName,
                   LeagueID, DivisionLevel
            FROM LeagueDivisions
            WHERE LeagueID = ?
        """
        results = self.executor.execute_query(query, (league_id,))
        return results[0] if results else None
    
    def get_competition_type(self, league_id: int) -> str:
        """
        Determine competition type from LeagueDivisions.
        Returns: 'League', 'Cup', or 'Continental'
        """
        info = self.get_league_info(league_id)
        if not info:
            return 'League'  # Default
        
        division_level = info.get('DivisionLevel', '')
        if division_level == 'Cup':
            return 'Cup'
        elif division_level == 'Continental':
            return 'Continental'
        else:
            return 'League'
    
    def get_leagues_by_country(self, country_id: int) -> List[Dict[str, Any]]:
        """Get all leagues for a country."""
        query = """
            SELECT LeagueID, LeagueName, DivisionLevel
            FROM LeagueDivisions
            WHERE CountryID = ?
            ORDER BY DivisionLevel
        """
        return self.executor.execute_query(query, (country_id,))
    
    def get_all_countries(self) -> List[Dict[str, Any]]:
        """Get list of all countries with leagues."""
        query = """
            SELECT DISTINCT CountryID, CountryName
            FROM LeagueDivisions
            ORDER BY CountryName
        """
        return self.executor.execute_query(query)