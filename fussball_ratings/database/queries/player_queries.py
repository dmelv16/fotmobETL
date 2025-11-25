"""
SQL queries for player, coach, and lineup data retrieval.
"""
from typing import Dict, List, Any, Optional, Set

from database.connection import QueryExecutor


class PlayerQueries:
    """Queries for player-related data."""
    
    def __init__(self, executor: QueryExecutor):
        self.executor = executor
    
    # =========================================================================
    # MATCH LINEUPS
    # =========================================================================
    
    def get_match_lineup(self, match_id: int) -> List[Dict[str, Any]]:
        """Get full lineup for a match (both teams)."""
        query = """
            SELECT match_id, team_id, player_id, player_name,
                   first_name, last_name, short_name, age,
                   position_id, usual_position_id, shirt_number,
                   is_captain, is_home_team, is_starter, formation,
                   country_name, country_code, market_value,
                   rating, fantasy_score, player_of_match
            FROM match_lineup_players
            WHERE match_id = ?
        """
        return self.executor.execute_query(query, (match_id,))
    
    def get_team_lineup(self, match_id: int, team_id: int) -> List[Dict[str, Any]]:
        """Get lineup for a specific team in a match."""
        query = """
            SELECT player_id, player_name, first_name, last_name,
                   age, position_id, usual_position_id, shirt_number,
                   is_captain, is_starter, formation,
                   country_name, country_code, market_value,
                   rating, fantasy_score, player_of_match
            FROM match_lineup_players
            WHERE match_id = ? AND team_id = ?
        """
        return self.executor.execute_query(query, (match_id, team_id))
    
    def get_starters(self, match_id: int, team_id: int) -> List[Dict[str, Any]]:
        """Get starting XI for a team."""
        query = """
            SELECT player_id, player_name, position_id, usual_position_id,
                   age, rating, is_captain
            FROM match_lineup_players
            WHERE match_id = ? AND team_id = ? AND is_starter = 1
        """
        return self.executor.execute_query(query, (match_id, team_id))
    
    def get_player_match_history(
        self, 
        player_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all matches a player has appeared in."""
        query = """
            SELECT mlp.match_id, mlp.team_id, mlp.position_id, mlp.is_starter,
                   mlp.rating, mlp.age, md.match_time_utc, md.league_id,
                   md.home_team_id, md.away_team_id,
                   md.home_team_score, md.away_team_score
            FROM match_lineup_players mlp
            JOIN match_details md ON mlp.match_id = md.match_id
            WHERE mlp.player_id = ?
            ORDER BY md.match_time_utc DESC
        """
        if limit:
            query = query.replace("SELECT", f"SELECT TOP {limit}")
        
        return self.executor.execute_query(query, (player_id,))
    
    def get_player_positions_played(self, player_id: int) -> List[Dict[str, Any]]:
        """Get all positions a player has played and count."""
        query = """
            SELECT position_id, COUNT(*) as match_count,
                   AVG(CAST(rating as FLOAT)) as avg_rating
            FROM match_lineup_players
            WHERE player_id = ? AND rating IS NOT NULL
            GROUP BY position_id
            ORDER BY match_count DESC
        """
        return self.executor.execute_query(query, (player_id,))
    
    def get_player_teams(self, player_id: int) -> List[Dict[str, Any]]:
        """Get all teams a player has played for with dates."""
        query = """
            SELECT team_id, MIN(md.match_time_utc) as first_match,
                   MAX(md.match_time_utc) as last_match,
                   COUNT(*) as appearances
            FROM match_lineup_players mlp
            JOIN match_details md ON mlp.match_id = md.match_id
            WHERE mlp.player_id = ?
            GROUP BY team_id
            ORDER BY first_match
        """
        return self.executor.execute_query(query, (player_id,))
    
    # =========================================================================
    # PLAYER STATS
    # =========================================================================
    
    def get_player_match_stats(
        self, 
        match_id: int, 
        player_id: int
    ) -> List[Dict[str, Any]]:
        """Get all stats for a player in a specific match."""
        query = """
            SELECT match_id, player_id, player_name, team_id, team_name,
                   is_goalkeeper, category, category_key,
                   stat_name, stat_key, stat_value, stat_total, stat_type
            FROM player_stats
            WHERE match_id = ? AND player_id = ?
        """
        return self.executor.execute_query(query, (match_id, player_id))
    
    def get_all_player_stats_for_match(self, match_id: int) -> List[Dict[str, Any]]:
        """Get all player stats for a match."""
        query = """
            SELECT player_id, player_name, team_id, is_goalkeeper,
                   stat_key, stat_value, stat_total
            FROM player_stats
            WHERE match_id = ?
        """
        return self.executor.execute_query(query, (match_id,))
    
    def get_player_stat_history(
        self, 
        player_id: int, 
        stat_key: str,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get history of a specific stat for a player."""
        query = """
            SELECT ps.match_id, ps.stat_value, ps.stat_total,
                   md.match_time_utc, md.league_id
            FROM player_stats ps
            JOIN match_details md ON ps.match_id = md.match_id
            WHERE ps.player_id = ? AND ps.stat_key = ?
            ORDER BY md.match_time_utc DESC
        """
        if limit:
            query = query.replace("SELECT", f"SELECT TOP {limit}")
        
        return self.executor.execute_query(query, (player_id, stat_key))
    
    def get_available_stat_keys_for_match(self, match_id: int) -> Set[str]:
        """Get which stat_keys are available for a match."""
        query = """
            SELECT DISTINCT stat_key
            FROM player_stats
            WHERE match_id = ? AND stat_value IS NOT NULL
        """
        results = self.executor.execute_query(query, (match_id,))
        return {r['stat_key'] for r in results}
    
    def get_goalkeeper_stats(
        self, 
        match_id: int, 
        player_id: int
    ) -> List[Dict[str, Any]]:
        """Get goalkeeper-specific stats."""
        query = """
            SELECT stat_key, stat_value, stat_total
            FROM player_stats
            WHERE match_id = ? AND player_id = ? AND is_goalkeeper = 1
        """
        return self.executor.execute_query(query, (match_id, player_id))
    
    # =========================================================================
    # PLAYER IDENTIFICATION
    # =========================================================================
    
    def get_all_unique_players(self) -> List[Dict[str, Any]]:
        """Get all unique players in the database."""
        query = """
            SELECT DISTINCT player_id, 
                   MAX(player_name) as player_name,
                   MAX(first_name) as first_name,
                   MAX(last_name) as last_name,
                   MAX(country_code) as country_code,
                   MAX(usual_position_id) as usual_position_id
            FROM match_lineup_players
            WHERE player_id IS NOT NULL
            GROUP BY player_id
        """
        return self.executor.execute_query(query)
    
    def find_player_by_name(
        self, 
        name: str, 
        team_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for players by name."""
        query = """
            SELECT DISTINCT player_id, player_name, usual_position_id
            FROM match_lineup_players
            WHERE player_name LIKE ?
        """
        params = [f"%{name}%"]
        
        if team_id:
            query += " AND team_id = ?"
            params.append(team_id)
        
        return self.executor.execute_query(query, tuple(params))

    def get_player_match_stats_bulk(
        self, 
        player_id: int, 
        match_ids: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """Get all stats for a player across multiple matches."""
        if not match_ids:
            return {}
        
        placeholders = ','.join(['?' for _ in match_ids])
        query = f"""
            SELECT match_id, stat_key, stat_value
            FROM player_stats
            WHERE player_id = ? AND match_id IN ({placeholders})
        """
        results = self.executor.execute_query(query, tuple([player_id] + match_ids))
        
        # Group by match_id
        stats_by_match = {}
        for row in results:
            match_id = row['match_id']
            if match_id not in stats_by_match:
                stats_by_match[match_id] = {}
            if row['stat_value'] is not None:
                stats_by_match[match_id][row['stat_key']] = row['stat_value']
        
        return stats_by_match
    
    def get_player_injuries(self, player_id: int) -> List[Dict[str, Any]]:
        """Get injury history for a player from match_unavailable_players."""
        query = """
            SELECT mup.match_id, mup.injury_id, mup.unavailability_type,
                   mup.expected_return, md.match_time_utc
            FROM match_unavailable_players mup
            JOIN match_details md ON mup.match_id = md.match_id
            WHERE mup.player_id = ? AND mup.injury_id IS NOT NULL
            ORDER BY md.match_time_utc DESC
        """
        return self.executor.execute_query(query, (player_id,))
    
    def get_player_injuries_bulk(
        self, 
        player_id: int, 
        match_ids: List[int]
    ) -> Dict[int, int]:
        """Get injury_id for player across multiple matches. Returns {match_id: injury_id}."""
        if not match_ids:
            return {}
        
        placeholders = ','.join(['?' for _ in match_ids])
        query = f"""
            SELECT match_id, injury_id
            FROM match_unavailable_players
            WHERE player_id = ? AND match_id IN ({placeholders}) AND injury_id IS NOT NULL
        """
        results = self.executor.execute_query(query, tuple([player_id] + match_ids))
        return {row['match_id']: row['injury_id'] for row in results}
    
    def get_player_full_match_context(
        self, 
        player_id: int, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get complete match context for a player including:
        - Lineup data (position, rating, captain status)
        - Match details (league, scores)
        - Team ratings (for big game calculation)
        - Injury status
        """
        query = """
            SELECT 
                mlp.match_id,
                mlp.player_id,
                mlp.team_id,
                mlp.position_id,
                mlp.usual_position_id,
                mlp.is_starter,
                mlp.is_captain,
                mlp.rating,
                mlp.age,
                mlp.is_home_team,
                
                md.league_id,
                md.league_name,
                md.match_time_utc,
                md.home_team_id,
                md.away_team_id,
                md.home_team_score,
                md.away_team_score,
                
                mts_own.rating AS own_team_rating,
                mts_opp.rating AS opponent_team_rating,
                
                mup.injury_id
                
            FROM match_lineup_players mlp
            INNER JOIN match_details md ON mlp.match_id = md.match_id
            LEFT JOIN match_team_stats mts_own 
                ON mlp.match_id = mts_own.match_id AND mlp.team_id = mts_own.team_id
            LEFT JOIN match_team_stats mts_opp 
                ON mlp.match_id = mts_opp.match_id AND mlp.team_id != mts_opp.team_id
            LEFT JOIN match_unavailable_players mup
                ON mlp.player_id = mup.player_id AND mlp.match_id = mup.match_id
            WHERE mlp.player_id = ? AND md.finished = 1
            ORDER BY md.match_time_utc DESC
        """
        if limit:
            query = query.replace("SELECT", f"SELECT TOP {limit}")
        
        return self.executor.execute_query(query, (player_id,))
    

class CoachQueries:
    """Queries for coach-related data."""
    
    def __init__(self, executor: QueryExecutor):
        self.executor = executor
    
    def get_match_coaches(self, match_id: int) -> List[Dict[str, Any]]:
        """Get coaches for both teams in a match."""
        query = """
            SELECT match_id, team_id, coach_id, coach_name,
                   first_name, last_name, age, is_home_team,
                   country_name, country_code
            FROM match_coaches
            WHERE match_id = ? AND is_coach = 1
        """
        return self.executor.execute_query(query, (match_id,))
    
    def get_coach_match_history(
        self, 
        coach_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all matches for a coach."""
        query = """
            SELECT mc.match_id, mc.team_id, mc.is_home_team,
                   md.match_time_utc, md.league_id,
                   md.home_team_score, md.away_team_score,
                   mts.formation, mts.rating as team_rating
            FROM match_coaches mc
            JOIN match_details md ON mc.match_id = md.match_id
            LEFT JOIN match_team_stats mts ON mc.match_id = mts.match_id 
                AND mc.team_id = mts.team_id
            WHERE mc.coach_id = ? AND mc.is_coach = 1
            ORDER BY md.match_time_utc DESC
        """
        if limit:
            query = query.replace("SELECT", f"SELECT TOP {limit}")
        
        return self.executor.execute_query(query, (coach_id,))
    
    def get_coach_teams(self, coach_id: int) -> List[Dict[str, Any]]:
        """Get all teams a coach has managed."""
        query = """
            SELECT mc.team_id,
                   MIN(md.match_time_utc) as first_match,
                   MAX(md.match_time_utc) as last_match,
                   COUNT(*) as matches_managed
            FROM match_coaches mc
            JOIN match_details md ON mc.match_id = md.match_id
            WHERE mc.coach_id = ? AND mc.is_coach = 1
            GROUP BY mc.team_id
            ORDER BY first_match
        """
        return self.executor.execute_query(query, (coach_id,))
    
    def get_all_unique_coaches(self) -> List[Dict[str, Any]]:
        """Get all unique coaches in the database."""
        query = """
            SELECT DISTINCT coach_id,
                   MAX(coach_name) as coach_name,
                   MAX(first_name) as first_name,
                   MAX(last_name) as last_name,
                   MAX(country_code) as country_code
            FROM match_coaches
            WHERE coach_id IS NOT NULL AND is_coach = 1
            GROUP BY coach_id
        """
        return self.executor.execute_query(query)
    
    def get_coach_record(self, coach_id: int) -> Dict[str, int]:
        """Get win/draw/loss record for a coach."""
        query = """
            SELECT 
                SUM(CASE 
                    WHEN (mc.is_home_team = 1 AND md.home_team_score > md.away_team_score)
                      OR (mc.is_home_team = 0 AND md.away_team_score > md.home_team_score)
                    THEN 1 ELSE 0 END) as wins,
                SUM(CASE 
                    WHEN md.home_team_score = md.away_team_score 
                    THEN 1 ELSE 0 END) as draws,
                SUM(CASE 
                    WHEN (mc.is_home_team = 1 AND md.home_team_score < md.away_team_score)
                      OR (mc.is_home_team = 0 AND md.away_team_score < md.home_team_score)
                    THEN 1 ELSE 0 END) as losses
            FROM match_coaches mc
            JOIN match_details md ON mc.match_id = md.match_id
            WHERE mc.coach_id = ? AND mc.is_coach = 1 AND md.finished = 1
        """
        results = self.executor.execute_query(query, (coach_id,))
        if results:
            return {
                'wins': results[0]['wins'] or 0,
                'draws': results[0]['draws'] or 0,
                'losses': results[0]['losses'] or 0
            }
        return {'wins': 0, 'draws': 0, 'losses': 0}

    def get_coach_full_match_context(
        self, 
        coach_id: int, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get complete match context for a coach including:
        - Team stats (resolved to coach's team perspective)
        - Formation
        - Match result
        """
        query = """
            SELECT 
                mc.coach_id,
                mc.match_id,
                mc.team_id,
                mc.is_home_team,
                
                md.match_time_utc,
                md.home_team_id,
                md.away_team_id,
                md.home_team_score,
                md.away_team_score,
                md.league_id,
                
                mts.formation,
                mts.rating AS team_rating,
                
                CASE WHEN mc.is_home_team = 1 THEN mss.home_xg ELSE mss.away_xg END AS xg,
                CASE WHEN mc.is_home_team = 1 THEN mss.away_xg ELSE mss.home_xg END AS xg_conceded,
                CASE WHEN mc.is_home_team = 1 THEN mss.home_possession ELSE mss.away_possession END AS possession,
                CASE WHEN mc.is_home_team = 1 THEN mss.home_yellow_cards ELSE mss.away_yellow_cards END AS yellow_cards,
                CASE WHEN mc.is_home_team = 1 THEN mss.home_red_cards ELSE mss.away_red_cards END AS red_cards,
                CASE WHEN mc.is_home_team = 1 THEN mss.home_fouls ELSE mss.away_fouls END AS fouls,
                CASE WHEN mc.is_home_team = 1 THEN mss.home_xg_set_play ELSE mss.away_xg_set_play END AS xg_set_play,
                CASE WHEN mc.is_home_team = 1 THEN mss.away_xg_set_play ELSE mss.home_xg_set_play END AS xg_set_play_conceded
                
            FROM match_coaches mc
            INNER JOIN match_details md ON mc.match_id = md.match_id
            LEFT JOIN match_team_stats mts 
                ON mc.match_id = mts.match_id AND mc.team_id = mts.team_id
            LEFT JOIN match_stats_summary mss ON mc.match_id = mss.match_id
            WHERE mc.coach_id = ? AND mc.is_coach = 1 AND md.finished = 1
            ORDER BY md.match_time_utc DESC
        """
        if limit:
            query = query.replace("SELECT", f"SELECT TOP {limit}")
        
        return self.executor.execute_query(query, (coach_id,))
    
    def get_coach_lineup_data_bulk(
        self, 
        coach_id: int, 
        match_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get lineup data (player_ids, ratings, ages) for coach's team across matches.
        Returns: {match_id: {'player_ids': [...], 'ratings': {pid: rating}, 'ages': {pid: age}}}
        """
        if not match_ids:
            return {}
        
        # First get team_id for each match
        placeholders = ','.join(['?' for _ in match_ids])
        team_query = f"""
            SELECT match_id, team_id
            FROM match_coaches
            WHERE coach_id = ? AND match_id IN ({placeholders}) AND is_coach = 1
        """
        team_results = self.executor.execute_query(team_query, tuple([coach_id] + match_ids))
        team_by_match = {r['match_id']: r['team_id'] for r in team_results}
        
        # Get lineup data
        lineup_query = f"""
            SELECT match_id, team_id, player_id, rating, age
            FROM match_lineup_players
            WHERE match_id IN ({placeholders})
        """
        lineup_results = self.executor.execute_query(lineup_query, tuple(match_ids))
        
        result = {}
        for match_id in match_ids:
            result[match_id] = {'player_ids': [], 'ratings': {}, 'ages': {}}
        
        for row in lineup_results:
            match_id = row['match_id']
            # Only include players from coach's team
            if row['team_id'] == team_by_match.get(match_id):
                result[match_id]['player_ids'].append(row['player_id'])
                if row['rating']:
                    result[match_id]['ratings'][row['player_id']] = row['rating']
                if row['age']:
                    result[match_id]['ages'][row['player_id']] = row['age']
        
        return result
    
    def get_coach_substitutions_bulk(
        self, 
        coach_id: int, 
        match_ids: List[int]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Get substitution data for coach's team across matches."""
        if not match_ids:
            return {}
        
        # Get team_id for each match
        placeholders = ','.join(['?' for _ in match_ids])
        team_query = f"""
            SELECT match_id, team_id
            FROM match_coaches
            WHERE coach_id = ? AND match_id IN ({placeholders}) AND is_coach = 1
        """
        team_results = self.executor.execute_query(team_query, tuple([coach_id] + match_ids))
        team_by_match = {r['match_id']: r['team_id'] for r in team_results}
        
        # Get substitutions
        sub_query = f"""
            SELECT match_id, player_id, team_id, substitution_time, substitution_type
            FROM match_substitutions
            WHERE match_id IN ({placeholders}) AND substitution_type = 'in'
            ORDER BY match_id, substitution_time
        """
        sub_results = self.executor.execute_query(sub_query, tuple(match_ids))
        
        result = {mid: [] for mid in match_ids}
        for row in sub_results:
            match_id = row['match_id']
            if row['team_id'] == team_by_match.get(match_id):
                result[match_id].append({
                    'player_in_id': row['player_id'],
                    'minute': row['substitution_time']
                })
        
        return result
    
    def get_goal_events_bulk(self, match_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """Get goal events with timing for multiple matches."""
        if not match_ids:
            return {}
        
        placeholders = ','.join(['?' for _ in match_ids])
        query = f"""
            SELECT match_id, time_minute, is_home_team,
                   home_score_before, away_score_before,
                   home_score_after, away_score_after, own_goal
            FROM match_events
            WHERE match_id IN ({placeholders}) AND event_type = 'Goal'
            ORDER BY match_id, time_minute
        """
        results = self.executor.execute_query(query, tuple(match_ids))
        
        result = {mid: [] for mid in match_ids}
        for row in results:
            result[row['match_id']].append({
                'minute': row['time_minute'],
                'is_home_goal': row['is_home_team'],
                'home_score_before': row['home_score_before'],
                'away_score_before': row['away_score_before'],
                'home_score_after': row['home_score_after'],
                'away_score_after': row['away_score_after'],
                'is_own_goal': row['own_goal']
            })
        
        return result
    

class TeamQueries:
    """Queries for team-related data."""
    
    def __init__(self, executor: QueryExecutor):
        self.executor = executor
    
    def get_all_unique_teams(self) -> List[Dict[str, Any]]:
        """Get all unique teams in the database."""
        query = """
            SELECT DISTINCT home_team_id as team_id, 
                   home_team_name as team_name
            FROM match_details
            WHERE home_team_id IS NOT NULL
            UNION
            SELECT DISTINCT away_team_id, away_team_name
            FROM match_details
            WHERE away_team_id IS NOT NULL
        """
        return self.executor.execute_query(query)
    
    def get_team_match_history(
        self, 
        team_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get match history for a team."""
        query = """
            SELECT match_id, match_time_utc, league_id,
                   home_team_id, away_team_id,
                   home_team_score, away_team_score,
                   CASE WHEN home_team_id = ? THEN 1 ELSE 0 END as is_home
            FROM match_details
            WHERE (home_team_id = ? OR away_team_id = ?) AND finished = 1
            ORDER BY match_time_utc DESC
        """
        if limit:
            query = query.replace("SELECT", f"SELECT TOP {limit}")
        
        return self.executor.execute_query(query, (team_id, team_id, team_id))
    
    def get_team_leagues(self, team_id: int) -> List[Dict[str, Any]]:
        """Get all leagues a team has played in."""
        query = """
            SELECT DISTINCT md.league_id, md.league_name,
                   ld.DivisionLevel, ld.CountryName
            FROM match_details md
            LEFT JOIN LeagueDivisions ld ON md.league_id = ld.LeagueID
            WHERE md.home_team_id = ? OR md.away_team_id = ?
        """
        return self.executor.execute_query(query, (team_id, team_id))
    
    def get_team_record_in_league(
        self, 
        team_id: int, 
        league_id: int
    ) -> Dict[str, int]:
        """Get team's record in a specific league."""
        query = """
            SELECT 
                SUM(CASE 
                    WHEN (home_team_id = ? AND home_team_score > away_team_score)
                      OR (away_team_id = ? AND away_team_score > home_team_score)
                    THEN 1 ELSE 0 END) as wins,
                SUM(CASE 
                    WHEN home_team_score = away_team_score 
                    THEN 1 ELSE 0 END) as draws,
                SUM(CASE 
                    WHEN (home_team_id = ? AND home_team_score < away_team_score)
                      OR (away_team_id = ? AND away_team_score < home_team_score)
                    THEN 1 ELSE 0 END) as losses
            FROM match_details
            WHERE (home_team_id = ? OR away_team_id = ?) 
                  AND league_id = ? AND finished = 1
        """
        params = (team_id, team_id, team_id, team_id, team_id, team_id, league_id)
        results = self.executor.execute_query(query, params)
        
        if results:
            return {
                'wins': results[0]['wins'] or 0,
                'draws': results[0]['draws'] or 0,
                'losses': results[0]['losses'] or 0
            }
        return {'wins': 0, 'draws': 0, 'losses': 0}

    def get_team_full_match_context(
        self, 
        team_id: int, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get complete match context for a team including:
        - Match details (scores, league)
        - Team stats (resolved to team's perspective)
        - Opponent rating
        """
        query = """
            SELECT 
                md.match_id,
                md.match_time_utc,
                md.league_id,
                CASE WHEN md.home_team_id = ? THEN 1 ELSE 0 END AS is_home_team,
                md.home_team_id,
                md.away_team_id,
                md.home_team_score,
                md.away_team_score,
                
                mts.rating AS own_rating,
                mts_opp.rating AS opponent_rating,
                
                CASE WHEN md.home_team_id = ? THEN mss.home_possession ELSE mss.away_possession END AS possession,
                CASE WHEN md.home_team_id = ? THEN mss.home_xg ELSE mss.away_xg END AS xg,
                CASE WHEN md.home_team_id = ? THEN mss.away_xg ELSE mss.home_xg END AS xg_conceded,
                CASE WHEN md.home_team_id = ? THEN mss.home_xg_set_play ELSE mss.away_xg_set_play END AS xg_set_play,
                CASE WHEN md.home_team_id = ? THEN mss.away_xg_set_play ELSE mss.home_xg_set_play END AS xg_set_play_conceded,
                CASE WHEN md.home_team_id = ? THEN mss.home_tackles ELSE mss.away_tackles END AS tackles,
                CASE WHEN md.home_team_id = ? THEN mss.home_interceptions ELSE mss.away_interceptions END AS interceptions,
                CASE WHEN md.home_team_id = ? THEN mss.home_duels_won ELSE mss.away_duels_won END AS duels_won,
                CASE WHEN md.home_team_id = ? THEN mss.home_aerial_duels_won ELSE mss.away_aerial_duels_won END AS aerial_duels_won,
                CASE WHEN md.home_team_id = ? THEN mss.home_yellow_cards ELSE mss.away_yellow_cards END AS yellow_cards,
                CASE WHEN md.home_team_id = ? THEN mss.home_red_cards ELSE mss.away_red_cards END AS red_cards,
                CASE WHEN md.home_team_id = ? THEN mss.home_fouls ELSE mss.away_fouls END AS fouls
                
            FROM match_details md
            LEFT JOIN match_team_stats mts 
                ON md.match_id = mts.match_id AND mts.team_id = ?
            LEFT JOIN match_team_stats mts_opp 
                ON md.match_id = mts_opp.match_id AND mts_opp.team_id != ?
            LEFT JOIN match_stats_summary mss ON md.match_id = mss.match_id
            WHERE (md.home_team_id = ? OR md.away_team_id = ?) AND md.finished = 1
            ORDER BY md.match_time_utc DESC
        """
        # team_id appears 17 times in the query
        params = tuple([team_id] * 17)
        
        if limit:
            query = query.replace("SELECT", f"SELECT TOP {limit}")
        
        return self.executor.execute_query(query, params)
    
    def get_team_lineups_bulk(
        self, 
        team_id: int, 
        match_ids: List[int]
    ) -> Dict[int, Dict[str, List[int]]]:
        """
        Get lineup data for team across matches.
        Returns: {match_id: {'all': [...], 'starters': [...], 'subs': [...]}}
        """
        if not match_ids:
            return {}
        
        placeholders = ','.join(['?' for _ in match_ids])
        query = f"""
            SELECT match_id, player_id, is_starter
            FROM match_lineup_players
            WHERE team_id = ? AND match_id IN ({placeholders})
        """
        results = self.executor.execute_query(query, tuple([team_id] + match_ids))
        
        result = {mid: {'all': [], 'starters': [], 'subs': []} for mid in match_ids}
        for row in results:
            match_id = row['match_id']
            player_id = row['player_id']
            result[match_id]['all'].append(player_id)
            if row['is_starter']:
                result[match_id]['starters'].append(player_id)
            else:
                result[match_id]['subs'].append(player_id)
        
        return result