"""
Optimized Data Loading Module v3

Key changes from v2:
- Filter attributes to only those needed for features
- Add attribute whitelist configuration
- Reduce memory usage with targeted queries
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Optional, List, Tuple, Set
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define which attributes are actually used in feature engineering
# UPDATE THESE based on what's actually predictive in your data
PLAYER_ATTRIBUTES_WHITELIST = {
    # Core performance attributes
    'overall_rating', 'potential', 'pace', 'shooting', 'passing',
    'dribbling', 'defending', 'physical', 'goalkeeping',
    # Position-specific
    'finishing', 'heading_accuracy', 'short_passing', 'volleys',
    'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
    'agility', 'reactions', 'balance', 'shot_power', 'jumping',
    'stamina', 'strength', 'long_shots', 'aggression', 'interceptions',
    'positioning', 'vision', 'penalties', 'composure',
    'marking', 'standing_tackle', 'sliding_tackle',
    # xG related if available
    'xg_per_90', 'xa_per_90', 'npxg_per_90',
}

COACH_ATTRIBUTES_WHITELIST = {
    'tactical_ability', 'experience', 'win_rate', 'avg_possession',
    'avg_goals_scored', 'avg_goals_conceded', 'formation_preference',
}

TEAM_ATTRIBUTES_WHITELIST = {
    'attack_rating', 'midfield_rating', 'defence_rating', 'overall_rating',
    'home_advantage', 'form_rating', 'squad_depth',
    'avg_possession', 'avg_shots', 'avg_shots_on_target',
    'xg_for', 'xg_against', 'ppg',  # points per game
}


class BatchDataLoader:
    """
    Optimized data loader with batch query support and attribute filtering.
    """
    
    def __init__(self, config, 
                 player_attrs: Optional[Set[str]] = None,
                 coach_attrs: Optional[Set[str]] = None,
                 team_attrs: Optional[Set[str]] = None):
        self.config = config
        
        # Use provided whitelists or defaults
        self.player_attrs_whitelist = player_attrs or PLAYER_ATTRIBUTES_WHITELIST
        self.coach_attrs_whitelist = coach_attrs or COACH_ATTRIBUTES_WHITELIST
        self.team_attrs_whitelist = team_attrs or TEAM_ATTRIBUTES_WHITELIST
        
        # Connection setup
        connection_string = config.db.connection_string
        if '?' in connection_string:
            connection_string += '&timeout=1800'
        else:
            connection_string += '?timeout=1800'
        
        self.engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        self.match_batch_size = 500
        self.entity_batch_size = 1000
    
    def _format_attr_list(self, attrs: Set[str]) -> str:
        """Format attribute set for SQL IN clause."""
        return ','.join(f"'{a}'" for a in attrs)
    
    # =========================================================================
    # MATCH QUERIES (unchanged)
    # =========================================================================
    
    def get_finished_matches(self, 
                            min_date: Optional[datetime] = None,
                            max_date: Optional[datetime] = None,
                            league_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load all finished matches with optional filters."""
        query = """
        SELECT 
            id, match_id, match_name, league_id, league_name,
            match_time_utc, home_team_id, home_team_name, home_team_score,
            away_team_id, away_team_name, away_team_score,
            coverage_level,
            number_of_home_red_cards, number_of_away_red_cards
        FROM match_details
        WHERE finished = 1 AND cancelled = 0
        """
        params = {}
        if min_date:
            query += " AND match_time_utc >= :min_date"
            params['min_date'] = min_date
        if max_date:
            query += " AND match_time_utc <= :max_date"
            params['max_date'] = max_date
        if league_ids:
            query += f" AND league_id IN ({','.join(map(str, league_ids))})"
        
        query += " ORDER BY match_time_utc"
        
        logger.info("Loading finished matches...")
        df = pd.read_sql(text(query), self.engine, params=params)
        logger.info(f"Loaded {len(df)} matches")
        return df
    
    # =========================================================================
    # BATCH LINEUP & COACH QUERIES (unchanged)
    # =========================================================================
    
    def get_lineups_batch(self, match_ids: List[int]) -> pd.DataFrame:
        """Load lineups for multiple matches at once."""
        if not match_ids:
            return pd.DataFrame()
        
        dfs = []
        total_batches = (len(match_ids) + self.match_batch_size - 1) // self.match_batch_size
        
        for batch_num, i in enumerate(range(0, len(match_ids), self.match_batch_size)):
            batch_ids = match_ids[i:i + self.match_batch_size]
            ids_str = ','.join(map(str, batch_ids))
            
            if batch_num % 10 == 0:
                logger.info(f"  Loading lineups batch {batch_num + 1}/{total_batches}...")
            
            query = f"""
            SELECT 
                match_id, player_id, team_id, is_home_team, is_starter,
                position_id, age, rating, market_value
            FROM match_lineup_players
            WHERE match_id IN ({ids_str})
            """
            dfs.append(pd.read_sql(text(query), self.engine))
        
        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        logger.info(f"Loaded {len(result)} lineup entries for {len(match_ids)} matches")
        return result
    
    def get_coaches_batch(self, match_ids: List[int]) -> pd.DataFrame:
        """Load coaches for multiple matches at once."""
        if not match_ids:
            return pd.DataFrame()
        
        dfs = []
        for i in range(0, len(match_ids), self.match_batch_size):
            batch_ids = match_ids[i:i + self.match_batch_size]
            ids_str = ','.join(map(str, batch_ids))
            
            query = f"""
            SELECT 
                match_id, coach_id, team_id, is_home_team, age
            FROM match_coaches
            WHERE match_id IN ({ids_str}) AND is_coach = 1
            """
            dfs.append(pd.read_sql(text(query), self.engine))
        
        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        logger.info(f"Loaded {len(result)} coach entries for {len(match_ids)} matches")
        return result
    
    # =========================================================================
    # PLAYER HISTORY - NOW WITH ATTRIBUTE FILTERING
    # =========================================================================
    
    def get_player_attributes_range(self,
                                    player_ids: List[int],
                                    min_date: datetime,
                                    max_date: datetime) -> pd.DataFrame:
        """Load ONLY whitelisted player attributes within a date range."""
        if not player_ids:
            return pd.DataFrame()
        
        attr_filter = self._format_attr_list(self.player_attrs_whitelist)
        
        dfs = []
        total_batches = (len(player_ids) + self.entity_batch_size - 1) // self.entity_batch_size
        
        for batch_num, i in enumerate(range(0, len(player_ids), self.entity_batch_size)):
            batch_ids = player_ids[i:i + self.entity_batch_size]
            ids_str = ','.join(map(str, batch_ids))
            
            if batch_num % 5 == 0:
                logger.info(f"  Loading player attrs batch {batch_num + 1}/{total_batches}...")
            
            # KEY CHANGE: Filter by attribute_code
            query = f"""
            SELECT 
                player_id, attribute_code, value, confidence, last_date
            FROM player_attribute_history
            WHERE player_id IN ({ids_str})
              AND attribute_code IN ({attr_filter})
              AND last_date >= :min_date 
              AND last_date <= :max_date
            """
            dfs.append(pd.read_sql(
                text(query), self.engine,
                params={'min_date': min_date, 'max_date': max_date}
            ))
        
        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        logger.info(f"Loaded {len(result)} player attribute records ({len(self.player_attrs_whitelist)} attr types)")
        return result
    
    def get_player_ratings_range(self,
                                 player_ids: List[int],
                                 min_date: datetime,
                                 max_date: datetime,
                                 sample_rate: float = 1.0) -> pd.DataFrame:
        """Load player ratings within a date range."""
        if not player_ids:
            return pd.DataFrame()
        
        dfs = []
        total_batches = (len(player_ids) + self.entity_batch_size - 1) // self.entity_batch_size
        
        for batch_num, i in enumerate(range(0, len(player_ids), self.entity_batch_size)):
            batch_ids = player_ids[i:i + self.entity_batch_size]
            ids_str = ','.join(map(str, batch_ids))
            
            if batch_num % 5 == 0:
                logger.info(f"  Loading player ratings batch {batch_num + 1}/{total_batches}...")
            
            if sample_rate < 1.0:
                query = f"""
                WITH RankedRatings AS (
                    SELECT 
                        player_id, rating, last_date,
                        ROW_NUMBER() OVER (
                            PARTITION BY player_id, 
                            DATEPART(year, last_date), 
                            DATEPART(week, last_date)
                            ORDER BY last_date DESC
                        ) as rn
                    FROM player_ratings
                    WHERE player_id IN ({ids_str})
                      AND last_date >= :min_date 
                      AND last_date <= :max_date
                )
                SELECT player_id, rating, last_date
                FROM RankedRatings WHERE rn = 1
                """
            else:
                query = f"""
                SELECT player_id, rating, last_date
                FROM player_ratings
                WHERE player_id IN ({ids_str})
                  AND last_date >= :min_date 
                  AND last_date <= :max_date
                """
            dfs.append(pd.read_sql(
                text(query), self.engine,
                params={'min_date': min_date, 'max_date': max_date}
            ))
        
        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        logger.info(f"Loaded {len(result)} player rating records")
        return result
    
    # =========================================================================
    # COACH HISTORY - NOW WITH ATTRIBUTE FILTERING
    # =========================================================================
    
    def get_coach_attributes_range(self,
                                   coach_ids: List[int],
                                   min_date: datetime,
                                   max_date: datetime) -> pd.DataFrame:
        """Load ONLY whitelisted coach attributes within a date range."""
        if not coach_ids:
            return pd.DataFrame()
        
        attr_filter = self._format_attr_list(self.coach_attrs_whitelist)
        
        dfs = []
        for i in range(0, len(coach_ids), self.entity_batch_size):
            batch_ids = coach_ids[i:i + self.entity_batch_size]
            ids_str = ','.join(map(str, batch_ids))
            
            query = f"""
            SELECT 
                coach_id, attribute_code, value, last_date
            FROM coach_attribute_history
            WHERE coach_id IN ({ids_str})
              AND attribute_code IN ({attr_filter})
              AND last_date >= :min_date 
              AND last_date <= :max_date
            """
            dfs.append(pd.read_sql(
                text(query), self.engine,
                params={'min_date': min_date, 'max_date': max_date}
            ))
        
        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return result
    
    def get_coach_ratings_range(self,
                                coach_ids: List[int],
                                min_date: datetime,
                                max_date: datetime) -> pd.DataFrame:
        """Load all coach ratings within a date range."""
        if not coach_ids:
            return pd.DataFrame()
        
        dfs = []
        for i in range(0, len(coach_ids), self.entity_batch_size):
            batch_ids = coach_ids[i:i + self.entity_batch_size]
            ids_str = ','.join(map(str, batch_ids))
            
            query = f"""
            SELECT coach_id, rating, last_date
            FROM coach_ratings
            WHERE coach_id IN ({ids_str})
              AND last_date >= :min_date 
              AND last_date <= :max_date
            """
            dfs.append(pd.read_sql(
                text(query), self.engine,
                params={'min_date': min_date, 'max_date': max_date}
            ))
        
        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return result
    
    # =========================================================================
    # TEAM HISTORY - NOW WITH ATTRIBUTE FILTERING
    # =========================================================================
    
    def get_team_attributes_range(self,
                                  team_ids: List[int],
                                  min_date: datetime,
                                  max_date: datetime) -> pd.DataFrame:
        """Load ONLY whitelisted team attributes within a date range."""
        if not team_ids:
            return pd.DataFrame()
        
        attr_filter = self._format_attr_list(self.team_attrs_whitelist)
        ids_str = ','.join(map(str, team_ids))
        
        query = f"""
        SELECT 
            team_id, attribute_code, value, last_date
        FROM team_attribute_history
        WHERE team_id IN ({ids_str})
          AND attribute_code IN ({attr_filter})
          AND last_date >= :min_date 
          AND last_date <= :max_date
        """
        result = pd.read_sql(
            text(query), self.engine,
            params={'min_date': min_date, 'max_date': max_date}
        )
        return result
    
    def get_team_ratings_range(self,
                               team_ids: List[int],
                               min_date: datetime,
                               max_date: datetime) -> pd.DataFrame:
        """Load all team ratings within a date range."""
        if not team_ids:
            return pd.DataFrame()
        
        ids_str = ','.join(map(str, team_ids))
        
        query = f"""
        SELECT team_id, rating, last_date
        FROM team_ratings
        WHERE team_id IN ({ids_str})
          AND last_date >= :min_date 
          AND last_date <= :max_date
        """
        result = pd.read_sql(
            text(query), self.engine,
            params={'min_date': min_date, 'max_date': max_date}
        )
        return result
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def get_all_history_for_matches(self,
                                    matches: pd.DataFrame,
                                    form_window_days: int = 42,
                                    memory_efficient: bool = True) -> dict:
        """Load all historical data needed for a set of matches."""
        matches = matches.copy()
        matches['match_time_utc'] = pd.to_datetime(matches['match_time_utc'])
        
        match_ids = matches['match_id'].tolist()
        min_date = matches['match_time_utc'].min() - timedelta(days=form_window_days + 7)
        max_date = matches['match_time_utc'].max()
        
        logger.info("Loading lineups and coaches...")
        lineups = self.get_lineups_batch(match_ids)
        coaches = self.get_coaches_batch(match_ids)
        
        player_ids = lineups['player_id'].dropna().astype(int).unique().tolist() if not lineups.empty else []
        coach_ids = coaches['coach_id'].dropna().astype(int).unique().tolist() if not coaches.empty else []
        team_ids = list(set(matches['home_team_id'].tolist() + matches['away_team_id'].tolist()))
        
        logger.info(f"Loading player history ({len(player_ids)} players, {len(self.player_attrs_whitelist)} attrs)...")
        player_attrs = self.get_player_attributes_range(player_ids, min_date, max_date)
        
        sample_rate = 0.1 if memory_efficient else 1.0
        player_ratings = self.get_player_ratings_range(player_ids, min_date, max_date, sample_rate=sample_rate)
        
        logger.info(f"Loading coach history ({len(coach_ids)} coaches)...")
        coach_attrs = self.get_coach_attributes_range(coach_ids, min_date, max_date)
        coach_ratings = self.get_coach_ratings_range(coach_ids, min_date, max_date)
        
        logger.info(f"Loading team history ({len(team_ids)} teams)...")
        team_attrs = self.get_team_attributes_range(team_ids, min_date, max_date)
        team_ratings = self.get_team_ratings_range(team_ids, min_date, max_date)
        
        return {
            'lineups': lineups,
            'coaches': coaches,
            'player_attrs': player_attrs,
            'player_ratings': player_ratings,
            'coach_attrs': coach_attrs,
            'coach_ratings': coach_ratings,
            'team_attrs': team_attrs,
            'team_ratings': team_ratings
        }
    
    def get_available_attribute_codes(self) -> dict:
        """Get all available attribute codes (for reference)."""
        codes = {}
        for table, entity in [
            ('player_attribute_history', 'player'),
            ('coach_attribute_history', 'coach'),
            ('team_attribute_history', 'team')
        ]:
            query = f"SELECT DISTINCT attribute_code FROM {table}"
            df = pd.read_sql(text(query), self.engine)
            codes[entity] = df['attribute_code'].tolist()
        return codes
    
    def print_attribute_stats(self):
        """Print stats about which attributes are being used vs available."""
        available = self.get_available_attribute_codes()
        
        print("\n" + "="*60)
        print("ATTRIBUTE USAGE STATS")
        print("="*60)
        
        for entity, avail_codes in available.items():
            if entity == 'player':
                whitelist = self.player_attrs_whitelist
            elif entity == 'coach':
                whitelist = self.coach_attrs_whitelist
            else:
                whitelist = self.team_attrs_whitelist
            
            used = set(avail_codes) & whitelist
            unused = set(avail_codes) - whitelist
            missing = whitelist - set(avail_codes)
            
            print(f"\n{entity.upper()} ATTRIBUTES:")
            print(f"  Available in DB: {len(avail_codes)}")
            print(f"  In whitelist: {len(whitelist)}")
            print(f"  Actually used: {len(used)}")
            print(f"  Skipped (not in whitelist): {len(unused)}")
            if missing:
                print(f"  Missing from DB: {missing}")