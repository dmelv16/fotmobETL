import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from config.settings import CHAMPIONS_LEAGUE_WEIGHT, EUROPA_LEAGUE_WEIGHT, CONFERENCE_LEAGUE_WEIGHT
from config.league_mappings import EUROPEAN_COMPETITIONS
import warnings
# Suppress the specific SQLAlchemy warning from pandas
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*SQLAlchemy.*')
logger = logging.getLogger(__name__)

class EuropeanResultsAnalyzer:
    """
    Analyzes European competition results to infer domestic league strength.
    Uses head-to-head results between teams from different leagues.
    """
    
    def __init__(self, connection, data_loader):
        self.conn = connection
        self.data_loader = data_loader
        self.league_ratings = {}  # {league_id: rating}
    
    def get_competition_weight(self, competition_id: int) -> float:
        """Get weight for competition importance."""
        comp_info = EUROPEAN_COMPETITIONS.get(competition_id, {})
        return comp_info.get('weight', 0.8)
    
    def initialize_league_ratings(self, league_ids: List[int], season_year: int, base_ratings: Dict[int, float] = None):
        """
        Initialize league ratings using previous season's final values if available.
        """
        if base_ratings:
            self.league_ratings = base_ratings.copy()
            return

        # Start default at 1500
        for lid in league_ids:
            self.league_ratings[lid] = 1500.0

        # Try to load previous season's ratings from database
        prev_season = int(season_year) - 1
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                SELECT league_id, overall_strength 
                FROM [dbo].[league_strength] 
                WHERE season_year = ?
            """, prev_season)
            
            rows = cursor.fetchall()
            prev_ratings = {row[0]: row[1] for row in rows}
            
            for league_id in league_ids:
                if league_id in prev_ratings:
                    prev_strength = float(prev_ratings[league_id])
                    base_elo = 1000 + (prev_strength * 10)
                    self.league_ratings[league_id] = (base_elo * 0.7) + (1500 * 0.3)
        except Exception as e:
            logger.warning(f"Could not load previous league ratings: {e}")
    
    def get_team_domestic_league(self, team_id: int, season_year: int) -> Optional[int]:
        """
        Determine which domestic league a team plays in.
        """
        query = """
            SELECT TOP 1 md.parent_league_id
            FROM [dbo].[match_details] md
            LEFT JOIN [dbo].[LeagueDivisions] ld ON md.parent_league_id = ld.LeagueID
            WHERE (md.home_team_id = ? OR md.away_team_id = ?)
            AND YEAR(md.match_time_utc) = ?
            AND (ld.DivisionLevel != 'Cup' OR ld.DivisionLevel IS NULL)
            AND md.parent_league_id NOT IN (42, 73, 848)
            GROUP BY md.parent_league_id
            ORDER BY COUNT(*) DESC
        """
        
        try:
            # FIX: Explicit int() casting for params
            params = [int(team_id), int(team_id), int(season_year)]
            df = pd.read_sql(query, self.conn, params=params)
            
            if len(df) > 0:
                val = df['parent_league_id'].iloc[0]
                return int(val) if pd.notnull(val) else None
            return None
        except Exception as e:
            # logger.warning(f"Could not determine domestic league for team {team_id}: {e}")
            return None
    
    def expected_result(self, rating_a: float, rating_b: float, is_home: bool = False) -> float:
        rating_diff = rating_b - rating_a
        if is_home:
            rating_diff -= 100
        return 1 / (1 + 10 ** (rating_diff / 400))
    
    def update_league_ratings_from_match(self, home_league_id: int, away_league_id: int,
                                        home_score: int, away_score: int,
                                        competition_weight: float, k_factor: float = 30):
        home_rating = self.league_ratings.get(home_league_id, 1500.0)
        away_rating = self.league_ratings.get(away_league_id, 1500.0)
        
        if home_score > away_score: actual = 1.0
        elif home_score < away_score: actual = 0.0
        else: actual = 0.5
        
        expected = self.expected_result(home_rating, away_rating, is_home=True)
        weighted_k = k_factor * competition_weight
        
        home_change = weighted_k * (actual - expected)
        away_change = weighted_k * ((1 - actual) - (1 - expected))
        
        self.league_ratings[home_league_id] = home_rating + home_change
        self.league_ratings[away_league_id] = away_rating + away_change
    
    def process_season(self, season_year: int, save_to_db: bool = True) -> Dict[int, float]:
        logger.info(f"Processing European results for season {season_year}")
        matches_df = self.data_loader.get_european_matches(int(season_year))
        
        if len(matches_df) == 0:
            logger.warning(f"No European matches found for season {season_year}")
            return {}
        
        team_leagues = {}
        all_teams = pd.concat([matches_df['home_team_id'], matches_df['away_team_id']]).unique()
        
        for tid in all_teams:
            lid = self.get_team_domestic_league(int(tid), int(season_year))
            if lid: team_leagues[int(tid)] = lid
        
        unique_leagues = list(set(team_leagues.values()))
        self.initialize_league_ratings(unique_leagues, int(season_year))
        
        processed_matches = []
        
        for _, match in matches_df.iterrows():
            home_id = int(match['home_team_id'])
            away_id = int(match['away_team_id'])
            home_league = team_leagues.get(home_id)
            away_league = team_leagues.get(away_id)
            
            if not home_league or not away_league or home_league == away_league:
                continue
            
            comp_id = int(match['competition_id'])
            weight = self.get_competition_weight(comp_id)
            h_score = int(match['home_team_score'])
            a_score = int(match['away_team_score'])
            
            h_rating_before = self.league_ratings[home_league]
            a_rating_before = self.league_ratings[away_league]
            expected = self.expected_result(h_rating_before, a_rating_before, is_home=True)
            
            if h_score > a_score: actual = 1.0
            elif h_score < a_score: actual = 0.0
            else: actual = 0.5
            
            self.update_league_ratings_from_match(home_league, away_league, h_score, a_score, weight)
            
            if save_to_db:
                processed_matches.append({
                    'match_id': int(match['match_id']),
                    'season_year': int(season_year),
                    'competition_id': comp_id,
                    'competition_name': match['competition_name'],
                    'competition_weight': weight,
                    'home_team_id': home_id,
                    'home_team_name': match['home_team_name'],
                    'home_league_id': home_league,
                    'home_score': h_score,
                    'home_xg': float(match['home_xg']) if pd.notnull(match['home_xg']) else None,
                    'away_team_id': away_id,
                    'away_team_name': match['away_team_name'],
                    'away_league_id': away_league,
                    'away_score': a_score,
                    'away_xg': float(match['away_xg']) if pd.notnull(match['away_xg']) else None,
                    'match_date': match['match_time_utc'],
                    'expected_result': expected,
                    'actual_result': actual,
                    'surprise_factor': abs(actual - expected)
                })
        
        if save_to_db and processed_matches:
            self._save_matches_to_db(processed_matches)
            self._save_league_network(int(season_year))
        
        return self.league_ratings
    
    def _save_matches_to_db(self, matches: List[Dict]):
        cursor = self.conn.cursor()
        for match in matches:
            cursor.execute("SELECT id FROM [dbo].[european_competition_results] WHERE match_id = ?", match['match_id'])
            if cursor.fetchone(): continue
            
            cursor.execute("""
                INSERT INTO [dbo].[european_competition_results] (
                    match_id, season_year, competition_id, competition_name, competition_weight,
                    home_team_id, home_team_name, home_league_id, home_league_name,
                    home_score, home_xg,
                    away_team_id, away_team_name, away_league_id, away_league_name,
                    away_score, away_xg,
                    match_date, result, expected_result, actual_result, surprise_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 
                         CASE WHEN ? > ? THEN 'home' WHEN ? < ? THEN 'away' ELSE 'draw' END,
                         ?, ?, ?)
            """, match['match_id'], match['season_year'], match['competition_id'], 
                 match['competition_name'], match['competition_weight'],
                 match['home_team_id'], match['home_team_name'], match['home_league_id'],
                 match['home_score'], match['home_xg'],
                 match['away_team_id'], match['away_team_name'], match['away_league_id'],
                 match['away_score'], match['away_xg'],
                 match['match_date'],
                 match['home_score'], match['away_score'], match['home_score'], match['away_score'],
                 match['expected_result'], match['actual_result'], match['surprise_factor'])
        self.conn.commit()
    
    def _save_league_network(self, season_year: int):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT home_league_id, away_league_id, COUNT(*),
                AVG(CASE WHEN result = 'home' THEN 1.0 WHEN result = 'draw' THEN 0.5 ELSE 0.0 END)
            FROM [dbo].[european_competition_results]
            WHERE season_year = ?
            GROUP BY home_league_id, away_league_id
        """, int(season_year))
        
        pairs = cursor.fetchall()
        for pair in pairs:
            la, lb, count, rate = pair
            rate = float(rate)
            safe_rate = max(0.05, min(0.95, rate))
            odds_ratio = safe_rate / (1.0 - safe_rate)
            gap = 1.0 / odds_ratio
            conf = min(count / 10.0, 1.0)
            
            cursor.execute("""
                MERGE [dbo].[league_network_edges] AS target
                USING (SELECT ? AS la, ? AS lb, ? AS yr) AS source
                ON (target.league_a_id = source.la AND target.league_b_id = source.lb AND target.season_year = source.yr)
                WHEN MATCHED THEN
                    UPDATE SET european_matches = ?, strength_gap_estimate = ?, gap_confidence = ?, gap_method = 'european', last_updated = GETDATE()
                WHEN NOT MATCHED THEN
                    INSERT (league_a_id, league_b_id, season_year, european_matches, strength_gap_estimate, gap_confidence, gap_method, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, 'european', GETDATE());
            """, int(la), int(lb), int(season_year), int(count), float(gap), float(conf), 
                 int(la), int(lb), int(season_year), int(count), float(gap), float(conf))
        self.conn.commit()
    
    def get_league_ratings_normalized(self, scale_min: float = 0, scale_max: float = 100) -> Dict[int, float]:
        """
        Get league ratings normalized to a 0-100 scale.
        """
        if not self.league_ratings:
            return {}
        
        ratings = np.array(list(self.league_ratings.values()))
        min_rating = ratings.min()
        max_rating = ratings.max()
        
        if max_rating == min_rating:
            return {k: (scale_max + scale_min) / 2 for k in self.league_ratings}
        
        normalized = {}
        for league_id, rating in self.league_ratings.items():
            normalized_value = ((rating - min_rating) / (max_rating - min_rating)) * (scale_max - scale_min) + scale_min
            normalized[league_id] = normalized_value
        
        return normalized