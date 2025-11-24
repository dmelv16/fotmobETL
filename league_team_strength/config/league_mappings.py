"""
League classifications and hierarchies - now loaded dynamically from database.
"""

import logging
from typing import Dict, Optional, List
import pandas as pd
import warnings
# Suppress the specific SQLAlchemy warning from pandas
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*SQLAlchemy.*')
logger = logging.getLogger(__name__)

# European Competitions (still hardcoded as these are special)
EUROPEAN_COMPETITIONS = {
    42: {'name': 'Champions League', 'type': 'european', 'weight': 1.0},
    73: {'name': 'Europa League', 'type': 'european', 'weight': 0.8},
    848: {'name': 'Conference League', 'type': 'european', 'weight': 0.6}
}

class LeagueRegistry:
    """
    Dynamic league registry that loads from database.
    Replaces hardcoded tier mappings.
    """
    
    def __init__(self, connection):
        self.conn = connection
        self.leagues = {}
        self.countries = {}
        self._load_from_database()
    
    def _load_from_database(self):
        """Load league information from LeagueDivisions table."""
        query = """
            SELECT 
                LeagueID,
                LeagueName,
                CountryID,
                CountryName,
                DivisionLevel
            FROM [dbo].[LeagueDivisions]
            WHERE DivisionLevel IS NOT NULL
            ORDER BY CountryName, DivisionLevel
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            
            if len(df) == 0:
                logger.warning("No leagues found in LeagueDivisions table")
                return
            
            for _, row in df.iterrows():
                league_id = row['LeagueID']
                division_level = row['DivisionLevel']
                
                # Parse division level - can be numeric, 'Cup', or 'Continental'
                if division_level in ('Cup', 'Continental'):
                    # Keep as string for special competitions
                    parsed_division = division_level
                else:
                    # Try to parse as numeric
                    try:
                        parsed_division = int(division_level)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse division level for {row['LeagueName']}: {division_level}")
                        continue
                
                self.leagues[league_id] = {
                    'league_id': league_id,
                    'name': row['LeagueName'],
                    'country_id': row['CountryID'],
                    'country': row['CountryName'],
                    'division_level': parsed_division,  # Can be int, 'Cup', or 'Continental'
                    'competition_type': self._get_competition_type(parsed_division),
                    'tier': None,  # Will be calculated dynamically
                    'base_strength': None  # Will be calculated
                }
                
                # Group by country
                country = row['CountryName']
                if country not in self.countries:
                    self.countries[country] = []
                self.countries[country].append(league_id)
            
            logger.info(f"Loaded {len(self.leagues)} competitions from {len(self.countries)} countries")
            
        except Exception as e:
            logger.error(f"Error loading league divisions: {e}")
            raise

    def _get_competition_type(self, division_level) -> str:
        """
        Determine competition type from division level.
        Returns: 'league', 'cup', or 'continental'
        """
        if division_level == 'Cup':
            return 'cup'
        elif division_level == 'Continental':
            return 'continental'
        elif isinstance(division_level, int):
            return 'league'
        else:
            return 'unknown'
    
    def get_league_info(self, league_id: int) -> Dict:
        """Get league information."""
        if league_id in EUROPEAN_COMPETITIONS:
            return {
                **EUROPEAN_COMPETITIONS[league_id],
                'league_id': league_id,
                'tier': 0,
                'division_level': 0
            }
        
        return self.leagues.get(league_id, {
            'league_id': league_id,
            'name': 'Unknown',
            'country': 'Unknown',
            'division_level': 99,
            'tier': 99,
            'base_strength': 30
        })
    
    def get_league_tier(self, league_id: int) -> int:
        """
        Get tier classification for a league.
        Special handling for cups and continental competitions.
        """
        league_info = self.get_league_info(league_id)
        
        # Special handling for competition types
        competition_type = league_info.get('competition_type', 'league')
        
        if competition_type == 'continental':
            return 0  # Continental competitions are "above" national leagues
        
        if competition_type == 'cup':
            # Cups inherit tier from their country's top division
            country = league_info.get('country')
            if country:
                # Find the top division league in this country
                country_leagues = self.get_leagues_by_country(country)
                for lid in country_leagues:
                    other_info = self.leagues.get(lid, {})
                    if other_info.get('division_level') == 1:
                        # Found top division - use its tier
                        top_div_tier = other_info.get('tier')
                        if top_div_tier is not None:
                            return top_div_tier
            
            # Fallback: estimate from country's general level
            return 2  # Default for cups
        
        # Regular league - if tier hasn't been calculated yet, estimate from division level
        if league_info.get('tier') is None:
            division = league_info.get('division_level', 99)
            
            if not isinstance(division, int):
                return 99
            
            # Rough initial estimate based on division
            if division == 1:
                return 2  # Could be tier 1 or 2, will be refined
            elif division == 2:
                return 2.5
            elif division == 3:
                return 3
            else:
                return 4
        
        return league_info.get('tier', 99)
    
    def update_league_tier(self, league_id: int, new_tier: float):
        """Update a league's tier based on calculated strength."""
        if league_id in self.leagues:
            self.leagues[league_id]['tier'] = new_tier
    
    def update_league_base_strength(self, league_id: int, strength: float):
        """Update a league's base strength estimate."""
        if league_id in self.leagues:
            self.leagues[league_id]['base_strength'] = strength
    
    def get_all_leagues(self) -> List[int]:
        """Get all league IDs."""
        return list(self.leagues.keys())
    
    def get_leagues_by_country(self, country: str) -> List[int]:
        """Get all leagues for a specific country."""
        return self.countries.get(country, [])
    
    def get_top_division_leagues(self) -> List[int]:
        """Get all top division (division_level=1) leagues."""
        return [
            league_id for league_id, info in self.leagues.items()
            if info['division_level'] == 1
        ]
    
    def get_leagues_by_division(self, division_level: int) -> List[int]:
        """Get all leagues at a specific division level."""
        return [
            league_id for league_id, info in self.leagues.items()
            if info['division_level'] == division_level
        ]
    
    def is_european_competition(self, league_id: int) -> bool:
        """Check if league is a European competition."""
        return league_id in EUROPEAN_COMPETITIONS
    
    def assign_tiers_from_strength(self, strength_data: pd.DataFrame):
        """
        Assign tiers to leagues based on calculated strength.
        Called after league strengths are calculated.
        
        Args:
            strength_data: DataFrame with columns [league_id, overall_strength]
        """
        if len(strength_data) == 0:
            return
        
        # Sort by strength
        sorted_leagues = strength_data.sort_values('overall_strength', ascending=False)
        
        # Assign tiers based on strength percentiles
        total_leagues = len(sorted_leagues)
        
        for i, row in sorted_leagues.iterrows():
            league_id = row['league_id']
            strength = row['overall_strength']
            
            # Tier assignment based on strength
            if strength >= 85:
                tier = 1
            elif strength >= 70:
                tier = 2
            elif strength >= 55:
                tier = 2.5
            elif strength >= 40:
                tier = 3
            else:
                tier = 4
            
            self.update_league_tier(league_id, tier)
            self.update_league_base_strength(league_id, strength)
        
        logger.info("Updated league tiers based on calculated strengths")

    def get_cups(self) -> List[int]:
        """Get all cup competitions."""
        return [
            league_id for league_id, info in self.leagues.items()
            if info.get('competition_type') == 'cup'
        ]

    def get_continental_competitions(self) -> List[int]:
        """Get all continental competitions."""
        return [
            league_id for league_id, info in self.leagues.items()
            if info.get('competition_type') == 'continental'
        ]

    def get_domestic_leagues(self) -> List[int]:
        """Get all domestic league competitions (excludes cups and continental)."""
        return [
            league_id for league_id, info in self.leagues.items()
            if info.get('competition_type') == 'league'
        ]

    def get_competitions_by_type(self, comp_type: str) -> List[int]:
        """
        Get competitions by type.
        
        Args:
            comp_type: 'league', 'cup', 'continental', or 'all'
        """
        if comp_type == 'all':
            return list(self.leagues.keys())
        
        return [
            league_id for league_id, info in self.leagues.items()
            if info.get('competition_type') == comp_type
        ]

    def validate_registry(self) -> bool:
        """
        Validate that registry loaded correctly.
        Returns True if valid, False otherwise.
        """
        if not self.leagues:
            logger.error("No leagues loaded in registry!")
            return False
        
        if not self.countries:
            logger.error("No countries loaded in registry!")
            return False
        
        logger.info(f"Registry validation: {len(self.leagues)} leagues, {len(self.countries)} countries")
        
        # Check for leagues with missing country info
        invalid_leagues = [
            (lid, info) for lid, info in self.leagues.items()
            if not info.get('country') or not info.get('country_id')
        ]
        
        if invalid_leagues:
            logger.warning(f"Found {len(invalid_leagues)} leagues with incomplete country info")
            for league_id, info in invalid_leagues[:5]:  # Show first 5
                logger.warning(f"  League {league_id}: {info.get('name')} - missing country data")
        
        # Check competition type distribution
        comp_types = {}
        for info in self.leagues.values():
            comp_type = info.get('competition_type', 'unknown')
            comp_types[comp_type] = comp_types.get(comp_type, 0) + 1
        
        logger.info("Competition types:")
        for comp_type, count in comp_types.items():
            logger.info(f"  - {comp_type}: {count}")
        
        return True

# Helper functions for backward compatibility
def get_league_info(league_id: int, league_registry: Optional[LeagueRegistry] = None) -> Dict:
    """
    Get league information.
    Now requires league_registry instance.
    """
    if league_registry:
        return league_registry.get_league_info(league_id)
    
    # Fallback for European competitions
    if league_id in EUROPEAN_COMPETITIONS:
        return {
            **EUROPEAN_COMPETITIONS[league_id],
            'league_id': league_id,
            'tier': 0
        }
    
    logger.warning(f"get_league_info called without league_registry for league {league_id}")
    return {
        'league_id': league_id,
        'name': 'Unknown',
        'country': 'Unknown',
        'tier': 99,
        'base_strength': 50
    }


def is_european_competition(league_id: int) -> bool:
    """Check if league is a European competition."""
    return league_id in EUROPEAN_COMPETITIONS