"""
Attribute calculators for Teams and Coaches.
Uses pre-resolved stats (no home/away prefixes - already handled by StatResolver).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import math

from config.constants import DataTier, ATTR_MIN, ATTR_MAX, MIN_MATCHES_FOR_ATTR
from config.coach_team_attributes import COACH_ATTRIBUTES, TEAM_ATTRIBUTES
from config.attribute_definitions import AttributeDefinition, CalculationMethod
from models.entities import Coach, Team, AttributeScore


@dataclass
class CoachMatchStats:
    """Stats collected for a coach from a single match."""
    coach_id: int
    match_id: int
    team_id: int
    is_home_team: bool
    data_tier: DataTier
    
    # Pre-resolved team stats (neutral keys like 'xg', 'possession')
    team_stats: Dict[str, Any] = field(default_factory=dict)
    
    formation: Optional[str] = None
    lineup_player_ids: List[int] = field(default_factory=list)
    substitutions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TeamMatchStats:
    """Stats collected for a team from a single match."""
    team_id: int
    match_id: int
    is_home_team: bool
    data_tier: DataTier
    
    # Pre-resolved stats (neutral keys)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    opponent_team_id: Optional[int] = None
    opponent_rating: Optional[float] = None


class CoachAttributeCalculator:
    """
    Calculates coach attributes from match history.
    
    Stats are pre-resolved - 'xg' is already the coach's team's xG,
    'xg_conceded' is the opponent's xG, etc.
    """
    
    def calculate_all_attributes(
        self,
        coach: Coach,
        match_history: List[CoachMatchStats]
    ) -> Dict[str, AttributeScore]:
        """Calculate all applicable attributes for a coach."""
        results = {}
        
        for attr_code, definition in COACH_ATTRIBUTES.items():
            score = self._calculate_attribute(definition, match_history)
            if score:
                results[attr_code] = score
        
        return results
    
    def _calculate_attribute(
        self,
        definition: AttributeDefinition,
        match_history: List[CoachMatchStats]
    ) -> Optional[AttributeScore]:
        """Calculate a single attribute."""
        
        # Filter matches by data tier
        valid_matches = [
            m for m in match_history
            if m.data_tier.value >= definition.min_tier.value
        ]
        
        if len(valid_matches) < definition.min_matches:
            return None
        
        # Route to appropriate calculator
        if definition.calculation_method == CalculationMethod.DIFFERENTIAL:
            return self._calc_differential(definition, valid_matches)
        elif definition.calculation_method == CalculationMethod.CONTEXTUAL:
            return self._calc_contextual(definition, valid_matches)
        elif definition.calculation_method == CalculationMethod.AGGREGATE:
            return self._calc_aggregate(definition, valid_matches)
        
        return None
    
    def _calc_differential(
        self,
        definition: AttributeDefinition,
        matches: List[CoachMatchStats]
    ) -> Optional[AttributeScore]:
        """Calculate differential-based attributes (actual vs expected)."""
        
        if definition.code == 'ATT':  # Attacking Tactics
            total_goals = 0
            total_xg = 0
            matches_with_xg = 0
            
            for m in matches:
                goals = m.team_stats.get('team_score', 0)
                xg = m.team_stats.get('xg')
                
                total_goals += goals
                if xg is not None:
                    total_xg += xg
                    matches_with_xg += 1
            
            if matches_with_xg < definition.min_matches // 2:
                # Not enough xG data, fall back to goals per game
                avg_goals = total_goals / len(matches)
                # Normalize: 1.5 goals/game = average (10), 3 = excellent (18)
                value = 10 + (avg_goals - 1.5) * 5
            else:
                # Goals vs xG differential
                diff = total_goals - total_xg
                diff_per_match = diff / matches_with_xg
                # +0.3 per match overperformance = +2 attribute
                value = 10 + diff_per_match * 6.67
            
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        elif definition.code == 'DEF':  # Defensive Tactics
            total_conceded = 0
            total_xg_against = 0
            matches_with_xg = 0
            
            for m in matches:
                conceded = m.team_stats.get('team_score_conceded', 0)
                xg_against = m.team_stats.get('xg_conceded')
                
                total_conceded += conceded
                if xg_against is not None:
                    total_xg_against += xg_against
                    matches_with_xg += 1
            
            if matches_with_xg < definition.min_matches // 2:
                avg_conceded = total_conceded / len(matches)
                # 1.0 goals conceded/game = average (10), 0.5 = excellent (15)
                value = 10 + (1.0 - avg_conceded) * 10
            else:
                diff = total_xg_against - total_conceded  # Positive = good (conceded less than expected)
                diff_per_match = diff / matches_with_xg
                value = 10 + diff_per_match * 6.67
            
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        elif definition.code == 'DIS':  # Discipline
            total_yellows = 0
            total_reds = 0
            total_fouls = 0
            
            for m in matches:
                total_yellows += m.team_stats.get('yellow_cards', 0) or 0
                total_reds += m.team_stats.get('red_cards', 0) or 0
                total_fouls += m.team_stats.get('fouls', 0) or 0
            
            # Weighted discipline score (lower = better)
            avg_discipline_cost = (total_yellows + total_reds * 3 + total_fouls * 0.1) / len(matches)
            # Average ~3 (1 yellow + ~10 fouls), range 1-6
            value = 10 + (3 - avg_discipline_cost) * 3
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        return None
    
    def _calc_contextual(
        self,
        definition: AttributeDefinition,
        matches: List[CoachMatchStats]
    ) -> Optional[AttributeScore]:
        """Calculate context-dependent attributes."""
        
        if definition.code == 'ADP':  # Adaptability (comebacks)
            comebacks = 0
            trailing_situations = 0
            
            # This would need event data - simplified version
            for m in matches:
                result = m.team_stats.get('result')
                # Crude proxy: wins from behind (would need event-level data for real impl)
                # For now, just use win rate as proxy
                if result == 'W':
                    comebacks += 1
            
            win_rate = comebacks / len(matches) if matches else 0
            value = 10 + (win_rate - 0.4) * 20  # 40% = average, 60% = 14
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=0.5,  # Lower confidence without full event data
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        elif definition.code == 'ROT':  # Squad Rotation
            unique_players_by_match = []
            all_players = set()
            
            for m in matches:
                match_players = set(m.lineup_player_ids)
                unique_players_by_match.append(match_players)
                all_players.update(match_players)
            
            if len(matches) < 2:
                return None
            
            # Calculate rotation: how many different players used / matches
            avg_unique = len(all_players) / len(matches)
            # ~13-15 unique players per match = good rotation
            value = 10 + (avg_unique - 12) * 2
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        elif definition.code == 'FOR':  # Formation Mastery
            formation_results = defaultdict(lambda: {'W': 0, 'D': 0, 'L': 0, 'total': 0})
            
            for m in matches:
                formation = m.formation
                if formation:
                    result = m.team_stats.get('result', 'D')
                    formation_results[formation][result] += 1
                    formation_results[formation]['total'] += 1
            
            if not formation_results:
                return None
            
            # Find best formation and its win rate
            best_win_rate = 0
            formations_mastered = 0
            
            for formation, results in formation_results.items():
                if results['total'] >= 5:  # Need at least 5 matches
                    win_rate = results['W'] / results['total']
                    if win_rate > 0.5:
                        formations_mastered += 1
                    best_win_rate = max(best_win_rate, win_rate)
            
            # Score based on mastery breadth and best performance
            value = 8 + formations_mastered * 2 + best_win_rate * 4
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        return None
    
    def _calc_aggregate(
        self,
        definition: AttributeDefinition,
        matches: List[CoachMatchStats]
    ) -> Optional[AttributeScore]:
        """Calculate aggregate-based attributes."""
        # Most coach attributes are differential or contextual
        return None


class TeamAttributeCalculator:
    """
    Calculates team attributes from match history.
    Stats are pre-resolved to neutral keys.
    """
    
    def calculate_all_attributes(
        self,
        team: Team,
        match_history: List[TeamMatchStats]
    ) -> Dict[str, AttributeScore]:
        """Calculate all applicable attributes for a team."""
        results = {}
        
        for attr_code, definition in TEAM_ATTRIBUTES.items():
            score = self._calculate_attribute(definition, match_history)
            if score:
                results[attr_code] = score
        
        return results
    
    def _calculate_attribute(
        self,
        definition: AttributeDefinition,
        match_history: List[TeamMatchStats]
    ) -> Optional[AttributeScore]:
        """Calculate a single attribute."""
        
        valid_matches = [
            m for m in match_history
            if m.data_tier.value >= definition.min_tier.value
        ]
        
        if len(valid_matches) < definition.min_matches:
            return None
        
        if definition.calculation_method == CalculationMethod.COMPOSITE:
            return self._calc_composite(definition, valid_matches)
        elif definition.calculation_method == CalculationMethod.AGGREGATE:
            return self._calc_aggregate(definition, valid_matches)
        elif definition.calculation_method == CalculationMethod.CONTEXTUAL:
            return self._calc_contextual(definition, valid_matches)
        
        return None
    
    def _calc_composite(
        self,
        definition: AttributeDefinition,
        matches: List[TeamMatchStats]
    ) -> Optional[AttributeScore]:
        """Calculate composite attributes."""
        
        if definition.code == 'ATK':  # Attack
            total_goals = sum(m.stats.get('team_score', 0) for m in matches)
            total_xg = sum(m.stats.get('xg', 0) or 0 for m in matches)
            total_shots = sum(m.stats.get('total_shots', 0) or 0 for m in matches)
            
            avg_goals = total_goals / len(matches)
            # 1.5 goals/game = 10, 2.5 = 15, 3+ = 18+
            value = 10 + (avg_goals - 1.5) * 5
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        elif definition.code == 'DEF':  # Defense
            total_conceded = sum(m.stats.get('team_score_conceded', 0) for m in matches)
            avg_conceded = total_conceded / len(matches)
            # 1.0 conceded/game = 10, 0.5 = 15, 1.5 = 5
            value = 10 + (1.0 - avg_conceded) * 10
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        elif definition.code == 'PRS':  # Pressing
            total_tackles = sum(m.stats.get('tackles', 0) or 0 for m in matches)
            total_interceptions = sum(m.stats.get('interceptions', 0) or 0 for m in matches)
            
            avg_defensive_actions = (total_tackles + total_interceptions) / len(matches)
            # ~25 actions/game = average (10)
            value = 10 + (avg_defensive_actions - 25) * 0.4
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        elif definition.code == 'PHY':  # Physicality
            total_duels = sum(m.stats.get('duels_won', 0) or 0 for m in matches)
            total_aerials = sum(m.stats.get('aerial_duels_won', 0) or 0 for m in matches)
            
            avg_duels = (total_duels + total_aerials) / len(matches)
            value = 10 + (avg_duels - 30) * 0.3
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        return None
    
    def _calc_aggregate(
        self,
        definition: AttributeDefinition,
        matches: List[TeamMatchStats]
    ) -> Optional[AttributeScore]:
        """Calculate aggregate attributes."""
        
        if definition.code == 'POS':  # Possession
            possession_values = [m.stats.get('possession') for m in matches if m.stats.get('possession')]
            
            if not possession_values:
                return None
            
            avg_possession = sum(possession_values) / len(possession_values)
            # 50% = 10, 60% = 14, 70% = 18
            value = 10 + (avg_possession - 50) * 0.4
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(possession_values) / (definition.min_matches * 2)),
                matches_used=len(possession_values),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        elif definition.code == 'DIS':  # Discipline
            total_yellows = sum(m.stats.get('yellow_cards', 0) or 0 for m in matches)
            total_reds = sum(m.stats.get('red_cards', 0) or 0 for m in matches)
            total_fouls = sum(m.stats.get('fouls', 0) or 0 for m in matches)
            
            avg_cost = (total_yellows + total_reds * 3 + total_fouls * 0.1) / len(matches)
            value = 10 + (3 - avg_cost) * 3
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        return None
    
    def _calc_contextual(
        self,
        definition: AttributeDefinition,
        matches: List[TeamMatchStats]
    ) -> Optional[AttributeScore]:
        """Calculate context-dependent attributes."""
        
        if definition.code == 'CON':  # Consistency
            points = [m.stats.get('points', 1) for m in matches]
            
            if len(points) < 5:
                return None
            
            avg = sum(points) / len(points)
            variance = sum((p - avg) ** 2 for p in points) / len(points)
            std_dev = math.sqrt(variance)
            
            # Lower std dev = more consistent. Range roughly 0.5-1.5
            value = 10 + (1.0 - std_dev) * 6
            value = max(ATTR_MIN, min(ATTR_MAX, value))
            
            return AttributeScore(
                value=value,
                confidence=min(1.0, len(matches) / (definition.min_matches * 2)),
                matches_used=len(matches),
                last_updated=datetime.now(),
                data_tiers_used=list(set(m.data_tier for m in matches))
            )
        
        return None