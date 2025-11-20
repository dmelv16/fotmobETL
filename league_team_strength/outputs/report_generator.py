"""
Generate human-readable reports from strength calculations.
"""

import pandas as pd
from typing import Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generate reports and summaries from strength data.
    """
    
    def __init__(self, connection, temporal_tracker):
        self.conn = connection
        self.temporal_tracker = temporal_tracker
    
    def generate_league_strength_report(self, season_year: int) -> str:
        """
        Generate a comprehensive league strength report.
        
        Returns: Formatted text report
        """
        query = """
            SELECT 
                league_id,
                league_name,
                tier,
                overall_strength,
                transfer_matrix_score,
                european_results_score,
                calculation_confidence,
                sample_size,
                trend_1yr
            FROM [dbo].[league_strength]
            WHERE season_year = ?
            ORDER BY overall_strength DESC
        """
        
        df = pd.read_sql(query, self.conn, params=[season_year])
        
        if len(df) == 0:
            return f"No league strength data available for season {season_year}"
        
        report = []
        report.append("=" * 100)
        report.append(f"LEAGUE STRENGTH REPORT - Season {season_year}/{season_year+1}")
        report.append("=" * 100)
        report.append("")
        
        # Top 20 leagues
        report.append("TOP 20 LEAGUES BY STRENGTH:")
        report.append("-" * 100)
        report.append(f"{'Rank':<6} {'League':<30} {'Strength':<12} {'Trend':<10} {'Confidence':<12} {'Sample':<8}")
        report.append("-" * 100)
        
        for i, row in df.head(20).iterrows():
            rank = i + 1
            league = row['league_name'][:28]
            strength = f"{row['overall_strength']:.1f}"
            
            trend = row['trend_1yr']
            if trend is not None:
                if trend > 0:
                    trend_str = f"↑ {trend:+.1f}"
                elif trend < 0:
                    trend_str = f"↓ {trend:+.1f}"
                else:
                    trend_str = "→ 0.0"
            else:
                trend_str = "N/A"
            
            confidence = f"{row['calculation_confidence']:.2f}"
            sample = str(row['sample_size'])
            
            report.append(f"{rank:<6} {league:<30} {strength:<12} {trend_str:<10} {confidence:<12} {sample:<8}")
        
        report.append("")
        
        # Tier breakdown
        report.append("BREAKDOWN BY TIER:")
        report.append("-" * 60)
        
        for tier in sorted(df['tier'].unique()):
            tier_df = df[df['tier'] == tier]
            avg_strength = tier_df['overall_strength'].mean()
            count = len(tier_df)
            
            report.append(f"Tier {tier}: {count} leagues, Average Strength: {avg_strength:.1f}")
        
        report.append("")
        
        # Biggest movers (if trend data available)
        if df['trend_1yr'].notna().sum() > 0:
            report.append("BIGGEST MOVERS (Year-over-Year):")
            report.append("-" * 80)
            
            # Biggest gainers
            gainers = df[df['trend_1yr'].notna()].nlargest(5, 'trend_1yr')
            report.append("\nTop Gainers:")
            for _, row in gainers.iterrows():
                report.append(f"  {row['league_name']}: +{row['trend_1yr']:.1f} points")
            
            # Biggest decliners
            decliners = df[df['trend_1yr'].notna()].nsmallest(5, 'trend_1yr')
            report.append("\nTop Decliners:")
            for _, row in decliners.iterrows():
                report.append(f"  {row['league_name']}: {row['trend_1yr']:.1f} points")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def generate_top_teams_report(self, season_year: int, top_n: int = 50) -> str:
        """
        Generate a report of top teams across all leagues.
        
        Returns: Formatted text report
        """
        # Get latest team strengths for the season
        query = """
            WITH RankedTeams AS (
                SELECT 
                    ts.team_id,
                    ts.team_name,
                    ls.league_name,
                    ts.overall_strength,
                    ts.elo_rating,
                    ts.xg_performance_rating,
                    ts.squad_quality_rating,
                    ts.coaching_effect,
                    ts.league_position,
                    ts.points,
                    ROW_NUMBER() OVER (PARTITION BY ts.team_id ORDER BY ts.as_of_date DESC) as rn
                FROM [dbo].[team_strength] ts
                INNER JOIN [dbo].[league_strength] ls 
                    ON ts.league_id = ls.league_id 
                    AND ts.season_year = ls.season_year
                WHERE ts.season_year = ?
            )
            SELECT TOP (?)
                team_name,
                league_name,
                overall_strength,
                elo_rating,
                xg_performance_rating,
                squad_quality_rating,
                coaching_effect,
                league_position,
                points
            FROM RankedTeams
            WHERE rn = 1
            ORDER BY overall_strength DESC
        """
        
        df = pd.read_sql(query, self.conn, params=[season_year, top_n])
        
        if len(df) == 0:
            return f"No team strength data available for season {season_year}"
        
        report = []
        report.append("=" * 120)
        report.append(f"TOP {top_n} TEAMS - Season {season_year}/{season_year+1}")
        report.append("=" * 120)
        report.append("")
        
        report.append(f"{'Rank':<6} {'Team':<25} {'League':<20} {'Overall':<10} {'Elo':<8} {'xG':<8} {'Squad':<8} {'Coach':<8}")
        report.append("-" * 120)
        
        for i, row in df.iterrows():
            rank = i + 1
            team = row['team_name'][:23]
            league = row['league_name'][:18]
            overall = f"{row['overall_strength']:.1f}"
            elo = f"{row['elo_rating']:.1f}" if pd.notna(row['elo_rating']) else "N/A"
            xg = f"{row['xg_performance_rating']:.1f}" if pd.notna(row['xg_performance_rating']) else "N/A"
            squad = f"{row['squad_quality_rating']:.1f}" if pd.notna(row['squad_quality_rating']) else "N/A"
            coach = f"{row['coaching_effect']:+.1f}" if pd.notna(row['coaching_effect']) else "N/A"
            
            report.append(f"{rank:<6} {team:<25} {league:<20} {overall:<10} {elo:<8} {xg:<8} {squad:<8} {coach:<8}")
        
        report.append("")
        report.append("=" * 120)
        
        return "\n".join(report)
    
    def generate_team_detailed_report(self, team_id: int, season_year: int) -> str:
        """
        Generate detailed report for a specific team.
        """
        # Get team data
        query = """
            SELECT TOP 1
                ts.*,
                ls.league_name,
                ls.overall_strength as league_strength
            FROM [dbo].[team_strength] ts
            INNER JOIN [dbo].[league_strength] ls 
                ON ts.league_id = ls.league_id 
                AND ts.season_year = ls.season_year
            WHERE ts.team_id = ?
              AND ts.season_year = ?
            ORDER BY ts.as_of_date DESC
        """
        
        df = pd.read_sql(query, self.conn, params=[team_id, season_year])
        
        if len(df) == 0:
            return f"No data found for team {team_id} in season {season_year}"
        
        row = df.iloc[0]
        
        report = []
        report.append("=" * 80)
        report.append(f"TEAM ANALYSIS: {row['team_name']}")
        report.append(f"Season {season_year}/{season_year+1} - {row['league_name']}")
        report.append("=" * 80)
        report.append("")
        
        report.append("STRENGTH BREAKDOWN:")
        report.append(f"  Overall Strength:      {row['overall_strength']:.1f}/100")
        report.append(f"  League Strength:       {row['league_strength']:.1f}/100")
        report.append("")
        
        report.append("COMPONENT SCORES:")
        report.append(f"  Elo Rating:            {row['elo_rating']:.1f}/100" if pd.notna(row['elo_rating']) else "  Elo Rating:            N/A")
        report.append(f"  xG Performance:        {row['xg_performance_rating']:.1f}/100" if pd.notna(row['xg_performance_rating']) else "  xG Performance:        N/A")
        report.append(f"  Squad Quality:         {row['squad_quality_rating']:.1f}/100" if pd.notna(row['squad_quality_rating']) else "  Squad Quality:         N/A")
        report.append(f"  Coaching Effect:       {row['coaching_effect']:+.1f}" if pd.notna(row['coaching_effect']) else "  Coaching Effect:       N/A")
        report.append("")
        
        report.append("SEASON PERFORMANCE:")
        report.append(f"  League Position:       {row['league_position']}" if pd.notna(row['league_position']) else "  League Position:       N/A")
        report.append(f"  Points:                {row['points']}" if pd.notna(row['points']) else "  Points:                N/A")
        report.append(f"  Matches Played:        {row['matches_played']}")
        report.append(f"  Goals For:             {row['goals_for']}" if pd.notna(row['goals_for']) else "  Goals For:             N/A")
        report.append(f"  Goals Against:         {row['goals_against']}" if pd.notna(row['goals_against']) else "  Goals Against:         N/A")
        report.append(f"  xG For:                {row['xg_for']:.2f}" if pd.notna(row['xg_for']) else "  xG For:                N/A")
        report.append(f"  xG Against:            {row['xg_against']:.2f}" if pd.notna(row['xg_against']) else "  xG Against:            N/A")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
