import pyodbc
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class StrengthDatabaseManager:
    """Manages database tables for strength calculations."""
    
    def __init__(self, connection):
        self.conn = connection
    
    def create_all_tables(self):
        """Create all strength-related tables."""
        self.create_league_strength_table()
        self.create_team_strength_table()
        self.create_transfer_performance_table()
        self.create_european_results_table()
        self.create_team_elo_history_table()
        self.create_league_network_table()
        self.create_style_profile_table()
        # Verify all tables were created
        logger.info("\nVerifying tables...")
        if self.verify_tables():
            logger.info("✓ All strength calculation tables created successfully")
        else:
            logger.error("✗ Some tables failed to create")
            raise Exception("Database table creation incomplete")    
        
    def create_league_strength_table(self):
        """Table for storing league strength scores over time."""
        cursor = self.conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                          WHERE object_id = OBJECT_ID(N'[dbo].[league_strength]') 
                          AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[league_strength] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [league_id] INT NOT NULL,
                    [league_name] NVARCHAR(200),
                    [season_year] INT NOT NULL,
                    [tier] FLOAT,
                    
                    -- Component scores
                    [transfer_matrix_score] FLOAT,
                    [european_results_score] FLOAT,
                    [network_inference_score] FLOAT,
                    [historical_consistency_score] FLOAT,
                    
                    -- Final composite score
                    [overall_strength] FLOAT NOT NULL,
                    
                    -- Confidence metrics
                    [calculation_confidence] FLOAT,
                    [sample_size] INT,
                    
                    -- Temporal tracking
                    [trend_1yr] FLOAT,  -- Year-over-year change
                    [trend_3yr] FLOAT,  -- 3-year trend
                    
                    -- Metadata
                    [last_updated] DATETIME DEFAULT GETDATE(),
                    [calculation_notes] NVARCHAR(MAX),
                    
                    CONSTRAINT UQ_league_strength UNIQUE(league_id, season_year),
                    INDEX IX_league_strength_season (season_year),
                    INDEX IX_league_strength_overall (overall_strength DESC)
                );
            END
        """)
        self.conn.commit()
    
    def create_team_strength_table(self):
        """Table for storing team strength scores over time."""
        cursor = self.conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                          WHERE object_id = OBJECT_ID(N'[dbo].[team_strength]') 
                          AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[team_strength] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [team_id] INT NOT NULL,
                    [team_name] NVARCHAR(200),
                    [league_id] INT NOT NULL,
                    [season_year] INT NOT NULL,
                    [as_of_date] DATE NOT NULL,
                    
                    -- Component scores
                    [elo_rating] FLOAT,
                    [xg_performance_rating] FLOAT,
                    [squad_quality_rating] FLOAT,
                    [coaching_effect] FLOAT,
                    
                    -- Final composite score
                    [overall_strength] FLOAT NOT NULL,
                    
                    -- Context metrics
                    [matches_played] INT,
                    [league_position] INT,
                    [points] INT,
                    
                    -- Performance metrics
                    [goals_for] INT,
                    [goals_against] INT,
                    [xg_for] FLOAT,
                    [xg_against] FLOAT,
                    
                    -- Metadata
                    [last_updated] DATETIME DEFAULT GETDATE(),
                    
                    INDEX IX_team_strength_team_season (team_id, season_year),
                    INDEX IX_team_strength_league (league_id, season_year),
                    INDEX IX_team_strength_date (as_of_date DESC)
                );
            END
        """)
        self.conn.commit()
    
    def create_transfer_performance_table(self):
        """Track player performance changes across transfers for league comparison."""
        cursor = self.conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                          WHERE object_id = OBJECT_ID(N'[dbo].[transfer_performance_analysis]') 
                          AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[transfer_performance_analysis] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [player_id] INT NOT NULL,
                    [player_name] NVARCHAR(200),
                    [position_id] INT,
                    
                    -- Source context
                    [source_league_id] INT NOT NULL,
                    [source_team_id] INT,
                    [source_season] INT NOT NULL,
                    [source_matches] INT,
                    [source_minutes] INT,
                    [source_goals_per90] FLOAT,
                    [source_assists_per90] FLOAT,
                    [source_xg_per90] FLOAT,
                    [source_team_strength] FLOAT,
                    
                    -- Destination context
                    [dest_league_id] INT NOT NULL,
                    [dest_team_id] INT,
                    [dest_season] INT NOT NULL,
                    [dest_matches] INT,
                    [dest_minutes] INT,
                    [dest_goals_per90] FLOAT,
                    [dest_assists_per90] FLOAT,
                    [dest_xg_per90] FLOAT,
                    [dest_team_strength] FLOAT,
                    
                    -- Transfer details
                    [transfer_date] DATE,
                    [age_at_transfer] INT,
                    
                    -- Performance change metrics
                    [goals_change_pct] FLOAT,
                    [assists_change_pct] FLOAT,
                    [xg_change_pct] FLOAT,
                    [overall_performance_change] FLOAT,
                    
                    -- Quality flags
                    [sufficient_sample_size] BIT,
                    [age_adjusted] BIT,
                    [team_quality_adjusted] BIT,
                    
                    [created_at] DATETIME DEFAULT GETDATE(),
                    
                    INDEX IX_transfer_perf_leagues (source_league_id, dest_league_id),
                    INDEX IX_transfer_perf_season (source_season, dest_season)
                );
            END
        """)
        self.conn.commit()
    
    def create_european_results_table(self):
        """Track European competition results for league strength calculation."""
        cursor = self.conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                          WHERE object_id = OBJECT_ID(N'[dbo].[european_competition_results]') 
                          AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[european_competition_results] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [match_id] INT NOT NULL UNIQUE,
                    [season_year] INT NOT NULL,
                    [competition_id] INT NOT NULL,
                    [competition_name] NVARCHAR(100),
                    [competition_weight] FLOAT,
                    
                    -- Home team
                    [home_team_id] INT NOT NULL,
                    [home_team_name] NVARCHAR(200),
                    [home_league_id] INT NOT NULL,
                    [home_league_name] NVARCHAR(200),
                    [home_score] INT,
                    [home_xg] FLOAT,
                    
                    -- Away team
                    [away_team_id] INT NOT NULL,
                    [away_team_name] NVARCHAR(200),
                    [away_league_id] INT NOT NULL,
                    [away_league_name] NVARCHAR(200),
                    [away_score] INT,
                    [away_xg] FLOAT,
                    
                    -- Match details
                    [match_date] DATE,
                    [stage] NVARCHAR(100),
                    [result] NVARCHAR(10),  -- 'home', 'away', 'draw'
                    
                    -- Strength inference
                    [expected_result] FLOAT,  -- Based on league strengths
                    [actual_result] FLOAT,    -- 1, 0.5, or 0
                    [surprise_factor] FLOAT,  -- How unexpected was result
                    
                    [created_at] DATETIME DEFAULT GETDATE(),
                    
                    INDEX IX_european_results_season (season_year),
                    INDEX IX_european_results_leagues (home_league_id, away_league_id)
                );
            END
        """)
        self.conn.commit()
    
    def create_team_elo_history_table(self):
        """Track Elo ratings match-by-match."""
        cursor = self.conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                          WHERE object_id = OBJECT_ID(N'[dbo].[team_elo_history]') 
                          AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[team_elo_history] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [team_id] INT NOT NULL,
                    [match_id] INT NOT NULL,
                    [season_year] INT NOT NULL,
                    [match_date] DATE NOT NULL,
                    
                    -- Elo before this match
                    [elo_before] FLOAT NOT NULL,
                    
                    -- Match details
                    [opponent_id] INT NOT NULL,
                    [opponent_elo_before] FLOAT NOT NULL,
                    [is_home] BIT NOT NULL,
                    [competition_id] INT,
                    [k_factor] FLOAT,
                    
                    -- Result
                    [actual_result] FLOAT NOT NULL,  -- 1, 0.5, or 0
                    [expected_result] FLOAT NOT NULL,
                    
                    -- Elo after this match
                    [elo_after] FLOAT NOT NULL,
                    [elo_change] FLOAT NOT NULL,
                    
                    [created_at] DATETIME DEFAULT GETDATE(),
                    
                    INDEX IX_elo_history_team (team_id, match_date),
                    INDEX IX_elo_history_match (match_id)
                );
            END
        """)
        self.conn.commit()
    
    def create_league_network_table(self):
        """Store league connectivity for network-based inference."""
        cursor = self.conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                          WHERE object_id = OBJECT_ID(N'[dbo].[league_network_edges]') 
                          AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[league_network_edges] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [league_a_id] INT NOT NULL,
                    [league_b_id] INT NOT NULL,
                    [season_year] INT NOT NULL,
                    
                    -- Connection strength
                    [transfer_count_a_to_b] INT DEFAULT 0,
                    [transfer_count_b_to_a] INT DEFAULT 0,
                    [total_transfers] INT DEFAULT 0,
                    
                    [european_matches] INT DEFAULT 0,
                    
                    -- Calculated gap (league_a relative to league_b)
                    [strength_gap_estimate] FLOAT,  -- Multiplier (1.15 = A is 15% harder)
                    [gap_confidence] FLOAT,         -- 0-1 confidence score
                    [gap_method] NVARCHAR(50),      -- 'transfer', 'european', 'inferred'
                    
                    [last_updated] DATETIME DEFAULT GETDATE(),
                    
                    CONSTRAINT UQ_league_network UNIQUE(league_a_id, league_b_id, season_year),
                    INDEX IX_network_edges_leagues (league_a_id, league_b_id)
                );
            END
        """)
        self.conn.commit()
    
    def create_style_profile_table(self):
        """Store league and team tactical style profiles."""
        cursor = self.conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects 
                          WHERE object_id = OBJECT_ID(N'[dbo].[style_profiles]') 
                          AND type in (N'U'))
            BEGIN
                CREATE TABLE [dbo].[style_profiles] (
                    [id] INT IDENTITY(1,1) PRIMARY KEY,
                    [entity_type] NVARCHAR(20) NOT NULL,  -- 'league' or 'team'
                    [entity_id] INT NOT NULL,
                    [entity_name] NVARCHAR(200),
                    [season_year] INT NOT NULL,
                    
                    -- Style metrics (0-100 scales)
                    [pace_of_play] FLOAT,
                    [possession_focus] FLOAT,
                    [defensive_intensity] FLOAT,
                    [pressing_intensity] FLOAT,
                    [direct_play_index] FLOAT,
                    [crossing_frequency] FLOAT,
                    [physical_intensity] FLOAT,
                    [technical_quality] FLOAT,
                    
                    -- Aggregated from matches
                    [avg_possession_pct] FLOAT,
                    [avg_passes_per_game] FLOAT,
                    [avg_long_balls_per_game] FLOAT,
                    [avg_crosses_per_game] FLOAT,
                    [avg_tackles_per_game] FLOAT,
                    [avg_fouls_per_game] FLOAT,
                    
                    -- Style cluster assignment (from ML clustering)
                    [style_cluster_id] INT,
                    [style_cluster_name] NVARCHAR(100),
                    
                    [matches_analyzed] INT,
                    [last_updated] DATETIME DEFAULT GETDATE(),
                    
                    INDEX IX_style_profiles_entity (entity_type, entity_id, season_year)
                );
            END
        """)
        self.conn.commit()

    def verify_tables(self) -> bool:
        """
        Verify all strength tables exist.
        Returns True if all tables exist, False otherwise.
        """
        cursor = self.conn.cursor()
        
        required_tables = [
            'league_strength',
            'team_strength',
            'transfer_performance_analysis',
            'european_competition_results',
            'team_elo_history',
            'league_network_edges',
            'style_profiles'
        ]
        
        existing_tables = []
        missing_tables = []
        
        for table in required_tables:
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = '{table}'
            """)
            
            exists = cursor.fetchone()[0]
            
            if exists:
                existing_tables.append(table)
                logger.info(f"  ✓ {table}")
            else:
                missing_tables.append(table)
                logger.error(f"  ✗ {table} - NOT FOUND")
        
        if missing_tables:
            logger.error(f"Missing {len(missing_tables)} tables: {missing_tables}")
            return False
        
        logger.info(f"All {len(existing_tables)} strength calculation tables verified")
        return True