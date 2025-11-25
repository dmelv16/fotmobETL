-- =============================================================================
-- RATING SYSTEM TABLES
-- =============================================================================

-- Player Ratings History
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'player_ratings')
CREATE TABLE player_ratings (
    id INT IDENTITY(1,1) PRIMARY KEY,
    player_id INT NOT NULL,
    match_id INT NOT NULL,
    rating DECIMAL(10,4) NOT NULL,
    rating_change DECIMAL(10,4) NOT NULL DEFAULT 0,
    k_factor_used DECIMAL(8,4) NULL,
    data_tier TINYINT NOT NULL DEFAULT 1,
    competition_multiplier DECIMAL(6,4) NOT NULL DEFAULT 1.0,
    created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    INDEX IX_player_ratings_player_id (player_id),
    INDEX IX_player_ratings_match_id (match_id),
    INDEX IX_player_ratings_created_at (created_at DESC)
);

-- Team Ratings History
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'team_ratings')
CREATE TABLE team_ratings (
    id INT IDENTITY(1,1) PRIMARY KEY,
    team_id INT NOT NULL,
    match_id INT NOT NULL,
    rating DECIMAL(10,4) NOT NULL,
    rating_change DECIMAL(10,4) NOT NULL DEFAULT 0,
    expected_result DECIMAL(6,4) NULL,
    actual_result DECIMAL(6,4) NULL,
    k_factor_used DECIMAL(8,4) NULL,
    data_tier TINYINT NOT NULL DEFAULT 1,
    created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    INDEX IX_team_ratings_team_id (team_id),
    INDEX IX_team_ratings_match_id (match_id),
    INDEX IX_team_ratings_created_at (created_at DESC)
);

-- Coach Ratings History
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'coach_ratings')
CREATE TABLE coach_ratings (
    id INT IDENTITY(1,1) PRIMARY KEY,
    coach_id INT NOT NULL,
    match_id INT NOT NULL,
    rating DECIMAL(10,4) NOT NULL,
    rating_change DECIMAL(10,4) NOT NULL DEFAULT 0,
    k_factor_used DECIMAL(8,4) NULL,
    data_tier TINYINT NOT NULL DEFAULT 1,
    created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    INDEX IX_coach_ratings_coach_id (coach_id),
    INDEX IX_coach_ratings_match_id (match_id),
    INDEX IX_coach_ratings_created_at (created_at DESC)
);

-- League Ratings History
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'league_ratings')
CREATE TABLE league_ratings (
    id INT IDENTITY(1,1) PRIMARY KEY,
    league_id INT NOT NULL,
    rating DECIMAL(10,4) NOT NULL,
    team_count INT NULL,
    avg_team_rating DECIMAL(10,4) NULL,
    rating_std_dev DECIMAL(10,4) NULL,
    season_year VARCHAR(20) NULL,
    created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    INDEX IX_league_ratings_league_id (league_id),
    INDEX IX_league_ratings_created_at (created_at DESC)
);
GO

-- =============================================================================
-- ATTRIBUTE TABLES
-- =============================================================================

-- Player Attributes
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'player_attributes')
CREATE TABLE player_attributes (
    id INT IDENTITY(1,1) PRIMARY KEY,
    player_id INT NOT NULL,
    attribute_code VARCHAR(100) NOT NULL,
    value DECIMAL(6,2) NOT NULL,
    confidence DECIMAL(4,3) NOT NULL DEFAULT 0.5,
    matches_used INT NOT NULL DEFAULT 0,
    data_tiers_used VARCHAR(50) NULL,
    created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    CONSTRAINT UQ_player_attributes UNIQUE (player_id, attribute_code),
    INDEX IX_player_attributes_player_id (player_id),
    INDEX IX_player_attributes_attribute_code (attribute_code)
);

-- Team Attributes
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'team_attributes')
CREATE TABLE team_attributes (
    id INT IDENTITY(1,1) PRIMARY KEY,
    team_id INT NOT NULL,
    attribute_code VARCHAR(100) NOT NULL,
    value DECIMAL(6,2) NOT NULL,
    matches_used INT NOT NULL DEFAULT 0,
    created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    CONSTRAINT UQ_team_attributes UNIQUE (team_id, attribute_code),
    INDEX IX_team_attributes_team_id (team_id)
);

-- Coach Attributes
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'coach_attributes')
CREATE TABLE coach_attributes (
    id INT IDENTITY(1,1) PRIMARY KEY,
    coach_id INT NOT NULL,
    attribute_code VARCHAR(100) NOT NULL,
    value DECIMAL(6,2) NOT NULL,
    matches_used INT NOT NULL DEFAULT 0,
    created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    CONSTRAINT UQ_coach_attributes UNIQUE (coach_id, attribute_code),
    INDEX IX_coach_attributes_coach_id (coach_id)
);
GO

-- =============================================================================
-- PROCESSING STATE TABLES
-- =============================================================================

-- Processing Log
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'processing_log')
CREATE TABLE processing_log (
    id INT IDENTITY(1,1) PRIMARY KEY,
    match_id INT NOT NULL,
    processing_type VARCHAR(200) NOT NULL,
    success BIT NOT NULL DEFAULT 0,
    error_message NVARCHAR(MAX) NULL,
    processed_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    INDEX IX_processing_log_match_id (match_id),
    INDEX IX_processing_log_type (processing_type),
    INDEX IX_processing_log_processed_at (processed_at DESC)
);
GO

-- =============================================================================
-- CURRENT STATE VIEWS (for easy querying of latest values)
-- =============================================================================

GO
CREATE OR ALTER VIEW vw_current_player_ratings AS
WITH latest AS (
    SELECT player_id, rating, rating_change, created_at,
           ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY created_at DESC) as rn
    FROM player_ratings
)
SELECT player_id, rating, rating_change, created_at as last_updated
FROM latest WHERE rn = 1;
GO

CREATE OR ALTER VIEW vw_current_team_ratings AS
WITH latest AS (
    SELECT team_id, rating, rating_change, created_at,
           ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY created_at DESC) as rn
    FROM team_ratings
)
SELECT team_id, rating, rating_change, created_at as last_updated
FROM latest WHERE rn = 1;
GO

CREATE OR ALTER VIEW vw_current_coach_ratings AS
WITH latest AS (
    SELECT coach_id, rating, rating_change, created_at,
           ROW_NUMBER() OVER (PARTITION BY coach_id ORDER BY created_at DESC) as rn
    FROM coach_ratings
)
SELECT coach_id, rating, rating_change, created_at as last_updated
FROM latest WHERE rn = 1;
GO

CREATE OR ALTER VIEW vw_player_full_profile AS
SELECT 
    mlp.player_id,
    mlp.player_name,
    mlp.usual_position_id,
    pr.rating,
    pa_fin.value as finishing,
    pa_pas.value as passing,
    pa_dri.value as dribbling,
    pa_tac.value as tackling,
    pa_sta.value as stamina,
    pa_str.value as strength
FROM (
    SELECT player_id, 
           MAX(player_name) as player_name,
           MAX(usual_position_id) as usual_position_id
    FROM match_lineup_players
    GROUP BY player_id
) mlp
LEFT JOIN vw_current_player_ratings pr ON mlp.player_id = pr.player_id
LEFT JOIN player_attributes pa_fin ON mlp.player_id = pa_fin.player_id AND pa_fin.attribute_code = 'finishing'
LEFT JOIN player_attributes pa_pas ON mlp.player_id = pa_pas.player_id AND pa_pas.attribute_code = 'passing'
LEFT JOIN player_attributes pa_dri ON mlp.player_id = pa_dri.player_id AND pa_dri.attribute_code = 'dribbling'
LEFT JOIN player_attributes pa_tac ON mlp.player_id = pa_tac.player_id AND pa_tac.attribute_code = 'tackling'
LEFT JOIN player_attributes pa_sta ON mlp.player_id = pa_sta.player_id AND pa_sta.attribute_code = 'stamina'
LEFT JOIN player_attributes pa_str ON mlp.player_id = pa_str.player_id AND pa_str.attribute_code = 'strength';
GO

-- =============================================================================
-- STAT AVAILABILITY TRACKING
-- =============================================================================

-- Track which stat_keys are available per match
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'match_stat_availability')
CREATE TABLE match_stat_availability (
    id INT IDENTITY(1,1) PRIMARY KEY,
    match_id INT NOT NULL,
    stat_key VARCHAR(100) NOT NULL,
    player_count INT NOT NULL DEFAULT 0,
    created_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    CONSTRAINT UQ_match_stat_availability UNIQUE (match_id, stat_key),
    INDEX IX_match_stat_availability_match_id (match_id),
    INDEX IX_match_stat_availability_stat_key (stat_key)
);
GO