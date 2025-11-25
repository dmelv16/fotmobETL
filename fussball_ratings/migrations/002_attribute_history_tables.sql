-- =============================================================================
-- ATTRIBUTE HISTORY TABLES
-- Store historical attribute values to track changes over time
-- =============================================================================

-- Player Attribute History
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'player_attribute_history')
CREATE TABLE player_attribute_history (
    id INT IDENTITY(1,1) PRIMARY KEY,
    player_id INT NOT NULL,
    attribute_code VARCHAR(100) NOT NULL,
    value DECIMAL(6,2) NOT NULL,
    confidence DECIMAL(4,3) NOT NULL DEFAULT 0.5,
    matches_used INT NOT NULL DEFAULT 0,
    data_tiers_used VARCHAR(50) NULL,
    
    -- Context for when this was calculated
    calculation_batch_id UNIQUEIDENTIFIER NULL,  -- Groups all attrs from same recalc
    trigger_match_id INT NULL,                    -- Match that triggered recalc (if any)
    calculated_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    INDEX IX_player_attr_hist_player (player_id),
    INDEX IX_player_attr_hist_code (attribute_code),
    INDEX IX_player_attr_hist_date (calculated_at DESC),
    INDEX IX_player_attr_hist_batch (calculation_batch_id)
);
GO

-- Team Attribute History
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'team_attribute_history')
CREATE TABLE team_attribute_history (
    id INT IDENTITY(1,1) PRIMARY KEY,
    team_id INT NOT NULL,
    attribute_code VARCHAR(100) NOT NULL,
    value DECIMAL(6,2) NOT NULL,
    matches_used INT NOT NULL DEFAULT 0,
    
    calculation_batch_id UNIQUEIDENTIFIER NULL,
    trigger_match_id INT NULL,
    calculated_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    INDEX IX_team_attr_hist_team (team_id),
    INDEX IX_team_attr_hist_code (attribute_code),
    INDEX IX_team_attr_hist_date (calculated_at DESC),
    INDEX IX_team_attr_hist_batch (calculation_batch_id)
);
GO

-- Coach Attribute History
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'coach_attribute_history')
CREATE TABLE coach_attribute_history (
    id INT IDENTITY(1,1) PRIMARY KEY,
    coach_id INT NOT NULL,
    attribute_code VARCHAR(100) NOT NULL,
    value DECIMAL(6,2) NOT NULL,
    matches_used INT NOT NULL DEFAULT 0,
    
    calculation_batch_id UNIQUEIDENTIFIER NULL,
    trigger_match_id INT NULL,
    calculated_at DATETIME2 NOT NULL DEFAULT GETDATE(),
    
    INDEX IX_coach_attr_hist_coach (coach_id),
    INDEX IX_coach_attr_hist_code (attribute_code),
    INDEX IX_coach_attr_hist_date (calculated_at DESC),
    INDEX IX_coach_attr_hist_batch (calculation_batch_id)
);
GO

-- =============================================================================
-- VIEWS FOR CURRENT ATTRIBUTE VALUES
-- These give you the latest value for each entity/attribute combination
-- =============================================================================

CREATE OR ALTER VIEW vw_current_player_attributes AS
WITH ranked AS (
    SELECT 
        player_id,
        attribute_code,
        value,
        confidence,
        matches_used,
        data_tiers_used,
        calculated_at,
        ROW_NUMBER() OVER (
            PARTITION BY player_id, attribute_code 
            ORDER BY calculated_at DESC
        ) as rn
    FROM player_attribute_history
)
SELECT 
    player_id,
    attribute_code,
    value,
    confidence,
    matches_used,
    data_tiers_used,
    calculated_at as last_updated
FROM ranked 
WHERE rn = 1;
GO

CREATE OR ALTER VIEW vw_current_team_attributes AS
WITH ranked AS (
    SELECT 
        team_id,
        attribute_code,
        value,
        matches_used,
        calculated_at,
        ROW_NUMBER() OVER (
            PARTITION BY team_id, attribute_code 
            ORDER BY calculated_at DESC
        ) as rn
    FROM team_attribute_history
)
SELECT 
    team_id,
    attribute_code,
    value,
    matches_used,
    calculated_at as last_updated
FROM ranked 
WHERE rn = 1;
GO

CREATE OR ALTER VIEW vw_current_coach_attributes AS
WITH ranked AS (
    SELECT 
        coach_id,
        attribute_code,
        value,
        matches_used,
        calculated_at,
        ROW_NUMBER() OVER (
            PARTITION BY coach_id, attribute_code 
            ORDER BY calculated_at DESC
        ) as rn
    FROM coach_attribute_history
)
SELECT 
    coach_id,
    attribute_code,
    value,
    matches_used,
    calculated_at as last_updated
FROM ranked 
WHERE rn = 1;
GO

-- =============================================================================
-- HELPER VIEW: Full Player Profile with Latest Attributes (Pivoted)
-- =============================================================================

CREATE OR ALTER VIEW vw_player_attribute_profile AS
SELECT 
    p.player_id,
    p.player_name,
    p.usual_position_id,
    pr.rating as current_rating,
    
    -- Technical
    MAX(CASE WHEN a.attribute_code = 'finishing' THEN a.value END) as finishing,
    MAX(CASE WHEN a.attribute_code = 'long_shots' THEN a.value END) as long_shots,
    MAX(CASE WHEN a.attribute_code = 'crossing' THEN a.value END) as crossing,
    MAX(CASE WHEN a.attribute_code = 'dribbling' THEN a.value END) as dribbling,
    MAX(CASE WHEN a.attribute_code = 'first_touch' THEN a.value END) as first_touch,
    MAX(CASE WHEN a.attribute_code = 'passing' THEN a.value END) as passing,
    MAX(CASE WHEN a.attribute_code = 'technique' THEN a.value END) as technique,
    MAX(CASE WHEN a.attribute_code = 'heading' THEN a.value END) as heading,
    MAX(CASE WHEN a.attribute_code = 'marking' THEN a.value END) as marking,
    MAX(CASE WHEN a.attribute_code = 'long_passing' THEN a.value END) as long_passing,
    
    -- Mental
    MAX(CASE WHEN a.attribute_code = 'aggression' THEN a.value END) as aggression,
    MAX(CASE WHEN a.attribute_code = 'anticipation' THEN a.value END) as anticipation,
    MAX(CASE WHEN a.attribute_code = 'bravery' THEN a.value END) as bravery,
    MAX(CASE WHEN a.attribute_code = 'composure' THEN a.value END) as composure,
    MAX(CASE WHEN a.attribute_code = 'concentration' THEN a.value END) as concentration,
    MAX(CASE WHEN a.attribute_code = 'decisions' THEN a.value END) as decisions,
    MAX(CASE WHEN a.attribute_code = 'flair' THEN a.value END) as flair,
    MAX(CASE WHEN a.attribute_code = 'leadership' THEN a.value END) as leadership,
    MAX(CASE WHEN a.attribute_code = 'off_the_ball' THEN a.value END) as off_the_ball,
    MAX(CASE WHEN a.attribute_code = 'positioning' THEN a.value END) as positioning,
    MAX(CASE WHEN a.attribute_code = 'teamwork' THEN a.value END) as teamwork,
    MAX(CASE WHEN a.attribute_code = 'vision' THEN a.value END) as vision,
    MAX(CASE WHEN a.attribute_code = 'work_rate' THEN a.value END) as work_rate,
    
    -- Physical
    MAX(CASE WHEN a.attribute_code = 'stamina' THEN a.value END) as stamina,
    MAX(CASE WHEN a.attribute_code = 'jumping' THEN a.value END) as jumping,
    MAX(CASE WHEN a.attribute_code = 'natural_fitness' THEN a.value END) as natural_fitness,
    MAX(CASE WHEN a.attribute_code = 'strength' THEN a.value END) as strength,
    
    -- Meta
    MAX(CASE WHEN a.attribute_code = 'consistency' THEN a.value END) as consistency,
    MAX(CASE WHEN a.attribute_code = 'versatility' THEN a.value END) as versatility,
    MAX(CASE WHEN a.attribute_code = 'adaptability' THEN a.value END) as adaptability,
    MAX(CASE WHEN a.attribute_code = 'big_game' THEN a.value END) as big_game,
    
    MAX(a.last_updated) as attributes_updated_at

FROM (
    SELECT player_id, MAX(player_name) as player_name, MAX(usual_position_id) as usual_position_id
    FROM match_lineup_players
    GROUP BY player_id
) p
LEFT JOIN vw_current_player_ratings pr ON p.player_id = pr.player_id
LEFT JOIN vw_current_player_attributes a ON p.player_id = a.player_id
GROUP BY p.player_id, p.player_name, p.usual_position_id, pr.rating;
GO

-- =============================================================================
-- USEFUL QUERIES FOR ATTRIBUTE HISTORY
-- =============================================================================

-- Example: Get attribute progression for a player
-- SELECT * FROM player_attribute_history 
-- WHERE player_id = 12345 AND attribute_code = 'finishing'
-- ORDER BY calculated_at;

-- Example: Compare attributes between two dates
-- WITH before AS (
--     SELECT player_id, attribute_code, value
--     FROM player_attribute_history
--     WHERE calculated_at <= '2024-01-01' 
--     AND calculated_at = (
--         SELECT MAX(calculated_at) FROM player_attribute_history h2 
--         WHERE h2.player_id = player_attribute_history.player_id 
--         AND h2.attribute_code = player_attribute_history.attribute_code
--         AND h2.calculated_at <= '2024-01-01'
--     )
-- ),
-- after AS (
--     SELECT player_id, attribute_code, value
--     FROM vw_current_player_attributes
-- )
-- SELECT 
--     a.player_id, a.attribute_code,
--     b.value as value_before, a.value as value_after,
--     a.value - b.value as change
-- FROM after a
-- JOIN before b ON a.player_id = b.player_id AND a.attribute_code = b.attribute_code
-- WHERE ABS(a.value - b.value) > 1;  -- Show significant changes only