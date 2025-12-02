"""
Configuration settings for the Soccer Match Prediction System
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DatabaseConfig:
    server: str = "DESKTOP-J9IV3OH"
    database: str = "fussballDB"
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = True
    
    @property
    def connection_string(self) -> str:
        if self.trusted_connection:
            return f"mssql+pyodbc://{self.server}/{self.database}?driver={self.driver}&trusted_connection=yes"
        return f"mssql+pyodbc://{self.server}/{self.database}?driver={self.driver}"

@dataclass
class FeatureConfig:
    # Player aggregation settings
    aggregate_player_stats: bool = True
    include_bench_players: bool = True
    bench_weight: float = 0.3  # Weight for bench player contributions
    starter_weight: float = 1.0
    
    # Which attribute codes to use (None = use all available)
    player_attribute_codes: Optional[List[str]] = None
    coach_attribute_codes: Optional[List[str]] = None
    team_attribute_codes: Optional[List[str]] = None
    
    # Missing data handling
    fill_missing_with: str = "median"  # "median", "mean", "zero"
    min_player_coverage: float = 0.0  # Min % of players needed (0 = accept any)
    
@dataclass
class ModelConfig:
    # Model types to train
    train_score_model: bool = True
    train_total_goals_model: bool = True
    train_result_model: bool = True
    
    # Train/test split
    test_size: float = 0.2
    random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    
    # Model hyperparameters
    xgb_params: dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    })

@dataclass 
class PipelineConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Output paths
    model_output_dir: str = "./models"
    feature_cache_dir: str = "./cache"
    use_cache: bool = True