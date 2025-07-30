"""
Main settings configuration for Project Hyperion
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Settings:
    """Main application settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    CHECKPOINTS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "checkpoints")
    
    # API Settings
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET_KEY: Optional[str] = None
    BINANCE_TESTNET: bool = True
    
    # Rate Limiting
    RATE_LIMIT_WEIGHT_PER_MINUTE: int = 1200
    RATE_LIMIT_SAFETY_MARGIN: float = 1.0  # 100% of limit
    RATE_LIMIT_DELAY_BETWEEN_CALLS: float = 0.1
    RATE_LIMIT_DELAY_BETWEEN_SYMBOLS: float = 1.0
    
    # Training Settings
    DEFAULT_SYMBOL: str = "ETHFDUSD"
    DEFAULT_INTERVAL: str = "1m"
    MAX_KLINES_PER_CALL: int = 1000
    DEFAULT_TRADING_PAIRS: list = field(default_factory=lambda: ["ETHFDUSD", "BTCFDUSD", "ADAUSDT", "DOTUSDT"])
    
    # Model Settings
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.1
    
    # Performance Settings
    OPTIMAL_CORES: int = 4
    BATCH_SIZE: int = 32
    MAX_MEMORY_USAGE: float = 0.8
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    def __post_init__(self):
        """Create directories after initialization"""
        for path in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.CHECKPOINTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
    
    def load_from_env(self):
        """Load settings from environment variables"""
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', self.BINANCE_API_KEY)
        self.BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', self.BINANCE_SECRET_KEY)
        self.BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Override with environment variables if present
        if os.getenv('RATE_LIMIT_WEIGHT_PER_MINUTE'):
            self.RATE_LIMIT_WEIGHT_PER_MINUTE = int(os.getenv('RATE_LIMIT_WEIGHT_PER_MINUTE'))
        
        if os.getenv('RATE_LIMIT_SAFETY_MARGIN'):
            self.RATE_LIMIT_SAFETY_MARGIN = float(os.getenv('RATE_LIMIT_SAFETY_MARGIN'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'project_root': str(self.PROJECT_ROOT),
            'data_dir': str(self.DATA_DIR),
            'models_dir': str(self.MODELS_DIR),
            'logs_dir': str(self.LOGS_DIR),
            'checkpoints_dir': str(self.CHECKPOINTS_DIR),
            'binance_testnet': self.BINANCE_TESTNET,
            'rate_limit_weight_per_minute': self.RATE_LIMIT_WEIGHT_PER_MINUTE,
            'rate_limit_safety_margin': self.RATE_LIMIT_SAFETY_MARGIN,
            'default_symbol': self.DEFAULT_SYMBOL,
            'default_interval': self.DEFAULT_INTERVAL,
            'random_state': self.RANDOM_STATE,
            'optimal_cores': self.OPTIMAL_CORES
        }

# Global settings instance
settings = Settings()
settings.load_from_env() 