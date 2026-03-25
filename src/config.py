"""
Configuration loader for trading system.
Reads from .env file and config/trading_params.yaml
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Alpaca API Configuration
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
TRADING_MODE = os.getenv("TRADING_MODE", "paper")

# Directory Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Load trading parameters from YAML
TRADING_PARAMS_PATH = PROJECT_ROOT / "config" / "trading_params.yaml"

def load_trading_params():
    """Load trading parameters from YAML file."""
    with open(TRADING_PARAMS_PATH, 'r') as f:
        return yaml.safe_load(f)

TRADING_PARAMS = load_trading_params()

# Extract commonly used parameters for easier access
STOCKS = TRADING_PARAMS.get('STOCKS', [])
DATA_CONFIG = TRADING_PARAMS.get('DATA', {})
MODEL_CONFIG = TRADING_PARAMS.get('MODEL', {})
STRATEGY_CONFIG = TRADING_PARAMS.get('STRATEGY', {})
RISK_CONFIG = TRADING_PARAMS.get('RISK_MANAGEMENT', {})
MARKET_CONFIG = TRADING_PARAMS.get('MARKET', {})
BACKTEST_CONFIG = TRADING_PARAMS.get('BACKTEST', {})

# Validation
def validate_config():
    """Validate that required API keys are set."""
    if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
        raise ValueError(
            "Alpaca API credentials not set. "
            "Please create a .env file with APCA_API_KEY_ID and APCA_API_SECRET_KEY. "
            "Get them from https://app.alpaca.markets (paper trading keys)"
        )
    
    if not STOCKS:
        raise ValueError("No stocks configured in trading_params.yaml")

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Trading Mode: {TRADING_MODE}")
    print(f"Base URL: {APCA_API_BASE_URL}")
    print(f"Stocks: {STOCKS}")
    print(f"Model Config: {MODEL_CONFIG}")
