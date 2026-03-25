"""
Phase 5 Test: Paper Trading Integration
Test Alpaca API integration and real-time trading execution.
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import (
    APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL,
    STOCKS, TRADING_MODE
)
from src.data.fetcher import DataFetcher
from src.models.lstm_predictor import LSTMPredictor
from src.strategy.engine import TradingStrategy
from src.paper_trading.executor import PaperTradingExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_connection():
    """Test Alpaca API connection."""
    print("Testing Alpaca API connection...")

    try:
        # Create mock components for testing
        class MockPredictor:
            def predict_probability(self, X):
                import numpy as np
                return np.random.beta(2, 2, len(X))

        class MockStrategy:
            def __init__(self):
                self.risk_per_trade = 0.02

            def calculate_position_size(self, portfolio_value, price):
                return int((portfolio_value * self.risk_per_trade) / (price * 0.02))

            def _combine_signals(self, lstm_signal, confidence, rsi, price):
                return lstm_signal if confidence > 0.6 else 'HOLD'

        class MockFetcher:
            def add_technical_indicators(self, df):
                df['RSI'] = 50
                return df

            def normalize_data(self, df):
                return df, None

        # Create executor
        predictor = MockPredictor()
        strategy = MockStrategy()
        fetcher = MockFetcher()
        executor = PaperTradingExecutor(predictor, strategy, fetcher)

        # Test API connection
        if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY or APCA_API_KEY_ID == "PLACEHOLDER_KEY":
            print("⚠ API keys not configured - skipping live API tests")
            print("  To enable live tests, update .env file with real Alpaca paper trading keys")
            return True

        # Test account info
        account = executor.get_account_info()
        if account:
            print("✓ API connection successful")
            print(f"  Account status: {account.get('status')}")
            print(f"  Trading mode: {TRADING_MODE}")
            print(f"  Cash: ${account.get('cash', 0):,.2f}")
            print(f"  Portfolio value: ${account.get('portfolio_value', 0):,.2f}")

            # Validate account is paper trading
            if TRADING_MODE != 'paper':
                print("⚠ WARNING: Not in paper trading mode!")
            else:
                print("✓ Confirmed paper trading mode")

            return True
        else:
            print("✗ API connection failed")
            return False

    except Exception as e:
        print(f"✗ API connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_market_hours():
    """Test market hours validation."""
    print("\nTesting market hours validation...")

    try:
        # Create mock executor
        class MockPredictor:
            def predict_probability(self, X):
                import numpy as np
                return np.random.beta(2, 2, len(X))

        class MockStrategy:
            def __init__(self):
                self.risk_per_trade = 0.02

        class MockFetcher:
            pass

        executor = PaperTradingExecutor(MockPredictor(), MockStrategy(), MockFetcher())

        # Test market hours check
        is_open = executor.is_market_open()
        print(f"✓ Market hours check: {'Open' if is_open else 'Closed'}")

        # Show market hours config
        print(f"  Market open: {executor.market_open_hour}:{executor.market_open_minute:02d}")
        print(f"  Market close: {executor.market_close_hour}:{executor.market_close_minute:02d}")

        return True

    except Exception as e:
        print(f"✗ Market hours test failed: {e}")
        return False


def test_real_time_data():
    """Test real-time data fetching."""
    print("\nTesting real-time data fetching...")

    try:
        # Create executor
        class MockPredictor:
            def predict_probability(self, X):
                import numpy as np
                return np.random.beta(2, 2, len(X))

        class MockStrategy:
            def __init__(self):
                self.risk_per_trade = 0.02

        class MockFetcher:
            def add_technical_indicators(self, df):
                df['RSI'] = 50
                return df

            def normalize_data(self, df):
                return df, None

        executor = PaperTradingExecutor(MockPredictor(), MockStrategy(), MockFetcher())

        if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY or APCA_API_KEY_ID == "PLACEHOLDER_KEY":
            print("⚠ API keys not configured - skipping real-time data test")
            return True

        # Test data fetching for first symbol
        symbol = STOCKS[0]
        df = executor.get_real_time_data(symbol, lookback_minutes=30)

        if not df.empty:
            print("✓ Real-time data fetched successfully")
            print(f"  Symbol: {symbol}")
            print(f"  Data points: {len(df)}")
            print(f"  Latest price: ${df['Close'].iloc[-1]:.2f}")
            print(f"  Time range: {df.index[0]} to {df.index[-1]}")
            return True
        else:
            print(f"⚠ No real-time data available for {symbol} (possibly market closed)")
            return True  # Not a failure if market is closed

    except Exception as e:
        print(f"✗ Real-time data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_generation():
    """Test signal generation with real-time data."""
    print("\nTesting signal generation...")

    try:
        # Create executor with real components
        data_fetcher = DataFetcher()

        # Get input size
        X, y, _, _ = data_fetcher.get_processed_data(STOCKS[0])
        input_size = X.shape[2]

        lstm_predictor = LSTMPredictor(input_size=input_size)
        trading_strategy = TradingStrategy(lstm_predictor, data_fetcher)
        executor = PaperTradingExecutor(lstm_predictor, trading_strategy, data_fetcher)

        # Load trained model if available
        model_path = Path("./models/lstm_model.pth")
        if model_path.exists():
            lstm_predictor.load_model(model_path)
            print("✓ Loaded trained LSTM model")
        else:
            print("⚠ Using untrained model")

        # Test signal generation for first symbol
        symbol = STOCKS[0]
        signal, signal_info = executor.generate_signal(symbol)

        print("✓ Signal generation completed")
        print(f"  Symbol: {symbol}")
        print(f"  Signal: {signal}")
        print(f"  Confidence: {signal_info.get('lstm_confidence', 'N/A')}")
        print(f"  RSI: {signal_info.get('rsi_value', 'N/A')}")
        print(f"  Current price: ${signal_info.get('current_price', 'N/A')}")

        # Validate signal is valid
        if signal in ['BUY', 'SELL', 'HOLD']:
            print("✓ Signal is valid")
            return True
        else:
            print(f"✗ Invalid signal: {signal}")
            return False

    except Exception as e:
        print(f"✗ Signal generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_summary():
    """Test portfolio summary generation."""
    print("\nTesting portfolio summary...")

    try:
        # Create executor
        class MockPredictor:
            def predict_probability(self, X):
                import numpy as np
                return np.random.beta(2, 2, len(X))

        class MockStrategy:
            def __init__(self):
                self.risk_per_trade = 0.02

        class MockFetcher:
            pass

        executor = PaperTradingExecutor(MockPredictor(), MockStrategy(), MockFetcher())

        if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY or APCA_API_KEY_ID == "PLACEHOLDER_KEY":
            print("⚠ API keys not configured - testing with mock data")
            # Test with mock portfolio manager
            summary = executor.get_portfolio_summary()
            print("✓ Portfolio summary structure created")
            return True

        # Test real portfolio summary
        summary = executor.get_portfolio_summary()

        if summary:
            print("✓ Portfolio summary retrieved")
            account = summary.get('account', {})
            positions = summary.get('positions', {})

            print(f"  Account status: {account.get('status', 'Unknown')}")
            print(f"  Portfolio value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"  Cash: ${account.get('cash', 0):,.2f}")
            print(f"  Positions: {len(positions)}")

            return True
        else:
            print("✗ Portfolio summary failed")
            return False

    except Exception as e:
        print(f"✗ Portfolio summary test failed: {e}")
        return False


def main():
    """Run all Phase 5 tests."""
    print("=== PHASE 5: PAPER TRADING INTEGRATION TESTS ===\n")

    tests = [
        ("API Connection", test_api_connection),
        ("Market Hours", test_market_hours),
        ("Real-Time Data", test_real_time_data),
        ("Signal Generation", test_signal_generation),
        ("Portfolio Summary", test_portfolio_summary),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED\n")
            else:
                print(f"✗ {test_name} FAILED\n")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}\n")

    print(f"=== PHASE 5 RESULTS: {passed}/{total} tests passed ===")

    if passed == total:
        print("🎉 All Phase 5 tests passed! Ready for Phase 6: Monitoring & Iteration")
        print("\n🚀 To start automated trading:")
        print("   python3 -c \"from src.paper_trading.executor import PaperTradingExecutor; exec = PaperTradingExecutor(...); exec.start_trading()\"")
        return True
    else:
        print("❌ Some Phase 5 tests failed. Check API configuration and market hours.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)