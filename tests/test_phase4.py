"""
Phase 4 Test: Backtesting Framework
Test the backtesting integration with historical data and strategy validation.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import STOCKS, BACKTEST_CONFIG
from src.data.fetcher import DataFetcher
from src.models.lstm_predictor import LSTMPredictor
from src.strategy.engine import TradingStrategy
from src.backtest.runner import BacktestRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_backtest_runner():
    """Test the backtest runner with real components."""
    print("Testing BacktestRunner with real components...")
    
    try:
        # Initialize components
        data_fetcher = DataFetcher()
        
        # Get input size from processed data
        X, y, _, _ = data_fetcher.get_processed_data(STOCKS[0])
        input_size = X.shape[2]
        
        lstm_predictor = LSTMPredictor(input_size=input_size)
        trading_strategy = TradingStrategy(lstm_predictor, data_fetcher)
        
        # Load trained model
        model_path = Path("./models/lstm_model.pth")
        if model_path.exists():
            lstm_predictor.load_model(model_path)
            print("✓ Loaded trained LSTM model")
        else:
            print("⚠ No trained model found, using untrained model")
        
        # Create backtest runner
        runner = BacktestRunner(lstm_predictor, trading_strategy, data_fetcher)
        
        # Test single symbol backtest
        symbol = STOCKS[0]  # AAPL
        print(f"Running backtest for {symbol}...")
        
        results = runner.run_backtest(symbol)
        
        if results:
            print("✓ Backtest completed successfully!")
            print(f"  Initial cash: ${results.get('initial_cash', 0):,.0f}")
            print(f"  Final value: ${results.get('final_value', 0):,.0f}")
            print(f"  Total return: {results.get('total_return', 0):.2%}")
            print(f"  Sharpe ratio: {results.get('sharpe_ratio', 'N/A')}")
            print(f"  Max drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"  Total trades: {results.get('total_trades', 0)}")
            print(f"  Win rate: {results.get('win_rate', 0):.1%}")
            print(f"  Avg trade P&L: ${results.get('avg_trade_pnl', 0):.2f}")
            
            # Check for reasonable results
            assert results.get('total_return', 0) > -1.0, "Return should be reasonable"
            assert results.get('max_drawdown', 1.0) < 1.0, "Drawdown should be < 100%"
            assert results.get('total_trades', 0) >= 0, "Trades should be non-negative"
            
            print("✓ Backtest results look reasonable")
        else:
            print("✗ Backtest failed - no results returned")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Backtest runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_symbol_backtest():
    """Test backtesting across multiple symbols."""
    print("\nTesting multi-symbol backtest...")
    
    try:
        # Initialize components
        data_fetcher = DataFetcher()
        
        # Get input size from processed data
        X, y, _, _ = data_fetcher.get_processed_data(STOCKS[0])
        input_size = X.shape[2]
        
        lstm_predictor = LSTMPredictor(input_size=input_size)
        trading_strategy = TradingStrategy(lstm_predictor, data_fetcher)
        
        # Load trained model if available
        model_path = Path("./models/lstm_model.pth")
        if model_path.exists():
            lstm_predictor.load_model(model_path)
        
        # Create backtest runner
        runner = BacktestRunner(lstm_predictor, trading_strategy, data_fetcher)
        
        # Test with first 2 symbols
        test_symbols = STOCKS[:2]
        print(f"Running multi-symbol backtest for {test_symbols}...")
        
        results = runner.run_multi_symbol_backtest(test_symbols)
        
        if results and 'combined' in results:
            print("✓ Multi-symbol backtest completed!")
            combined = results['combined']
            print(f"  Symbols tested: {combined.get('total_symbols', 0)}")
            print(f"  Combined Sharpe ratio: {combined.get('combined_sharpe_ratio', 'N/A')}")
            print(f"  Avg daily return: {combined.get('avg_daily_return', 0):.4%}")
            print(f"  Daily volatility: {combined.get('daily_volatility', 0):.4%}")
            
            # Check individual symbol results
            for symbol in test_symbols:
                if symbol in results:
                    symbol_result = results[symbol]
                    print(f"  {symbol}: Return={symbol_result.get('total_return', 0):.2%}, "
                          f"Trades={symbol_result.get('total_trades', 0)}")
            
            print("✓ Multi-symbol results look reasonable")
            return True
        else:
            print("✗ Multi-symbol backtest failed")
            return False
            
    except Exception as e:
        print(f"✗ Multi-symbol backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_configuration():
    """Test backtest configuration loading."""
    print("\nTesting backtest configuration...")
    
    try:
        # Check BACKTEST_CONFIG
        required_keys = ['INITIAL_CASH', 'COMMISSION', 'SLIPPAGE']
        
        for key in required_keys:
            if key not in BACKTEST_CONFIG:
                print(f"✗ Missing required backtest config: {key}")
                return False
            print(f"✓ {key}: {BACKTEST_CONFIG[key]}")
        
        # Validate values
        assert BACKTEST_CONFIG['INITIAL_CASH'] > 0, "Initial cash must be positive"
        assert 0 <= BACKTEST_CONFIG['COMMISSION'] <= 0.01, "Commission should be reasonable"
        assert 0 <= BACKTEST_CONFIG['SLIPPAGE'] <= 0.01, "Slippage should be reasonable"
        
        print("✓ Backtest configuration is valid")
        return True
        
    except Exception as e:
        print(f"✗ Backtest configuration test failed: {e}")
        return False


def main():
    """Run all Phase 4 tests."""
    print("=== PHASE 4: BACKTESTING FRAMEWORK TESTS ===\n")
    
    tests = [
        ("Backtest Configuration", test_backtest_configuration),
        ("Backtest Runner", test_backtest_runner),
        ("Multi-Symbol Backtest", test_multi_symbol_backtest),
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
    
    print(f"=== PHASE 4 RESULTS: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All Phase 4 tests passed! Ready for Phase 5: Paper Trading Integration")
        return True
    else:
        print("❌ Some Phase 4 tests failed. Please fix before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)