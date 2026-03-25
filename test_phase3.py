#!/usr/bin/env python3
"""
Phase 3 Test: Trading Strategy Engine & Risk Management.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.engine import TradingStrategy, PortfolioManager
from src.models.lstm_predictor import LSTMPredictor
from src.data.fetcher import DataFetcher
import numpy as np
import pandas as pd

def test_trading_strategy():
    """Test trading strategy components."""
    print("=" * 60)
    print("Phase 3 Test: Trading Strategy Engine & Risk Management")
    print("=" * 60)
    
    # Test 1: Initialize components
    print("\n[Test 1] Initializing strategy components...")
    try:
        # Create mock predictor (since we don't have a trained one yet)
        class MockPredictor:
            def predict_probability(self, X):
                # Return random confidence scores
                return np.random.beta(2, 2, len(X))  # Beta distribution for realistic scores
        
        # Create data fetcher
        fetcher = DataFetcher()
        
        # Create strategy
        strategy = TradingStrategy(MockPredictor(), fetcher)
        
        print("✓ Strategy components initialized")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 2: Generate signals with mock data
    print("\n[Test 2] Generating trading signals...")
    try:
        # Create mock market data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(105, 115, 100),
            'Low': np.random.uniform(95, 105, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Add technical indicators
        mock_data = fetcher.add_technical_indicators(mock_data)
        
        # Generate signal
        signal = strategy.generate_signal('AAPL', mock_data)
        
        print(f"✓ Signal generated: {signal['signal']}")
        print(f"  LSTM confidence: {signal.get('lstm_confidence', 'N/A'):.3f}")
        print(f"  RSI value: {signal.get('rsi_value', 'N/A'):.1f}")
        print(f"  Current price: ${signal.get('current_price', 'N/A'):.2f}")
        
        # Test multiple signals
        signals = []
        for i in range(10):
            # Slightly modify data to get different signals
            test_data = mock_data.copy()
            test_data['Close'] = test_data['Close'] * (1 + np.random.normal(0, 0.01, len(test_data)))
            test_data = fetcher.add_technical_indicators(test_data)
            
            sig = strategy.generate_signal('AAPL', test_data)
            signals.append(sig['signal'])
        
        print(f"  Signal distribution: {pd.Series(signals).value_counts().to_dict()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Position sizing
    print("\n[Test 3] Testing position sizing...")
    try:
        portfolio_value = 100000
        current_price = 150
        
        position_size = strategy.calculate_position_size(portfolio_value, current_price)
        
        # Verify risk calculation
        risk_amount = portfolio_value * strategy.risk_per_trade
        expected_shares = int(risk_amount / (current_price * strategy.stop_loss_pct))
        
        print(f"✓ Position sizing calculated")
        print(f"  Portfolio value: ${portfolio_value:,.0f}")
        print(f"  Stock price: ${current_price:.2f}")
        print(f"  Risk per trade: {strategy.risk_per_trade:.1%}")
        print(f"  Stop loss: {strategy.stop_loss_pct:.1%}")
        print(f"  Calculated shares: {position_size}")
        print(f"  Expected shares: {expected_shares}")
        print(f"  Risk amount: ${risk_amount:.2f}")
        
        assert position_size == expected_shares, "Position sizing calculation incorrect"
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 4: Exit conditions
    print("\n[Test 4] Testing exit conditions...")
    try:
        entry_price = 100
        
        # Test stop loss (LONG position)
        stop_loss_price = entry_price * (1 - strategy.stop_loss_pct)
        should_exit, reason = strategy.should_exit_position(entry_price, stop_loss_price - 0.01, 'LONG')
        print(f"✓ Stop loss test (LONG): {should_exit} - {reason}")
        assert should_exit and reason == 'STOP_LOSS', "Stop loss not triggered correctly"
        
        # Test take profit (LONG position)
        take_profit_price = entry_price * (1 + strategy.take_profit_pct)
        should_exit, reason = strategy.should_exit_position(entry_price, take_profit_price + 0.01, 'LONG')
        print(f"✓ Take profit test (LONG): {should_exit} - {reason}")
        assert should_exit and reason == 'TAKE_PROFIT', "Take profit not triggered correctly"
        
        # Test no exit condition
        should_exit, reason = strategy.should_exit_position(entry_price, entry_price * 1.01, 'LONG')
        print(f"✓ No exit test: {should_exit} - {reason}")
        assert not should_exit, "Exit triggered when it shouldn't"
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 5: Portfolio management
    print("\n[Test 5] Testing portfolio management...")
    try:
        portfolio = PortfolioManager(initial_cash=50000)
        
        # Test initial state
        initial_value = portfolio.get_portfolio_value({})
        print(f"✓ Initial portfolio value: ${initial_value:.2f}")
        assert initial_value == 50000, "Initial portfolio value incorrect"
        
        # Test opening position
        symbol = 'AAPL'
        shares = 10
        price = 150
        success = portfolio.open_position(symbol, shares, price, 'LONG')
        print(f"✓ Opened position: {success}")
        assert success, "Failed to open position"
        
        # Check updated portfolio
        portfolio_value = portfolio.get_portfolio_value({symbol: price})
        expected_value = 50000 - (shares * price) + (shares * price)  # cash - cost + position value
        print(f"  Portfolio value after opening: ${portfolio_value:.2f}")
        assert abs(portfolio_value - expected_value) < 0.01, "Portfolio value calculation incorrect"
        
        # Test closing position
        exit_price = 160
        success = portfolio.close_position(symbol, exit_price, 'TEST')
        print(f"✓ Closed position: {success}")
        assert success, "Failed to close position"
        
        # Check final portfolio
        final_value = portfolio.get_portfolio_value({})
        expected_pnl = (exit_price - price) * shares
        expected_final = 50000 - (shares * price) + (shares * exit_price)
        print(f"  Final portfolio value: ${final_value:.2f}")
        print(f"  Expected P&L: ${expected_pnl:.2f}")
        assert abs(final_value - expected_final) < 0.01, "Final portfolio value incorrect"
        
        # Test performance metrics
        metrics = portfolio.get_performance_metrics()
        print(f"✓ Performance metrics: {metrics}")
        assert metrics['total_trades'] == 1, "Trade count incorrect"
        assert abs(metrics['total_pnl'] - expected_pnl) < 0.01, "P&L calculation incorrect"
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All Phase 3 tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_trading_strategy()
    sys.exit(0 if success else 1)