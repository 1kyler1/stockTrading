#!/usr/bin/env python3
"""
Demo: Paper Trading System
Shows how to use the automated trading system with Alpaca paper trading.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import STOCKS
from src.data.fetcher import DataFetcher
from src.models.lstm_predictor import LSTMPredictor
from src.strategy.engine import TradingStrategy
from src.paper_trading.executor import PaperTradingExecutor

def demo_paper_trading():
    """Demonstrate paper trading capabilities."""
    print("🚀 AI Stock Trading System - Paper Trading Demo")
    print("=" * 60)

    # Initialize components
    print("\n📊 Initializing trading system...")

    # Data fetcher
    data_fetcher = DataFetcher()

    # Get input size from processed data
    X, y, _, _ = data_fetcher.get_processed_data(STOCKS[0])
    input_size = X.shape[2]

    # LSTM predictor
    lstm_predictor = LSTMPredictor(input_size=input_size)

    # Load trained model if available
    model_path = Path("./models/lstm_model.pth")
    if model_path.exists():
        lstm_predictor.load_model(model_path)
        print("✓ Loaded trained LSTM model")
    else:
        print("⚠ Using untrained model (random predictions)")

    # Trading strategy
    trading_strategy = TradingStrategy(lstm_predictor, data_fetcher)

    # Paper trading executor
    executor = PaperTradingExecutor(lstm_predictor, trading_strategy, data_fetcher)

    print("✓ Trading system initialized")

    # Check market status
    print("\n🕐 Market Status:")
    is_open = executor.is_market_open()
    print(f"  Market Open: {'Yes' if is_open else 'No'}")
    print(f"  Trading Hours: 9:30 AM - 4:00 PM ET")

    # Get account info (if API keys configured)
    print("\n💰 Account Information:")
    account = executor.get_account_info()
    if account and 'status' in account:
        print(f"  Status: {account['status']}")
        print(f"  Cash: ${account.get('cash', 0):,.2f}")
        print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
    else:
        print("  API keys not configured - using paper trading simulation")
        print("  To enable live trading, update .env with real Alpaca keys")

    # Get positions
    print("\n📈 Current Positions:")
    positions = executor.get_positions()
    if positions:
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
    else:
        print("  No open positions")

    # Generate sample signals
    print("\n🎯 Sample Trading Signals:")
    for symbol in STOCKS[:3]:  # Show first 3 symbols
        signal, signal_info = executor.generate_signal(symbol)
        confidence = signal_info.get('lstm_confidence', 0)
        rsi = signal_info.get('rsi_value', 'N/A')
        price = signal_info.get('current_price', 'N/A')

        print(f"  {symbol}: {signal} (Confidence: {confidence:.2f}, RSI: {rsi}, Price: ${price})")

    # Portfolio summary
    print("\n📊 Portfolio Summary:")
    summary = executor.get_portfolio_summary()
    pm = summary.get('portfolio_manager', {})
    perf = pm.get('performance', {})

    print(f"  Total Trades: {perf.get('total_trades', 0)}")
    print(f"  Win Rate: {perf.get('win_rate', 0):.1%}")
    print(f"  Total P&L: ${perf.get('total_pnl', 0):.2f}")
    print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")

    # Trading instructions
    print("\n🚀 How to Start Automated Trading:")
    print("1. Configure Alpaca API keys in .env file")
    print("2. Train the LSTM model: python3 test_phase2.py")
    print("3. Run backtests: python3 test_phase4.py")
    print("4. Start trading:")
    print("   from src.paper_trading.executor import PaperTradingExecutor")
    print("   executor = PaperTradingExecutor(predictor, strategy, fetcher)")
    print("   executor.start_trading()  # Runs every 5 minutes during market hours")

    print("\n⚠️  Important Notes:")
    print("- This is PAPER TRADING - no real money at risk")
    print("- Always backtest strategies before live trading")
    print("- Monitor performance and adjust parameters as needed")
    print("- The system includes comprehensive risk management")

    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    demo_paper_trading()