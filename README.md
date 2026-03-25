# AI Stock Trading Algorithm

A fully autonomous AI-powered stock trading system using LSTM neural networks and technical analysis for paper trading.

## 🚀 Project Status

### ✅ Completed Phases

- **Phase 1**: Data Pipeline ✓
  - yfinance integration for historical data
  - Technical indicators (RSI, MACD, Bollinger Bands, SMAs)
  - Data normalization and windowing for LSTM input
  - Comprehensive data validation and caching

- **Phase 2**: LSTM Model ✓
  - PyTorch LSTM implementation with 128 hidden units
  - Training pipeline with early stopping
  - Model evaluation and persistence
  - ~50% baseline accuracy achieved

- **Phase 3**: Trading Strategy ✓
  - LSTM confidence + RSI confirmation signals
  - Risk management (2% per trade, 2% stop loss, 5% take profit)
  - Portfolio management with position tracking
  - Trade journaling and P&L calculation

- **Phase 4**: Backtesting Framework ✓
  - Backtrader integration for historical simulation
  - Multi-symbol backtesting capability
  - Performance metrics (Sharpe ratio, drawdown, win rate)
  - Result persistence and analysis

- **Phase 5**: Paper Trading Integration ✓
  - Alpaca API integration for real-time data
  - Order execution and position management
  - Market hours validation and error handling
  - Automated trading cycle execution

- **Phase 6**: Monitoring & Iteration ✓
  - Performance monitoring with metrics and drawdown analysis
  - Alert generation for risk thresholds
  - Performance report generation
  - Support for iterative tuning

### ✅ Current Phase: Phase 7 - Deployment & Continuous Improvement

- Auto retrain scheduler (24h by default; configurable)
- Minimal status endpoint (`/status`) for model and health info
- Light deployment manager for orchestrating retrain + live status
- Phase-driven integration tests (test_phase7.py)

### 📋 Upcoming Phases

- **Phase 8**: Visualization & Monitoring Dashboard
  - Streamlit/Flask/Grafana reporting
  - Alerting (email/slack)
  - Continuous integration with live paper trading loops

## 🏗️ Architecture

```
src/
├── config.py              # Configuration management
├── data/
│   └── fetcher.py         # Data acquisition & preprocessing
├── models/
│   └── lstm_predictor.py  # PyTorch LSTM implementation
├── strategy/
│   └── engine.py          # Trading logic & risk management
└── backtest/
    └── runner.py          # Backtesting framework
```

## 🛠️ Technology Stack

- **Language**: Python 3.12
- **ML Framework**: PyTorch
- **Data Processing**: pandas, numpy, scikit-learn
- **Technical Analysis**: pandas-ta
- **Backtesting**: backtrader
- **Broker API**: Alpaca (paper trading)
- **Data Source**: yfinance

## 📊 Key Features

- **LSTM Predictions**: Deep learning model for price direction forecasting
- **Risk Management**: Position sizing, stop-loss, take-profit orders
- **Technical Indicators**: RSI, MACD, Bollinger Bands for signal confirmation
- **Backtesting**: Historical performance validation
- **Paper Trading**: Safe testing with real market data

## 🚀 Quick Start

1. **Setup Environment**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**

   ```bash
   cp .env.example .env
   # Edit .env with your Alpaca API credentials
   ```

3. **Run Tests**

   ```bash
   # Test data pipeline
   python3 test_phase1.py

   # Test LSTM model
   python3 test_phase2.py

   # Test trading strategy
   python3 test_phase3.py

   # Test backtesting
   python3 test_phase4.py
   ```

## 📈 Performance Baseline

- **Data**: 2+ years of daily OHLCV data per symbol
- **LSTM Accuracy**: ~50% direction prediction (coin-flip baseline)
- **Risk Management**: 2% max risk per trade
- **Backtest**: Realistic slippage (0.1%) and commissions

## ⚠️ Disclaimer

This is a paper trading system for educational and research purposes only. Not intended for real money trading. Past performance does not guarantee future results. Always test thoroughly before any financial decisions.
