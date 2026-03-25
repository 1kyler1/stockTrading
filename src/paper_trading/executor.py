"""
Paper Trading Integration: Execute trades using Alpaca API
Connects to Alpaca paper trading account for real-time execution.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np
from alpaca_trade_api import REST, TimeFrame
from alpaca_trade_api.stream import Stream
import pytz

from src.config import (
    APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL,
    STOCKS, MARKET_CONFIG, RISK_CONFIG, TRADING_MODE
)
from src.data.fetcher import DataFetcher
from src.models.lstm_predictor import LSTMPredictor
from src.strategy.engine import TradingStrategy, PortfolioManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Eastern Time Zone for market hours
EASTERN = pytz.timezone('US/Eastern')


class PaperTradingExecutor:
    """Execute paper trades using Alpaca API."""

    def __init__(self, predictor: LSTMPredictor, trading_strategy: TradingStrategy, data_fetcher: DataFetcher):
        """
        Initialize paper trading executor.

        Args:
            predictor: Trained LSTM predictor
            trading_strategy: Trading strategy logic
            data_fetcher: Data fetcher for market data
        """
        self.predictor = predictor
        self.trading_strategy = trading_strategy
        self.data_fetcher = data_fetcher

        # Initialize Alpaca API
        self.api = REST(
            key_id=APCA_API_KEY_ID,
            secret_key=APCA_API_SECRET_KEY,
            base_url=APCA_API_BASE_URL
        )

        # Portfolio manager for position tracking
        self.portfolio_manager = PortfolioManager()

        # Trading state
        self.is_trading_active = False
        self.last_signal_time = {}
        self.market_data_cache = {}

        # Market hours
        self.market_open_hour = MARKET_CONFIG.get('MARKET_OPEN_HOUR', 9)
        self.market_open_minute = MARKET_CONFIG.get('MARKET_OPEN_MINUTE', 30)
        self.market_close_hour = MARKET_CONFIG.get('MARKET_CLOSE_HOUR', 16)
        self.market_close_minute = MARKET_CONFIG.get('MARKET_CLOSE_MINUTE', 0)

        logger.info("PaperTradingExecutor initialized")

    def is_market_open(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now(EASTERN)

        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check market hours
        market_open = now.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
        market_close = now.replace(hour=self.market_close_hour, minute=self.market_close_minute, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_account_info(self) -> Dict:
        """
        Get current account information.

        Returns:
            Dictionary with account details
        """
        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def get_positions(self) -> Dict[str, Dict]:
        """
        Get current positions.

        Returns:
            Dictionary of symbol -> position details
        """
        try:
            positions = self.api.list_positions()
            position_data = {}

            for pos in positions:
                position_data[pos.symbol] = {
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                }

            return position_data
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    def get_real_time_data(self, symbol: str, lookback_minutes: int = 60) -> pd.DataFrame:
        """
        Get real-time market data for a symbol.

        Args:
            symbol: Stock symbol
            lookback_minutes: Minutes of historical data to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get recent bars
            end_time = datetime.now(EASTERN)
            start_time = end_time - timedelta(minutes=lookback_minutes)

            bars = self.api.get_bars(
                symbol,
                TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=1000
            )

            if not bars:
                logger.warning(f"No bars received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.timestamp,
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume
                })

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error getting real-time data for {symbol}: {e}")
            return pd.DataFrame()

    def generate_signal(self, symbol: str) -> Tuple[str, Dict]:
        """
        Generate trading signal for a symbol using real-time data.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (signal, signal_info)
        """
        try:
            # Get recent data
            df = self.get_real_time_data(symbol, lookback_minutes=120)  # 2 hours

            if df.empty or len(df) < 60:
                return 'HOLD', {'reason': 'insufficient_data'}

            # Add technical indicators
            df_with_indicators = self.data_fetcher.add_technical_indicators(df)

            # Normalize data
            df_normalized, _ = self.data_fetcher.normalize_data(df_with_indicators)

            # Create windowed input (last 60 minutes)
            if len(df_normalized) >= 60:
                X = df_normalized.values[-60:].reshape(1, 60, -1)

                # Get LSTM prediction
                try:
                    confidence = self.predictor.predict_probability(X)[0]
                except Exception as e:
                    logger.warning(f"LSTM prediction failed for {symbol}: {e}, using 0.5")
                    confidence = 0.5

                # Get current RSI
                current_rsi = df_with_indicators['RSI'].iloc[-1] if 'RSI' in df_with_indicators.columns else 50
                current_price = df['Close'].iloc[-1]

                # Generate signal
                signal_info = {
                    'lstm_confidence': confidence,
                    'rsi_value': current_rsi,
                    'current_price': current_price,
                    'timestamp': datetime.now(EASTERN)
                }

                # Use trading strategy to decide
                signal = self.trading_strategy._combine_signals(
                    'BUY' if confidence > 0.5 else 'SELL',
                    confidence,
                    current_rsi,
                    current_price
                )

                return signal, signal_info
            else:
                return 'HOLD', {'reason': 'insufficient_normalized_data'}

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return 'HOLD', {'error': str(e)}

    def execute_signal(self, symbol: str, signal: str, signal_info: Dict) -> bool:
        """
        Execute a trading signal.

        Args:
            symbol: Stock symbol
            signal: 'BUY', 'SELL', or 'HOLD'
            signal_info: Signal details

        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            if signal == 'HOLD':
                return True

            # Check market hours
            if not self.is_market_open():
                logger.info(f"Market closed, skipping {signal} signal for {symbol}")
                return False

            # Get current positions and account info
            positions = self.get_positions()
            account = self.get_account_info()

            if not account:
                logger.error("Could not get account information")
                return False

            current_position = positions.get(symbol, {}).get('qty', 0)
            current_price = signal_info['current_price']
            portfolio_value = account['portfolio_value']

            # Check if we should execute
            if signal == 'BUY' and current_position == 0:
                # Calculate position size
                shares = self.trading_strategy.calculate_position_size(portfolio_value, current_price)

                if shares > 0:
                    # Check buying power
                    cost = shares * current_price
                    if cost <= account['buying_power']:
                        # Place buy order
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=shares,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )

                        logger.info(f"BUY ORDER PLACED: {symbol} {shares} shares at ~${current_price:.2f}")
                        self.portfolio_manager.open_position(symbol, shares, current_price, signal_info)
                        return True
                    else:
                        logger.warning(f"Insufficient buying power for {symbol}: need ${cost:.2f}, have ${account['buying_power']:.2f}")
                        return False

            elif signal == 'SELL' and current_position > 0:
                # Close position
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=current_position,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )

                logger.info(f"SELL ORDER PLACED: {symbol} {current_position} shares at ~${current_price:.2f}")
                self.portfolio_manager.close_position(symbol, current_price, signal_info)
                return True

            return False

        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}")
            return False

    def run_trading_cycle(self) -> Dict[str, Dict]:
        """
        Run one complete trading cycle for all symbols.

        Returns:
            Dictionary of results for each symbol
        """
        results = {}

        for symbol in STOCKS:
            try:
                # Generate signal
                signal, signal_info = self.generate_signal(symbol)

                # Execute signal
                executed = self.execute_signal(symbol, signal, signal_info)

                results[symbol] = {
                    'signal': signal,
                    'signal_info': signal_info,
                    'executed': executed,
                    'timestamp': datetime.now(EASTERN)
                }

                logger.info(f"{symbol}: Signal={signal}, Executed={executed}")

            except Exception as e:
                logger.error(f"Error in trading cycle for {symbol}: {e}")
                results[symbol] = {
                    'signal': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now(EASTERN)
                }

        return results

    def start_trading(self, cycle_interval_seconds: int = 300):
        """
        Start automated trading loop.

        Args:
            cycle_interval_seconds: Seconds between trading cycles (default 5 minutes)
        """
        logger.info("Starting automated paper trading...")
        self.is_trading_active = True

        try:
            while self.is_trading_active:
                # Check if market is open
                if not self.is_market_open():
                    logger.info("Market closed, waiting...")
                    time.sleep(60)  # Check every minute
                    continue

                # Run trading cycle
                cycle_start = time.time()
                results = self.run_trading_cycle()

                # Log summary
                executed_count = sum(1 for r in results.values() if r.get('executed', False))
                logger.info(f"Trading cycle completed: {executed_count}/{len(STOCKS)} orders executed")

                # Wait for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, cycle_interval_seconds - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.is_trading_active = False
            logger.info("Trading stopped")

    def stop_trading(self):
        """Stop automated trading."""
        logger.info("Stopping trading...")
        self.is_trading_active = False

    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary.

        Returns:
            Dictionary with portfolio details
        """
        try:
            account = self.get_account_info()
            positions = self.get_positions()

            # Calculate portfolio metrics
            total_value = account.get('portfolio_value', 0)
            cash = account.get('cash', 0)
            positions_value = sum(pos.get('market_value', 0) for pos in positions.values())

            # Get portfolio manager summary
            pm_summary = self.portfolio_manager.get_summary()

            return {
                'account': account,
                'positions': positions,
                'portfolio_manager': pm_summary,
                'total_positions': len(positions),
                'total_value': total_value,
                'cash_allocation': cash / total_value if total_value > 0 else 0,
                'positions_allocation': positions_value / total_value if total_value > 0 else 0,
                'timestamp': datetime.now(EASTERN)
            }

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}


if __name__ == "__main__":
    # Test paper trading executor
    print("Testing PaperTradingExecutor...")

    # Create mock components
    class MockPredictor:
        def predict_probability(self, X):
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

    # Create components
    predictor = MockPredictor()
    strategy = MockStrategy()
    fetcher = MockFetcher()

    # Create executor
    executor = PaperTradingExecutor(predictor, strategy, fetcher)

    # Test market hours
    is_open = executor.is_market_open()
    print(f"Market open: {is_open}")

    # Test account info (requires API keys)
    if APCA_API_KEY_ID and APCA_API_SECRET_KEY:
        account = executor.get_account_info()
        if account:
            print(f"Account status: {account.get('status')}")
            print(f"Cash: ${account.get('cash', 0):,.2f}")
            print(f"Portfolio value: ${account.get('portfolio_value', 0):,.2f}")
        else:
            print("Could not connect to Alpaca API")
    else:
        print("API keys not configured")

    print("✓ Paper trading executor test completed")