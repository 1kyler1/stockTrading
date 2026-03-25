"""
Trading Strategy Engine: Combines LSTM predictions with technical indicators
and risk management rules to generate buy/sell/hold signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from src.config import STRATEGY_CONFIG, RISK_CONFIG, MARKET_CONFIG
from src.models.lstm_predictor import LSTMPredictor
from src.data.fetcher import DataFetcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingStrategy:
    """Trading strategy that combines LSTM predictions with technical indicators."""
    
    def __init__(self, predictor: LSTMPredictor, data_fetcher: DataFetcher):
        """
        Initialize trading strategy.
        
        Args:
            predictor: Trained LSTM predictor
            data_fetcher: Data fetcher for live data
        """
        self.predictor = predictor
        self.data_fetcher = data_fetcher
        
        # Strategy parameters
        self.lstm_threshold = STRATEGY_CONFIG.get('LSTM_CONFIDENCE_THRESHOLD', 0.65)
        self.rsi_period = STRATEGY_CONFIG.get('RSI_PERIOD', 14)
        self.rsi_oversold = STRATEGY_CONFIG.get('RSI_OVERSOLD', 30)
        self.rsi_overbought = STRATEGY_CONFIG.get('RSI_OVERBOUGHT', 70)
        
        # Risk management parameters
        self.risk_per_trade = RISK_CONFIG.get('RISK_PER_TRADE', 0.02)
        self.stop_loss_pct = RISK_CONFIG.get('STOP_LOSS_PERCENT', 0.02)
        self.take_profit_pct = RISK_CONFIG.get('TAKE_PROFIT_PERCENT', 0.05)
        self.max_positions = RISK_CONFIG.get('MAX_POSITIONS', 5)
        self.max_portfolio_exposure = RISK_CONFIG.get('MAX_PORTFOLIO_EXPOSURE', 0.50)
        
        logger.info("TradingStrategy initialized")
    
    def generate_signal(self, symbol: str, current_data: pd.DataFrame) -> Dict:
        """
        Generate trading signal for a symbol based on current market data.
        
        Args:
            symbol: Stock symbol
            current_data: Recent OHLCV data with technical indicators
        
        Returns:
            Dictionary with signal information
        """
        try:
            # Get LSTM prediction
            if len(current_data) < 60:
                return {'signal': 'HOLD', 'reason': 'Insufficient data for LSTM', 'confidence': 0.0}
            
            # Prepare data for LSTM (last 60 days)
            recent_data = current_data.tail(60)
            
            # Add technical indicators if not present
            if 'RSI' not in recent_data.columns:
                recent_data = self.data_fetcher.add_technical_indicators(recent_data)
            
            # Normalize data
            normalized_data, _ = self.data_fetcher.normalize_data(recent_data)
            
            # Create windowed input
            X = normalized_data.values.reshape(1, 60, -1)  # Single sample
            
            # Get LSTM prediction
            confidence = self.predictor.predict_probability(X)[0]
            lstm_signal = 'BUY' if confidence > 0.5 else 'SELL'
            
            # Get current technical indicators
            current_rsi = recent_data['RSI'].iloc[-1] if 'RSI' in recent_data.columns else 50
            current_close = recent_data['Close'].iloc[-1]
            current_sma200 = recent_data['SMA_200'].iloc[-1] if 'SMA_200' in recent_data.columns else None

            # Combine signals
            signal = self._combine_signals(lstm_signal, confidence, current_rsi, current_close, current_sma200)
            
            return {
                'signal': signal,
                'lstm_confidence': confidence,
                'rsi_value': current_rsi,
                'current_price': current_close,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'reason': f'Error: {str(e)}', 'confidence': 0.0}
    
    def _combine_signals(self, lstm_signal: str, confidence: float, rsi: float,
                         current_price: float, sma_200: Optional[float] = None) -> str:
        """
        Combine LSTM and technical indicator signals.

        Args:
            lstm_signal: BUY or SELL from LSTM
            confidence: LSTM confidence score
            rsi: Current RSI value
            current_price: Current stock price
            sma_200: 200-day SMA value (trend filter gate)

        Returns:
            Final signal: BUY, SELL, or HOLD
        """
        # Check LSTM confidence threshold
        if confidence < self.lstm_threshold:
            return 'HOLD'

        # Apply technical confirmation
        if lstm_signal == 'BUY':
            # Trend filter: only buy in uptrend (price above 200-day SMA)
            if sma_200 is not None and current_price < sma_200:
                logger.info(f"BUY blocked by trend filter: price ${current_price:.2f} below SMA200 ${sma_200:.2f}")
                return 'HOLD'
            if rsi < self.rsi_oversold or confidence > 0.7:
                return 'BUY'
            else:
                return 'HOLD'

        elif lstm_signal == 'SELL':
            if rsi > self.rsi_overbought or confidence > 0.7:
                return 'SELL'
            else:
                return 'HOLD'

        return 'HOLD'
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) from OHLCV data.

        Args:
            df: DataFrame with High, Low, Close columns
            period: ATR period (default 14)

        Returns:
            ATR value, or None if insufficient data
        """
        if len(df) < period + 1:
            return None

        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(period).mean().iloc[-1]

    def calculate_position_size(self, portfolio_value: float, current_price: float,
                              atr: Optional[float] = None) -> float:
        """
        Calculate position size based on risk management rules.

        Args:
            portfolio_value: Current portfolio value
            current_price: Current stock price
            atr: ATR value for dynamic stop loss (uses fixed % if None)

        Returns:
            Number of shares to trade
        """
        risk_amount = portfolio_value * self.risk_per_trade

        if atr is not None:
            # ATR-based stop: 2x ATR below entry
            stop_loss_dollars = atr * 2
        else:
            # Fallback to fixed percentage
            stop_loss_dollars = current_price * self.stop_loss_pct

        shares = int(risk_amount / stop_loss_dollars)
        return max(1, shares)

    def calculate_stop_price(self, entry_price: float, atr: Optional[float] = None,
                             position_type: str = 'LONG') -> float:
        """
        Calculate the stop loss price for a position.

        Args:
            entry_price: Entry price
            atr: ATR value (uses fixed % if None)
            position_type: 'LONG' or 'SHORT'

        Returns:
            Stop loss price
        """
        if atr is not None:
            stop_distance = atr * 2
        else:
            stop_distance = entry_price * self.stop_loss_pct

        if position_type == 'LONG':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def should_exit_position(self, entry_price: float, current_price: float,
                           position_type: str, stop_price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if position should be exited based on stop loss/take profit.

        Args:
            entry_price: Price when position was entered
            current_price: Current price
            position_type: 'LONG' or 'SHORT'
            stop_price: ATR-based stop price (uses fixed % if None)

        Returns:
            Tuple of (should_exit, reason)
        """
        if position_type == 'LONG':
            # Stop loss: use ATR-based price if available, else fixed %
            stop = stop_price if stop_price is not None else entry_price * (1 - self.stop_loss_pct)
            if current_price <= stop:
                return True, 'STOP_LOSS'
            if current_price >= entry_price * (1 + self.take_profit_pct):
                return True, 'TAKE_PROFIT'

        elif position_type == 'SHORT':
            stop = stop_price if stop_price is not None else entry_price * (1 + self.stop_loss_pct)
            if current_price >= stop:
                return True, 'STOP_LOSS'
            if current_price <= entry_price * (1 - self.take_profit_pct):
                return True, 'TAKE_PROFIT'

        return False, ''


class PortfolioManager:
    """Manages portfolio positions, cash, and performance tracking."""
    
    def __init__(self, initial_cash: float = 100000):
        """
        Initialize portfolio manager.
        
        Args:
            initial_cash: Starting cash amount
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> {'shares': int, 'entry_price': float, 'entry_time': datetime}
        self.trade_history = []  # List of completed trades
        
        logger.info(f"PortfolioManager initialized with ${initial_cash:,.2f}")
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_prices: Dictionary of symbol -> current price
        
        Returns:
            Total portfolio value
        """
        position_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value += position['shares'] * current_prices[symbol]
        
        return self.cash + position_value
    
    def can_open_position(self, symbol: str, shares: int, price: float, current_prices: Optional[Dict[str, float]] = None) -> bool:
        """
        Check if we can open a new position.

        Args:
            symbol: Stock symbol
            shares: Number of shares
            price: Current price
            current_prices: Optional mapping of existing symbols to current price

        Returns:
            True if position can be opened
        """
        cost = shares * price

        # Check cash availability
        if cost > self.cash:
            return False

        # Check max positions limit
        if len(self.positions) >= RISK_CONFIG.get('MAX_POSITIONS', 5):
            return False

        # Check portfolio exposure (only count position values, not cash)
        if current_prices is None:
            current_prices = {}

        position_value = sum(
            pos['shares'] * current_prices.get(sym, pos['entry_price'])
            for sym, pos in self.positions.items()
        )
        max_exposure = self.initial_cash * RISK_CONFIG.get('MAX_PORTFOLIO_EXPOSURE', 0.50)

        if position_value + cost > max_exposure:
            return False

        return True
    
    def open_position(self, symbol: str, shares: int, price: float,
                      position_type: str = 'LONG', stop_price: Optional[float] = None):
        """
        Open a new position.

        Args:
            symbol: Stock symbol
            shares: Number of shares
            price: Entry price
            position_type: 'LONG' or 'SHORT'
            stop_price: ATR-based stop loss price
        """
        if not self.can_open_position(symbol, shares, price):
            logger.warning(f"Cannot open position for {symbol}: insufficient funds or limits reached")
            return False

        cost = shares * price
        self.cash -= cost

        self.positions[symbol] = {
            'shares': shares,
            'entry_price': price,
            'entry_time': datetime.now(),
            'type': position_type,
            'stop_price': stop_price
        }
        
        logger.info(f"Opened {position_type} position: {shares} shares of {symbol} at ${price:.2f}")
        return True
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'MANUAL'):
        """
        Close an existing position.
        
        Args:
            symbol: Stock symbol
            exit_price: Exit price
            reason: Reason for closing
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        shares = position['shares']
        entry_price = position['entry_price']
        
        # Calculate P&L
        if position['type'] == 'LONG':
            pnl = (exit_price - entry_price) * shares
        else:  # SHORT
            pnl = (entry_price - exit_price) * shares
        
        # Update cash
        proceeds = shares * exit_price
        self.cash += proceeds
        
        # Record trade
        trade = {
            'symbol': symbol,
            'type': position['type'],
            'shares': shares,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now()
        }
        
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed {position['type']} position: {symbol}, P&L: ${pnl:.2f} ({reason})")
        return True
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0}
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(trade['pnl'] for trade in self.trade_history)
        
        # Calculate returns for Sharpe ratio (simplified)
        returns = [trade['pnl'] / (trade['shares'] * trade['entry_price']) for trade in self.trade_history]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary.
        
        Returns:
            Dictionary with portfolio state and performance
        """
        # For summary without current prices, use entry prices as approximation
        current_prices = {symbol: pos['entry_price'] for symbol, pos in self.positions.items()}
        current_value = self.get_portfolio_value(current_prices)
        metrics = self.get_performance_metrics()
        
        return {
            'cash': self.cash,
            'portfolio_value': current_value,
            'total_return': (current_value - self.initial_cash) / self.initial_cash if self.initial_cash > 0 else 0,
            'open_positions': len(self.positions),
            'positions': self.positions.copy(),
            'trade_history': self.trade_history.copy(),
            'performance': metrics,
            'timestamp': datetime.now()
        }


if __name__ == "__main__":
    # Test TradingStrategy and PortfolioManager
    print("Testing TradingStrategy and PortfolioManager...")
    
    # Mock components
    class MockPredictor:
        def predict_probability(self, X):
            return np.random.random(len(X))
    
    class MockFetcher:
        def add_technical_indicators(self, df):
            df['RSI'] = 50  # Neutral RSI
            return df
    
    # Create mock data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    mock_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(105, 115, 100),
        'Low': np.random.uniform(95, 105, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Test strategy
    strategy = TradingStrategy(MockPredictor(), MockFetcher())
    signal = strategy.generate_signal('AAPL', mock_data)
    print(f"Generated signal: {signal}")
    
    # Test portfolio manager
    portfolio = PortfolioManager(10000)
    print(f"Initial portfolio value: ${portfolio.get_portfolio_value({}):.2f}")
    
    # Test position sizing
    position_size = strategy.calculate_position_size(10000, 100)
    print(f"Position size for $100 stock: {position_size} shares")
    
    # Test opening position
    success = portfolio.open_position('AAPL', position_size, 100)
    print(f"Opened position: {success}")
    
    # Test closing position
    success = portfolio.close_position('AAPL', 105, 'TEST')
    print(f"Closed position: {success}")
    
    # Test metrics
    metrics = portfolio.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    print("✓ Strategy and Portfolio tests completed")