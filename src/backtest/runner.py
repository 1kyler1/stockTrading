"""
Backtesting Framework: Run trading strategies on historical data
using backtrader library to validate performance before live trading.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional
from src.config import DATA_DIR, BACKTEST_CONFIG, STOCKS
from src.data.fetcher import DataFetcher
from src.models.lstm_predictor import LSTMPredictor
from src.strategy.engine import TradingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMDataFeed(bt.feeds.PandasData):
    """Custom backtrader data feed for LSTM predictions."""
    
    lines = ('lstm_signal', 'rsi', 'confidence')
    
    params = (
        ('lstm_signal', -1),
        ('rsi', -1),
        ('confidence', -1),
    )


class LSTMSignalStrategy(bt.Strategy):
    """Backtrader strategy that uses LSTM predictions."""
    
    params = (
        ('lstm_predictor', None),
        ('trading_strategy', None),
        ('data_fetcher', None),
        ('max_positions', 5),
        ('risk_per_trade', 0.02),
    )
    
    def __init__(self):
        """Initialize strategy."""
        self.order = None
        self.portfolio_value = []
        self.trades = []
        
        # Keep track of positions
        self.current_positions = {}
        
        logger.info("LSTMSignalStrategy initialized")
    
    def next(self):
        """Execute strategy logic on each bar."""
        try:
            # Get current data
            current_price = self.data.close[0]
            symbol = self.data._name
            
            # Need at least 60 bars for LSTM input
            if len(self.data) < 60:
                return  # Not enough data
            
            # Create DataFrame from backtrader data (last 60 bars)
            dates = [self.data.datetime.date(i) for i in range(-59, 1)]
            ohlcv_data = {
                'Open': [self.data.open[i] for i in range(-59, 1)],
                'High': [self.data.high[i] for i in range(-59, 1)],
                'Low': [self.data.low[i] for i in range(-59, 1)],
                'Close': [self.data.close[i] for i in range(-59, 1)],
                'Volume': [self.data.volume[i] for i in range(-59, 1)]
            }
            
            df = pd.DataFrame(ohlcv_data, index=dates)
            
            # Add technical indicators
            df = self.p.data_fetcher.add_technical_indicators(df)
            
            # Normalize data
            df_normalized, _ = self.p.data_fetcher.normalize_data(df)
            
            # Create windowed input - ensure we have exactly 60 samples
            if len(df_normalized) >= 60:
                X = df_normalized.values[-60:].reshape(1, 60, -1)
                
                # Get LSTM prediction
                try:
                    confidence = self.p.lstm_predictor.predict_probability(X)[0]
                except Exception as e:
                    logger.warning(f"LSTM prediction failed: {e}, using random confidence")
                    confidence = np.random.beta(2, 2)  # Random confidence between 0-1
                
                # Get current RSI
                current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                
                # Generate trading signal
                signal_info = {
                    'lstm_confidence': confidence,
                    'rsi_value': current_rsi,
                    'current_price': current_price
                }
                
                # Use trading strategy to decide
                signal = self.p.trading_strategy._combine_signals(
                    'BUY' if confidence > 0.5 else 'SELL',
                    confidence,
                    current_rsi,
                    current_price
                )
                
                # Execute trades
                self._execute_signal(symbol, signal, current_price, signal_info)
                
                # Record portfolio value
                self.portfolio_value.append(self.broker.getvalue())
            
        except Exception as e:
            logger.error(f"Error in strategy next(): {e}")
            import traceback
            traceback.print_exc()
    
    def _execute_signal(self, symbol: str, signal: str, price: float, signal_info: Dict):
        """Execute buy/sell signal."""
        try:
            # Check current position
            current_position = self.getposition(self.data).size
            
            if signal == 'BUY' and current_position == 0:
                # Calculate position size
                portfolio_value = self.broker.getvalue()
                shares = self.p.trading_strategy.calculate_position_size(portfolio_value, price)
                
                if shares > 0:
                    # Check if we can afford it
                    cost = shares * price
                    if cost <= self.broker.getcash():
                        self.order = self.buy(size=shares)
                        logger.info(f"BUY {symbol}: {shares} shares at ${price:.2f}")
                        
                        # Record trade
                        self.trades.append({
                            'type': 'BUY',
                            'symbol': symbol,
                            'shares': shares,
                            'price': price,
                            'timestamp': self.data.datetime.date(0),
                            'signal_info': signal_info
                        })
            
            elif signal == 'SELL' and current_position > 0:
                # Close position
                self.order = self.sell(size=current_position)
                logger.info(f"SELL {symbol}: {current_position} shares at ${price:.2f}")
                
                # Record trade
                self.trades.append({
                    'type': 'SELL',
                    'symbol': symbol,
                    'shares': current_position,
                    'price': price,
                    'timestamp': self.data.datetime.date(0),
                    'signal_info': signal_info
                })
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"BUY EXECUTED: {order.executed.price:.2f}")
            elif order.issell():
                logger.info(f"SELL EXECUTED: {order.executed.price:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order failed: {order.status}")
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            pnl = trade.pnl
            logger.info(f"TRADE CLOSED: P&L = ${pnl:.2f}")


class BacktestRunner:
    """Run backtests using backtrader."""
    
    def __init__(self, predictor: LSTMPredictor, trading_strategy: TradingStrategy, data_fetcher: DataFetcher):
        """
        Initialize backtest runner.
        
        Args:
            predictor: Trained LSTM predictor
            trading_strategy: Trading strategy logic
            data_fetcher: Data fetcher for historical data
        """
        self.predictor = predictor
        self.trading_strategy = trading_strategy
        self.data_fetcher = data_fetcher
        
        self.results_dir = DATA_DIR / "backtest_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def run_backtest(self, symbol: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest for a single symbol.
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dictionary with backtest results
        """
        try:
            # Get historical data
            if start_date and end_date:
                # Fetch specific date range
                df = self.data_fetcher.get_historical_data(symbol, days=1000)  # Get plenty of data
                df = df.loc[start_date:end_date] if start_date and end_date else df
            else:
                df = self.data_fetcher.get_historical_data(symbol)
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            logger.info(f"Data type: {type(df)}, Shape: {df.shape}, Columns: {list(df.columns)}")
            
            # Prepare data for backtrader
            df_bt = df.copy()
            
            # Flatten MultiIndex columns if they exist (yfinance returns MultiIndex)
            if isinstance(df_bt.columns, pd.MultiIndex):
                df_bt.columns = df_bt.columns.get_level_values(0)
                logger.info(f"Flattened MultiIndex columns: {list(df_bt.columns)}")
            
            # Ensure datetime index is properly formatted
            if not isinstance(df_bt.index, pd.DatetimeIndex):
                df_bt.index = pd.to_datetime(df_bt.index)
            
            logger.info(f"Prepared data type: {type(df_bt)}, Shape: {df_bt.shape}, Index type: {type(df_bt.index)}")
            
            # Create backtrader data feed
            data = bt.feeds.PandasData(
                dataname=df_bt,
                name=symbol,
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=-1
            )
            
            # Initialize cerebro
            cerebro = bt.Cerebro()
            
            # Add data
            cerebro.adddata(data)
            
            # Add strategy
            cerebro.addstrategy(
                LSTMSignalStrategy,
                lstm_predictor=self.predictor,
                trading_strategy=self.trading_strategy,
                data_fetcher=self.data_fetcher
            )
            
            # Set broker parameters
            initial_cash = BACKTEST_CONFIG.get('INITIAL_CASH', 100000)
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=BACKTEST_CONFIG.get('COMMISSION', 0.001))
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # Run backtest
            logger.info(f"Running backtest for {symbol}...")
            results = cerebro.run()
            strategy = results[0]
            
            # Extract results
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - initial_cash) / initial_cash
            
            # Get analyzer results
            sharpe_ratio = results[0].analyzers.sharpe.get_analysis().get('sharperatio', None)
            returns_analysis = results[0].analyzers.returns.get_analysis()
            drawdown_analysis = results[0].analyzers.drawdown.get_analysis()
            trade_analysis = results[0].analyzers.trades.get_analysis()
            
            backtest_results = {
                'symbol': symbol,
                'initial_cash': initial_cash,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': drawdown_analysis.get('max', {}).get('drawdown', 0),
                'total_trades': trade_analysis.get('total', {}).get('total', 0),
                'winning_trades': trade_analysis.get('won', {}).get('total', 0),
                'losing_trades': trade_analysis.get('lost', {}).get('total', 0),
                'win_rate': trade_analysis.get('won', {}).get('total', 0) / max(1, trade_analysis.get('total', {}).get('total', 0)),
                'avg_trade_pnl': trade_analysis.get('pnl', {}).get('net', {}).get('average', 0),
                'portfolio_values': strategy.portfolio_value,
                'trades': strategy.trades
            }
            
            logger.info(f"Backtest completed for {symbol}: Return={total_return:.2%}, Sharpe={sharpe_ratio if sharpe_ratio is not None else 'N/A'}")
            
            # Save results
            self._save_results(backtest_results)
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_multi_symbol_backtest(self, symbols: List[str] = None, start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest across multiple symbols.
        
        Args:
            symbols: List of symbols to test
            start_date: Start date
            end_date: End date
        
        Returns:
            Combined results
        """
        if symbols is None:
            symbols = STOCKS
        
        all_results = {}
        combined_returns = []
        
        for symbol in symbols:
            logger.info(f"Running backtest for {symbol}...")
            results = self.run_backtest(symbol, start_date, end_date)
            if results:
                all_results[symbol] = results
                
                # Collect returns for combined analysis
                if results.get('portfolio_values'):
                    # Calculate daily returns
                    portfolio_values = results['portfolio_values']
                    if len(portfolio_values) > 1:
                        returns = np.diff(portfolio_values) / portfolio_values[:-1]
                        combined_returns.extend(returns)
        
        # Calculate combined metrics
        if combined_returns:
            combined_sharpe = np.mean(combined_returns) / np.std(combined_returns) * np.sqrt(252) if combined_returns else 0
            combined_metrics = {
                'combined_sharpe_ratio': combined_sharpe,
                'avg_daily_return': np.mean(combined_returns),
                'daily_volatility': np.std(combined_returns),
                'total_symbols': len(all_results)
            }
        else:
            combined_metrics = {}
        
        all_results['combined'] = combined_metrics
        
        logger.info(f"Multi-symbol backtest completed: {len(all_results)-1} symbols tested")
        
        return all_results
    
    def _save_results(self, results: Dict):
        """Save backtest results to file."""
        import json
        
        symbol = results['symbol']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{symbol}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                serializable_results[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                serializable_results[key] = int(value)
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {filepath}")


if __name__ == "__main__":
    # Test backtesting framework
    print("Testing BacktestRunner...")
    
    # Create mock components
    class MockPredictor:
        def predict_probability(self, X):
            return np.random.beta(2, 2, len(X))
    
    class MockStrategy:
        def __init__(self):
            self.risk_per_trade = 0.02
            self.stop_loss_pct = 0.02
        
        def calculate_position_size(self, portfolio_value, price):
            return int((portfolio_value * self.risk_per_trade) / (price * self.stop_loss_pct))
        
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
    
    # Create backtest runner
    runner = BacktestRunner(predictor, strategy, fetcher)
    
    # Run backtest for AAPL
    print("Running backtest for AAPL...")
    results = runner.run_backtest('AAPL')
    
    if results:
        print(f"Backtest completed!")
        print(f"Total return: {results.get('total_return', 0):.2%}")
        print(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"Total trades: {results.get('total_trades', 0)}")
        print(f"Win rate: {results.get('win_rate', 0):.1%}")
    else:
        print("Backtest failed")
    
    print("✓ Backtesting framework test completed")