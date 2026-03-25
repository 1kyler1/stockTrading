"""
Monitoring & Reporting: Performance analytics, alerts, and iterative metrics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    daily_return_mean: float = 0.0
    daily_return_std: float = 0.0
    trade_history: List[Dict[str, Any]] = field(default_factory=list)


class PerformanceMonitor:
    """Compute and report performance metrics for trades/portfolio."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_series: List[Dict[str, Any]] = []
        self.metrics = PerformanceMetrics()

    def add_trade(self, trade: Dict[str, Any]):
        """Record a completed trade. trade should include 'pnl' and 'exit_time'."""
        self.trades.append(trade)

    def add_portfolio_value(self, timestamp: datetime, value: float):
        self.portfolio_series.append({'timestamp': timestamp, 'value': value})

    def update_metrics(self):
        self._compute_trade_metrics()
        self._compute_time_series_metrics()

    def _compute_trade_metrics(self):
        if not self.trades:
            self.metrics = PerformanceMetrics(trade_history=[])
            return

        total_pnl = sum(t.get('pnl', 0.0) for t in self.trades)
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0.0) > 0)
        losing_trades = total_trades - winning_trades
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        self.metrics.total_trades = total_trades
        self.metrics.winning_trades = winning_trades
        self.metrics.losing_trades = losing_trades
        self.metrics.win_rate = win_rate
        self.metrics.total_pnl = total_pnl
        self.metrics.avg_pnl = avg_pnl
        self.metrics.trade_history = self.trades.copy()

    def _compute_time_series_metrics(self):
        if len(self.portfolio_series) < 2:
            return

        # Sort by timestamp
        series = sorted(self.portfolio_series, key=lambda x: x['timestamp'])
        values = np.array([point['value'] for point in series], dtype=float)

        # Daily returns (assuming continuous period, can be intraday)
        returns = np.diff(values) / values[:-1]
        self.metrics.daily_return_mean = float(np.mean(returns)) if len(returns) > 0 else 0.0
        self.metrics.daily_return_std = float(np.std(returns)) if len(returns) > 0 else 0.0
        self.metrics.sharpe_ratio = float((np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0)

        # Max drawdown
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        self.metrics.max_drawdown = float(max_dd)

    def get_metrics(self) -> PerformanceMetrics:
        self.update_metrics()
        return self.metrics

    def generate_report(self) -> Dict[str, Any]:
        metrics = self.get_metrics()
        report = {
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': metrics.win_rate,
            'total_pnl': metrics.total_pnl,
            'avg_pnl': metrics.avg_pnl,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'daily_return_mean': metrics.daily_return_mean,
            'daily_return_std': metrics.daily_return_std,
            'trade_history_len': len(metrics.trade_history)
        }
        logger.info(f"Performance report: {report}")
        return report

    def check_alerts(self,
                     drawdown_threshold: float = 0.2,
                     loss_threshold: float = -1000.0,
                     max_drawdown_threshold: float = 0.15) -> List[str]:
        """Generate alert messages for risky conditions."""
        alerts = []
        if self.metrics.total_pnl < loss_threshold:
            alerts.append(f"Total P&L below threshold: {self.metrics.total_pnl:.2f}")
        if self.metrics.max_drawdown > max_drawdown_threshold:
            alerts.append(f"Max drawdown exceeds threshold: {self.metrics.max_drawdown:.2%}")
        if self.metrics.win_rate < 0.3 and self.metrics.total_trades > 10:
            alerts.append(f"Low win rate: {self.metrics.win_rate:.2%}")

        for a in alerts:
            logger.warning(f"Performance alert: {a}")

        return alerts
