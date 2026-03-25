#!/usr/bin/env python3
"""
Phase 6 Test: Monitoring & Iteration
Test performance monitor and reporting.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from src.monitoring.monitor import PerformanceMonitor


def test_performance_monitor_metrics():
    print("Testing PerformanceMonitor metrics computation...")
    monitor = PerformanceMonitor()

    # Create mock trades
    now = datetime.now()
    trades = [
        {'symbol': 'AAPL', 'pnl': 120.0, 'exit_time': now},
        {'symbol': 'AAPL', 'pnl': -30.0, 'exit_time': now + timedelta(minutes=1)},
        {'symbol': 'MSFT', 'pnl': 50.0, 'exit_time': now + timedelta(minutes=2)},
    ]

    for t in trades:
        monitor.add_trade(t)

    # Create portfolio values
    values = [100000.0, 100120.0, 100090.0, 100140.0]
    for i, v in enumerate(values):
        monitor.add_portfolio_value(now + timedelta(days=i), v)

    metrics = monitor.get_metrics()
    report = monitor.generate_report()

    assert metrics.total_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert abs(metrics.win_rate - (2 / 3)) < 1e-6
    assert abs(metrics.total_pnl - 140.0) < 1e-6
    assert report['trade_history_len'] == 3

    print("✓ Performance metrics computed correctly")
    print(report)
    return True


def test_alert_generation():
    print("Testing PerformanceMonitor alert generation...")
    monitor = PerformanceMonitor()

    # Insert poor performance > threshold
    now = datetime.now()
    monitor.add_trade({'symbol': 'AAPL', 'pnl': -5000.0, 'exit_time': now})
    monitor.add_portfolio_value(now, 100000)
    monitor.add_portfolio_value(now + timedelta(days=1), 90000)

    metrics = monitor.get_metrics()
    alerts = monitor.check_alerts(loss_threshold=-1000.0, max_drawdown_threshold=0.05)

    assert 'Total P&L below threshold' in alerts[0]
    assert any('Max drawdown' in a for a in alerts)

    print("✓ Alerts generated correctly")
    print(alerts)
    return True


def main():
    tests = [
        ("Performance metrics", test_performance_monitor_metrics),
        ("Alert generation", test_alert_generation),
    ]

    passed = 0
    for name, func in tests:
        print(f"Running: {name}")
        if func():
            passed += 1
            print(f"✓ {name} PASSED\n")
        else:
            print(f"✗ {name} FAILED\n")

    print(f"Phase 6 tests passed: {passed}/{len(tests)}")

    return passed == len(tests)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
