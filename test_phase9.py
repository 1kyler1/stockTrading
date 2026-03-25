#!/usr/bin/env python3
"""
Phase 9 Test: Alerting + CI readiness
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.alerting.alerts import format_alert_message, send_email_alert, send_slack_alert


def test_format_alert_message():
    print('Testing alert text formatting...')
    alerts = ['Drawdown > 20%', 'Win rate < 20%']
    metrics = {'total_pnl': -2500.5, 'win_rate': 0.18}
    text = format_alert_message(alerts, metrics)
    assert 'Drawdown > 20%' in text
    assert 'total_pnl' in text
    print('✓ Format OK')
    return True


def test_send_email_alert_no_server():
    print('Testing email alert fallback behavior...')
    result = send_email_alert(
        subject='Test',
        body='Test alert',
        sender='test@example.com',
        recipient='test@example.com',
        smtp_server='localhost',
        smtp_port=2525
    )
    assert result is False
    print('✓ Email fallback behavior OK')
    return True


def test_send_slack_alert_no_webhook():
    print('Testing Slack alert fallback behavior...')
    result = send_slack_alert('https://hooks.slack.com/services/FAKE/FAKE/FAKE', 'Test')
    assert result is False
    print('✓ Slack fallback behavior OK')
    return True


def main():
    tests = [
        ('Format', test_format_alert_message),
        ('Email fallback', test_send_email_alert_no_server),
        ('Slack fallback', test_send_slack_alert_no_webhook),
    ]

    passed = 0
    for name, fn in tests:
        print('Running', name)
        if fn():
            passed += 1
            print('✓', name, 'PASSED\n')

    print(f'Phase 9 tests passed: {passed}/{len(tests)}')
    return passed == len(tests)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
