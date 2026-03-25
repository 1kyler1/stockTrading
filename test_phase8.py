#!/usr/bin/env python3
"""
Phase 8 Test: Visualization & Monitoring Dashboard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.dashboard import main


def test_dashboard_import():
    print('Testing dashboard import...')
    assert callable(main)
    print('✓ dashboard.main is callable')
    return True


def test_status_data():
    print('Testing deployment status endpoint data format...')
    from src.deployment.manager import StatusHandler
    data = StatusHandler.status_data
    assert 'phase' in data
    assert data['phase'] == '7' or data['phase'] == '8'
    assert 'status' in data
    print('✓ status_data keys verified')
    return True


def main_test():
    tests = [
        ('Dashboard import', test_dashboard_import),
        ('Status data', test_status_data),
    ]

    passed = 0
    for name, func in tests:
        print('Running', name)
        if func():
            passed += 1
            print('✓', name, 'PASSED\n')
        else:
            print('✗', name, 'FAILED\n')
    print(f'Phase 8 tests passed: {passed}/{len(tests)}')
    return passed == len(tests)


if __name__ == '__main__':
    success = main_test()
    sys.exit(0 if success else 1)