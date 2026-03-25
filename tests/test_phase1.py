#!/usr/bin/env python3
"""
Simple test script for Phase 1: DataFetcher functionality.
Tests: fetching data, adding indicators, normalization, and windowing.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.config import STOCKS

def test_data_fetcher():
    """Test DataFetcher functionality."""
    print("=" * 60)
    print("Phase 1 Test: DataFetcher")
    print("=" * 60)
    
    fetcher = DataFetcher()
    
    # Test 1: Fetch historical data
    print("\n[Test 1] Fetching historical data for AAPL...")
    try:
        df = fetcher.get_historical_data("AAPL", days=100)
        print(f"✓ Downloaded {len(df)} rows of data")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Sample close prices:\n{df['Close'].head()}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 2: Add technical indicators
    print("\n[Test 2] Adding technical indicators...")
    try:
        df_indicators = fetcher.add_technical_indicators(df)
        print(f"✓ Added indicators successfully")
        print(f"  Data shape after indicators: {df_indicators.shape}")
        print(f"  New columns: {[c for c in df_indicators.columns if c not in df.columns]}")
        print(f"  Sample indicator values:\n{df_indicators[['RSI', 'MACD', 'SMA_20']].tail(5)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 3: Normalize data
    print("\n[Test 3] Normalizing data...")
    try:
        df_normalized, scaler = fetcher.normalize_data(df_indicators)
        print(f"✓ Data normalized successfully")
        print(f"  Normalized Close range: [{df_normalized['Close'].min():.3f}, {df_normalized['Close'].max():.3f}]")
        print(f"  Normalized RSI range: [{df_normalized['RSI'].min():.3f}, {df_normalized['RSI'].max():.3f}]")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 4: Create windowed data
    print("\n[Test 4] Creating windowed data for LSTM...")
    try:
        X, y = fetcher.prepare_windowed_data(df_normalized)
        print(f"✓ Created windowed data successfully")
        print(f"  X shape: {X.shape} (samples, window_size, features)")
        print(f"  y shape: {y.shape} (samples,)")
        print(f"  y values (direction): {np.unique(y, return_counts=True)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 5: Full pipeline (get_processed_data)
    print("\n[Test 5] Full pipeline: get_processed_data()...")
    try:
        X, y, df_norm, scaler = fetcher.get_processed_data("AAPL")
        print(f"✓ Full pipeline completed successfully")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Number of features in input window: {X.shape[2]}")
        print(f"  Class distribution (0=down, 1=up): {np.unique(y, return_counts=True)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All Phase 1 tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    import numpy as np
    success = test_data_fetcher()
    sys.exit(0 if success else 1)
