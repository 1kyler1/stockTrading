#!/usr/bin/env python3
"""
Phase 2 Test: LSTM Model Development and Training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.models.lstm_predictor import LSTMPredictor
from src.config import STOCKS
import numpy as np
from sklearn.model_selection import train_test_split

def test_lstm_model():
    """Test LSTM model training and evaluation."""
    print("=" * 60)
    print("Phase 2 Test: LSTM Model Development")
    print("=" * 60)

    # Test 1: Data preparation
    print(f"\n[Test 1] Preparing data from all {len(STOCKS)} stocks...")
    try:
        fetcher = DataFetcher()
        all_X, all_y = [], []
        for symbol in STOCKS:
            try:
                X_s, y_s, _, _ = fetcher.get_processed_data(symbol)
                if len(X_s) > 0:
                    all_X.append(X_s)
                    all_y.append(y_s)
                    print(f"  ✓ {symbol}: {len(X_s)} samples")
                else:
                    print(f"  ⚠ {symbol}: no samples, skipping")
            except Exception as e:
                print(f"  ⚠ {symbol}: error ({e}), skipping")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        # Shuffle so stocks are mixed throughout train/val/test
        shuffle_idx = np.random.permutation(len(X))
        X, y = X[shuffle_idx], y[shuffle_idx]

        print(f"✓ Data prepared: X shape = {X.shape}, y shape = {y.shape}")
        print(f"  Class balance - Down: {(y==0).sum()}, Up: {(y==1).sum()}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 2: Train/val/test split
    print("\n[Test 2] Splitting data (70/15/15)...")
    try:
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"✓ Data split successfully")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 3: Initialize LSTM model
    print("\n[Test 3] Initializing LSTM model...")
    try:
        input_size = X.shape[2]  # Number of features
        predictor = LSTMPredictor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
        print(f"✓ LSTM model initialized")
        print(f"  Input size: {input_size}, Hidden size: 128, Layers: 2")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 4: Train model (short training for test)
    print("\n[Test 4] Training LSTM model (20 epochs)...")
    try:
        history = predictor.train(
            X_train, y_train, 
            X_val, y_val,
            epochs=20,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=5
        )
        print(f"✓ Training completed")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Make predictions
    print("\n[Test 5] Making predictions...")
    try:
        predictions = predictor.predict(X_test[:20])
        probabilities = predictor.predict_probability(X_test[:20])
        print(f"✓ Predictions made successfully")
        print(f"  Sample predictions: {predictions[:5]}")
        print(f"  Sample probabilities: {probabilities[:5]}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 6: Evaluate model
    print("\n[Test 6] Evaluating model on test set...")
    try:
        metrics = predictor.evaluate(X_test, y_test)
        print(f"✓ Model evaluated")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 7: Save and load model
    print("\n[Test 7] Saving and loading model...")
    try:
        predictor.save_model("test_model.pt")
        
        # Create new predictor and load
        predictor2 = LSTMPredictor(input_size=input_size)
        predictor2.load_model("test_model.pt")
        
        # Make predictions with loaded model
        predictions2 = predictor2.predict(X_test[:5])
        print(f"✓ Model saved and loaded successfully")
        print(f"  Predictions match: {np.allclose(predictions[:5], predictions2)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All Phase 2 tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_lstm_model()
    sys.exit(0 if success else 1)
