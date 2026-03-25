"""Deployment & Continuous Improvement

Phase 7: Provide retraining scheduler, model health checks, and lightweight status API.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Optional

from src.config import STOCKS, MODEL_DIR, DATA_CONFIG
from src.data.fetcher import DataFetcher
from src.models.lstm_predictor import LSTMPredictor
from src.strategy.engine import TradingStrategy

logger = logging.getLogger(__name__)


class StatusHandler(BaseHTTPRequestHandler):
    status_data = {
        'model_last_trained': None,
        'retrain_count': 0,
        'last_retrain_result': None,
        'phase': '7',
        'status': 'ok'
    }

    def do_GET(self):
        if self.path != '/status':
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
            return

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(StatusHandler.status_data).encode('utf-8'))

    def log_message(self, format: str, *args):
        return  # Silence default http server logging


class DeploymentManager:
    """Manage live deployment operations and scheduled retraining."""

    def __init__(self,
                 data_fetcher: Optional[DataFetcher] = None,
                 predictor: Optional[LSTMPredictor] = None,
                 strategy: Optional[TradingStrategy] = None):
        self.data_fetcher = data_fetcher or DataFetcher()
        self.predictor = predictor
        self.strategy = strategy
        self.retrain_lock = threading.Lock()
        self.retrain_thread = None
        self.server_thread = None

        self.model_dir = Path(MODEL_DIR)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.model_path = self.model_dir / 'lstm_model.pth'

        logger.info('DeploymentManager initialized')

    def _rotate_model(self):
        if self.model_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive = self.model_dir / f'lstm_model_{timestamp}.pth'
            os.replace(self.model_path, archive)
            logger.info(f'Rotated model file to {archive}')

    def retrain_model(self,
                      symbol: Optional[str] = None,
                      epochs: int = 5,
                      batch_size: int = 32,
                      learning_rate: float = 0.001) -> Dict[str, any]:
        """Retrain the model using historical data and persist model weights."""
        with self.retrain_lock:
            symbol = symbol or STOCKS[0]
            try:
                logger.info(f'Starting retrain for {symbol} (epochs={epochs})')
                X, y, _, _ = self.data_fetcher.get_processed_data(symbol)

                if len(X) < 1:
                    raise ValueError('No training data available')

                n = len(X)
                train_end = int(0.7 * n)
                val_end = int(0.85 * n)

                X_train = X[:train_end]
                y_train = y[:train_end]
                X_val = X[train_end:val_end]
                y_val = y[train_end:val_end]

                if self.predictor is None:
                    input_size = X.shape[2]
                    self.predictor = LSTMPredictor(input_size=input_size)
                
                if self.strategy is None:
                    self.strategy = TradingStrategy(self.predictor, self.data_fetcher)

                history = self.predictor.train(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )

                self._rotate_model()
                self.predictor.save_model(self.model_path)

                score = self.predictor.evaluate(X_val, y_val)
                status = {
                    'retrain_time': datetime.now().isoformat(),
                    'symbol': symbol,
                    'epochs': epochs,
                    'history': history,
                    'validation_metrics': score,
                }

                StatusHandler.status_data['model_last_trained'] = status['retrain_time']
                StatusHandler.status_data['retrain_count'] += 1
                StatusHandler.status_data['last_retrain_result'] = score

                logger.info('Retrain complete')
                return status

            except Exception as e:
                logger.error(f'Retraining failed: {e}')
                return {'error': str(e)}

    def schedule_retrain(self,
                         interval_hours: int = 24,
                         symbol: Optional[str] = None,
                         epochs: int = 5,
                         batch_size: int = 32,
                         learning_rate: float = 0.001):
        """Schedule retraining repeatedly every n hours."""
        def _job():
            logger.info('Retrain job triggered')
            self.retrain_model(symbol, epochs, batch_size, learning_rate)
            if self.retrain_thread is not None:
                self.retrain_thread = threading.Timer(interval_hours * 3600, _job)
                self.retrain_thread.daemon = True
                self.retrain_thread.start()

        self.retrain_thread = threading.Timer(0, _job)  # Immediate first run
        self.retrain_thread.daemon = True
        self.retrain_thread.start()
        logger.info(f'Scheduled retrain every {interval_hours} hours')

    def stop_retrain(self):
        if self.retrain_thread is not None:
            self.retrain_thread.cancel()
            self.retrain_thread = None
            logger.info('Retrain scheduler stopped')

    def start_status_server(self, host: str = '0.0.0.0', port: int = 9000):
        """Start a minimal HTTP server for status endpoint (/status)."""
        def _run():
            server = HTTPServer((host, port), StatusHandler)
            logger.info(f'Status server running at http://{host}:{port}/status')
            try:
                server.serve_forever()
            except Exception as e:
                logger.error(f'Status server stopped: {e}')

        self.server_thread = threading.Thread(target=_run, daemon=True)
        self.server_thread.start()

    def stop_status_server(self):
        # no direct stop in base server; OK for demo mode
        self.server_thread = None
        logger.info('Status server stopped (thread will terminate on process exit)')


if __name__ == '__main__':
    dm = DeploymentManager()
    dm.start_status_server(port=9000)
    dm.schedule_retrain(interval_hours=1, epochs=2)

    print('Deployment manager running. Check /status endpoint. Press Ctrl+C to exit.')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dm.stop_retrain()
        print('Stopped')
