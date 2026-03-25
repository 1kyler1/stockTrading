"""Interactive dashboard for performance visualization and strategy monitoring."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from src.monitoring.monitor import PerformanceMonitor
from src.deployment.manager import DeploymentManager

MODEL_DIR = Path("./models")


def load_backtest_results(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_json(path)
        return df
    except Exception:
        return pd.DataFrame()


def main():
    st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
    st.title("AI Stock Trading - Phase 8 Dashboard")
    
    st.sidebar.header("Status")
    st.sidebar.info("Phase 8: Visualization and Monitoring Dashboard")

    # Deploy manager status
    st.subheader("Deployment Status")
    dm = DeploymentManager()
    st.write("Model dir:", MODEL_DIR)
    status = {
        'model_last_trained': dm.data_fetcher.lookback_days,
        'phase': '8',
        'status': 'ok'
    }
    st.json(status)

    # Load latest backtest result file
    st.subheader("Backtest Performance")
    result_files = sorted((Path("./data/backtest_results").glob("*.json")), reverse=True)
    if result_files:
        chosen_file = st.selectbox("Select backtest result", result_files, format_func=lambda x: x.name)
        data = load_backtest_results(chosen_file)
        if not data.empty:
            st.write(data)

            if "portfolio_values" in data:
                values = data["portfolio_values"]
                if isinstance(values, list):
                    df = pd.DataFrame({'value': values})
                    df['step'] = df.index
                    fig = px.line(df, x='step', y='value', title='Portfolio Value over Time')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data present in selected file")
    else:
        st.warning("No backtest files found in data/backtest_results")

    # Monitoring metrics
    st.subheader("Live Monitor")
    monitor = PerformanceMonitor()
    # Example data (synthetic) to show graphs
    for p in [100000, 100100, 99900, 100250]:
        monitor.add_portfolio_value(pd.Timestamp.now(), p)

    report = monitor.generate_report()
    st.json(report)

    # Trade history
    if report['trade_history_len'] > 0:
        st.write("## Trade history currently not tracked in this UI demo")


if __name__ == '__main__':
    main()