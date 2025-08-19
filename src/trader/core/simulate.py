import sys
import json
from datetime import datetime
import joblib
from pathlib import Path

from .prepare_data import load_and_prepare_data
from .simulate_trading import simulate_trading
from trader.viz.analysis_tools import analyze_performance, save_logs_and_summary, analyze_by_regime_segments
from trader.viz.plotting_tools import write_performance_artifacts
from .ml_feature_config import ml_features
from trader.utils.paths import project_root, models_dir, config_dir, make_run_dir

REQUIRED_KEYS = [
    "sma_short", "sma_long", "rsi_window_min", "rsi_window_max", "rsi_buy_threshold", "rsi_strong_buy_threshold",
    "rsi_sell_threshold", "rsi_strong_sell_threshold", "signal_eval_horizon", "ml_buy", "ml_sell"
]

def simulate(input_file):
    run_dir, run_id, started_at = make_run_dir(label="sim")
    
    with open(config_dir() / 'config.json') as f:
        params = json.load(f)

    buy_model  = joblib.load(models_dir() / params["ml_buy"]["model_path"])
    sell_model = joblib.load(models_dir() / params["ml_sell"]["model_path"])

    missing = [k for k in REQUIRED_KEYS if k not in params]
    if missing:
        print(f"[WARN] Missing config keys (using defaults where applicable): {missing}")

    print(f"Start processing: {datetime.now()}")
    df = load_and_prepare_data(input_file, params, apply_labels=False)

    if hasattr(sell_model, "feature_names_in_"):
        diff = set(sell_model.feature_names_in_) ^ set(ml_features)
        if diff:
            print("[WARN] Sell model feature mismatch:", diff)
            exit(1)

    print(f"Simulating trading: {datetime.now()}")
    trades, portfolio_values, trade_logs, bitcoins = simulate_trading(df, params, buy_model, sell_model, ml_features, ml_features, run_dir)

    print(f"Analyzing performance: {datetime.now()}")
    metrics = analyze_performance(df, trades, portfolio_values, df['c'].iloc[0])

    # NEW: bull vs bear breakdown
    if 'bull_regime' in df.columns:
        seg_stats = analyze_by_regime_segments(df, portfolio_values)
        print("\n=== Regime Segment Stats ===")
        for regime, stats in seg_stats.items():
            print(f"{regime.capitalize()} regime:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    print(f"Plotting: {datetime.now()}")
    save_logs_and_summary(df, trades, trade_logs, portfolio_values, bitcoins, input_file, metrics, run_dir)

    paths = write_performance_artifacts(
        df, portfolio_values, trades, input_file, output_dir=run_dir
    )
    print("Wrote:", paths)
    print(f"End: {datetime.now()}")
