import os
import json
import numpy as np
import pandas as pd

def analyze_performance(df, trades, portfolio_values, starting_price):
    # --- keep all your existing computations here ---
    metrics = {}  # whatever you currently fill

    # --- NEW: extra metrics; doesn't touch your existing keys ---
    rets = pd.Series(portfolio_values).pct_change().dropna()
    if len(rets) > 0:
        # Adjust annualization factor if hourly bars, e.g., 24*365
        ann_factor = 365
        mean = float(rets.mean())
        std = float(rets.std()) if rets.std() is not None else 0.0
        downside = float(rets[rets < 0].std()) if (rets < 0).any() else 0.0

        sharpe = (mean / (std + 1e-12)) * np.sqrt(ann_factor)
        sortino = (mean / (downside + 1e-12)) * np.sqrt(ann_factor)

        try:
            cagr = (portfolio_values[-1] / portfolio_values[0]) ** (ann_factor / max(len(rets), 1)) - 1
        except Exception:
            cagr = np.nan

        metrics.update({
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "cagr": round(cagr, 4) if np.isfinite(cagr) else None,
        })
    else:
        metrics.update({"sharpe": None, "sortino": None, "cagr": None})

    return metrics

def save_logs_and_summary(df, trades, trade_logs, portfolio_values, bitcoins, input_file, metrics, output_dir="output", horizon=36):
    os.makedirs(output_dir, exist_ok=True)

    #log_path = f"{output_dir}/{os.path.splitext(os.path.basename(input_file))[0]}_trade_analysis_log.json"
    log_path = f"{output_dir}/trade_analysis_log.json"
    with open(log_path, "w") as f:
        for log in trade_logs:
            if isinstance(log.get("entry_time"), pd.Timestamp):
                log["entry_time"] = log["entry_time"].isoformat()
            if isinstance(log.get("exit_time"), pd.Timestamp):
                log["exit_time"] = log["exit_time"].isoformat()
        json.dump(trade_logs, f, indent=2)

    summary_file = f"{output_dir}/results_summary.json"
    summary_entry = {
        'input_file': input_file,
        'initial_cash': df['c'].iloc[0],
        'final_portfolio_value': round(portfolio_values[-1], 2),
        'initial_btc_price': round(df['c'].iloc[0], 2),
        'final_btc_price': round(df['c'].iloc[-1], 2),
        'final_bitcoins_held': round(bitcoins, 8),
        'trades_executed': int(df['Signal'].abs().sum()),
        'portfolio_vs_btc_diff_pct': round((portfolio_values[-1] / df['c'].iloc[-1] - 1) * 100, 2),
        "signal_horizon_hours": horizon
    }
    summary_entry.update(metrics)

    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            all_summaries = json.load(f)
    else:
        all_summaries = []

    all_summaries.append(summary_entry)
    with open(summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)

def analyze_by_regime_segments(df: pd.DataFrame, portfolio_values: list):
    """Break portfolio into contiguous bull/bear stretches and report per regime."""
    results = {"bull": [], "bear": []}
    pv_series = pd.Series(portfolio_values, index=df.index)

    current_regime = df['bull_regime'].iloc[0]
    seg_start_idx = df.index[0]

    for idx in range(1, len(df)):
        regime_now = df['bull_regime'].iloc[idx]
        if regime_now != current_regime:
            # Segment ended
            seg_mask = (df.index >= seg_start_idx) & (df.index < df.index[idx])
            pv_seg = pv_series[seg_mask]
            if len(pv_seg) >= 2:
                start_val = pv_seg.iloc[0]
                end_val = pv_seg.iloc[-1]
                change = (end_val / start_val - 1) * 100
                label = "bull" if current_regime == 1 else "bear"
                results[label].append(change)

            # Start new segment
            seg_start_idx = df.index[idx]
            current_regime = regime_now

    # Handle final segment
    seg_mask = (df.index >= seg_start_idx)
    pv_seg = pv_series[seg_mask]
    if len(pv_seg) >= 2:
        start_val = pv_seg.iloc[0]
        end_val = pv_seg.iloc[-1]
        change = (end_val / start_val - 1) * 100
        label = "bull" if current_regime == 1 else "bear"
        results[label].append(change)

    # Summarize
    summary = {}
    for label in ["bull", "bear"]:
        if results[label]:
            summary[label] = {
                "segments": len(results[label]),
                "avg_pct_change": round(sum(results[label]) / len(results[label]), 2),
                "max_gain": round(max(results[label]), 2),
                "max_loss": round(min(results[label]), 2)
            }
        else:
            summary[label] = {"segments": 0}
    return summary
