# plotting_tools.py
from __future__ import annotations
from pathlib import Path
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def norm_reason(r: str) -> str:
    r = (r or "").strip().lower()
    return r.split("=", 1)[0] 

REASON_COLORS = {
    "buy_adaptive_rsi":   "green",
    "buy_signal":         "green",
    "strong_buy_signal":  "blue",
    "ml_preemptive_exit_p":"black",
    "sell_signal":        "red",
    "strong_sell_signal": "red",
    "trailing_stop":      "red",
    "max_hold_candles":   "yellow",
}

from collections import Counter

def count_reasons(trades: list[dict]) -> Counter:
    counts = Counter()
    for t in trades:
        r = norm_reason(t.get("reason", ""))
        if r in REASON_COLORS:           # only your valid reasons
            counts[r] += 1
    return counts

def write_reason_counts(trades: list[dict], output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    counts = count_reasons(trades)

    # include zeros for reasons that didn’t occur
    ordered = {k: int(counts.get(k, 0)) for k in sorted(REASON_COLORS.keys())}

    json_path = out_dir / "reason_counts.json"
    csv_path  = out_dir / "reason_counts.csv"
    txt_path  = out_dir / "reason_counts.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    pd.Series(ordered, name="count").to_csv(csv_path, header=True)

    with txt_path.open("w", encoding="utf-8") as f:
        for k, v in ordered.items():
            f.write(f"{k}: {v}\n")

    return {"reason_counts_json": json_path, "reason_counts_csv": csv_path, "reason_counts_txt": txt_path}

def export_performance_data(df, portfolio_values, trades, input_file, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path(input_file).stem
    df2 = df.copy()
    if "Portfolio" not in df2.columns:
        df2["Portfolio"] = list(portfolio_values)

    payload = {
        "schema": {"version": 1, "time_col": "t", "price_col": "c"},
        "df": df2,
        "trades": trades,
        "input_file": str(input_file),
        "portfolio_values": list(portfolio_values),  # redundancy = safer
    }
    paths = {
        "pickle": out_dir / "data.pkl",
        #"timeseries_csv": out_dir / f"{base}_timeseries.csv",
        #"trades_json": out_dir / f"{base}_trades.json",
        "timeseries_csv": out_dir / f"timeseries.csv",
        "trades_json": out_dir / f"trades.json",
    }

    with paths["pickle"].open("wb") as f:
        pickle.dump(payload, f)

    df2.to_csv(paths["timeseries_csv"], index=False)
    pd.DataFrame(trades).to_json(paths["trades_json"], orient="records")
    return paths

"""
def export_performance_data(df: pd.DataFrame,
                            trades: list[dict],
                            input_file: str | Path,
                            output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path(input_file).stem
    paths = {
        "pickle": out_dir / "data.pkl",
        #"timeseries_csv": out_dir / f"{base}_timeseries.csv",
        #"trades_json": out_dir / f"{base}_trades.json",
        "timeseries_csv": out_dir / f"timeseries.csv",
        "trades_json": out_dir / f"trades.json",
    }

    with paths["pickle"].open("wb") as f:
        pickle.dump({"df": df, "trades": trades, "input_file": str(input_file)}, f)

    df.to_csv(paths["timeseries_csv"], index=False)
    pd.DataFrame(trades).to_json(paths["trades_json"], orient="records")
    return paths
"""

def render_performance_plot(df: pd.DataFrame,
                            portfolio_values: list | pd.Series,
                            trades: list[dict],
                            input_file: str | Path,
                            output_dir: str | Path,
                            time_col: str = "t",
                            price_col: str = "c",
                            dpi: int = 110) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(input_file).stem
    #plot_path = out_dir / f"{base}_plot.png"
    plot_path = out_dir / f"plot.png"

    df = df.copy()
    # ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["Portfolio"] = list(portfolio_values)

    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df["Portfolio"], label="Portfolio Value (USD)", linewidth=0.6)
    plt.plot(df[time_col], df[price_col], label="Bitcoin Price (USD)", linewidth=0.6)

    # scatter trades by valid reasons only
    reasons = sorted({
        r for r in (norm_reason(t.get("reason","")) for t in trades)
        if r in REASON_COLORS
    })
    for reason in reasons:
        color = REASON_COLORS[reason]
        buys  = [t for t in trades if t.get("type") == "buy"  and norm_reason(t.get("reason","")) == reason]
        sells = [t for t in trades if t.get("type") == "sell" and norm_reason(t.get("reason","")) == reason]

        if buys:
            times  = pd.to_datetime([t["parsed_time"] for t in buys])
            prices = [t["price"] for t in buys]
            plt.scatter(times, prices, marker="^", label=f"Buy: {reason}", s=32, zorder=3, color=color)

        if sells:
            times  = pd.to_datetime([t["parsed_time"] for t in sells])
            prices = [t["price"] for t in sells]
            plt.scatter(times, prices, marker="v", label=f"Sell: {reason}", s=32, zorder=3, color=color)

    plt.xlabel("Time")
    plt.ylabel("Value (USD)")
    plt.title("Portfolio vs Bitcoin Price (with Executed Trades)")

    # de-duplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    seen, dh, dl = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            dh.append(h); dl.append(l)
    plt.legend(dh, dl, loc="best")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    plt.close()
    return plot_path

def write_performance_artifacts(df: pd.DataFrame,
                                portfolio_values: list | pd.Series,
                                trades: list[dict],
                                input_file: str | Path,
                                output_dir: str | Path) -> dict[str, Path]:
    """Writes CSV/JSON/pickle + PNG plot to output_dir and returns their paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_paths = export_performance_data(df, portfolio_values, trades, input_file, out_dir)
    plot_path = render_performance_plot(df, portfolio_values, trades, input_file, out_dir)
    counts_paths = write_reason_counts(trades, out_dir)
    return {**data_paths, "plot_png": plot_path, **counts_paths}
