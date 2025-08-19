# train_sell_model.py
import sys, os, json, joblib
import pandas as pd

from trader.utils.paths import project_root, models_dir, config_dir, make_run_dir

from .prepare_data import load_and_prepare_data
from .ml_model import train_ml_model
from .ml_feature_config import ml_features  # reuse same features

with open(config_dir() / 'config.json') as f:
    params = json.load(f)

INPUT_FILE = project_root() / "data/inputdata/train_data.json"
SELL_CFG = params.get("ml_sell", {}) or {}
MODEL_OUTPUT_PATH = models_dir() / params['ml_sell']['model_path']

def make_sell_labels(df: pd.DataFrame, sell_h: int, drop_pct: float) -> pd.DataFrame:
    """
    Label 1 if the worst forward drawdown over next H bars <= -drop_pct.
    This captures 'should exit soon' risk.
    """
    # min future close over next H bars (shift(-1) so next bar is the first look-ahead)
    min_fwd = df['c'].shift(-1).rolling(sell_h).min()
    fwd_drawdown = (min_fwd / df['c']) - 1.0
    df = df.copy()
    df['ml_sell_label'] = (fwd_drawdown <= -float(drop_pct)).astype(int)
    return df

def train(input_file):
    # prepare df with indicators/features; DO NOT include buy labels during sim, but for training it's fine
    df = load_and_prepare_data(input_file, params, apply_labels=True)

    # build sell labels (separate from your buy 'ml_label')
    sell_h = int(SELL_CFG.get("sell_eval_horizon", params.get("signal_eval_horizon", 36)))
    drop_pct = float(SELL_CFG.get("sell_drop_threshold", 0.03))
    df = make_sell_labels(df, sell_h, drop_pct)

    # (optional) drop buy label so we don't accidentally use it
    df = df.drop(columns=['ml_label'], errors='ignore')

    # sanity prints
    print("Sell label counts:\n", df["ml_sell_label"].value_counts(dropna=False))
    print("DF length after prepare_df:", len(df))
    print("Date range:", df["parsed_time"].min(), "→", df["parsed_time"].max())

    # train model (same trainer; just pass label_column)
    model, feature_list = train_ml_model(df, features=ml_features, label_column='ml_sell_label')

    # save
    print("[sell] Saving model to:", MODEL_OUTPUT_PATH)
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("✅ Done. Sell model saved.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = INPUT_FILE
    train(input_file)

