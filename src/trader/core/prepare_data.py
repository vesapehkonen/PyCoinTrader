from .ml_preprocessing import generate_ml_features, label_for_ml
from .signal_generator import generate_and_label_signals, compute_indicators
import pandas as pd
import json

# -------------------- Data Loading --------------------
def load_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["parsed_time"] = pd.to_datetime(df["t"])
    return df

# -------------------- Pipeline Wrappers --------------------
def load_and_prepare_data(input_file, params, apply_labels=True):
    df = load_data(input_file)
    df = compute_indicators(df, params)
    df = generate_and_label_signals(df, params)
    df = generate_ml_features(df)
    if apply_labels:
        df = label_for_ml(df, horizon=params.get("signal_eval_horizon", 36))
    else:
        # Ensure any ML training-only columns are absent during sim
        df = df.drop(columns=['ml_label'], errors='ignore')
    return df
