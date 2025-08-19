import sys
import pandas as pd
import joblib
import os
import json

from trader.utils.paths import project_root, models_dir, config_dir, make_run_dir

from .ml_model import train_ml_model
from .prepare_data import load_and_prepare_data
from .ml_feature_config import ml_features

with open(config_dir() / 'config.json') as f:
    params = json.load(f)

INPUT_FILE = project_root() / "data/inputdata/train_data.json"
MODEL_OUTPUT_PATH = models_dir() / params['ml_buy']['model_path']

def train(input_file):
    df = load_and_prepare_data(input_file, params, apply_labels=True)

    print("Signal counts:\n", df["Signal"].value_counts())
    print("DF length after prepare_df:", len(df))
    print("Date range:", df["parsed_time"].min(), "→", df["parsed_time"].max())
    
    model, feature_list = train_ml_model(df, features=ml_features)
    print("[6] Saving model to:", MODEL_OUTPUT_PATH)
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("✅ Done. Model saved.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = INPUT_FILE
    train(input_file)
