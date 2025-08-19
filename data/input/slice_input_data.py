import sys
import pandas as pd
from datetime import datetime

def slice_input_file(input_file, output_file, start_date=None, end_date=None):
    df = pd.read_json(input_file)
    df['parsed_time'] = pd.to_datetime(df['t'])

    if start_date:
        try:
            start = pd.to_datetime(start_date).tz_localize('UTC')
            df = df[df['parsed_time'] >= start]
        except Exception as e:
            print(f"[!] Invalid start_date: {start_date} → {e}")

    if end_date:
        try:
            end = pd.to_datetime(end_date).tz_localize('UTC')
            df = df[df['parsed_time'] <= end]
        except Exception as e:
            print(f"[!] Invalid end_date: {end_date} → {e}")

    df.to_json(output_file, orient='records', indent=2)
    print(f"✅ Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python slice_input_data.py <input_file> <output_file> [start_date] [end_date]")
        print("       python slice_input_data.py input_data.json output_data.json 2021-01-01 2024-07-01")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    start_date = sys.argv[3] if len(sys.argv) > 3 else None
    end_date = sys.argv[4] if len(sys.argv) > 4 else None

    slice_input_file(input_file, output_file, start_date, end_date)
