import numpy as np
import pandas as pd

# -------------------- Indicator Computation --------------------
def compute_rsi(data, window):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    eps = 1e-12
    rs = gain / (loss + eps)
    rsi = 100 - (100 / (1 + rs))
    # Avoid NaNs at warmup so comparisons don't break downstream
    return rsi.bfill()

def compute_indicators(df, params):
    df['SMA_short'] = df['c'].rolling(window=params['sma_short']).mean()
    df['SMA_long']  = df['c'].rolling(window=params['sma_long']).mean()

    df['bull_regime'] = (df['SMA_short'] > df['SMA_long']).astype(int)

    df['volatility'] = df['c'].pct_change().rolling(5).std()
    denom = (df['volatility'].max() - df['volatility'].min())
    if not np.isfinite(denom) or denom == 0:
        denom = 1.0
    norm_vol = (df['volatility'] - df['volatility'].min()) / denom

    df['rsi_window_dynamic'] = (
        params['rsi_window_min'] + (1 - norm_vol) * (params['rsi_window_max'] - params['rsi_window_min'])
    ).round().fillna(params['rsi_window_max']).astype(int)

    df['RSI'] = np.nan
    for i in range(params['rsi_window_max'], len(df)):
        window = df.loc[i, 'rsi_window_dynamic']
        rsi_series = compute_rsi(df['c'][i - window + 1:i + 1], window=window)
        df.loc[i, 'RSI'] = rsi_series.iloc[-1]

    return df

def generate_and_label_signals(df, params):
    df = generate_signals(df, params)
    df = evaluate_signals(df, params)
    return df

# -------------------- Signal Generation --------------------
def generate_signals(df, params):
    df['Signal'] = 0
    rsi_buy_mask = (df['RSI'] < params['rsi_buy_threshold'])
    buy_mask = (
        (df['RSI'] < params['rsi_buy_threshold']) &
        (df['SMA_short'] > df['SMA_long'])
    )
    strong_buy_mask = (
        (df['RSI'] < params['rsi_strong_buy_threshold']) &
        (df['SMA_short'] > df['SMA_long'])
    )
    sell_mask = (
        (df['RSI'] > params['rsi_sell_threshold']) &
        (df['SMA_short'] < df['SMA_long'])
    )
    strong_sell_mask = (
        (df['RSI'] > params['rsi_strong_sell_threshold']) &
        (df['SMA_short'] < df['SMA_long'])
    )
    if params.get('use_price_change_filter', False):
        df['price_change_pct'] = df['c'].pct_change() * 100
        sell_mask &= df['price_change_pct'] < params.get('sell_pct_threshold', -1)
        strong_sell_mask &= df['price_change_pct'] < params.get('strong_sell_pct_threshold', -3)

    df.loc[rsi_buy_mask, 'Signal'] = 3
    df.loc[buy_mask, 'Signal'] = 1
    df.loc[strong_buy_mask, 'Signal'] = 2
    df.loc[sell_mask, 'Signal'] = -1
    df.loc[strong_sell_mask, 'Signal'] = -2

    print("\nSignal counts:\n", df['Signal'].value_counts())
    return df

def evaluate_signals(df, params):
    horizon = params['signal_eval_horizon']
    df['btc_price_later'] = df['c'].shift(-horizon)
    df['price_change'] = df['btc_price_later'] - df['c']

    def evaluate_trade(signal, change):
        if signal > 0:
            return "correct" if change > 0 else "wrong"
        elif signal < 0:
            return "correct" if change < 0 else "wrong"
        return None

    df['trade_eval'] = df.apply(lambda row: evaluate_trade(row['Signal'], row['price_change']), axis=1)
    return df

