import pandas as pd

def generate_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds machine learning features to the dataframe in-place.
    Features include returns, momentum, RSI flags, and SMA relationships.
    """
    df['return_1h'] = df['c'].pct_change(1)
    df['return_6h'] = df['c'].pct_change(6)
    df['momentum'] = df['c'] - df['c'].shift(6)
    df['price_above_sma'] = (df['c'] > df['SMA_short']).astype(int)
    df['sma_ratio'] = df['SMA_short'] / df['SMA_long']
    df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
    df['rsi_overbought'] = (df['RSI'] > 70).astype(int)

    return df

def label_for_ml(df: pd.DataFrame, horizon: int = 36, profit_threshold: float = 0.02) -> pd.DataFrame:
    """
    Label each row with whether a buy at that time would have been profitable.
    
    Adds a new column:
    - 'ml_label': 1 if price increases by profit_threshold in next `horizon` steps, else 0
    """
    future_price = df['c'].shift(-horizon)
    price_change_pct = (future_price - df['c']) / df['c']
    df['ml_label'] = (price_change_pct > profit_threshold).astype(int)
    return df
