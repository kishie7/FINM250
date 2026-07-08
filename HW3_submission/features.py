"""Shared feature-engineering pipeline.

Used by both hw3_backtest.py (historical backtest) and
hw3_paper_trading.py (live signal) so the two use the same
"""
import numpy as np
import pandas as pd

feature_cols = [
    "SMA_short",
    "SMA_long",
    "MACD",
    "MACD_signal",
    "ATR",
    "ADX",
    "RSI",
    "WILLIAMS_R",
    "BB_upper",
    "BB_lower",
    "CMF",
    "log_return",
    "rolling_mean_10",
    "rolling_std_10",
    "rolling_mean_20",
    "rolling_std_20",
]


def compute_features(bars_df):
    """Compute all technical-indicator feature columns on a raw OHLCV
    DataFrame (needs timestamp/open/high/low/close/volume columns)."""
    df = bars_df.copy()

    df['SMA_short'] = df['close'].rolling(window=20).mean()
    df['SMA_long'] = df['close'].rolling(window=200).mean()

    df['EMA_short'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=200, adjust=False).mean()

    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ADX (14-period) -- also saves ATR as its own column
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)

    df['ATR'] = tr.ewm(span=14, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / df['ATR'])
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / df['ATR'])
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df['ADX'] = dx.ewm(span=14, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)  # guard divide-by-zero on flat/no-loss windows
    df['RSI'] = 100 - (100 / (1 + rs))
    df.loc[loss == 0, 'RSI'] = np.where(gain[loss == 0] > 0, 100, 50)

    high_14 = df['high'].rolling(window=14).max()
    low_14 = df['low'].rolling(window=14).min()
    range_14 = (high_14 - low_14).replace(0, np.nan)
    df['WILLIAMS_R'] = -100 * (high_14 - df['close']) / range_14

    df['BB_mid'] = df['SMA_short']
    df['BB_upper'] = df['SMA_short'] + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['SMA_short'] - 2 * df['close'].rolling(window=20).std()

    hl_range = (df['high'] - df['low']).replace(0, np.nan)
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_range
    mf_volume = mf_multiplier * df['volume']
    df['CMF'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

    df['market_return'] = df['close'].pct_change()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["rolling_mean_10"] = df["close"].rolling(window=10).mean()
    df["rolling_std_10"] = df["close"].rolling(window=10).std()

    df["rolling_mean_20"] = df["close"].rolling(window=20).mean()
    df["rolling_std_20"] = df["close"].rolling(window=20).std()

    return df
