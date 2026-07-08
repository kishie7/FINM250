# Imports
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from alpaca.data.enums import Adjustment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Allow for API and secret keys file read
from dotenv import load_dotenv
import os
import joblib

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if API_KEY is None or SECRET_KEY is None:
    raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in environment (.env)")

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

INITIAL_CAPITAL = 100_000
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # set to an annualized rate if you want excess-return Sharpe/Sortino

allowed_tickers = ["AAPL", "MSFT", "SPY", "QQQ", "NVDA"]

while True:
    user_input = input("Enter a ticker symbol: ").strip().upper()
    if user_input in allowed_tickers:
        print(f"Ticker {user_input} selected successfully!")
        ticker = user_input
        break
    else:
        print("Invalid ticker. Please try again.")

request = StockBarsRequest(
    symbol_or_symbols=ticker,
    timeframe=TimeFrame.Day,
    start=datetime(2021, 6, 1),
    end=datetime(2026, 6, 1),
    adjustment=Adjustment.ALL,
)
bars = client.get_stock_bars(request).df
df = bars.reset_index()

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
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
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
df['ADX'] = dx.ewm(span=14, adjust=False).mean()

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss.replace(0, np.nan)  # guard divide-by-zero on flat/no-loss windows
df['RSI'] = 100 - (100 / (1 + rs))

high_14 = df['high'].rolling(window=14).max()
low_14 = df['low'].rolling(window=14).min()
range_14 = (high_14 - low_14).replace(0, np.nan)  # guard divide-by-zero
df['WILLIAMS_R'] = -100 * (high_14 - df['close']) / range_14

df['BB_mid'] = df['SMA_short']
df['BB_upper'] = df['SMA_short'] + 2 * df['close'].rolling(window=20).std()
df['BB_lower'] = df['SMA_short'] - 2 * df['close'].rolling(window=20).std()

hl_range = (df['high'] - df['low']).replace(0, np.nan)  # guard divide-by-zero
mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_range
mf_volume = mf_multiplier * df['volume']
df['CMF'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

df['market_return'] = df['close'].pct_change()

df["log_return"] = np.log(df["close"] / df["close"].shift(1))

df["rolling_mean_10"] = df["close"].rolling(window=10).mean()
df["rolling_std_10"] = df["close"].rolling(window=10).std()

df["rolling_mean_20"] = df["close"].rolling(window=20).mean()
df["rolling_std_20"] = df["close"].rolling(window=20).std()

# Replace any inf produced by the ratio guards above, then drop NaNs for ML rows
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Target is binary: 1 if tomorrow's return is positive, else 0
df["target"] = (df["market_return"].shift(-1) > 0).astype(int)

ml_df = df.dropna(subset=feature_cols + ["target"]).copy()

X = ml_df[feature_cols]
y = ml_df["target"]

# ---------------------------------------------------------------------------
# Chronological train/test split -- BEFORE any fitting, to avoid leakage
# ---------------------------------------------------------------------------
split = int(len(ml_df) * 0.8)

X_train_raw, X_test_raw = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Fit scaler and PCA on TRAIN ONLY, then transform train and test separately
scaler = StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(n_components=0.80).fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Number of PCA components:", pca.n_components_)
print("Explained variance (train-fitted):", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

# Train the ML Model: Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)

train_acc = model.score(X_train_pca, y_train)
test_acc = model.score(X_test_pca, y_test)
print(f"Train accuracy: {train_acc:.3f} | Test (out-of-sample) accuracy: {test_acc:.3f}")

# Predict probabilities on train and test separately (no leakage), then
# stitch back together only for bookkeeping/plotting purposes.
train_probs = model.predict_proba(X_train_pca)[:, 1]
test_probs = model.predict_proba(X_test_pca)[:, 1]

ml_df["probability"] = np.concatenate([train_probs, test_probs])
ml_df["signal_ml"] = np.where(ml_df["probability"] > 0.60, 1, 0)
ml_df["position_ml"] = ml_df["signal_ml"]
ml_df["is_test"] = np.concatenate([
    np.zeros(len(train_probs), dtype=bool),
    np.ones(len(test_probs), dtype=bool),
])

df["signal_ml"] = 0
df["position_ml"] = 0
df["probability"] = np.nan

df.loc[ml_df.index, "signal_ml"] = ml_df["signal_ml"]
df.loc[ml_df.index, "position_ml"] = ml_df["position_ml"]
df.loc[ml_df.index, "probability"] = ml_df["probability"]

# Index (in df) where the out-of-sample test period begins -- used to slice
# the backtest so the performance table reflects genuine OOS behavior.
test_start_idx = ml_df.index[split]


def run_backtest(price_df, position_col, initial_capital=INITIAL_CAPITAL):
    """
    Long-only, no-leverage, no-shorting backtest.

    Parameters
    ----------
    price_df : DataFrame with 'timestamp', 'close', 'market_return', position_col
    position_col : name of the column holding target position (0 or 1)
    initial_capital : starting portfolio value in dollars

    Returns
    -------
    result_df : DataFrame with portfolio metrics: portfolio_value, strategy_return, drawdown, pnl
    trades : list of dicts, one per executed trade (entry/exit/pnl)
    """
    result_df = price_df[['timestamp', 'close', 'market_return', position_col]].copy()
    result_df = result_df.rename(columns={position_col: 'position'})

    result_df['strategy_return'] = result_df['position'].shift(1) * result_df['market_return']
    result_df['strategy_return'] = result_df['strategy_return'].fillna(0)

    result_df['portfolio_value'] = initial_capital * (1 + result_df['strategy_return']).cumprod()
    result_df['cumulative_return'] = result_df['portfolio_value'] / initial_capital - 1

    result_df['pnl'] = result_df['portfolio_value'] - initial_capital
    result_df['daily_pnl'] = result_df['portfolio_value'].diff().fillna(0)

    running_max = result_df['portfolio_value'].cummax()
    result_df['drawdown'] = result_df['portfolio_value'] / running_max - 1

    # Use the SHIFTED position (the one that actually earns strategy_return)
    # to define entries/exits, so entry_price lines up with the day the
    # trade starts accruing P&L rather than the signal day.
    pos_shifted = result_df['position'].shift(1).fillna(0)
    pos_shifted_prev = pos_shifted.shift(1).fillna(0)

    entries = result_df.index[(pos_shifted_prev == 0) & (pos_shifted == 1)]
    exits = result_df.index[(pos_shifted_prev == 1) & (pos_shifted == 0)]

    trades = []
    entry_iter = iter(entries)
    exit_iter = iter(exits)
    entry_idx = next(entry_iter, None)
    exit_idx = next(exit_iter, None)

    open_trade = None
    for i in result_df.index:
        if entry_idx is not None and i == entry_idx:
            open_trade = {
                'entry_date': result_df.loc[i, 'timestamp'],
                'entry_price': result_df.loc[i, 'close'],
                'entry_portfolio': result_df.loc[i, "portfolio_value"],
            }
            entry_idx = next(entry_iter, None)
        if exit_idx is not None and i == exit_idx and open_trade is not None:
            exit_portfolio = result_df.loc[i, "portfolio_value"]

            open_trade['exit_date'] = result_df.loc[i, 'timestamp']
            open_trade['exit_price'] = result_df.loc[i, 'close']
            open_trade["exit_portfolio"] = exit_portfolio

            open_trade['return_pct'] = (
                open_trade['exit_price'] / open_trade['entry_price'] - 1
            )
            open_trade["pnl_$"] = exit_portfolio - open_trade["entry_portfolio"]
            open_trade["portfolio_return_pct"] = (
                exit_portfolio / open_trade["entry_portfolio"] - 1
            )

            trades.append(open_trade)
            open_trade = None
            exit_idx = next(exit_iter, None)

    if open_trade is not None:
        exit_portfolio = result_df.iloc[-1]["portfolio_value"]
        open_trade["exit_date"] = result_df.iloc[-1]["timestamp"]
        open_trade["exit_price"] = result_df.iloc[-1]["close"]
        open_trade["exit_portfolio"] = exit_portfolio
        open_trade["return_pct"] = open_trade["exit_price"] / open_trade["entry_price"] - 1
        open_trade["pnl_$"] = exit_portfolio - open_trade["entry_portfolio"]
        open_trade["portfolio_return_pct"] = exit_portfolio / open_trade["entry_portfolio"] - 1
        trades.append(open_trade)

    return result_df, trades


def calculate_performance_metrics(result_df, trades, risk_free_rate=RISK_FREE_RATE):
    """Compute standard risk-adjusted performance metrics from a
    run_backtest() result_df and trade log."""
    returns = result_df['strategy_return'].dropna()

    total_return = result_df['portfolio_value'].iloc[-1] / result_df['portfolio_value'].iloc[0] - 1

    n_days = len(result_df)
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = ((result_df['portfolio_value'].iloc[-1] / result_df['portfolio_value'].iloc[0])
             ** (1 / years) - 1) if years > 0 else np.nan

    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf
    sharpe = ((excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
              if excess_returns.std() not in (0, np.nan) and not np.isnan(excess_returns.std())
              else np.nan)

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    sortino = ((excess_returns.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
               if downside_std not in (0, np.nan) and not np.isnan(downside_std)
               else np.nan)

    max_drawdown = result_df['drawdown'].min()

    if trades:
        wins = sum(1 for t in trades if t['return_pct'] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = np.nan

    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Num Trades': len(trades),
    }


def format_table(comparison):
    pct_cols = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 'Win Rate']
    display_table = comparison.copy()
    for col in pct_cols:
        display_table[col] = display_table[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    for col in ['Sharpe Ratio', 'Sortino Ratio']:
        display_table[col] = display_table[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    return display_table


# Buy & Hold: always fully invested (position == 1 every day)
df['position_bh'] = 1

# ---------------------------------------------------------------------------
# Full-period backtest (kept for the equity curve / drawdown charts, but
# labeled clearly -- it mixes in-sample and out-of-sample days)
# ---------------------------------------------------------------------------
bh_result, bh_trades = run_backtest(df, 'position_bh')
s1_result, s1_trades = run_backtest(df, 'position_ml')

bh_metrics = calculate_performance_metrics(bh_result, bh_trades)
s1_metrics = calculate_performance_metrics(s1_result, s1_trades)

comparison_full = pd.DataFrame({
    'Buy & Hold': bh_metrics,
    'ML Signal (full period)': s1_metrics
}).T

print("\n" + "=" * 90)
print(f"PERFORMANCE COMPARISON (FULL PERIOD, includes in-sample days): {ticker}")
print("=" * 90)
print(format_table(comparison_full).to_string())
print("=" * 90)

# ---------------------------------------------------------------------------
# Out-of-sample-only backtest -- the one that actually reflects predictive
# skill, since the model never saw these labels during training.
# ---------------------------------------------------------------------------
df_test_only = df.loc[test_start_idx:].copy()

bh_test_result, bh_test_trades = run_backtest(df_test_only, 'position_bh')
s1_test_result, s1_test_trades = run_backtest(df_test_only, 'position_ml')

bh_test_metrics = calculate_performance_metrics(bh_test_result, bh_test_trades)
s1_test_metrics = calculate_performance_metrics(s1_test_result, s1_test_trades)

comparison_oos = pd.DataFrame({
    'Buy & Hold': bh_test_metrics,
    'ML Signal (out-of-sample)': s1_test_metrics
}).T

print("\n" + "=" * 90)
print(f"PERFORMANCE COMPARISON (OUT-OF-SAMPLE ONLY, last 20% of dates): {ticker}")
print("=" * 90)
print(format_table(comparison_oos).to_string())
print("=" * 90)

best_sharpe = comparison_oos['Sharpe Ratio'].idxmax() if comparison_oos['Sharpe Ratio'].notna().any() else "N/A (degenerate returns)"
print(f"\nBest OOS risk-adjusted performer (by Sharpe Ratio): {best_sharpe}")

# ---------------------------------------------------------------------------
# Save fitted scaler / PCA / model so the paper-trading script can reuse
# them instead of refitting on live data (which would reintroduce leakage
# and drift from what was actually backtested).
# ---------------------------------------------------------------------------
joblib.dump(scaler, f"scaler_{ticker}.joblib")
joblib.dump(pca, f"pca_{ticker}.joblib")
joblib.dump(model, f"model_{ticker}.joblib")
print(f"\nSaved fitted scaler/pca/model for {ticker} to disk for reuse in paper trading.")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1, 1]})

axes[0].plot(df['timestamp'], df['close'], label='Close', color='black', linewidth=1)
axes[0].plot(df['timestamp'], df['SMA_short'], label='SMA 20', alpha=0.7)
axes[0].plot(df['timestamp'], df['SMA_long'], label='SMA 200', alpha=0.7)
axes[0].plot(df['timestamp'], df['BB_upper'], label='BB Upper', linestyle='--', alpha=0.4, color='gray')
axes[0].plot(df['timestamp'], df['BB_lower'], label='BB Lower', linestyle='--', alpha=0.4, color='gray')

buy_points = df[df['position_ml'].diff() == 1]
sell_points = df[df['position_ml'].diff() == -1]
axes[0].scatter(buy_points['timestamp'], buy_points['close'], marker='^', color='green', s=80, label='Buy (ML)', zorder=5)
axes[0].scatter(sell_points['timestamp'], sell_points['close'], marker='v', color='red', s=80, label='Sell (ML)', zorder=5)

# Mark where the out-of-sample period begins
axes[0].axvline(df.loc[test_start_idx, 'timestamp'], color='blue', linestyle=':', alpha=0.7, label='Test split start')

axes[0].set_title(f'{ticker} Price, Moving Averages, Bollinger Bands, and ML Signals')
axes[0].set_ylabel('Price ($)')
axes[0].legend(loc='upper left', fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['timestamp'], df['RSI'], label='RSI', color='purple')
axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
axes[1].set_ylabel('RSI')
axes[1].legend(loc='upper left', fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].plot(df['timestamp'], df['MACD'], label='MACD', color='blue')
axes[2].plot(df['timestamp'], df['MACD_signal'], label='Signal', color='orange')
axes[2].axhline(0, color='black', linewidth=0.5)
axes[2].set_ylabel('MACD')
axes[2].set_xlabel('Date')
axes[2].legend(loc='upper left', fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('price_indicators_signals.png', dpi=150)
plt.show()

# Equity curve -- out-of-sample only, since that's the honest comparison
plt.figure(figsize=(14, 6))
plt.plot(bh_test_result['timestamp'], bh_test_result['portfolio_value'], label='Buy & Hold', alpha=0.8)
plt.plot(s1_test_result['timestamp'], s1_test_result['portfolio_value'], label='ML Signal', alpha=0.8)
plt.title(f'Out-of-Sample Equity Curve ({ticker}) -- Starting Capital: ${INITIAL_CAPITAL:,.0f}')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('equity_curve_comparison.png', dpi=150)
plt.show()

# Drawdown chart -- out-of-sample only
plt.figure(figsize=(14, 6))
plt.plot(bh_test_result['timestamp'], bh_test_result['drawdown'], label='Buy & Hold', alpha=0.8)
plt.plot(s1_test_result['timestamp'], s1_test_result['drawdown'], label='ML Signal', alpha=0.8)
plt.title(f'Out-of-Sample Drawdown Comparison ({ticker})')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('drawdown_comparison.png', dpi=150)
plt.show()

print("\nSaved charts: price_indicators_signals.png, equity_curve_comparison.png, drawdown_comparison.png")
print("Saved model artifacts: scaler_%s.joblib, pca_%s.joblib, model_%s.joblib" % (ticker, ticker, ticker))
