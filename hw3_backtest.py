
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

from features import feature_cols, compute_features


from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

print(API_KEY is not None)

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

INITIAL_CAPITAL = 100_000
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  

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


df = compute_features(df)

# Target is binary: 1 if tomorrow's return is positive, else 0
df["target"] = (df["market_return"].shift(-1) > 0).astype(int)

ml_df = df.dropna(subset=feature_cols + ["target"]).copy()

X = ml_df[feature_cols]
y = ml_df["target"]

# Chronological train/test split -- BEFORE any fitting, to avoid leakage
split = int(len(ml_df) * 0.8)

X_train_raw, X_test_raw = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(n_components=0.80)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Number of PCA components:", pca.n_components_)
print("Explained variance:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())


model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)

test_probabilities = model.predict_proba(X_test_pca)[:, 1]

ml_df["probability"] = np.nan
ml_df.loc[X_test_raw.index, "probability"] = test_probabilities

ml_df["signal_ml"] = 0
ml_df.loc[X_test_raw.index, "signal_ml"] = np.where(test_probabilities > 0.60, 1, 0)

ml_df["position_ml"] = ml_df["signal_ml"]

df["signal_ml"] = 0
df["position_ml"] = 0
df["probability"] = np.nan

df.loc[ml_df.index, "signal_ml"] = ml_df["signal_ml"]
df.loc[ml_df.index, "position_ml"] = ml_df["position_ml"]
df.loc[ml_df.index, "probability"] = ml_df["probability"]


test_start_date = ml_df.loc[X_test_raw.index, "timestamp"].iloc[0]
df_test = df[df["timestamp"] >= test_start_date].copy()
print(f"\nOut-of-sample backtest window: {df_test['timestamp'].iloc[0].date()} "
      f"to {df_test['timestamp'].iloc[-1].date()}")


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

    #PnL
    result_df['pnl'] = result_df['portfolio_value'] - initial_capital

    #Daily PnL
    result_df['daily_pnl'] = (result_df['portfolio_value'].diff().fillna(0))


    #Drawdown
    running_max = result_df['portfolio_value'].cummax()
    result_df['drawdown'] = result_df['portfolio_value'] / running_max - 1

    trades = []
    # Use the SHIFTED position (the one that actually earns strategy_return)
    # to define entries/exits, so entry_price lines up with the day the
    # trade starts accruing P&L rather than the signal day.
    pos_shifted = result_df['position'].shift(1).fillna(0)
    pos_shifted_prev = pos_shifted.shift(1).fillna(0)
    entries = result_df.index[(pos_shifted_prev == 0) & (pos_shifted == 1)]
    exits = result_df.index[(pos_shifted_prev == 1) & (pos_shifted == 0)]

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

            #P&L
            open_trade["pnl_$"] = (exit_portfolio - open_trade["entry_portfolio"])

            #Portfolio return during trade
            open_trade["portfolio_return_pct"] = (
                exit_portfolio / open_trade["entry_portfolio"] - 1)

            trades.append(open_trade)
            open_trade = None
            exit_idx = next(exit_iter, None)

    # Close any open trade at the final bar
    if open_trade is not None:

        exit_portfolio = result_df.iloc[-1]["portfolio_value"]

        open_trade["exit_date"] = result_df.iloc[-1]["timestamp"]
        open_trade["exit_price"] = result_df.iloc[-1]["close"]
        open_trade["exit_portfolio"] = exit_portfolio

        open_trade["return_pct"] = (open_trade["exit_price"] / open_trade["entry_price"] - 1
        )

        open_trade["pnl_$"] = (exit_portfolio - open_trade["entry_portfolio"]
        )

        open_trade["portfolio_return_pct"] = (exit_portfolio / open_trade["entry_portfolio"] - 1
        )

        trades.append(open_trade)

    return result_df, trades


def calculate_performance_metrics(result_df, trades, risk_free_rate=RISK_FREE_RATE):
    """Compute standard risk-adjusted performance metrics from a
    run_backtest() result_df and trade log."""
    returns = result_df['strategy_return'].dropna()

    total_return = result_df['portfolio_value'].iloc[-1] / result_df['portfolio_value'].iloc[0] - 1

    n_days = len(result_df)
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (result_df['portfolio_value'].iloc[-1] / result_df['portfolio_value'].iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if excess_returns.std() != 0 else np.nan

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_std not in (0, np.nan) and not np.isnan(downside_std) else np.nan

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



df_test['position_bh'] = 1

bh_result, bh_trades = run_backtest(df_test, 'position_bh')
s1_result, s1_trades = run_backtest(df_test, 'position_ml')

bh_metrics = calculate_performance_metrics(bh_result, bh_trades)
s1_metrics = calculate_performance_metrics(s1_result, s1_trades)

comparison = pd.DataFrame({
    'Buy & Hold': bh_metrics,
    'ML Signal': s1_metrics
}).T

pct_cols = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 'Win Rate']
display_table = comparison.copy()
for col in pct_cols:
    display_table[col] = display_table[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
for col in ['Sharpe Ratio', 'Sortino Ratio']:
    display_table[col] = display_table[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

print("\n" + "=" * 90)
print(f"PERFORMANCE COMPARISON (out-of-sample): {ticker}")
print("=" * 90)
print(display_table.to_string())
print("=" * 90)

best_sharpe = comparison['Sharpe Ratio'].idxmax()
print(f"\nBest risk-adjusted performer (by Sharpe Ratio): {best_sharpe}")


fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1, 1]})

axes[0].plot(df['timestamp'], df['close'], label='Close', color='black', linewidth=1)
axes[0].plot(df['timestamp'], df['SMA_short'], label='SMA 20', alpha=0.7)
axes[0].plot(df['timestamp'], df['SMA_long'], label='SMA 200', alpha=0.7)
axes[0].plot(df['timestamp'], df['BB_upper'], label='BB Upper', linestyle='--', alpha=0.4, color='gray')
axes[0].plot(df['timestamp'], df['BB_lower'], label='BB Lower', linestyle='--', alpha=0.4, color='gray')

buy_points = df[df['position_ml'].diff() == 1]
sell_points = df[df['position_ml'].diff() == -1]
axes[0].scatter(buy_points['timestamp'], buy_points['close'], marker='^', color='green', s=80, label='Buy (ML Signal)', zorder=5)
axes[0].scatter(sell_points['timestamp'], sell_points['close'], marker='v', color='red', s=80, label='Sell (ML Signal)', zorder=5)

axes[0].set_title(f'{ticker} Price, Moving Averages, Bollinger Bands, and ML Signal Trades')
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


plt.figure(figsize=(14, 6))
plt.plot(bh_result['timestamp'], bh_result['portfolio_value'], label='Buy & Hold', linewidth=3, linestyle='--', alpha=0.9)
plt.plot(s1_result['timestamp'], s1_result['portfolio_value'], label='ML Signal', linewidth=1.25, alpha=0.9)
plt.title(f'Equity Curve Comparison, Out-of-Sample ({ticker}) -- Starting Capital: ${INITIAL_CAPITAL:,.0f}')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('equity_curve_comparison.png', dpi=150)
plt.show()


plt.figure(figsize=(14, 6))
plt.plot(bh_result['timestamp'], bh_result['drawdown'], label='Buy & Hold', linewidth=3, linestyle='--', alpha=0.9)
plt.plot(s1_result['timestamp'], s1_result['drawdown'], label='ML Signal', linewidth=1.25, alpha=0.9)

plt.title(f'Drawdown Comparison, Out-of-Sample ({ticker})')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('drawdown_comparison.png', dpi=150)
plt.show()


plt.figure(figsize=(10, 5))
components = range(1, len(pca.explained_variance_ratio_) + 1)
plt.bar(components, pca.explained_variance_ratio_, alpha=0.6, label='Individual')
plt.plot(components, np.cumsum(pca.explained_variance_ratio_), marker='o', color='darkred', label='Cumulative')
plt.axhline(0.80, color='gray', linestyle='--', alpha=0.7, label='80% threshold')
plt.title(f'PCA Explained Variance ({ticker}, fit on train slice)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(list(components))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_explained_variance.png', dpi=150)
plt.show()

print("\nSaved charts: price_indicators_signals.png, equity_curve_comparison.png, "
      "drawdown_comparison.png, pca_explained_variance.png")
