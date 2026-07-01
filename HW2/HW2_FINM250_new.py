import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

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
)
bars = client.get_stock_bars(request).df
df = bars.reset_index()

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

df['ATR'] = tr.ewm(span=14, adjust=False).mean()  # FIX: ATR now saved as its own column

plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / df['ATR'])
minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / df['ATR'])
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
df['ADX'] = dx.ewm(span=14, adjust=False).mean()

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))

high_14 = df['high'].rolling(window=14).max()
low_14 = df['low'].rolling(window=14).min()
df['WILLIAMS_R'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

df['BB_mid'] = df['SMA_short']
df['BB_upper'] = df['SMA_short'] + 2 * df['close'].rolling(window=20).std()
df['BB_lower'] = df['SMA_short'] - 2 * df['close'].rolling(window=20).std()

mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
mf_volume = mf_multiplier * df['volume']
df['CMF'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

df['market_return'] = df['close'].pct_change()

# --- Strategy 1: Trend Following ---
# Buy: MACD > Signal AND ADX > 25   |   Sell: MACD < Signal
df['signal_1'] = 0
df.loc[(df['MACD'] > df['MACD_signal']) & (df['ADX'] > 25), 'signal_1'] = 1
df.loc[df['MACD'] < df['MACD_signal'], 'signal_1'] = -1
df['position_1'] = df['signal_1'].replace(0, np.nan).ffill().fillna(0).clip(lower=0)
# clip(lower=0): long-only per assignment constraints -- a -1 signal means
# "exit to cash", not "go short"

# --- Strategy 2: Mean Reversion ---
# Buy: RSI < 40 AND close < BB_lower   |   Sell/flatten: RSI > 60 AND close > BB_upper
# Exit to flat: price reverts to BB_mid
df['signal_2'] = 0
df.loc[(df['RSI'] < 40) & (df['close'] < df['BB_lower']), 'signal_2'] = 1
df.loc[(df['RSI'] > 60) & (df['close'] > df['BB_upper']), 'signal_2'] = -1

crossed_up = (df['close'].shift(1) < df['BB_mid'].shift(1)) & (df['close'] >= df['BB_mid'])
crossed_down = (df['close'].shift(1) > df['BB_mid'].shift(1)) & (df['close'] <= df['BB_mid'])
df.loc[crossed_up | crossed_down, 'signal_2'] = 0
# NOTE: signal_2 == 0 is ambiguous between "no new signal yet" and
# "explicit flatten." ffill() below treats every 0 as "carry position
# forward," so the flatten rows need to be forced to a real flat state.
flatten_mask = crossed_up | crossed_down
df['position_2'] = df['signal_2'].replace(0, np.nan)
df.loc[flatten_mask, 'position_2'] = 0
df['position_2'] = df['position_2'].ffill().fillna(0).clip(lower=0)

# --- Strategy 3: Custom (SMA + Williams %R + CMF breakout) ---
# Buy: close > SMA_short AND Williams %R > -50 AND CMF > 0
# Exit to flat: any two of the three conditions turn bearish
df['signal_3'] = 0
buy_cond = (df['close'] > df['SMA_short']) & (df['WILLIAMS_R'] > -50) & (df['CMF'] > 0)
exit_cond = (
    ((df['close'] < df['SMA_short']) & (df['WILLIAMS_R'] < -50)) |
    ((df['close'] < df['SMA_short']) & (df['CMF'] < 0)) |
    ((df['WILLIAMS_R'] < -50) & (df['CMF'] < 0))
)
df.loc[buy_cond, 'signal_3'] = 1
df.loc[exit_cond, 'signal_3'] = 0
position_3 = df['signal_3'].where(buy_cond | exit_cond)
df['position_3'] = position_3.ffill().fillna(0).clip(lower=0)


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
    result_df : DataFrame with portfolio_value, strategy_return, drawdown
    trades : list of dicts, one per executed trade (entry/exit/pnl)
    """
    result_df = price_df[['timestamp', 'close', 'market_return', position_col]].copy()
    result_df = result_df.rename(columns={position_col: 'position'})

    result_df['strategy_return'] = result_df['position'].shift(1) * result_df['market_return']
    result_df['strategy_return'] = result_df['strategy_return'].fillna(0)

    result_df['portfolio_value'] = initial_capital * (1 + result_df['strategy_return']).cumprod()
    result_df['cumulative_return'] = result_df['portfolio_value'] / initial_capital - 1

    running_max = result_df['portfolio_value'].cummax()
    result_df['drawdown'] = result_df['portfolio_value'] / running_max - 1

    trades = []
    pos_shifted = result_df['position'].shift(1).fillna(0)
    entries = result_df.index[(pos_shifted == 0) & (result_df['position'] == 1)]
    exits = result_df.index[(pos_shifted == 1) & (result_df['position'] == 0)]

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
            }
            entry_idx = next(entry_iter, None)
        if exit_idx is not None and i == exit_idx and open_trade is not None:
            open_trade['exit_date'] = result_df.loc[i, 'timestamp']
            open_trade['exit_price'] = result_df.loc[i, 'close']
            open_trade['return_pct'] = (
                open_trade['exit_price'] / open_trade['entry_price'] - 1
            )
            trades.append(open_trade)
            open_trade = None
            exit_idx = next(exit_iter, None)

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


# Buy & Hold: always fully invested (position == 1 every day)
df['position_bh'] = 1

bh_result, bh_trades = run_backtest(df, 'position_bh')
s1_result, s1_trades = run_backtest(df, 'position_1')
s2_result, s2_trades = run_backtest(df, 'position_2')
s3_result, s3_trades = run_backtest(df, 'position_3')

bh_metrics = calculate_performance_metrics(bh_result, bh_trades)
s1_metrics = calculate_performance_metrics(s1_result, s1_trades)
s2_metrics = calculate_performance_metrics(s2_result, s2_trades)
s3_metrics = calculate_performance_metrics(s3_result, s3_trades)

comparison = pd.DataFrame({
    'Buy & Hold': bh_metrics,
    'Trend Following': s1_metrics,
    'Mean Reversion': s2_metrics,
    'Custom (Livermore Breakout)': s3_metrics,
}).T

pct_cols = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 'Win Rate']
display_table = comparison.copy()
for col in pct_cols:
    display_table[col] = display_table[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
for col in ['Sharpe Ratio', 'Sortino Ratio']:
    display_table[col] = display_table[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

print("\n" + "=" * 90)
print(f"PERFORMANCE COMPARISON: {ticker}")
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

# Buy/sell markers from Strategy 1 as a representative example
buy_points = df[df['position_1'].diff() == 1]
sell_points = df[df['position_1'].diff() == -1]
axes[0].scatter(buy_points['timestamp'], buy_points['close'], marker='^', color='green', s=80, label='Buy (Strat 1)', zorder=5)
axes[0].scatter(sell_points['timestamp'], sell_points['close'], marker='v', color='red', s=80, label='Sell (Strat 1)', zorder=5)

axes[0].set_title(f'{ticker} Price, Moving Averages, Bollinger Bands, and Strategy 1 Signals')
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

# --- 8b. Combined equity curve: all 4 strategies together ---
plt.figure(figsize=(14, 6))
plt.plot(bh_result['timestamp'], bh_result['portfolio_value'], label='Buy & Hold', alpha=0.8)
plt.plot(s1_result['timestamp'], s1_result['portfolio_value'], label='Trend Following', alpha=0.8)
plt.plot(s2_result['timestamp'], s2_result['portfolio_value'], label='Mean Reversion', alpha=0.8)
plt.plot(s3_result['timestamp'], s3_result['portfolio_value'], label='Custom (Livermore Breakout)', alpha=0.8)
plt.title(f'Equity Curve Comparison ({ticker}) -- Starting Capital: ${INITIAL_CAPITAL:,.0f}')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('equity_curve_comparison.png', dpi=150)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(bh_result['timestamp'], bh_result['drawdown'], label='Buy & Hold', alpha=0.8)
plt.plot(s1_result['timestamp'], s1_result['drawdown'], label='Trend Following', alpha=0.8)
plt.plot(s2_result['timestamp'], s2_result['drawdown'], label='Mean Reversion', alpha=0.8)
plt.plot(s3_result['timestamp'], s3_result['drawdown'], label='Custom (Livermore Breakout)', alpha=0.8)
plt.title(f'Drawdown Comparison ({ticker})')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('drawdown_comparison.png', dpi=150)
plt.show()

print("\nSaved charts: price_indicators_signals.png, equity_curve_comparison.png, drawdown_comparison.png")