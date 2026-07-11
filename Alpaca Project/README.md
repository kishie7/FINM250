# Alpaca Systematic Trading Core

This repository implements Sections 2 and 3 of the project: a systematic strategy and a modular trading-system architecture. It is restricted to Alpaca paper trading.

## Strategy

The system uses a long/flat moving-average trend strategy:

- Long when the fast moving average is above the slow moving average.
- Flat otherwise.
- Position size is inversely proportional to recent annualized volatility.
- Per-asset target weight is capped.
- Stop-loss, take-profit, maximum gross exposure, maximum order size, and a market-timezone daily drawdown kill switch are enforced by the risk module.
- Live order results distinguish submitted, partially filled, filled, canceled, expired, rejected, and timed-out/open states.

The intuition is that medium-term price trends can persist because investors and institutions adjust positions gradually. Volatility targeting reduces exposure when an asset becomes unusually risky.

## Architecture

```text
Alpaca Historical/Streaming Data
            |
            v
       data/ modules
            |
            v
 strategy/trend_following.py
            |
            v
      risk/manager.py
            |
            v
 execution/alpaca_broker.py ----> Alpaca paper account
            |
            v
 engine/trading_engine.py
```

Backtests use the same strategy and risk modules, but substitute `SimulatedBroker` for the Alpaca broker.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add paper-trading API keys to `.env`. Never commit this file.

## Run

```bash
python main.py --mode paper-check
python main.py --mode backtest
pytest
```

Backtest outputs are written to `data/equity_curve.csv` and `data/trades.csv`. Metrics include cumulative return, maximum drawdown, annualized volatility, Sharpe ratio, trade count, and realized hit rate.

## Modules

- `data/`: historical retrieval, live WebSocket wrapper, SQLite storage
- `strategy/`: indicators, signals, volatility-adjusted target weights
- `risk/`: position, exposure, order-size, stop, take-profit, and drawdown checks
- `execution/`: Alpaca paper broker and simulated broker
- `engine/`: live paper-trading orchestration
- `backtest/`: historical simulation and performance metrics
- `config/`: non-secret parameters
- `tests/`: unit tests for strategy and risk controls

## Safety

`TradingClient(..., paper=True)` is hard-coded in the Alpaca broker. This code does not provide a live-money execution path.
