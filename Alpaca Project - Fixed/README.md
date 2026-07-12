Alpaca Project

Strategy:

The system uses a long/flat moving-average trend strategy:

- Long when the fast moving average is above the slow moving average.
- Flat otherwise.
- Position size is inversely proportional to recent annualized volatility.
- Per-asset target weight is capped.
- Stop-loss, take-profit, maximum gross exposure, maximum order size, and a market-timezone daily drawdown kill switch are enforced by the risk module.
- Live order results distinguish submitted, partially filled, filled, canceled, expired, rejected, and timed-out/open states.

The intuition is that medium-term price trends can persist because investors and institutions adjust positions gradually. Volatility targeting reduces exposure when an asset becomes unusually risky.


Architecture


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
 execution/alpaca_broker.py 
            |
            v
 engine/trading_engine.py



Running Instructions:
source .venv_check/bin/activate
streamlit run ui/app.py