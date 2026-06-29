Homework 1 FINM250
Christian Turk
Rosalyn Jia 
Owakamare Princewill

LINK TO GITHUB: https://github.com/kishie7/FINM250.git

This assignment calls for a real-time stock market data terminal that displays historical data and updates a real-time bid, ask, and price UI. 

The first file data_connector.py connects to the Alpaca historical data API and displays the OHLCV data for a single stock over the past 30 days.  

The second file appv6.py is the real-time quote UI which uses streamlit in order to display the bid, ask, price, and last trade.  It auto-refreshes every 2 seconds and has a chart that displays this data.  It can be used for any tickers.

In order to keep our secret keys private we have the scripts read from a '.env' file.  So you should use the text editor and add your keys to a .env that looks like this:

ALPACA_API_KEY=your-api-key-here
ALPACA_SECRET_KEY=your-secret-key-here

Then you can run both files in the terminal.  To run the appv6.py you use:
streamlit run appv6.py

The app will open in your browser. Enter a ticker symbol and the quotes and chart will update in real time.
