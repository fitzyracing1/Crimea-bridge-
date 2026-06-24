"""Oil market analyst and trading bot.

Generates buy/sell/hold recommendations for exchange-traded oil benchmarks
(Brent, WTI, oil ETFs) from technical analysis of public price data,
backtests the strategy, and can execute the signals through a broker --
a built-in paper-trading simulator by default, or Alpaca's API.
"""

__version__ = "0.1.0"
