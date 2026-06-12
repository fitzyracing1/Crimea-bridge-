# Oil Analyst Bot

An analyst/trading bot for exchange-traded oil benchmarks. It downloads public
price history, scores the market with technical indicators, recommends
BUY / SELL / HOLD with a plain-language rationale, backtests the strategy,
and can execute the signals through a broker — a built-in paper-trading
simulator by default, or Alpaca's brokerage API.

It works on lawful, exchange-traded instruments only: Brent futures (`BZ=F`),
WTI futures (`CL=F`), and oil ETFs such as `USO` and `BNO`. It is not a tool
for trading physical or sanctioned crude.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .[dev]
```

## Usage

### Analyze: today's recommendation

```bash
python -m oil_analyst analyze --ticker BZ=F --period 1y
```

Example output:

```
SELL (composite score -0.28) at 88.18 as of 2026-06-12
  - trend [-1.00 x 0.35]: SMA20 91.30 is below SMA50 93.57 (-2.42%); close 88.18
  - momentum [+0.21 x 0.25]: RSI(14) = 45.8 (neutral zone)
  - macd [-0.60 x 0.25]: MACD -1.055 is below its signal -0.913
  - mean_reversion [+0.14 x 0.15]: %B = 0.36, price inside the bands
```

### Backtest: how the strategy would have done

```bash
python -m oil_analyst backtest --ticker CL=F --period 2y --cash 25000
```

Reports total return, max drawdown, Sharpe ratio, win rate, and a
buy-and-hold benchmark. Fills happen at the next bar's open (no look-ahead),
with commission and slippage applied.

### Trade: let the bot manage a position

```bash
# Single decision against the built-in paper portfolio (default)
python -m oil_analyst trade --ticker BNO

# Keep running, deciding once an hour
python -m oil_analyst trade --ticker BNO --loop --interval 3600

# Decide and journal, but never place an order
python -m oil_analyst trade --ticker BNO --dry-run
```

The paper portfolio persists to `paper_portfolio.json` between runs, and every
decision (with its indicator votes) is appended to `trade_journal.jsonl` as an
audit trail.

### Trading through Alpaca

The Alpaca adapter trades US-listed ETFs (use `USO` or `BNO`; futures are not
available there). Set credentials in the environment:

```bash
export ALPACA_API_KEY_ID=...
export ALPACA_API_SECRET_KEY=...
python -m oil_analyst trade --ticker USO --broker alpaca
```

This targets Alpaca's **paper-trading endpoint** by default. Pointing the bot
at the live endpoint requires two deliberate steps — setting
`ALPACA_BASE_URL=https://api.alpaca.markets` *and* passing `--allow-live` —
otherwise it refuses to start.

## How the signal works

Four indicator families each cast a vote in [-1, +1]; the weighted sum is the
composite score (BUY above +0.25, SELL below -0.25, HOLD between):

| Vote | Weight | Bullish when... |
|---|---|---|
| Trend (SMA 20/50) | 0.35 | the fast average is above the slow one |
| Momentum (RSI 14) | 0.25 | RSI is low/oversold |
| MACD (12/26/9) | 0.25 | MACD is above its signal line, especially on a fresh crossover |
| Mean reversion (Bollinger %B) | 0.15 | price is near or below the lower band |

BUY recommendations include ATR-based stop-loss and take-profit levels.
Position management is long-only: BUY deploys a configurable fraction of cash
when flat, SELL closes the whole position.

## Tests

```bash
python -m pytest tests/
```

## Disclaimer

This software is for research and education. It is not financial advice, and
backtested performance does not predict future results. Trading futures and
ETFs involves substantial risk of loss. You are responsible for complying
with the laws and trade sanctions of your jurisdiction; this project supports
only publicly listed benchmark instruments and regulated brokerages.
