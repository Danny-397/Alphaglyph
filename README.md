# ◈ TradeBot

**Autonomous algorithmic trading system with three quantitative strategies, a backtesting engine with Sharpe ratio and max drawdown analysis, automated risk management, and real-time dashboard — paper trading only via Alpaca API.**

[![CI](https://github.com/Danny-397/trading-bot-/actions/workflows/ci.yml/badge.svg)](https://github.com/Danny-397/trading-bot-/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask)](https://flask.palletsprojects.com)
[![Alpaca](https://img.shields.io/badge/Alpaca-Paper_Trading_Only-FFCD00?style=flat)](https://alpaca.markets)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=flat)](https://chartjs.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite)](https://sqlite.org)

> **Live demo:** `[Add Render/Vercel URL after deployment]`

---

## ⚠️ Disclaimer — Paper Trading Only

> **This project is for educational purposes only.**
> TradeBot connects exclusively to Alpaca's **paper-trading sandbox** environment.
> **No real money is ever placed at risk. This bot never trades real money.**
> Nothing in this project constitutes financial advice.
> Past backtesting performance does not guarantee future real-world results.

---

## What is this?

TradeBot is a fully autonomous algorithmic trading system that connects to Alpaca's paper trading environment (simulated market, fake money, zero financial risk). It executes trades based on three quantitative technical analysis strategies, enforces multi-layered automated risk management on every order, and includes a backtesting engine that simulates strategies against historical OHLCV data with full performance metrics including Sharpe ratio and max drawdown.

Built as an independent CS project demonstrating end-to-end software engineering: quantitative signal generation, REST API design, real-time data visualization, SQLite persistence, and cloud deployment.

---

## Features

| Feature | Description |
|---|---|
| **3 Quantitative Strategies** | Moving Average Crossover, RSI Mean Reversion, MACD Momentum — each selectable from the live dashboard |
| **Backtesting Engine** | Simulates any strategy against 1+ years of historical data; outputs Sharpe ratio, max drawdown, win rate, and full P&L log |
| **Automated Risk Management** | Stop-loss, take-profit, position sizing, cash reserve, daily trade cap — enforced automatically on every single order |
| **Live Dashboard** | Real-time portfolio stats, open positions table, trade feed, equity curve — auto-refreshes every 10 seconds |
| **Performance Metrics** | Win rate, Sharpe ratio, max drawdown, avg win/loss, comparison against SPY buy-and-hold benchmark |
| **SQLite Persistence** | Every trade and portfolio snapshot stored locally — metrics survive server restarts |
| **Paper Trading Only** | Hard-coded `paper=True` flag in Alpaca client — no path to live trading exists |

---

## Tech Stack

**Backend**
- Python 3.11 + Flask 3 (REST API)
- alpaca-py (Alpaca Markets Paper Trading API)
- yfinance (historical OHLCV data)
- pandas + numpy (all indicator maths implemented from scratch — no black-box libraries)
- SQLite (trade and performance persistence via database.py)
- Python threading (non-blocking 5-minute bot loop)

**Frontend**
- Vanilla HTML / CSS / JavaScript (zero frameworks)
- Chart.js 4 (equity curve + SPY benchmark visualization)
- Dark monospace terminal aesthetic

**Deployment**
- Backend → Render (free tier, gunicorn)
- Frontend → Vercel (free tier, static)

---

## How the Strategies Work

### Strategy 1 — Moving Average Crossover
The most classical algorithmic trading strategy. Two Simple Moving Averages track trend direction.

- **BUY** when the 20-day SMA crosses *above* the 50-day SMA ("golden cross") — short-term momentum is rising
- **SELL** when the 20-day SMA crosses *below* the 50-day SMA ("death cross") — momentum is falling
- Formula: `SMA(n) = sum of last n closing prices / n`
- Works best in trending, directional markets

### Strategy 2 — RSI Mean Reversion
The Relative Strength Index (0–100) identifies when a stock is temporarily over- or under-valued.

- **BUY** when RSI < 30 — stock is oversold, statistically likely to reverse upward
- **SELL** when RSI > 70 — stock is overbought, statistically likely to pull back
- Formula: `RSI = 100 - 100 / (1 + RS)` where `RS = avg gain / avg loss` over 14 periods
- Works best in range-bound, oscillating markets

### Strategy 3 — MACD Momentum
Compares two exponential moving averages to detect momentum shifts, with volume confirmation to filter noise.

- **BUY** when MACD line crosses above signal line *and* volume exceeds 20-day average
- **SELL** when MACD line crosses below signal line
- Formula: `MACD = EMA(12) - EMA(26)`, `Signal = EMA(9) of MACD`
- Volume filter significantly reduces false signals on low-liquidity days

---

## Risk Management Rules

Every order must pass through a centralized gating system before execution. All rules run simultaneously and cannot be bypassed.

| Rule | Parameter | Purpose |
|---|---|---|
| Stop-loss | −5% from entry | Hard exit to prevent runaway losses |
| Take-profit | +15% from entry | Lock in gains before they evaporate |
| Max position size | 10% of portfolio | Forced diversification across holdings |
| Minimum cash reserve | 20% of portfolio | Always keep dry powder for new signals |
| Daily trade cap | 10 orders/day | Prevent overtrading on noisy signals |
| Market hours only | 9:30 AM–4:00 PM EST | Avoid illiquid pre/after-market conditions |

---

## Backtesting Results

> *Run the backtester with your own date range and capital — results vary by period and strategy.*

Example results (replace with real numbers after running):

| Strategy | Period | Return | Win Rate | Sharpe | Max Drawdown |
|---|---|---|---|---|---|
| MA Crossover | Jan–Dec 2024 | — | — | — | — |
| RSI Mean Reversion | Jan–Dec 2024 | — | — | — | — |
| MACD Momentum | Jan–Dec 2024 | — | — | — | — |
| SPY Buy & Hold | Jan–Dec 2024 | — | — | — | — |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Danny-397/trading-bot-
cd trading-bot-
```

### 2. Get Alpaca paper trading API keys (free)
1. Go to [alpaca.markets](https://alpaca.markets) and create an account
2. Switch to **Paper Trading** in the top-right toggle
3. Go to **Overview → API Keys → Generate New Key**
4. Copy both the API key and secret key

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and fill in your ALPACA_API_KEY and ALPACA_SECRET_KEY
```

### 4. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 5. Start the backend
```bash
python backend/app.py
# Server starts on http://localhost:5000
```

### 6. Open the frontend
Open `frontend/index.html` in any browser.
For a proper dev server: `npx serve frontend` or `python -m http.server 3000 -d frontend`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | Yes | Alpaca **paper trading** API key |
| `ALPACA_SECRET_KEY` | Yes | Alpaca **paper trading** secret key |
| `PORT` | No | Flask port (default: 5000) |

---

## Project Structure

```
trading-bot-/
├── backend/
│   ├── app.py           Flask REST API — all endpoints
│   ├── bot.py           Core trading loop + Alpaca order execution
│   ├── strategies.py    Signal generation (MA Crossover, RSI, MACD)
│   ├── backtest.py      Historical simulation engine
│   ├── risk.py          Risk gate: stop-loss, sizing, daily limits
│   ├── database.py      SQLite persistence layer
│   └── requirements.txt
├── frontend/
│   ├── index.html       Live trading dashboard
│   ├── backtest.html    Backtesting interface
│   ├── about.html       Strategy & architecture explanation
│   ├── style.css        Dark terminal theme
│   └── app.js           API calls, charts, auto-refresh
├── .env.example
└── README.md
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Bot state, portfolio summary, live Sharpe & drawdown |
| `/api/start` | POST | Start the trading bot |
| `/api/stop` | POST | Stop the trading bot |
| `/api/strategy` | POST | Switch active strategy |
| `/api/trades` | GET | Trade history |
| `/api/portfolio/history` | GET | Portfolio value over time |
| `/api/activity` | GET | Recent bot activity log |
| `/api/indicators` | GET | Current indicator values for watchlist |
| `/api/backtest` | POST | Run a backtest simulation |
| `/health` | GET | Health check |

---

## Deployment

**Backend (Render):**
1. Push to GitHub
2. New Web Service → connect repo → set root to `backend/`
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add environment variables: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`

**Frontend (Vercel):**
1. New Project → connect repo → set root to `frontend/`
2. Update `API_BASE` in `app.js` with your Render URL before deploying

---

## License

[MIT](LICENSE) — free to use, fork, and build on with attribution.

---

## Author

**Danny** — high school developer building a fintech engineering portfolio.
Independent CS project demonstrating algorithmic systems, quantitative finance, and full-stack development.
