# ◈ AlphaGlyph — Quantitative Research Platform

**A paper-trading algorithmic system that not only executes quantitative strategies, but rigorously validates whether its own performance is statistically real — using the same mathematical tools employed by institutional quant funds.**

[![CI](https://github.com/Danny-397/alphaglyph/actions/workflows/ci.yml/badge.svg)](https://github.com/Danny-397/alphaglyph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask)](https://flask.palletsprojects.com)
[![Paper Trading](https://img.shields.io/badge/Paper_Trading-Simulated-FFCD00?style=flat)](#)
[![SciPy](https://img.shields.io/badge/SciPy-1.13-8CAAE6?style=flat&logo=scipy)](https://scipy.org)

> **Live demo:** [alphaglyph.org](https://alphaglyph.org)

---

## ⚠️ Disclaimer — Paper Trading Only

> This project is for **educational purposes only.**
> AlphaGlyph runs an **internal paper-trading simulator** — orders are filled at real market prices (from yfinance) but with simulated cash. No brokerage account, no API keys, no real money is ever involved.
> **This bot never trades real money.**
> Nothing here constitutes financial advice. Past backtesting performance does not guarantee future results.

---

## What is this?

AlphaGlyph is a full-stack quantitative research platform that does something most student trading projects don't: **it asks whether its own results are real.**

Most backtesting tools show you a Sharpe ratio and stop there. This system goes further — after every simulation it applies three independent statistical tests borrowed from professional quant finance:

1. **Monte Carlo Resampling** (1,000 bootstrap paths) — does this strategy actually beat random?
2. **Deflated Sharpe Ratio** (Lopez de Prado, 2014) — is the Sharpe genuine after correcting for multiple-testing bias and non-normal returns?
3. **Fama-French 3-Factor Decomposition** — is the return actually alpha, or is it just passive exposure to known market risk premia that a factor ETF would replicate for free?

The system synthesizes all three into a verdict card: **STATISTICALLY SIGNIFICANT**, **PROMISING — NEEDS MORE DATA**, or **INCONCLUSIVE — MAY BE NOISE**.

Beyond validation, the platform includes a live trading bot, a backtesting engine with walk-forward cross-validation and transaction cost modeling, and a Markowitz portfolio optimizer that computes the efficient frontier via quadratic programming.

---

## What Makes This Different

| Typical student trading bot | This project |
|---|---|
| Fixed stop-loss from entry | Trailing stop — floor rises as price climbs, locking in gains |
| Fixed position sizing | Kelly Criterion sizing — fraction derived from historical win rate and odds ratio |
| Single strategy, single backtest | 4 strategies + adaptive mode; walk-forward out-of-sample testing |
| "My Sharpe is 1.4" | "My Sharpe is 1.4 and the Deflated Sharpe gives 91% probability it's real after testing 5 strategies" |
| Backtest return metric | Fama-French alpha decomposition — separates skill from passive factor exposure |
| No portfolio theory | Markowitz efficient frontier via quadratic programming; max-Sharpe and min-variance portfolios |
| No statistical context | Monte Carlo fan chart + Sharpe percentile rank vs 1,000 resampled paths |
| No market context | ADX + Bollinger Band Width + realized volatility regime detection; adaptive strategy selection |

---

## Feature Overview

### Trading Engine
- **4 Quantitative Strategies**: MA Crossover, RSI Mean Reversion, MACD Momentum, ML stub (ready for a trained model)
- **Adaptive Mode**: Detects the current market regime from SPY data and selects the optimal strategy automatically
- **Trailing Stop-Loss**: Exit floor rises with price — a position that runs up 12% can't turn into a loss
- **Kelly Criterion Sizing**: Position size is computed from the Kelly formula using live win rate and avg win/loss; falls back to fixed sizing when fewer than 10 closed trades exist
- **Risk Profiles**: Conservative / Moderate / Aggressive — each controls stop distance, take-profit target, position cap, cash reserve, and high-volatility behaviour
- **5-Minute Cycle**: Every five minutes during market hours the bot detects the regime, checks risk gates, sweeps stops, generates signals, and executes BUY/SELL orders

### Market Regime Detection
Classifies the market into four states using three independent indicators computed from SPY:

| Regime | Trigger | Default Strategy |
|---|---|---|
| TRENDING UP | ADX ≥ 25, +DI > −DI | MA Crossover |
| TRENDING DOWN | ADX ≥ 25, −DI > +DI | MA Crossover |
| RANGING | ADX < 20 | RSI Mean Reversion |
| HIGH VOLATILITY | 30-day realized vol > 25% | RSI (reduced size) |

Indicators: **ADX** (Wilder's smoothing), **Bollinger Band Width** (consolidation proxy), **30-day Annualised Realised Volatility**.

### Backtesting Engine
- **Day-by-day simulation** over any date range with any set of tickers
- **Walk-forward cross-validation**: train on first 70%, evaluate on final 30% only — prevents in-sample bias
- **Transaction costs**: configurable commission + slippage applied to every buy and sell
- **Rolling Kelly sizing**: position size updates after each closed trade using only trades *before* that date (no look-ahead)
- **Regime tagging**: every trade labelled with the market regime at execution time
- **SPY benchmark**: parallel simulation of buy-and-hold for alpha comparison
- **Calmar ratio**: annualised return / max drawdown — risk-adjusted metric used by hedge funds

### Statistical Validation
- **Monte Carlo (1,000 paths)**: resamples the daily return sequence with replacement; computes where the actual result ranks in the distribution of random paths. Fan chart shows P5/P25/P50/P75/P95 equity bands.
- **Probabilistic Sharpe Ratio (PSR)**: P(SR\_true > SR*) corrected for non-normality using skewness and excess kurtosis (Lopez de Prado, 2014, eq. 1)
- **Deflated Sharpe Ratio (DSR)**: PSR where the benchmark is the *expected maximum Sharpe from N independent random strategies*, scaling correctly with sample size via √(252/T). Tells you whether the best result from a search over strategies is real or just the luckiest of N.
- **Fama-French 3-Factor Decomposition**: OLS regression of portfolio excess returns against Mkt-RF, SMB (size), and HML (value) factors from Ken French's data library. Reports Jensen's alpha (annualised), factor betas, R², and per-coefficient t-statistics.

### Portfolio Optimizer
- **Markowitz Mean-Variance Optimization** via `scipy.optimize.minimize` (SLSQP)
- Computes the **efficient frontier** (60 portfolios from min-variance to max return)
- Returns the **max-Sharpe (tangency) portfolio** and **global minimum-variance portfolio**
- Visualises individual asset risk/return scatter, optimal portfolio positions, and a **Pearson correlation heatmap**
- Optional integration with backtest: when `use_markowitz=true`, optimal weights replace the fixed `max_position_pct` in position sizing

### Risk Management (all enforced on every order)

| Rule | Conservative | Moderate | Aggressive |
|---|---|---|---|
| Trailing stop | 3% from peak | 5% from peak | 7% from peak |
| Take-profit | 10% from entry | 15% from entry | 20% from entry |
| Max position | 5% of portfolio | 10% of portfolio | 15% of portfolio |
| Cash reserve | 30% minimum | 20% minimum | 10% minimum |
| Daily trade cap | 6 | 10 | 15 |
| High-vol behaviour | Sits out entirely | Half-size positions | Full size |

### Dashboard & UI
- **Landing page**: clean green/black marketing page (`index.html`) — distinct from the blue app, links into the platform
- **Live dashboard**: portfolio value, open positions, equity curve, regime card, recent trades — auto-refreshes every 10 seconds
- **Backtest interface**: sticky sidebar form + 4-tab results (Performance / Monte Carlo / ⚗ Research / Trades)
- **Portfolio optimizer**: efficient frontier chart, optimal weight bar displays, correlation matrix heatmap
- **Strategy Validation Report**: synthesises Monte Carlo, DSR, and Fama-French into a single verdict with colour-coded confidence
- Vanilla HTML/CSS/JS + Chart.js — zero frontend frameworks, zero build step

---

## Tech Stack

### Backend
| Library | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| Flask | 3.0 | REST API |
| Tiingo | API | Primary market data (free key; reliable from cloud) |
| yfinance | 0.2 | Fallback market data (free, no key) |
| Stooq | — | Secondary fallback (free, no key) |
| pandas | 2.2 | Time-series data manipulation |
| numpy | 1.26 | Indicator maths, matrix operations |
| scipy | 1.13 | Quadratic programming (Markowitz SLSQP) |
| python-dotenv | 1.0 | Environment variable loading |
| gunicorn | 22 | Production WSGI server |
| pytz | 2024 | Market hours timezone handling |

All indicator maths (SMA, EMA, RSI, MACD, ADX, Bollinger Bands) are implemented from first principles — no TA-Lib or similar black-box dependency.

### Frontend
- Vanilla HTML5 / CSS3 / JavaScript (ES2022)
- Chart.js 4.4 (line, scatter, horizontal bar charts)
- Inter + JetBrains Mono (Google Fonts)
- No React, no Vue, no Webpack — open any `.html` file in a browser

### CI/CD
- GitHub Actions: syntax check (`py_compile`), flake8 lint, pytest (99 tests), DB + simulator smoke test
- Security audit via pip-audit (non-blocking, runs as separate job)

---

## Project Structure

```
alphaglyph/
├── backend/
│   ├── app.py           Flask REST API — all 13 endpoints
│   ├── bot.py           Trading loop, order execution, trailing stop tracking
│   ├── simulator.py     Internal paper-trading engine (fills at real prices, SQLite state)
│   ├── strategies.py    Signal generators (MA Crossover, RSI, MACD, ML stub)
│   ├── backtest.py      Historical simulation — walk-forward, Kelly, costs, regime tagging
│   ├── features.py      Centralised indicator engineering (SMA, RSI, MACD, ATR, returns)
│   ├── regime.py        Market regime detection (ADX, BB Width, realised volatility)
│   ├── risk.py          Risk gate: trailing stop, Kelly sizing, profiles, daily limits
│   ├── portfolio.py     Markowitz efficient frontier via SciPy SLSQP
│   ├── monte_carlo.py   Bootstrap resampling — 1,000 equity paths, fan chart bands
│   ├── stats.py         PSR, Deflated Sharpe Ratio, Fama-French 3-factor OLS
│   ├── database.py      SQLite persistence (WAL mode, Kelly computation, live metrics)
│   ├── gunicorn.conf.py workers=1 (one bot thread — prevents duplicate orders)
│   ├── requirements.txt
│   └── tests/
│       ├── test_risk.py       26 tests — stop/take, trailing stop, sizing, Kelly, risk gates
│       ├── test_simulator.py  20 tests — buy/sell fills, cash flow, P&L, position tracking
│       ├── test_features.py   13 tests — SMA, RSI, MACD correctness on synthetic data
│       ├── test_portfolio.py  17 tests — Markowitz constraints, frontier math, data layer
│       └── test_stats.py      23 tests — PSR/DSR math, FF3 CSV parsing, OLS regression
├── frontend/
│   ├── index.html       Landing page (green/black marketing page)
│   ├── dashboard.html   Live trading dashboard
│   ├── backtest.html    Backtest interface with Research tab
│   ├── portfolio.html   Markowitz portfolio optimizer
│   ├── landing.css      Landing page styles (green/black, distinct from app)
│   ├── style.css        App design system (Inter + JetBrains Mono, dark blue)
│   ├── config.js        Sets the Render backend URL for production
│   └── app.js           All client logic — dashboard, backtest, portfolio optimizer
├── .github/
│   └── workflows/ci.yml
├── .flake8
├── .env.example
└── README.md
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Bot state, portfolio summary, regime, Kelly fraction, live Sharpe & drawdown |
| `/api/start` | POST | Start the trading bot with a given strategy |
| `/api/stop` | POST | Send stop signal to the bot loop |
| `/api/strategy` | POST | Switch active strategy while running |
| `/api/risk_tolerance` | POST | Switch risk profile (conservative / moderate / aggressive) |
| `/api/regime` | GET | Detect current market regime from live SPY data |
| `/api/trades` | GET | Trade history with limit and strategy filters |
| `/api/portfolio/history` | GET | Portfolio value time-series snapshots |
| `/api/portfolio/optimize` | POST | Run Markowitz efficient frontier optimization |
| `/api/activity` | GET | Recent bot activity log |
| `/api/indicators` | GET | Current indicator values for all watchlist tickers |
| `/api/watchlist` | GET | Current watchlist tickers |
| `/api/backtest` | POST | Full backtest with walk-forward, Kelly, Monte Carlo, DSR, Fama-French |
| `/health` | GET | Health check (returns `{"status": "ok"}`) |

### Backtest request body

```json
{
  "strategy":        "adaptive",
  "tickers":         ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "JPM", "SPY"],
  "start_date":      "2023-01-01",
  "end_date":        "2024-01-01",
  "initial_capital": 100000,
  "walk_forward":    false,
  "risk_tolerance":  "moderate",
  "commission_pct":  0.001,
  "slippage_pct":    0.0005,
  "use_markowitz":   false
}
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Danny-397/alphaglyph
cd alphaglyph
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

No API keys or brokerage account needed — the internal simulator handles order execution.

### 3. Run the backend
```bash
python backend/app.py
# → http://localhost:5000
```

### 4. Open the frontend
```bash
# Option A: open directly — index.html is the landing page;
# click "Launch Platform" to reach the dashboard
open frontend/index.html

# Option B: local dev server (avoids CORS issues)
python -m http.server 3000 -d frontend
# → http://localhost:3000  (dashboard at /dashboard.html)
```

### 5. Run the test suite
```bash
cd backend
pytest tests/ -v
# 99 tests, all should pass
```

---

## Environment Variables

All environment variables are **optional** — the app runs out of the box with no configuration.

| Variable | Required | Description |
|---|---|---|
| `TIINGO_API_KEY` | No (recommended) | Free key from [tiingo.com](https://www.tiingo.com). Primary market-data source — yfinance/Stooq are rate-limited and often blocked on cloud IPs, so set this for reliable backtesting in production. |
| `PORT` | No | Flask port (default: 5000) |
| `DATABASE_PATH` | No | SQLite path (default: `backend/alphaglyph.db`). Set to `/data/alphaglyph.db` on Render for persistence. |
| `CORS_ORIGINS` | No | Allowed CORS origins (default: `*`). Set to your Vercel URL in production. |

### Market data sources

AlphaGlyph tries data sources in order: **Tiingo** (if `TIINGO_API_KEY` is set) → **yfinance** → **Stooq**. The first to return data wins, and results are cached in-process for 15 minutes. yfinance and Stooq are free but heavily rate-limited (especially from datacenter IPs); Tiingo's free tier is reliable from anywhere, which is why it's recommended for deployment.

---

## Deployment

### Backend → Render

1. Push to GitHub
2. New Web Service → connect repo → root directory: `backend/`
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app:app` (gunicorn.conf.py auto-discovered)
5. Environment variables (both optional):

| Variable | Value |
|---|---|
| `DATABASE_PATH` | `/data/alphaglyph.db` |
| `CORS_ORIGINS` | `https://your-project.vercel.app` |

`render.yaml` in the repo root configures all of this automatically — Render detects it on import.

**Persistent disk**: Add a Render disk mounted at `/data` — otherwise SQLite is wiped on every deploy. (`render.yaml` declares this disk for you.)

### Frontend → Vercel

1. New Project → root directory: `frontend/`
2. No build step needed (static files)
3. Set your Render backend URL in `frontend/config.js` (`window.RENDER_URL = '...'`) before deploying

### Deployment notes

- **`workers = 1` is mandatory.** `gunicorn.conf.py` enforces this. Multiple workers = multiple bot threads = duplicate orders against the same simulated account.
- **Render free tier spins down** after 15 min of inactivity. Use UptimeRobot (free) to ping `/health` every 10 minutes. The bot loop must be restarted from the dashboard after a cold start.
- **Backtest timeouts**: the Render default timeout is 30s; `gunicorn.conf.py` sets `timeout = 120` to accommodate long backtests with many tickers.

---

## Mathematical Background

### Kelly Criterion
`f* = (b·p − q) / b` where `b = avg_win / avg_loss`, `p = win_rate`, `q = 1 − p`.
Half-Kelly (`f* × 0.5`) is used in practice to reduce variance without sacrificing much expected growth. Falls back to fixed sizing until 10 closed trades exist.

### Probabilistic Sharpe Ratio (PSR)
`PSR(SR*) = Φ[(SR_hat − SR*) √(T−1) / √(1 − γ₃·SR_hat + (γ₄−1)/4·SR_hat²)]`
where γ₃ = skewness, γ₄ = raw kurtosis. Corrects the naive Sharpe comparison for fat tails and finite sample size.
Source: *Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." Journal of Portfolio Management.*

### Deflated Sharpe Ratio (DSR)
DSR = PSR where `SR* = E[max Sharpe | N strategies]`.
`SR* = (1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))`, scaled by `√(252/T)` for the actual sample size.
γ ≈ 0.5772 (Euler–Mascheroni constant). A DSR > 95% indicates the best strategy is unlikely to be the luckiest of N random strategies.

### Fama-French 3-Factor Model
`R_p − R_f = α + β_mkt(R_m−R_f) + β_smb·SMB + β_hml·HML + ε`
Solved via OLS (`numpy.linalg.lstsq`). Factor data downloaded daily from Kenneth French's data library (Dartmouth). Alpha with |t| > 2 is considered statistically significant.
Source: *Fama, E. F. & French, K. R. (1993). "Common risk factors in the returns on stocks and bonds." Journal of Financial Economics.*

### Markowitz Mean-Variance Optimization
`min w^T Σ w` s.t. `Σw = 1, w_i ≥ 0` (long-only, fully invested).
Max-Sharpe: minimise `−(w^T μ − r_f) / √(w^T Σ w)`.
Solved with `scipy.optimize.minimize(method='SLSQP')`. All annualisation uses 252 trading days.
Source: *Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.*

---

## Test Suite

```
backend/tests/
├── test_risk.py       26 tests  — trailing stop, Kelly sizing, risk profiles, daily limits
├── test_simulator.py  20 tests  — buy/sell fills, cash flow, P&L, position tracking
├── test_features.py   13 tests  — SMA/RSI/MACD correctness on synthetic OHLCV data
├── test_portfolio.py  17 tests  — Markowitz constraints, efficient frontier math
└── test_stats.py      23 tests  — PSR/DSR math, FF3 CSV parsing, OLS regression
```

All 99 tests pass with zero network calls — every test uses synthetic in-memory data or monkeypatched API calls. Run with `pytest tests/ -v` from the `backend/` directory.

---

## License

[MIT](LICENSE) — free to use, fork, and build on with attribution.

---

## Author

**Danny** — high school developer.
Independent project demonstrating quantitative finance, statistical inference, and full-stack engineering.

Key concepts implemented from scratch:
- Wilder's ADX, Bollinger Bands, RSI (Wilder EMA smoothing), MACD
- Kelly Criterion with Monte Carlo confirmation
- Markowitz quadratic programming
- Probabilistic and Deflated Sharpe Ratio (Lopez de Prado)
- Fama-French 3-factor OLS decomposition
