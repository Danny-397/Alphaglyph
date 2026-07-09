<p align="center">
  <img src="docs/banner.svg" alt="AlphaGlyph — backtest a strategy, then find out if the edge is real" width="100%">
</p>

# ◈ AlphaGlyph — A Backtesting & Strategy-Validation Lab

**A backtesting & strategy-validation lab.** Run classical quantitative strategies, a patient "dip-buyer" value strategy, build-your-own-rule custom strategies, or a multi-modal machine-learning transformer on real historical prices with simulated capital — every trade explained in plain English — then do what most backtests don't: **check whether the edge is genuine skill or just luck**, with the same statistical tests institutional quant funds use (a random-timing permutation test, Deflated Sharpe, Fama-French). The backend is fully stateless, so the demo is free-tier-proof.

[![CI](https://github.com/Danny-397/alphaglyph/actions/workflows/ci.yml/badge.svg)](https://github.com/Danny-397/alphaglyph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask)](https://flask.palletsprojects.com)
[![Paper Trading](https://img.shields.io/badge/Paper_Trading-Simulated-FFCD00?style=flat)](#)
[![SciPy](https://img.shields.io/badge/SciPy-1.13-8CAAE6?style=flat&logo=scipy)](https://scipy.org)

> **Live demo:** [alphaglyph.org](https://alphaglyph.org)

---

## ⚠️ Disclaimer — Educational, Simulated Only

> This project is for **educational purposes only.**
> AlphaGlyph backtests strategies at **real historical market prices** with **simulated capital** — there is no brokerage account, no API keys, and **no real money is ever involved.**
> Nothing here constitutes financial advice. Past backtested performance does not guarantee future results.

---

## What is this?

AlphaGlyph is a full-stack quantitative platform with one thing most student trading projects don't have: **honesty about whether its results are real.**

Anyone can pick a strategy, run it on real historical prices, and see **every decision explained** — what the strategy saw, in which market regime, and why it bought or sold (with an optional animated replay of the whole run). But most backtesting tools show you a Sharpe ratio and stop there. This one goes further — after every simulation it applies three independent statistical tests borrowed from professional quant finance:

1. **Skill Test vs Random Timing** (1,000-path permutation test) — does the strategy's Sharpe actually beat "monkey" traders that time the market at random with the same exposure? (This is the real "beats random?" test — see the note below on why a naive bootstrap *can't* answer it.)
2. **Deflated Sharpe Ratio** (Lopez de Prado, 2014) — is the Sharpe genuine after correcting for multiple-testing bias and non-normal returns?
3. **Fama-French 3-Factor Decomposition** — is the return actually alpha, or just passive exposure to known risk premia a factor ETF would replicate for free?

A fourth panel — a **Monte Carlo fan chart** (1,000 stationary block-bootstrap paths) — shows the *range of outcomes*, but is honestly labelled as an outcome-spread view, **not** a skill test (a bootstrap of a strategy's own returns sits at ~the 50th percentile by construction, so it can't tell skill from luck — the permutation test is what does that).

…synthesised into a verdict card: **STATISTICALLY SIGNIFICANT**, **PROMISING — NEEDS MORE DATA**, or **INCONCLUSIVE — MAY BE NOISE**.

The whole experience is one page — the **Backtester** — where you pick a strategy (or build your own rules), run it with walk-forward cross-validation and transaction costs, watch an optional animated replay of every explained trade, and get the verdict. A small **Tools** tab adds a live signal scanner and a Markowitz portfolio optimizer. See **[Methodology & Honest Limitations](#methodology--honest-limitations)** for exactly what it does and doesn't claim.

---

## Why I Built This

I started out wanting to build a trading bot that "beat the market." I got one to beat the market — on the backtest. Then I changed the date range and it lost. Then I added a second strategy, cherry-picked the better of the two, and got an even nicer number. That was the moment the project actually became interesting: I realized I had built a machine for **fooling myself**, and that almost every student "my strategy returns 40%" project is doing exactly that without noticing.

So I went looking for how professionals guard against it, and fell down the quantitative-finance rabbit hole — Lopez de Prado's *Advances in Financial Machine Learning*, the Deflated Sharpe Ratio, Fama-French factor attribution, walk-forward validation, purged splits. The hard part wasn't writing the strategies; it was building the tests that try to prove my own results are luck, and being honest when they succeed. The ML model lands at roughly a coin flip, and I chose to put that number in the README rather than bury it, because a project about statistical honesty that hides its worst result would be a lie.

> *[Danny — drop one or two sentences here about your own path to this: the class, the market crash, the family member who trades, the first strategy you were sure would work. This is the part an admissions essay grows out of, and it should be in your voice, not mine.]*

What I take away from it: the interesting engineering in quant isn't the prediction — it's the **epistemics**. Knowing what you don't know, and building the instruments to measure it.

---

## What Makes This Different

| Typical student trading project | This project |
|---|---|
| Fixed stop-loss from entry | Trailing stop — floor rises as price climbs, locking in gains |
| Fixed position sizing | Kelly Criterion sizing — fraction derived from historical win rate and odds ratio |
| Single strategy, single backtest | 6 strategies + adaptive mode + a no-code custom rule builder; walk-forward out-of-sample testing |
| Black-box decisions | Every trade explained in plain English (what it saw, the regime, why it acted) |
| "My Sharpe is 1.4" | "My Sharpe is 1.4 and the Deflated Sharpe gives 91% probability it's real after testing 5 strategies" |
| Backtest return metric | Fama-French alpha decomposition — separates skill from passive factor exposure |
| No portfolio theory | Markowitz efficient frontier via quadratic programming; max-Sharpe and min-variance portfolios |
| No statistical context | Permutation skill test vs 1,000 random-timing traders + a Monte Carlo outcome-spread fan chart (honestly distinguished from each other) |
| No market context | ADX + Bollinger Band Width + realized volatility regime detection; adaptive strategy selection |

---

## Feature Overview

### Strategies (6 + adaptive + custom)
- **MA Crossover** — trend stance with a 1% hysteresis band: long while the 20-day average is ≥1% above the 50-day, exit only when it falls ≥1% below (the dead band stops the whipsaw that bleeds costs when the averages hug each other)
- **RSI Mean Reversion** — buy oversold dips (RSI < 40) *only within an uptrend* (20-day > 50-day average); exit when overbought or the trend breaks, so it doesn't catch falling knives on the way down
- **MACD Momentum** — trend stance: long while the MACD line is above zero (far less churn than the noisy signal-line cross)
- **Dip Buyer (52-week value)** — buys *more* as a stock falls toward its 52-week low, averages down on further drops, keeps cash in reserve for the next dip, and sells on recovery toward the high
- **ML Transformer** — a transformer with a multi-modal architecture (price + macro + news-sentiment blocks with modality dropout), served via ONNX (see below). **Honest note:** the *checkpoint shipped in this repo* was trained with the macro (FRED) and news (GDELT) blocks zero-filled — those free sources were unavailable during that training run — so it currently predicts from the **price block alone**. The multi-modal plumbing is real and tested; re-running the training pipeline with those sources live activates the extra channels with no code change.
- **Custom (build-your-own rules)** — a no-code rule builder: define BUY/SELL from indicators (price, SMAs, RSI, MACD, volume, returns, 52-week range) with operators (below / above / crosses above / crosses below) combined with ALL/ANY
- **Adaptive Mode** — detects the current market regime from SPY and auto-selects the fitting strategy

### Bots & sizing
- **Animated replay** — any backtest can be replayed day-by-day: the equity curve grows and each trade streams in with a plain-English reason. Runs client-side from the result, so it needs no server state.
- **Trailing Stop-Loss** — exit floor rises with price so winners are protected
- **Kelly Criterion Sizing** — position size from the Kelly formula on live win rate / odds; falls back to fixed sizing under 10 closed trades
- **Optional dip-weighted sizing** — bet bigger near the 52-week low, smaller near the high
- **Risk Profiles** — Conservative / Moderate / Aggressive control stop distance, take-profit, position cap, cash reserve, and high-volatility behaviour

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
- **Skill Test — random-timing permutation (the real "beats random?" test)**: builds 1,000 "monkey" traders that go long the SPY benchmark on a *random* subset of days, sized to the strategy's own market exposure, and ranks the strategy's Sharpe against that null. A percentile ≥ 95 means the *timing* beat ~95% of random schedules (p ≲ 0.05). **Why this and not the Monte Carlo below:** a bootstrap of a strategy's own returns is centered on the actual result *by construction* (the resampled mean equals the sample mean), so its percentile is ~50 for almost any strategy and cannot separate skill from luck. The permutation test uses a null with real structure, so its percentile actually moves with skill. *Honest limitation:* the null trades the benchmark, so it blends market-timing skill with asset selection — a genuine null, not a perfect attribution.
- **Monte Carlo outcome spread (1,000 paths)**: resamples the daily return sequence with a **stationary block bootstrap** (Politis & Romano, 1994) — blocks of consecutive days with geometrically-distributed lengths (mean ≈ n^⅓) — so each path preserves the serial correlation (volatility clustering, momentum) an i.i.d. resample would erase. Shown as a P5/P25/P50/P75/P95 fan chart and labelled in the UI as an **outcome-spread view, not a skill test** (see above).
- **Probabilistic Sharpe Ratio (PSR)**: P(SR\_true > SR*) corrected for non-normality using skewness and excess kurtosis (Lopez de Prado, 2014, eq. 1)
- **Deflated Sharpe Ratio (DSR)**: PSR where the benchmark is the *expected maximum Sharpe from N independent random strategies*, scaling correctly with sample size via √(252/T). Tells you whether the best result from a search over strategies is real or just the luckiest of N.
- **Fama-French 3-Factor Decomposition**: OLS regression of portfolio excess returns against Mkt-RF, SMB (size), and HML (value) factors from Ken French's data library. Reports Jensen's alpha (annualised), factor betas, R², and per-coefficient t-statistics.

### Portfolio Optimizer
- **Markowitz Mean-Variance Optimization** via `scipy.optimize.minimize` (SLSQP)
- Computes the **efficient frontier** (60 portfolios from min-variance to max return)
- Returns the **max-Sharpe (tangency) portfolio** and **global minimum-variance portfolio**
- Visualises individual asset risk/return scatter, optimal portfolio positions, and a **Pearson correlation heatmap**
- Optional integration with backtest: when `use_markowitz=true`, optimal weights replace the fixed `max_position_pct` in position sizing
- **Honest caveat (shown in the UI):** expected returns are historical means, so this is *naive* Markowitz — hypersensitive to the return estimate, prone to over-concentration and unrealistically high "expected" returns. Weights are illustrative; shrinkage (Ledoit-Wolf) or Black-Litterman would be the professional next step.

### Risk Management (all enforced on every order)

| Rule | Conservative | Moderate | Aggressive |
|---|---|---|---|
| Trailing stop | 10% from peak | 15% from peak | 22% from peak |
| Take-profit | 40% from entry | off — let winners run | off — let winners run |
| Max position | 12% of portfolio | 18% of portfolio | 30% of portfolio |
| Cash reserve | 15% minimum | 5% minimum | 2% minimum |
| Daily trade cap | 8 | 12 | 20 |
| High-vol behaviour | Sits out entirely | 60%-size positions | Full size |

Stops are intentionally wide and take-profit is largely disabled by design: the engine **lets winners run** rather than capping them, which is why trend strategies can ride a full move instead of selling early. (Dip Buyer overrides this with its own tranche/averaging logic.)

### Pages (intentionally just three)
- **Home** (`index.html`) — landing page; the whole pitch is "run a strategy, then find out if the edge is real."
- **Backtest** (`backtest.html`) — **the core.** Pick any of 6 strategies or build your own rules, run it on real history with transaction costs, and get: the equity curve, every trade explained in plain English, an **optional animated replay** of the run, the **Strategy Leaderboard**, and 4-tab results (Performance / Monte Carlo / ⚗ Research / Trades) ending in the **Strategy Validation Report** (Skill Test + Deflated Sharpe + Fama-French → one colour-coded verdict; the Monte Carlo tab is the outcome-spread fan chart).
- **Tools** (`tools.html`) — two utilities beyond backtesting, as tabs: a **Live Signal Scanner** (what every strategy + the ML model says about your watchlist now) and the **Markowitz Portfolio Optimizer** (efficient frontier, optimal weights, correlation heatmap).
- Vanilla HTML/CSS/JS + Chart.js — zero frontend frameworks, zero build step.

---

## Screenshots

The fastest way to see AlphaGlyph is the **[live demo → alphaglyph.org](https://alphaglyph.org)** — pick a strategy, hit run, and open the ⚗ Research tab for the verdict.

**Strategy Validation Report — the whole point of the project: after the backtest, three independent tests ask whether the edge is real. The skill test vs random timing, Deflated Sharpe, and Fama-French alpha each pass or fail into one colour-coded verdict:**

<p align="center"><img src="docs/screenshots/verdict.png" alt="Strategy Validation Report: skill test vs random timing, Deflated Sharpe, and Fama-French alpha combined into a PROMISING verdict, with the Skill Test detail card" width="90%"></p>

**Backtest — equity curve vs SPY with every trade explained in plain English:**

<p align="center"><img src="docs/screenshots/backtest.png" alt="Backtest: MA Crossover equity curve beating SPY, with the day-by-day 'Watch it trade' replay and plain-English trade reasons" width="90%"></p>

**Monte Carlo — the outcome-spread fan chart (P5–P95), honestly labelled as a spread view rather than a skill test:**

<p align="center"><img src="docs/screenshots/montecarlo.png" alt="Monte Carlo outcome-spread fan chart with P5–P95 equity bands and return/Sharpe distribution tables" width="90%"></p>

**Signal Scanner — every strategy's current stance on a watchlist (note the ML column sitting honestly near 55%, a coin flip):**

<p align="center"><img src="docs/screenshots/scanner.png" alt="Signal Scanner: current MA/RSI/MACD/ML stance and consensus for a watchlist, with the regime banner" width="90%"></p>

**Portfolio Optimizer — Markowitz efficient frontier, with the honest naive-mean caveat shown in the UI:**

<p align="center"><img src="docs/screenshots/optimizer.png" alt="Markowitz efficient frontier with the max-Sharpe and min-variance portfolios, and the naive-Markowitz caveat" width="90%"></p>

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
- GitHub Actions: syntax check (`py_compile`), flake8 lint, pytest (140 tests), DB + simulator smoke test
- Security audit via pip-audit (non-blocking, runs as separate job)

---

## Architecture

A deliberately simple, **stateless** design: the browser holds all UI state, the
backend is a pure compute layer (every request is a self-contained market
computation), and the machine-learning model is trained **offline** and shipped
as a portable ONNX artifact — so the live server never trains, never persists,
and survives a free-tier cold start with nothing to recover.

```mermaid
flowchart LR
    subgraph Client["🌐 Browser — Vercel (static)"]
        H[Home] --- B[Backtester] --- T[Tools]
    end

    subgraph API["⚙️ Flask API — Render (stateless)"]
        BT[backtest.py<br/>walk-forward · Kelly · costs]
        ST[stats.py<br/>PSR · Deflated Sharpe · Fama-French]
        MC[monte_carlo.py<br/>skill permutation test<br/>+ bootstrap fan chart]
        PF[portfolio.py<br/>Markowitz frontier]
        RG[regime.py<br/>ADX · BBW · vol]
        ML[ml_runtime.py<br/>ONNX inference]
    end

    subgraph Data["📊 Market data (cascading fallback)"]
        TI[Tiingo] --> YF[yfinance] --> SQ[Stooq]
    end

    subgraph Offline["🧪 Offline ML pipeline (Colab)"]
        DS[dataset.py<br/>12y · chrono 60/20/20] --> MD[model.py<br/>multi-modal transformer] --> TR[train.py<br/>+ ONNX export]
    end

    Client -->|JSON over HTTPS| API
    API --> Data
    TR -.->|ships model.onnx| ML
    ST --> FF[(Fama-French factors<br/>cached + disk fallback)]
```

### Design decisions & tradeoffs

A few things here are unconventional *on purpose* — flagging them so they read as choices, not oversights:

- **No frontend framework, one page-routed `app.js`.** The whole UI is vanilla HTML/CSS/JS with Chart.js and zero build step — you can open a file in a browser and it works, and the whole client is one artifact a reviewer can read top to bottom. The tradeoff is a large single JS file instead of a component tree; for a three-page app with no shared team, the framework's ceremony would cost more than it saves. If this grew past a handful of contributors, splitting `app.js` into modules would be the first refactor.
- **Fills modelled at the daily close.** Every order fills at the signal day's closing price with a flat commission + slippage per side. This is simple and reproducible, but optimistic versus reality (next-open fills, market impact, partial fills). It's disclosed in [Methodology](#methodology--honest-limitations); a next-open fill mode is the most obvious realism upgrade.
- **Stateless by design, `workers = 1`.** No database and no server-side session means the free-tier instance can cold-start with nothing to recover — but it also means all UI state lives in the browser and every request re-computes from market data. Single-worker gunicorn keeps the in-process cache and keep-warm thread coherent; horizontal scaling would trade that simplicity for a shared cache.
- **Model trained offline, shipped as ONNX.** The server never trains. Training lives in `ml/` (built for Colab) and exports a portable `.onnx` artifact, so the live API is pure inference. The cost is that refreshing the model is a manual pipeline run, not a server job — the right call for a demo that must survive on free infrastructure.

---

## Project Structure

```
alphaglyph/
├── backend/                  (stateless Flask API — no DB, no server bot)
│   ├── app.py           REST API — 8 stateless endpoints
│   ├── strategies.py    Signal generators (MA, RSI, MACD, Dip Buyer, ML, current-stance scanner)
│   ├── backtest.py      Historical simulation — walk-forward, Kelly, costs, regime tagging,
│   │                    dip-weighted sizing, and the safe custom-rule evaluator
│   ├── ml_runtime.py    ONNX inference for the transformer (lazy load, graceful degrade)
│   ├── ml_features.py   Multi-modal feature frames (price + macro + news sentiment)
│   ├── features.py      Indicator engineering (SMA, RSI, MACD, ATR, returns) — from scratch
│   ├── regime.py        Market regime detection (ADX, BB Width, realised volatility)
│   ├── risk.py          Risk profiles: trailing stop, Kelly sizing, caps, daily limits
│   ├── portfolio.py     Markowitz efficient frontier via SciPy SLSQP
│   ├── monte_carlo.py   Random-timing permutation skill test + stationary block-bootstrap fan chart
│   ├── stats.py         PSR, Deflated Sharpe Ratio, Fama-French 3-factor OLS
│   ├── simulator.py     Standalone paper-trading sim + database.py — retained & tested,
│   │                    not used by the stateless API
│   ├── gunicorn.conf.py
│   └── tests/           140 tests, fully offline
│       ├── test_backtest.py   no look-ahead, P&L accounting, custom-rule evaluator
│       ├── test_risk.py · test_simulator.py · test_features.py
│       └── test_portfolio.py · test_stats.py
├── ml/                       (offline training — run in Colab, not on the server)
│   ├── dataset.py       Builds the 12-year multi-ticker dataset (chronological 60/20/20)
│   ├── model.py         The multi-modal transformer
│   └── train.py         Train + ONNX export + parity check
├── frontend/                 (vanilla HTML/CSS/JS + Chart.js — green dark theme)
│   ├── index.html       Landing page
│   ├── backtest.html    The core: backtest + custom rule builder + leaderboard
│   │                    + validation report + optional animated replay
│   ├── tools.html       Signal Scanner + Markowitz optimizer (tabs)
│   ├── landing.css · style.css · config.js
│   └── app.js           All client logic (one file, page-routed)
├── .github/workflows/   ci.yml · keepwarm.yml
├── .flake8 · .env.example · README.md
```

---

## API Reference

The API is **fully stateless** — no database, no server-side bot. Every endpoint is a pure market computation, which is what makes it free-tier-proof.

| Endpoint | Method | Description |
|---|---|---|
| `/api/backtest` | POST | Full backtest with walk-forward, Kelly, skill test, Monte Carlo, DSR, Fama-French. The core of the app. |
| `/api/compare` | POST | Run every strategy on the same inputs → ranked leaderboard |
| `/api/scan` | GET | Live Signal Scanner: current stance of every strategy + ML per ticker |
| `/api/regime` | GET | Detect the current market regime from live SPY data |
| `/api/ml/info` | GET | ML model status, architecture, train/val/test metrics, thresholds |
| `/api/portfolio/optimize` | POST | Markowitz efficient frontier optimisation |
| `/api/validate_ticker` | GET | Check whether a ticker symbol exists |
| `/health` | GET | Health check (`{"status": "ok", "ml": "loaded"}`) |

### Backtest request body

```json
{
  "strategy":        "dip_buyer",
  "tickers":         ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "JPM", "SPY"],
  "start_date":      "2023-01-01",
  "end_date":        "2024-01-01",
  "initial_capital": 100000,
  "walk_forward":    false,
  "risk_tolerance":  "moderate",
  "commission_pct":  0.001,
  "slippage_pct":    0.0005,
  "use_markowitz":   false,
  "range_sizing":    false,
  "custom_rules":    null
}
```

For `"strategy": "custom"`, supply `custom_rules`:

```json
{
  "buy":  {"logic": "all", "conditions": [{"left": "rsi14", "op": "lt", "right": 30}]},
  "sell": {"logic": "any", "conditions": [{"left": "rsi14", "op": "gt", "right": 70}]}
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

No API keys or brokerage account needed — backtests and bots fill orders at real market prices with simulated cash. (A free Tiingo key is optional but recommended for reliable market data from cloud IPs.)

### 3. Run the backend
```bash
python backend/app.py
# → http://localhost:5000
```

### 4. Open the frontend
```bash
# Option A: open directly — index.html is the landing page;
# use the nav (Backtest, Tools) to reach the app
open frontend/index.html

# Option B: local dev server (avoids CORS issues)
python -m http.server 3000 -d frontend
# → http://localhost:3000  (Backtester at /backtest.html)
```

### 5. Run the test suite
```bash
cd backend
pytest tests/ -v
# 140 tests, all should pass
```

---

## Environment Variables

All environment variables are **optional** — the app runs out of the box with no configuration.

| Variable | Required | Description |
|---|---|---|
| `TIINGO_API_KEY` | No (recommended) | Free key from [tiingo.com](https://www.tiingo.com). Primary market-data source — yfinance/Stooq are rate-limited and often blocked on cloud IPs, so set this for reliable backtesting in production. |
| `PORT` | No | Flask port (default: 5000) |
| `CORS_ORIGINS` | No | Allowed CORS origins (default: `*`). Set to your Vercel URL in production. |
| `KEEPALIVE_SECONDS` | No | Self-ping interval to keep the free Render instance warm (default: 600). `RENDER_EXTERNAL_URL` is set by Render automatically. |

### Market data sources

AlphaGlyph tries data sources in order: **Tiingo** (if `TIINGO_API_KEY` is set) → **yfinance** → **Stooq**. The first to return data wins, and results are cached in-process for 15 minutes. yfinance and Stooq are free but heavily rate-limited (especially from datacenter IPs); Tiingo's free tier is reliable from anywhere, which is why it's recommended for deployment.

---

## Deployment

### Backend → Render

1. Push to GitHub
2. New Web Service → connect repo → root directory: `backend/`
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app:app` (gunicorn.conf.py auto-discovered)
5. Environment variables (all optional):

| Variable | Value |
|---|---|
| `TIINGO_API_KEY` | your free Tiingo key (recommended for reliable data) |
| `CORS_ORIGINS` | `https://your-project.vercel.app` |

`render.yaml` in the repo root configures the service automatically — Render detects it on import. **No database or persistent disk is required** — the API is fully stateless.

### Frontend → Vercel

1. New Project → root directory: `frontend/`
2. No build step needed (static files)
3. Set your Render backend URL in `frontend/config.js` (`window.RENDER_URL = '...'`) before deploying

### Deployment notes

- **`workers = 1`** — `gunicorn.conf.py` keeps the in-process rate limiter and the keep-warm self-ping thread consistent; threads handle concurrent requests. The API is stateless, so there's nothing to coordinate across processes.
- **Render free tier spins down** after 15 min of inactivity. The app pings its own `RENDER_EXTERNAL_URL` (set automatically by Render) every ~10 min from a background thread, which counts as inbound traffic and keeps the instance warm — no external uptime service needed. A bundled GitHub Actions cron (`.github/workflows/keepwarm.yml`) is a backup; for belt-and-suspenders you can also point a free pinger ([cron-job.org](https://cron-job.org)) at `/health`.
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
Expected returns μ are historical means — naive Markowitz, and deliberately flagged as such in the UI because it is hypersensitive to that estimate.
Source: *Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.*

### Random-Timing Permutation Test (skill test)
Null hypothesis: the strategy's timing is no better than chance. We draw 1,000 "monkey" schedules that are long the benchmark on a random subset of `k = round(exposure · T)` days (matched to the strategy's market exposure) and flat otherwise, and compute each schedule's annualised Sharpe the same way the backtest does. The reported statistic is the **percentile rank** of the strategy's Sharpe in that null distribution; `p ≈ 1 − percentile/100` (one-sided). Unlike a bootstrap of the strategy's own returns — whose percentile is ~50 by construction — this null has real structure, so the percentile genuinely tracks skill.
Related: *Masters, T. (2018). "Permutation and Randomization Tests for Trading System Development."*

---

## Methodology & Honest Limitations

Backtesting is easy to get wrong in ways that flatter the result. Here is exactly how this engine works, and — just as importantly — what it does **not** prove.

### How the backtest avoids look-ahead bias
- **Trailing indicators only.** Every signal at day *T* is computed from data up to and including day *T* (rolling SMAs, RSI, MACD, the 52-week range). No indicator reads a future bar. This is verified by an automated test (`test_backtest.py::TestNoLookahead`): truncating the future must not change a single past trade.
- **Rolling Kelly sizing.** Position sizing uses the Kelly fraction from trades that closed *before* the current date — never the full-sample win rate.
- **Walk-forward mode** trains/warms up on the first 70% and reports metrics only on the held-out final 30%.
- **ML train/validation/test split is chronological (60/20/20).** The transformer is trained on the oldest 60% of dates, tuned on the next 20%, and reported on the most recent 20% — split by date, never randomly. On the Backtest page, an ML backtest is **forced to start after the model's validation cutoff**, so it only ever runs on data the model never saw.

### What it honestly does *not* claim
- **The ML model is roughly a coin flip (test AUC ≈ 0.51).** That is not a bug to hide — next-day equity direction is genuinely close to unpredictable, and any project claiming otherwise should be distrusted. The value here is the **rigour of the evaluation** (chronological splits, purged boundaries, out-of-sample gating, Deflated Sharpe), not a magic edge.
- **The shipped model is currently price-only.** The transformer's architecture is multi-modal (price + macro + news-sentiment blocks, trained with modality dropout), but the committed checkpoint was trained with the macro/news blocks zero-filled because FRED/GDELT were unavailable during that run. So the "multi-modal" claim describes the *architecture and pipeline*, not the extra signal in this particular checkpoint — the macro/news channels light up only after a retrain with those sources reachable. (You can verify this yourself: the macro/news entries in `ml_model_meta.json` have mean/std of exactly `0.0`.)
- **"Beats the market" is period-dependent.** Simple technical strategies frequently *underperform* buy-and-hold out-of-sample, especially in strong trends. A good-looking single-period return means little — that's exactly why the skill test, Deflated Sharpe Ratio, and Fama-French alpha are shown: to ask whether a result is real or luck.
- **The Monte Carlo fan chart is not a skill test.** A bootstrap of a strategy's *own* returns is centered on its actual result by construction, so its percentile hovers near 50 regardless of skill. It is presented (and labelled) as an outcome-*spread* view; the **random-timing permutation test** is the one that actually asks "did this beat random?" — and its null trades the benchmark, so it mixes timing skill with asset selection rather than isolating either perfectly.
- **The portfolio optimizer is naive Markowitz.** Expected returns are historical means, which makes mean-variance optimization hypersensitive to estimation error — it over-concentrates and projects unrealistically high "expected" returns. Weights are illustrative, not advice; shrinkage or Black-Litterman would be the honest production fix.
- **Fills are modelled at the daily close** on the signal day, with a flat commission + slippage cost per side. Real execution (next-open fills, market impact, partial fills, borrow costs) is not modelled.
- **The 52-week range is taken within the fetched window**, so on a short backtest it approximates a shorter range; strategies that depend on it (Dip Buyer) default to a multi-year window.
- **Survivorship/selection:** results are shown for liquid large-caps that exist today. Free-tier market data can also be rate-limited or revised.
- **Paper trading only — no real money, not financial advice.** Past backtested performance does not predict future returns.

---

## Test Suite

```
backend/tests/
├── test_risk.py       — trailing stop, Kelly sizing, risk profiles, daily limits
├── test_simulator.py  — buy/sell fills, cash flow, P&L, position tracking
├── test_features.py   — SMA/RSI/MACD correctness on synthetic OHLCV data
├── test_portfolio.py  — Markowitz constraints, efficient frontier math
├── test_stats.py      — PSR/DSR math, FF3 CSV parsing, OLS regression
├── test_monte_carlo.py— block bootstrap + random-timing skill test (percentile tracks skill)
└── test_backtest.py   — NO LOOK-AHEAD, P&L accounting, custom-rule evaluator
```

All **140 tests** pass with zero network calls — every test uses synthetic in-memory data or monkeypatched market-data/Fama-French calls, so the suite is fast and deterministic. Run with `pytest tests/ -v` from the `backend/` directory.

---

## License

[MIT](LICENSE) — free to use, fork, and build on with attribution.

---

## Author

**Danny** — high school developer.
Independent project demonstrating quantitative finance, statistical inference, and full-stack engineering.

Key concepts implemented from scratch:
- Wilder's ADX, Bollinger Bands, RSI (Wilder EMA smoothing), MACD
- Kelly Criterion sizing (rolling, no look-ahead)
- Random-timing permutation skill test + stationary block-bootstrap Monte Carlo
- Markowitz quadratic programming (with an honest naive-mean caveat)
- Probabilistic and Deflated Sharpe Ratio (Lopez de Prado)
- Fama-French 3-factor OLS decomposition
