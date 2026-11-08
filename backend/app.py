"""
Flask REST API for AlphaGlyph.
All endpoints return JSON.  The frontend polls these on a 10-second interval.
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone

import pytz

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

import backtest as backtester
import calibration as calib
import costsweep
import cpcv
import database as db
import datamining
import features  # noqa: F401 — imported so startup errors surface early
import ledger
import ml_runtime
import options_bt
import options_robust
import pead as pead_study
import portfolio as portopt
import predictor as predictor_engine
import regime as reg
import strategies

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)-20s  %(levelname)s  %(message)s',
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# CORS. This used to default to "*", so with CORS_ORIGINS unset any site on
# the internet could call the API from a visitor's browser. The default is now
# the real frontends; CORS_ORIGINS still overrides, as a comma-separated list.
#
# Origins are matched exactly and carry no trailing slash -- the browser sends
# "https://alphaglyph.org", so a configured "https://alphaglyph.org/" would
# never match and every request would fail with "Failed to fetch".
_configured_origins = [
    o.strip().rstrip('/') for o in os.getenv('CORS_ORIGINS', '').split(',') if o.strip()
]
_default_origins = [
    'https://alphaglyph.org',
    'https://www.alphaglyph.org',
    re.compile(r'^https://[A-Za-z0-9-]+\.vercel\.app$'),  # production + preview deploys
    re.compile(r'^http://localhost:\d+$'),                 # local dev
    re.compile(r'^http://127\.0\.0\.1:\d+$'),
]
CORS(app, origins=_configured_origins + _default_origins)


def _client_ip() -> str:
    """
    Real client IP for rate limiting. Render's edge network sits in front of
    the app and its proxy IP *rotates* between requests, so keying on
    remote_addr (even via ProxyFix) puts every request in a different bucket
    and the limits never bite. The original visitor is the left-most entry of
    X-Forwarded-For, which is stable per client — so key on that.
    """
    fwd = request.headers.get('X-Forwarded-For', '')
    if fwd:
        return fwd.split(',')[0].strip()
    return get_remote_address()


# Per-IP rate limiting. The scanner and backtest pages make a handful of calls
# per visit, so the default ceiling is generous enough for normal use (several
# tabs) while still stopping a script that floods the API. The expensive endpoints (backtest,
# portfolio optimize) get a much stricter cap below — those do heavy compute and
# uncached data downloads, so they're the real abuse/availability risk.
# Storage is in-process memory, which is correct here because gunicorn runs a
# single worker (see gunicorn.conf.py).
limiter = Limiter(
    key_func=_client_ip,
    app=app,
    default_limits=['200 per minute', '5000 per day'],
    storage_uri='memory://',
)


@app.errorhandler(429)
def ratelimit_handler(exc):
    return jsonify({
        'error': 'Rate limit exceeded — please slow down and try again shortly.',
        'limit': str(exc.description),
    }), 429


#: Self-ping window, in America/New_York. 09:00 gives the instance a warm-up
#: margin before the 09:30 open (a cold start is ~50s); 16:30 covers the close.
KEEPALIVE_START_MIN = 9 * 60
KEEPALIVE_END_MIN = 16 * 60 + 30


def _within_keepalive_window(now=None) -> bool:
    """True on weekdays inside the self-ping window.

    Set KEEPALIVE_ALWAYS=1 to ping around the clock -- only sensible on a paid
    instance, where hours are not metered.
    """
    if os.getenv('KEEPALIVE_ALWAYS', '').strip().lower() in ('1', 'true', 'yes'):
        return True
    if now is None:
        now = datetime.now(pytz.timezone('America/New_York'))
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return KEEPALIVE_START_MIN <= minutes < KEEPALIVE_END_MIN


def _start_keepalive():
    """
    Keep the Render free instance awake by pinging its own public URL on a timer.

    Render spins the service down after ~15 min with no INBOUND traffic. A
    request to our own public URL is routed back through Render's edge as inbound
    traffic, so it resets the idle timer — keeping the site warm (fast ML
    forecasts, charts, backtests) with no external uptime service. (GitHub
    Actions cron is too throttled to rely on: scheduled runs can lag by hours.)

    WHY THE PING IS WINDOWED
    Render's 750 free instance-hours are per WORKSPACE per month, shared by
    every free service on the account, and when they run out Render suspends
    ALL of them. Pinging around the clock holds this one service awake ~730
    h/month, which is the entire allowance -- it leaves nothing for the
    QuantumSafe, Tradeski and RL-Trader backends and has already taken them
    down together once. The market-hours window costs ~165 h/month instead.

    Nothing is lost by narrowing it: the bot trades US equities and risk.py
    declines to trade when the market is shut, so an off-hours ping woke the
    instance only to decline. Outside the window the service sleeps and the
    first visitor pays one ~50s cold start, which the frontend already covers
    with a "waking the backend" pill.

    Active only in production, where Render sets RENDER_EXTERNAL_URL.
    """
    import threading
    import time

    url = os.getenv('RENDER_EXTERNAL_URL', '').strip()
    if not url:
        host = os.getenv('RENDER_EXTERNAL_HOSTNAME', '').strip()
        url = f'https://{host}' if host else ''
    if not url:
        return  # not on Render (local/dev) — nothing to keep warm

    target = url.rstrip('/') + '/health'
    interval = int(os.getenv('KEEPALIVE_SECONDS', '600'))  # 10 min < 15 min timeout

    def loop():
        import requests
        while True:
            time.sleep(interval)
            if not _within_keepalive_window():
                continue
            try:
                requests.get(target, timeout=20)
            except Exception as exc:
                logger.warning('keepalive ping failed: %s', exc)

    threading.Thread(target=loop, daemon=True, name='keepalive').start()
    logger.info(
        'Self-keepalive enabled → %s every %ds, weekdays %02d:%02d–%02d:%02d ET',
        target, interval,
        KEEPALIVE_START_MIN // 60, KEEPALIVE_START_MIN % 60,
        KEEPALIVE_END_MIN // 60, KEEPALIVE_END_MIN % 60,
    )


_start_keepalive()


def _start_cache_warmup():
    """
    Precompute the two slow research results in the background after boot.

    The track record (calibration) and the PEAD study each fetch years of
    history for a dozen-plus tickers. Locally that is ~15s; on Render's free
    instance the first run after a restart took several minutes, all of it
    spent by whichever visitor clicked first. Both cache their result for 12h,
    so computing them once at boot -- one after the other, to keep peak memory
    low -- and again before the cache lapses means visitors get the cached
    answer. Production only (same RENDER_EXTERNAL_URL gate as the keepalive);
    set WARM_CACHES=0 to disable.
    """
    import threading
    import time

    if not (os.getenv('RENDER_EXTERNAL_URL') or os.getenv('RENDER_EXTERNAL_HOSTNAME')):
        return
    if os.getenv('WARM_CACHES', '1').strip().lower() in ('0', 'false', 'no'):
        return

    def loop():
        time.sleep(30)  # let gunicorn finish booting and pass its health check
        while True:
            for name, fn in (('calibration', calib.compute_calibration),
                             ('pead', pead_study.compute_pead)):
                started = time.time()
                try:
                    fn()
                    logger.info('cache warm-up: %s ready in %.0fs', name, time.time() - started)
                except Exception as exc:
                    logger.warning('cache warm-up: %s failed: %s', name, exc)
            time.sleep(11 * 3600)  # refresh just before the 12h caches expire

    threading.Thread(target=loop, daemon=True, name='cache-warmup').start()


_start_cache_warmup()

VALID_STRATEGIES = strategies.VALID_STRATEGIES + ('adaptive', 'custom')

# Whitelist for user-built custom strategies (no eval — only these are allowed).
_CUSTOM_INDICATORS = {'close', 'volume', 'sma20', 'sma50', 'rsi14', 'macd_line',
                      'macd_signal', 'macd_hist', 'vol_ma20', 'return_1d',
                      'return_5d', 'range52'}
_CUSTOM_OPS = {'lt', 'gt', 'cross_up', 'cross_dn'}


def _sanitize_rules(rules):
    """Keep only whitelisted indicators/operators; bound the number of rules."""
    def clean_group(g):
        out = []
        for c in (g.get('conditions') or [])[:8]:
            left = c.get('left')
            op   = c.get('op')
            right = c.get('right')
            if left not in _CUSTOM_INDICATORS or op not in _CUSTOM_OPS:
                continue
            if isinstance(right, str):
                if right not in _CUSTOM_INDICATORS:
                    continue
            else:
                try:
                    right = float(right)
                except (TypeError, ValueError):
                    continue
            out.append({'left': left, 'op': op, 'right': right})
        return {'logic': 'all' if g.get('logic') == 'all' else 'any', 'conditions': out}

    rules = rules or {}
    return {'buy': clean_group(rules.get('buy') or {}),
            'sell': clean_group(rules.get('sell') or {})}


# ── Regime ────────────────────────────────────────────────────────────────────

@app.route('/api/regime')
def get_regime():
    """Detect the current market regime from SPY (stateless — no bot/DB)."""
    try:
        spy_df = features.fetch_ohlcv('SPY', period='6mo')
        if spy_df is None or len(spy_df) < 60:
            return jsonify({'error': 'Insufficient SPY data'}), 503
        result = reg.detect_regime(spy_df)
        return jsonify({
            'regime':      result.regime,
            'label':       result.label,
            'description': result.description,
            'strategy':    reg.get_regime_strategy(result.regime, 'moderate'),
            'adx':         result.adx,
            'plus_di':     result.plus_di,
            'minus_di':    result.minus_di,
            'bb_width':    result.bb_width,
            'vol_30d':     result.vol_30d,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.route('/api/scan')
@limiter.limit('20 per minute')
def scan():
    """
    Live Signal Scanner: for each ticker, the CURRENT stance of every strategy
    (BUY / SELL / HOLD) plus the ML transformer's direction probability — so a
    user can see, at a glance, what the strategies say about their stocks right
    now. Stateless; data fetches are cached so repeat scans are fast.
    """
    raw = request.args.get('tickers', '')
    tickers = [t.strip().upper() for t in raw.split(',') if t.strip()] or list(strategies.WATCHLIST)
    tickers = tickers[:20]

    ml_loaded = ml_runtime.get_info().get('loaded')
    out = []
    for t in tickers:
        sigs  = {}
        price = None
        for strat in ('ma_crossover', 'rsi', 'macd'):
            try:
                s, p = strategies.get_signal(strat, t)
            except Exception:
                s, p = None, None
            sigs[strat] = s
            if p:
                price = p

        row = {'ticker': t}
        if ml_loaded:
            try:
                mlp = ml_runtime.live_prediction(t)
            except Exception:
                mlp = None
            if mlp and mlp.get('available'):
                sigs['ml'] = mlp.get('signal')
                row['p_up'] = mlp.get('p_up')
                row['q50']  = mlp.get('quantiles', {}).get('q50')
                price = price or mlp.get('price')

        row['price']   = round(price, 2) if price else None
        row['signals'] = sigs
        row['buys']    = sum(1 for v in sigs.values() if v == 'BUY')
        row['sells']   = sum(1 for v in sigs.values() if v == 'SELL')
        out.append(row)

    return jsonify({'scanned': out, 'count': len(out), 'ml_loaded': bool(ml_loaded)})


@app.route('/api/predict')
@limiter.limit('30 per minute')
def predict():
    """
    Multi-signal Predictor for a single ticker: blends the ML transformer's
    price forecast, an earnings (PEAD + growth) tilt, and GDELT news sentiment
    into one explainable directional lean with per-component breakdown.

    Sentiment (GDELT) and — when the ML model falls back — feature computation
    can each take a few seconds, so this is capped tighter than /api/scan.
    """
    symbol = request.args.get('ticker', '').strip().upper()
    if not symbol:
        return jsonify({'error': 'ticker required'}), 400
    if not symbol.replace('.', '').replace('-', '').isalnum() or len(symbol) > 8:
        return jsonify({'error': 'invalid ticker'}), 400
    try:
        # 'unavailable' is a valid, expected state (thin free data), not an
        # error — the frontend renders it as an honest "no signal" card.
        result = predictor_engine.predict(symbol)
        # Append the call to the forward track record (deduped to once/day/ticker).
        # Best-effort: a ledger failure must never break the prediction itself.
        try:
            ledger.log_prediction(result)
        except Exception:
            logger.warning('ledger logging failed for %s', symbol, exc_info=True)
        return jsonify(result)
    except Exception as exc:
        logger.exception('predict failed for %s', symbol)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/predict/ledger')
@limiter.limit('30 per minute')
def predict_ledger():
    """
    Live forward track record: every Predictor call is logged at prediction time
    and graded once its horizon elapses. Returns the recent rows plus a summary
    (hit rate, Brier score, realized returns) — a genuine out-of-sample record
    that can't be retro-fitted. Lazily grades matured rows on read.
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 200)
    except (TypeError, ValueError):
        limit = 50
    try:
        return jsonify(ledger.get_ledger(limit=limit))
    except Exception as exc:
        logger.exception('ledger read failed')
        return jsonify({'available': False, 'reason': str(exc)}), 500


@app.route('/api/predict/calibration')
@limiter.limit('6 per minute')
def predict_calibration():
    """
    Out-of-sample reliability of the ML price forecaster: does "60% up" actually
    happen ~60% of the time? Returns decile bins (predicted vs realised) plus
    Brier / log-loss / Brier-skill scores. Heavy (multi-ticker inference over
    years) but cached, so repeat loads are instant.
    """
    raw = request.args.get('tickers', '')
    tickers = [t.strip().upper() for t in raw.split(',') if t.strip()] or None
    oos = request.args.get('oos', '1') != '0'
    try:
        return jsonify(calib.compute_calibration(tickers, oos_only=oos))
    except Exception as exc:
        logger.exception('calibration failed')
        return jsonify({'available': False, 'reason': str(exc)}), 500


@app.route('/api/research/pead')
@limiter.limit('6 per minute')
def research_pead():
    """Pooled post-earnings-announcement-drift (PEAD) event study across a
    basket: cumulative abnormal return by surprise tercile + a beat-minus-miss
    t-test. Heavy but cached."""
    raw = request.args.get('tickers', '')
    tickers = [t.strip().upper() for t in raw.split(',') if t.strip()] or None
    try:
        return jsonify(pead_study.compute_pead(tickers))
    except Exception as exc:
        logger.exception('pead study failed')
        return jsonify({'available': False, 'reason': str(exc)}), 500


@app.route('/api/research/datamine', methods=['POST'])
@limiter.limit('10 per hour')
def research_datamine():
    """
    Data-mining lab: sweep N strategy variants on one ticker, keep the best
    in-sample Sharpe, then show the Deflated Sharpe Ratio deflate it back toward
    reality — a live demonstration of multiple-testing / p-hacking.
    """
    data    = request.get_json() or {}
    ticker  = (data.get('ticker') or 'AAPL').strip().upper()
    n       = int(data.get('n_strategies', 200))
    try:
        return jsonify(datamining.run_sweep(ticker, n))
    except Exception as exc:
        logger.exception('datamine failed')
        return jsonify({'available': False, 'reason': str(exc)}), 500


@app.route('/api/research/cpcv', methods=['POST'])
@limiter.limit('10 per hour')
def research_cpcv():
    """
    Combinatorially-Symmetric Cross-Validation (CSCV) → Probability of Backtest Overfitting.
    Runs a real trend/mean-reversion hyperparameter grid on one ticker, then uses
    CSCV to measure how often the in-sample winner degrades out-of-sample. Heavy
    (hundreds of IS/OOS splits) but cached.
    """
    data     = request.get_json() or {}
    ticker   = (data.get('ticker') or 'SPY').strip().upper()
    n_groups = int(data.get('n_groups', 10))
    try:
        return jsonify(cpcv.compute_cpcv(ticker, n_groups=n_groups))
    except Exception as exc:
        logger.exception('cpcv failed')
        return jsonify({'available': False, 'reason': str(exc)}), 500


@app.route('/api/research/cost-sensitivity', methods=['POST'])
@limiter.limit('10 per hour')
def research_cost_sensitivity():
    """
    Transaction-cost sensitivity sweep: rerun one strategy across a grid of
    per-side cost levels and report the net-return / edge decay curve plus the
    break-even cost at which the edge over buy-and-hold vanishes.
    """
    data           = request.get_json() or {}
    strategy       = data.get('strategy', 'ma_crossover')
    tickers        = [t.strip().upper() for t in (data.get('tickers') or [])
                      if isinstance(t, str) and t.strip()]
    start_date     = data.get('start_date',
                              (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d'))
    end_date       = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    initial_capital = float(data.get('initial_capital', 100_000))
    risk_tolerance = data.get('risk_tolerance', 'moderate')

    if strategy not in VALID_STRATEGIES:
        return jsonify({'available': False, 'reason': 'Invalid strategy'}), 400
    if not tickers:
        return jsonify({'available': False, 'reason': 'At least one ticker required'}), 400
    if risk_tolerance not in ('conservative', 'moderate', 'aggressive'):
        return jsonify({'available': False, 'reason': 'Invalid risk tolerance'}), 400

    custom_rules = _sanitize_rules(data.get('custom_rules')) if strategy == 'custom' else None
    try:
        return jsonify(costsweep.run_cost_sweep(
            strategy, tickers, start_date, end_date,
            initial_capital, risk_tolerance, custom_rules))
    except Exception as exc:
        logger.exception('cost sweep failed')
        return jsonify({'available': False, 'reason': str(exc)}), 500


@app.route('/api/ml/info')
def ml_info():
    """ML transformer status: whether a trained model is deployed, its
    architecture, validation/test metrics, and decision thresholds.
    The frontend uses this to enable the ML strategy option."""
    return jsonify(ml_runtime.get_info())


@app.route('/api/compare', methods=['POST'])
@limiter.limit('15 per hour')
def compare_strategies():
    """
    Run every strategy on the SAME tickers/period/capital and return a ranked
    leaderboard (return, Sharpe, max drawdown, Calmar, win rate, trades) plus
    each strategy's equity curve for an overlay chart.

    Data is downloaded once per ticker and cached, so the N backtests share it.
    """
    data            = request.get_json() or {}
    tickers         = [t.strip().upper() for t in (data.get('tickers') or [])
                       if isinstance(t, str) and t.strip()]
    start_date      = data.get('start_date',
                               (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date        = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    initial_capital = float(data.get('initial_capital', 100_000))
    risk_tolerance  = data.get('risk_tolerance', 'moderate')

    if not tickers:
        return jsonify({'error': 'At least one ticker required'}), 400
    if not (1_000 <= initial_capital <= 10_000_000):
        return jsonify({'error': 'Capital must be between $1,000 and $10,000,000'}), 400
    if risk_tolerance not in ('conservative', 'moderate', 'aggressive'):
        return jsonify({'error': 'Invalid risk tolerance'}), 400

    candidates = ['adaptive', 'ma_crossover', 'rsi', 'macd', 'dip_buyer']
    if ml_runtime.get_info().get('loaded'):
        candidates.append('ml')

    LABELS = {'adaptive': 'Adaptive (Regime)', 'ma_crossover': 'MA Crossover',
              'rsi': 'RSI Mean Reversion', 'macd': 'MACD Momentum', 'ml': 'ML Transformer',
              'dip_buyer': 'Dip Buyer (52-Week Value)'}

    results = []
    benchmark_return = None
    spy_curve = []
    for strat in candidates:
        try:
            r = backtester.run_backtest(
                strat, tickers, start_date, end_date,
                initial_capital, False, risk_tolerance, 0.0003, 0.0003)
        except Exception as exc:
            logger.warning('compare: %s failed: %s', strat, exc)
            results.append({'strategy': strat, 'label': LABELS.get(strat, strat), 'error': str(exc)})
            continue
        if 'error' in r:
            results.append({'strategy': strat, 'label': LABELS.get(strat, strat), 'error': r['error']})
            continue
        m = r['metrics']
        benchmark_return = m.get('benchmark_return')
        if not spy_curve and r.get('spy_curve'):
            spy_curve = r['spy_curve']
        results.append({
            'strategy':         strat,
            'label':            LABELS.get(strat, strat),
            'total_return':     m.get('total_return'),
            'sharpe_ratio':     m.get('sharpe_ratio'),
            'max_drawdown':     m.get('max_drawdown'),
            'calmar_ratio':     m.get('calmar_ratio'),
            'win_rate':         m.get('win_rate'),
            'total_trades':     m.get('total_trades'),
            'final_value':      m.get('final_value'),
            'equity_curve':     r.get('equity_curve', []),
        })

    ranked = [r for r in results if 'error' not in r]
    ranked.sort(key=lambda r: (r.get('total_return') is None, -(r.get('total_return') or 0)))
    errored = [r for r in results if 'error' in r]

    return jsonify({
        'results':          ranked + errored,
        'benchmark_return': benchmark_return,
        'spy_curve':        spy_curve,
        'tickers':          tickers,
        'start_date':       start_date,
        'end_date':         end_date,
    })


@app.route('/api/validate_ticker')
def validate_ticker():
    """Check whether a ticker exists. status: valid | not_found | rate_limited."""
    symbol = request.args.get('symbol', '').strip().upper()
    if not symbol:
        return jsonify({'valid': False, 'symbol': '', 'status': 'not_found'}), 400
    status = features.validate_symbol(symbol)
    return jsonify({'valid': status == 'valid', 'symbol': symbol, 'status': status})


# ── Backtesting ───────────────────────────────────────────────────────────────

@app.route('/api/backtest', methods=['POST'])
@limiter.limit('20 per hour')
def run_backtest():
    data            = request.get_json() or {}
    strategy        = data.get('strategy', 'adaptive')
    tickers         = [t.strip().upper() for t in (data.get('tickers') or [])
                       if isinstance(t, str) and t.strip()]
    start_date      = data.get('start_date',
                               (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date        = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    initial_capital = float(data.get('initial_capital', 100_000))
    walk_forward    = bool(data.get('walk_forward', False))
    risk_tolerance  = data.get('risk_tolerance', 'moderate')
    # Realistic costs for liquid US large-caps in the commission-free era:
    # ~0.03% commission/fees + ~0.03% slippage per side. The old 0.10%/0.05%
    # default badly overstated costs and unfairly dragged active strategies.
    commission_pct  = float(data.get('commission_pct', 0.0003))
    slippage_pct    = float(data.get('slippage_pct',   0.0003))
    use_markowitz   = bool(data.get('use_markowitz', False))
    range_sizing    = bool(data.get('range_sizing', False))
    cash_in_market  = bool(data.get('cash_in_market', False))
    custom_rules    = _sanitize_rules(data.get('custom_rules')) if strategy == 'custom' else None

    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': 'Invalid strategy'}), 400
    if strategy == 'custom' and not custom_rules['buy']['conditions']:
        return jsonify({'error': 'Add at least one BUY condition for a custom strategy'}), 400
    if not (1_000 <= initial_capital <= 10_000_000):
        return jsonify({'error': 'Capital must be between $1,000 and $10,000,000'}), 400
    if not tickers:
        return jsonify({'error': 'At least one ticker required'}), 400
    if risk_tolerance not in ('conservative', 'moderate', 'aggressive'):
        return jsonify({'error': 'Invalid risk tolerance'}), 400
    if not (0 <= commission_pct <= 0.05):
        return jsonify({'error': 'Commission must be between 0% and 5%'}), 400
    if not (0 <= slippage_pct <= 0.05):
        return jsonify({'error': 'Slippage must be between 0% and 5%'}), 400

    result = backtester.run_backtest(
        strategy, tickers, start_date, end_date,
        initial_capital, walk_forward, risk_tolerance,
        commission_pct, slippage_pct,
        use_markowitz=use_markowitz,
        range_sizing=range_sizing,
        cash_in_market=cash_in_market,
        custom_rules=custom_rules,
    )
    return jsonify(result)


# ── Options strategy backtests ────────────────────────────────────────────────

_OPT_TICKER = re.compile(r'^[A-Z][A-Z0-9.\-]{0,9}$')


def _num(data, key, default, lo, hi, label, allow_none=False):
    """A bounded number from the request body; raises ValueError with a readable message."""
    v = data.get(key, default)
    if v is None or v == '':
        if allow_none:
            return None
        v = default
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'{label} must be a number')
    if not (lo <= v <= hi):
        raise ValueError(f'{label} must be between {lo:g} and {hi:g}')
    return v


def _options_args(data: dict) -> dict:
    """Validated settings shared by the options backtest and comparison endpoints."""
    ticker = str(data.get('ticker') or 'SPY').strip().upper()
    if not _OPT_TICKER.match(ticker):
        raise ValueError('Enter a valid ticker')
    end = data.get('end_date') or datetime.now().strftime('%Y-%m-%d')
    start = data.get('start_date') or (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
    try:
        s_dt, e_dt = datetime.strptime(start, '%Y-%m-%d'), datetime.strptime(end, '%Y-%m-%d')
    except (TypeError, ValueError):
        raise ValueError('Dates must be YYYY-MM-DD')
    if s_dt < datetime(1995, 1, 1) or e_dt <= s_dt + timedelta(days=60):
        raise ValueError('Pick a window of at least two months, starting 1995 or later')
    if (e_dt - s_dt).days > options_bt.MAX_YEARS * 366:
        raise ValueError(f'The window can be at most {options_bt.MAX_YEARS} years')
    spread = _num(data, 'spread_pct', None, 0, 20, 'Bid/ask cost', allow_none=True)
    return {
        'ticker': ticker, 'start': start, 'end': end,
        'dte': int(_num(data, 'dte', 30, 3, 180, 'Days to expiry')),
        'initial_capital': _num(data, 'initial_capital', 100_000, 1_000, 10_000_000, 'Capital'),
        'exit_dte': (None if data.get('exit_dte') in (None, '') else
                     int(_num(data, 'exit_dte', 0, 0, 179, 'Exit days before expiry'))),
        'profit_take': _num(data, 'profit_take', None, 1, 1000, 'Profit target', allow_none=True),
        'stop_loss': _num(data, 'stop_loss', None, 1, 1000, 'Stop loss', allow_none=True),
        'spread_pct': None if spread is None else spread / 100.0,
        'commission': _num(data, 'commission', 0.65, 0, 10, 'Commission per contract'),
        'skew': bool(data.get('skew', True)),
        'entry': options_bt.clean_entry(data.get('entry')),
    }


@app.route('/api/options/presets')
def options_presets():
    """Every preset options strategy, grouped by leg count, with what each backtest holds."""
    dte = request.args.get('dte', 30, type=int)
    return jsonify({'presets': options_bt.presets_summary(max(3, min(dte or 30, 180)))})


@app.route('/api/options/backtest', methods=['POST'])
@limiter.limit('30 per hour')
def options_backtest():
    """Backtest one options strategy (a preset or custom legs) on one underlying."""
    data = request.get_json() or {}
    try:
        args = _options_args(data)
        preset = data.get('preset')
        if preset:
            if preset not in options_bt.PRESETS:
                raise ValueError('Unknown strategy')
            p = options_bt.PRESETS[preset]
            legs, name = p['legs'], p['name']
            alloc = _num(data, 'alloc_pct', p['alloc'], 1, 100, 'Capital per position')
        else:
            legs = options_bt.clean_legs(data.get('legs'))
            name = str(data.get('name') or 'Custom strategy').strip()[:60] or 'Custom strategy'
            alloc = _num(data, 'alloc_pct', 10, 1, 100, 'Capital per position')
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    market = options_bt.load_market(args['ticker'], args['start'], args['end'])
    if 'error' in market:
        return jsonify(market), 422
    run_kw = {'legs': legs, 'name': name, 'alloc_pct': alloc, 'market': market, **args}
    result = options_bt.run(**run_kw)
    if 'error' in result:
        return jsonify(result), 422
    result['sensitivity'] = options_robust.model_sensitivity(run_kw, result)
    return jsonify(result)


@app.route('/api/options/compare', methods=['POST'])
@limiter.limit('15 per hour')
def options_compare():
    """Run every preset options strategy on the same underlying and window, ranked by return."""
    data = request.get_json() or {}
    try:
        args = _options_args(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    market = options_bt.load_market(args['ticker'], args['start'], args['end'])
    if 'error' in market:
        return jsonify(market), 422
    rows, rets, bench = [], [], None
    for key, p in options_bt.PRESETS.items():
        r = options_bt.run(legs=p['legs'], name=p['name'], alloc_pct=p['alloc'], market=market, **args)
        if 'error' in r:
            continue
        bench = r['benchmark']
        rets.append(options_robust.excess_returns(r))
        m = r['metrics']
        rows.append({'key': key, 'name': p['name'], 'group': p['group'], 'legs_count': len(p['legs']),
                     'headline': r['strategy']['headline'], 'alloc_pct': p['alloc'],
                     **{k: m[k] for k in ('total_return', 'cagr', 'sharpe', 'max_drawdown', 'win_rate',
                                          'trades', 'psr', 'total_costs')},
                     'equity': [e['value'] for e in r['equity']][::5]})
    deflation = options_robust.deflate(rows, rets)
    rows.sort(key=lambda r: -(r['total_return'] or -1e9))
    return jsonify({'ticker': args['ticker'], 'start_date': args['start'], 'end_date': args['end'],
                    'dte': args['dte'], 'benchmark': bench, 'results': rows, 'deflation': deflation})


@app.route('/api/options/sweep', methods=['POST'])
@limiter.limit('10 per hour')
def options_sweep():
    """One strategy across days to expiry x exit rules, with PBO and a Deflated Sharpe for the best."""
    data = request.get_json() or {}
    try:
        args = _options_args(data)
        s_dt = datetime.strptime(args['start'], '%Y-%m-%d')
        e_dt = datetime.strptime(args['end'], '%Y-%m-%d')
        if (e_dt - s_dt).days > options_robust.SWEEP_MAX_YEARS * 366:
            raise ValueError(f'The robustness test runs on at most {options_robust.SWEEP_MAX_YEARS} years')
        if (e_dt - s_dt).days < 365:
            raise ValueError('The robustness test needs at least a year of history')
        preset = data.get('preset')
        if preset:
            if preset not in options_bt.PRESETS:
                raise ValueError('Unknown strategy')
            p = options_bt.PRESETS[preset]
            legs, name = p['legs'], p['name']
            alloc = _num(data, 'alloc_pct', p['alloc'], 1, 100, 'Capital per position')
        else:
            legs = options_bt.clean_legs(data.get('legs'))
            name = str(data.get('name') or 'Custom strategy').strip()[:60] or 'Custom strategy'
            alloc = _num(data, 'alloc_pct', 10, 1, 100, 'Capital per position')
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    market = options_bt.load_market(args['ticker'], args['start'], args['end'])
    if 'error' in market:
        return jsonify(market), 422
    current = args['dte']
    result = options_robust.sweep(legs=legs, name=name, alloc_pct=alloc, market=market,
                                  current_dte=current, **args)
    if 'error' in result:
        return jsonify(result), 422
    return jsonify(result)


# ── Portfolio optimization ─────────────────────────────────────────────────────

@app.route('/api/portfolio/optimize', methods=['POST'])
@limiter.limit('20 per hour')
def optimize_portfolio():
    """
    Compute the Markowitz efficient frontier for the requested tickers.

    Body (all optional):
        tickers    — list of ticker symbols  (default: watchlist)
        start_date — YYYY-MM-DD              (default: 1 year ago)
        end_date   — YYYY-MM-DD              (default: today)
        n_points   — frontier resolution     (default: 60)
    """
    data       = request.get_json() or {}
    tickers    = data.get('tickers', strategies.WATCHLIST)
    start_date = data.get('start_date',
                          (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date   = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    n_points   = int(data.get('n_points', 60))
    shrink     = bool(data.get('shrink', False))

    if not tickers or len(tickers) < 2:
        return jsonify({'error': 'At least 2 tickers required'}), 400

    result = portopt.compute_efficient_frontier(tickers, start_date, end_date, n_points, shrink=shrink)
    if 'error' in result:
        return jsonify(result), 422
    return jsonify(result)


# ── Health check ──────────────────────────────────────────────────────────────

@app.route('/health')
@limiter.exempt
def health():
    # Exempt from rate limiting: the keep-warm self-ping (and any external
    # uptime monitor) hits this frequently to keep the free instance awake.
    return jsonify({
        'status': 'ok',
        'ts':     datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        'ml':     'loaded' if ml_runtime.get_info().get('loaded') else 'unavailable',
        # Which database the live ledger writes to. "sqlite" on Render's free
        # tier means an ephemeral disk: the live ledger resets on redeploy,
        # which is why the committed record (tools/forward_ledger.py) exists.
        'db':     'postgres' if db.is_postgres() else 'sqlite',
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
