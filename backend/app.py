"""
Flask REST API for AlphaGlyph.
All endpoints return JSON.  The frontend polls these on a 10-second interval.
"""

import logging
import os
from datetime import datetime, timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

import backtest as backtester
import calibration as calib
import datamining
import features  # noqa: F401 — imported so startup errors surface early
import ml_runtime
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

# CORS: in production, replace the origins list with your Vercel URL to lock
# down which frontends can call this API.
# e.g. CORS(app, origins=['https://your-project.vercel.app'])
# For local development, allow all origins.
_cors_origins = os.getenv('CORS_ORIGINS', '*')
CORS(app, origins=_cors_origins)


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


def _start_keepalive():
    """
    Keep the Render free instance awake by pinging its own public URL on a timer.

    Render spins the service down after ~15 min with no INBOUND traffic. A
    request to our own public URL is routed back through Render's edge as inbound
    traffic, so it resets the idle timer — keeping the site warm (fast ML
    forecasts, charts, backtests) with no external uptime service. (GitHub
    Actions cron is too throttled to rely on: scheduled runs can lag by hours.)

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
            try:
                requests.get(target, timeout=20)
            except Exception as exc:
                logger.warning('keepalive ping failed: %s', exc)

    threading.Thread(target=loop, daemon=True, name='keepalive').start()
    logger.info('Self-keepalive enabled → %s every %ds', target, interval)


_start_keepalive()

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
        return jsonify(predictor_engine.predict(symbol))
    except Exception as exc:
        logger.exception('predict failed for %s', symbol)
        return jsonify({'error': str(exc)}), 500


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
    # The API is fully stateless — no bot, no database to report on.
    return jsonify({
        'status': 'ok',
        'ts':     datetime.utcnow().isoformat(),
        'ml':     'loaded' if ml_runtime.get_info().get('loaded') else 'unavailable',
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
