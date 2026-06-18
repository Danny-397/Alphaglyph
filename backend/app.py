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
import bot
import database
import features  # noqa: F401 — imported so startup errors surface early
import ml_runtime
import portfolio as portopt
import regime as reg
import risk
import simulator
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


# Per-IP rate limiting. The dashboard polls ~24 req/min per open tab, so the
# default ceiling is generous enough for normal use (several tabs) while still
# stopping a script that floods the API. The expensive endpoints (backtest,
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


# ── Owner-only controls ─────────────────────────────────────────────────────
# Set ADMIN_TOKEN on the server to make the bot controls (start/stop/strategy/
# risk) owner-only on a public demo, so visitors can't stop or hijack the bot.
# When unset, controls stay open (local/dev). The owner unlocks the dashboard
# by visiting it once with ?admin=<token> (stored in the browser thereafter).
ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', '').strip()


def _admin_ok() -> bool:
    if not ADMIN_TOKEN:
        return True
    sent = (request.headers.get('X-Admin-Token', '')
            or request.args.get('admin', '')).strip()
    return sent == ADMIN_TOKEN


def _deny_admin():
    return jsonify({'error': 'Bot controls are owner-only on this live demo.'}), 403


database.init_db()
simulator.init_simulator()
# Render free tier spins the process down on inactivity and gunicorn can recycle
# the worker — either kills the in-memory bot thread. The bot is now tick-driven
# (cycles run on incoming requests, source of truth is the DB), so it can't stay
# stopped. ensure_running_default() keeps the demo bot ON by default and resumes
# the heartbeat after a restart. Set BOT_AUTOSTART=false to require a manual start.
bot.ensure_running_default()

VALID_STRATEGIES = strategies.VALID_STRATEGIES + ('adaptive',)


# ── Status / control ──────────────────────────────────────────────────────────

@app.route('/api/status')
def get_status():
    # Every dashboard poll nudges the engine — if a trading cycle is due, it runs
    # in the background. This is what keeps the bot alive without a long-running
    # thread: traffic drives it. Non-blocking, so the status response stays fast.
    try:
        bot.maybe_run_cycle()
    except Exception as exc:
        logger.warning('status tick failed: %s', exc)

    state         = database.get_bot_state()
    strategy      = state['strategy']       if state else 'adaptive'
    risk_tol      = state['risk_tolerance'] if state else 'moderate'
    portfolio     = bot.get_portfolio_summary()
    metrics       = database.get_performance_metrics(strategy)
    live_metrics  = database.get_live_metrics()
    regime_info   = bot.get_last_regime()
    kelly         = database.compute_kelly_fraction(strategy)

    return jsonify({
        'is_running':      bot.is_running(),
        'strategy':        strategy,
        'risk_tolerance':  risk_tol,
        'started_at':      state.get('started_at') if state else None,
        'market_open':     risk.is_market_open(),
        'daily_trades':    risk.get_daily_trade_count(),
        'max_daily':       risk.get_risk_profile(risk_tol)['max_daily_trades'],
        'portfolio':       portfolio,
        'metrics':         metrics,
        'live_metrics':    live_metrics,
        'regime':          regime_info,
        'kelly_fraction':  round(kelly * 100, 2) if kelly else None,
        'admin_required':  bool(ADMIN_TOKEN),
        'next_cycle_in':   bot.seconds_until_next_cycle(),
        'cycle_seconds':   bot._CYCLE_SECONDS,
    })


@app.route('/api/start', methods=['POST'])
def start():
    if not _admin_ok():
        return _deny_admin()
    data     = request.get_json() or {}
    strategy = data.get('strategy', 'adaptive')
    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': 'Invalid strategy'}), 400
    database.update_bot_state(strategy=strategy)
    success, msg = bot.start_bot()
    return jsonify({'success': success, 'message': msg})


@app.route('/api/stop', methods=['POST'])
def stop():
    if not _admin_ok():
        return _deny_admin()
    success, msg = bot.stop_bot()
    return jsonify({'success': success, 'message': msg})


@app.route('/api/strategy', methods=['POST'])
def set_strategy():
    if not _admin_ok():
        return _deny_admin()
    data     = request.get_json() or {}
    strategy = data.get('strategy')
    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': 'Invalid strategy'}), 400
    database.update_bot_state(strategy=strategy)
    return jsonify({'success': True, 'strategy': strategy})


@app.route('/api/risk_tolerance', methods=['POST'])
def set_risk_tolerance():
    if not _admin_ok():
        return _deny_admin()
    data      = request.get_json() or {}
    tolerance = data.get('tolerance', 'moderate')
    if tolerance not in ('conservative', 'moderate', 'aggressive'):
        return jsonify({'error': 'Invalid risk tolerance'}), 400
    database.update_bot_state(risk_tolerance=tolerance)
    return jsonify({'success': True, 'risk_tolerance': tolerance,
                    'profile': risk.get_risk_profile(tolerance)})


# ── Regime ────────────────────────────────────────────────────────────────────

@app.route('/api/regime')
def get_regime():
    """Detect current market regime from SPY and return the result."""
    try:
        spy_df = features.fetch_ohlcv('SPY', period='6mo')
        if spy_df is None or len(spy_df) < 60:
            return jsonify({'error': 'Insufficient SPY data'}), 503
        result = reg.detect_regime(spy_df)
        state    = database.get_bot_state()
        risk_tol = state.get('risk_tolerance', 'moderate') if state else 'moderate'
        return jsonify({
            'regime':      result.regime,
            'label':       result.label,
            'description': result.description,
            'strategy':    reg.get_regime_strategy(result.regime, risk_tol),
            'adx':         result.adx,
            'plus_di':     result.plus_di,
            'minus_di':    result.minus_di,
            'bb_width':    result.bb_width,
            'vol_30d':     result.vol_30d,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.route('/api/trades')
def get_trades():
    limit    = request.args.get('limit', 20, type=int)
    strategy = request.args.get('strategy')
    return jsonify(database.get_trades(limit=limit, strategy=strategy))


@app.route('/api/portfolio/history')
def portfolio_history():
    strategy = request.args.get('strategy')
    limit    = request.args.get('limit', 500, type=int)
    return jsonify(database.get_portfolio_history(strategy=strategy, limit=limit))


@app.route('/api/activity')
def activity():
    return jsonify(bot.get_activity_log())


@app.route('/api/indicators')
def indicators():
    strategy = request.args.get('strategy', 'ma_crossover')
    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': 'Invalid strategy'}), 400
    result = {}
    for ticker in strategies.WATCHLIST:
        result[ticker] = strategies.get_indicator_data(ticker, strategy)
    return jsonify(result)


@app.route('/api/watchlist')
def get_watchlist():
    return jsonify(strategies.WATCHLIST)


@app.route('/api/ml/info')
def ml_info():
    """ML transformer status: whether a trained model is deployed, its
    architecture, validation/test metrics, and decision thresholds.
    The frontend uses this to enable the ML strategy option."""
    return jsonify(ml_runtime.get_info())


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
    commission_pct  = float(data.get('commission_pct', 0.001))
    slippage_pct    = float(data.get('slippage_pct',   0.0005))
    use_markowitz   = bool(data.get('use_markowitz', False))

    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': 'Invalid strategy'}), 400
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

    if not tickers or len(tickers) < 2:
        return jsonify({'error': 'At least 2 tickers required'}), 400

    result = portopt.compute_efficient_frontier(tickers, start_date, end_date, n_points)
    if 'error' in result:
        return jsonify(result), 422
    return jsonify(result)


# ── Health check ──────────────────────────────────────────────────────────────

@app.route('/health')
@limiter.exempt
def health():
    # Exempt: UptimeRobot (or similar) pings this frequently to keep the free
    # Render instance warm — it must never be rate limited.
    #
    # Self-healing watchdog: a keep-warm ping (UptimeRobot etc.) both resumes the
    # heartbeat thread AND runs a trading cycle if one is due. With a ~5-minute
    # ping the bot can never stay down. Wrapped so a hiccup never fails the check.
    try:
        bot.resume_if_running()
        bot.maybe_run_cycle()
    except Exception as exc:
        logger.warning('health watchdog failed: %s', exc)
    return jsonify({
        'status':  'ok',
        'ts':      datetime.utcnow().isoformat(),
        'running': bot.is_running(),
        'engine_thread': bot.engine_thread_alive(),
        'next_cycle_in': bot.seconds_until_next_cycle(),
        # 'postgres' = persistent (survives redeploys); 'sqlite' = ephemeral.
        'db':      'postgres' if database._USE_PG else 'sqlite',
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
