"""
Flask REST API for TradeBot.
All endpoints return JSON.  The frontend polls these on a 10-second interval.
"""

import logging
import os
from datetime import datetime, timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

import backtest as backtester
import bot
import database
import features  # noqa: F401 — imported so startup errors surface early
import regime as reg
import risk
import strategies

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)-20s  %(levelname)s  %(message)s',
)

logger = logging.getLogger(__name__)

# ── Startup validation ────────────────────────────────────────────────────────
_missing = [k for k in ('ALPACA_API_KEY', 'ALPACA_SECRET_KEY')
            if not os.getenv(k, '').strip()]
if _missing:
    logger.warning(
        'Missing environment variables: %s — '
        'the bot will not be able to connect to Alpaca until these are set.',
        ', '.join(_missing)
    )

app = Flask(__name__)

# CORS: in production, replace the origins list with your Vercel URL to lock
# down which frontends can call this API.
# e.g. CORS(app, origins=['https://your-project.vercel.app'])
# For local development, allow all origins.
_cors_origins = os.getenv('CORS_ORIGINS', '*')
CORS(app, origins=_cors_origins)
database.init_db()

VALID_STRATEGIES = strategies.VALID_STRATEGIES + ('adaptive',)


# ── Status / control ──────────────────────────────────────────────────────────

@app.route('/api/status')
def get_status():
    state         = database.get_bot_state()
    strategy      = state['strategy']      if state else 'adaptive'
    risk_tol      = state['risk_tolerance'] if state else 'moderate'
    portfolio     = bot.get_portfolio_summary()
    metrics       = database.get_performance_metrics(strategy)
    live_metrics  = database.get_live_metrics()
    regime_info   = bot.get_last_regime()

    return jsonify({
        'is_running':     bot.is_running(),
        'strategy':       strategy,
        'risk_tolerance': risk_tol,
        'started_at':     state.get('started_at') if state else None,
        'market_open':    risk.is_market_open(),
        'daily_trades':   risk.get_daily_trade_count(),
        'max_daily':      risk.get_risk_profile(risk_tol)['max_daily_trades'],
        'portfolio':      portfolio,
        'metrics':        metrics,
        'live_metrics':   live_metrics,
        'regime':         regime_info,
    })


@app.route('/api/start', methods=['POST'])
def start():
    data     = request.get_json() or {}
    strategy = data.get('strategy', 'adaptive')
    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': 'Invalid strategy'}), 400
    database.update_bot_state(strategy=strategy)
    success, msg = bot.start_bot()
    return jsonify({'success': success, 'message': msg})


@app.route('/api/stop', methods=['POST'])
def stop():
    success, msg = bot.stop_bot()
    return jsonify({'success': success, 'message': msg})


@app.route('/api/strategy', methods=['POST'])
def set_strategy():
    data     = request.get_json() or {}
    strategy = data.get('strategy')
    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': 'Invalid strategy'}), 400
    database.update_bot_state(strategy=strategy)
    return jsonify({'success': True, 'strategy': strategy})


@app.route('/api/risk_tolerance', methods=['POST'])
def set_risk_tolerance():
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


# ── Backtesting ───────────────────────────────────────────────────────────────

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    data            = request.get_json() or {}
    strategy        = data.get('strategy', 'adaptive')
    tickers         = data.get('tickers', ['AAPL', 'MSFT', 'SPY'])
    start_date      = data.get('start_date',
                               (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date        = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    initial_capital = float(data.get('initial_capital', 100_000))
    walk_forward    = bool(data.get('walk_forward', False))
    risk_tolerance  = data.get('risk_tolerance', 'moderate')

    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': 'Invalid strategy'}), 400
    if not (1_000 <= initial_capital <= 10_000_000):
        return jsonify({'error': 'Capital must be between $1,000 and $10,000,000'}), 400
    if not tickers:
        return jsonify({'error': 'At least one ticker required'}), 400
    if risk_tolerance not in ('conservative', 'moderate', 'aggressive'):
        return jsonify({'error': 'Invalid risk tolerance'}), 400

    result = backtester.run_backtest(
        strategy, tickers, start_date, end_date,
        initial_capital, walk_forward, risk_tolerance
    )
    return jsonify(result)


# ── Health check ──────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'ts': datetime.utcnow().isoformat()})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
