from datetime import datetime
import pytz

# ── Risk profiles ─────────────────────────────────────────────────────────────
# Each profile controls position sizing, stop/take levels, cash reserve,
# and behaviour in high-volatility regimes.
RISK_PROFILES = {
    'conservative': {
        'stop_loss_pct':    0.03,   # 3%  — kept for legacy display
        'trail_pct':        0.10,   # 10% — wide enough to not whipsaw on noise
        'take_profit_pct':  0.40,   # 40% — let winners run (only caps blow-off tops)
        'max_position_pct': 0.12,   # per-position cap
        'min_cash_reserve': 0.15,   # 15% cash cushion (more deployed than before)
        'max_daily_trades': 8,
        'vol_size_mult':    0.40,   # 40% normal size in HIGH_VOLATILITY
        'trade_high_vol':   False,  # sit out HIGH_VOLATILITY regime entirely
    },
    'moderate': {
        'stop_loss_pct':    0.05,   # 5%  — kept for legacy display
        'trail_pct':        0.15,   # 15% — gives trends room to breathe
        'take_profit_pct':  1.00,   # 100% — effectively off; trend/stop is the exit
        'max_position_pct': 0.18,
        'min_cash_reserve': 0.05,   # stay close to fully invested
        'max_daily_trades': 12,
        'vol_size_mult':    0.60,
        'trade_high_vol':   True,
    },
    'aggressive': {
        'stop_loss_pct':    0.07,   # 7%  — kept for legacy display
        'trail_pct':        0.22,   # 22% — only a major reversal exits
        'take_profit_pct':  2.00,   # effectively off — ride the full trend
        'max_position_pct': 0.30,   # concentrated bets
        'min_cash_reserve': 0.02,
        'max_daily_trades': 20,
        'vol_size_mult':    1.00,
        'trade_high_vol':   True,
    },
}

# Default profile (backward compatible)
_DEFAULT = RISK_PROFILES['moderate']

# Legacy module-level constants (kept for backward compatibility)
STOP_LOSS_PCT    = _DEFAULT['stop_loss_pct']
TAKE_PROFIT_PCT  = _DEFAULT['take_profit_pct']
MAX_POSITION_PCT = _DEFAULT['max_position_pct']
MAX_DAILY_TRADES = _DEFAULT['max_daily_trades']
MIN_CASH_RESERVE = _DEFAULT['min_cash_reserve']
MARKET_OPEN_H    = 9
MARKET_OPEN_M    = 30
MARKET_CLOSE_H   = 16
MARKET_CLOSE_M   = 0

_daily_trade_count = 0
_last_trade_date   = None


def get_risk_profile(tolerance: str = 'moderate') -> dict:
    """Return the risk parameter dict for the given tolerance level."""
    return RISK_PROFILES.get(tolerance, RISK_PROFILES['moderate'])


def is_market_open() -> bool:
    est   = pytz.timezone('America/New_York')
    now   = datetime.now(est)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,  second=0, microsecond=0)
    close_ = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    return open_ <= now < close_


def _reset_if_new_day():
    global _daily_trade_count, _last_trade_date
    today = datetime.now().date()
    if _last_trade_date != today:
        _daily_trade_count = 0
        _last_trade_date   = today


def check_daily_trade_limit(profile: dict = None) -> bool:
    _reset_if_new_day()
    limit = (profile or _DEFAULT)['max_daily_trades']
    return _daily_trade_count < limit


def increment_trade_count():
    _reset_if_new_day()
    global _daily_trade_count
    _daily_trade_count += 1


def get_daily_trade_count() -> int:
    _reset_if_new_day()
    return _daily_trade_count


def calculate_position_size_kelly(portfolio_value: float, price: float, cash: float,
                                   kelly_fraction: float | None,
                                   profile: dict = None) -> int:
    """
    Kelly Criterion position sizing.

    Uses the Kelly fraction (derived from win rate + odds ratio of past trades)
    instead of a fixed percentage, capped at the profile's max_position_pct
    as a hard safety limit.

    Falls back to calculate_position_size() when kelly_fraction is None
    (insufficient trade history — fewer than 10 closed trades).
    """
    if kelly_fraction is None or kelly_fraction <= 0:
        return calculate_position_size(portfolio_value, price, cash, profile)

    p = profile or _DEFAULT
    # Kelly fraction as % of portfolio, capped at the profile's hard limit
    effective_pct = min(kelly_fraction, p['max_position_pct'])
    max_spend     = portfolio_value * effective_pct
    usable_cash   = cash - (portfolio_value * p['min_cash_reserve'])

    if usable_cash <= 0 or price <= 0:
        return 0
    spend = min(max_spend, usable_cash)
    return max(int(spend / price), 0)


def calculate_position_size(portfolio_value: float, price: float, cash: float,
                             profile: dict = None) -> int:
    """
    Return number of whole shares to buy.

    Respects the profile's max_position_pct and min_cash_reserve.
    Pass profile=get_risk_profile(tolerance) to override defaults.
    """
    p = profile or _DEFAULT
    max_by_pct  = portfolio_value * p['max_position_pct']
    usable_cash = cash - (portfolio_value * p['min_cash_reserve'])
    if usable_cash <= 0 or price <= 0:
        return 0
    spend  = min(max_by_pct, usable_cash)
    return max(int(spend / price), 0)


def calculate_stop_loss(entry_price: float, profile: dict = None,
                        high_price: float = None) -> float:
    p   = profile or _DEFAULT
    ref = high_price if high_price is not None else entry_price
    return round(ref * (1 - p['trail_pct']), 2)


def calculate_take_profit(entry_price: float, profile: dict = None) -> float:
    pct = (profile or _DEFAULT)['take_profit_pct']
    return round(entry_price * (1 + pct), 2)


def check_stop_take(current_price: float, entry_price: float,
                    profile: dict = None,
                    high_since_entry: float = None) -> str | None:
    """Returns 'stop_loss', 'take_profit', or None."""
    p          = profile or _DEFAULT
    trail_high = high_since_entry if high_since_entry is not None else entry_price
    if current_price <= trail_high * (1 - p['trail_pct']):
        return 'stop_loss'
    if current_price >= entry_price * (1 + p['take_profit_pct']):
        return 'take_profit'
    return None


def can_trade(portfolio_value: float, cash: float,
              profile: dict = None) -> tuple[bool, str]:
    """Master gate — returns (allowed, reason_string)."""
    p = profile or _DEFAULT
    if not is_market_open():
        return False, 'Market is closed'
    if not check_daily_trade_limit(p):
        return False, f'Daily trade limit ({p["max_daily_trades"]}) reached'
    if portfolio_value > 0 and (cash / portfolio_value) < p['min_cash_reserve']:
        return False, f'Cash below {int(p["min_cash_reserve"]*100)}% reserve minimum'
    return True, 'OK'
