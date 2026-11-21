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

# Smallest Kelly-sized position, as a share of the profile's max_position_pct.
KELLY_FLOOR_FRAC = 0.25

def get_risk_profile(tolerance: str = 'moderate') -> dict:
    """Return the risk parameter dict for the given tolerance level."""
    return RISK_PROFILES.get(tolerance, RISK_PROFILES['moderate'])


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
    # Kelly fraction as % of portfolio, between a floor and the profile's hard
    # limit. Without the floor a small positive edge (half-Kelly of 0.3% is
    # common) buys less than one share, the strategy stops trading, and since
    # no new trades close, the Kelly estimate can never recover: a 2021-2026
    # SMA-crossover backtest on SPY/QQQ went flat in May 2024 for that reason.
    floor = p['max_position_pct'] * KELLY_FLOOR_FRAC
    effective_pct = min(max(kelly_fraction, floor), p['max_position_pct'])
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
