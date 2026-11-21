"""
Unit tests for risk.py.

All tests are pure Python — no network, no simulator, no database.

Expected values are derived from the active risk profile (not hardcoded), so
the suite tests behaviour/invariants and survives parameter tuning.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import risk

_P     = risk.get_risk_profile('moderate')
TRAIL  = _P['trail_pct']           # trailing-stop distance from peak
TP     = _P['take_profit_pct']     # take-profit distance from entry
MAXPOS = _P['max_position_pct']    # per-position cap (fraction of portfolio)
RESERVE = _P['min_cash_reserve']   # cash reserve floor (fraction of portfolio)


class TestPositionSizing:
    def test_respects_max_position_pct(self):
        # Plenty of cash → capped by max_position_pct.
        shares = risk.calculate_position_size(100_000, 100.0, 100_000)
        assert shares == int(100_000 * MAXPOS / 100.0)

    def test_respects_cash_reserve(self):
        # Cash exactly at the reserve floor → nothing usable → 0 shares.
        shares = risk.calculate_position_size(100_000, 100.0, int(100_000 * RESERVE))
        assert shares == 0

    def test_returns_zero_when_price_is_zero(self):
        assert risk.calculate_position_size(100_000, 0.0, 50_000) == 0

    def test_returns_zero_when_portfolio_is_zero(self):
        assert risk.calculate_position_size(0.0, 100.0, 0.0) == 0

    def test_returns_whole_shares_only(self):
        shares = risk.calculate_position_size(100_000, 333.0, 100_000)
        assert isinstance(shares, int)
        assert shares == int(100_000 * MAXPOS / 333.0)


# ── Kelly sizing floor ────────────────────────────────────────────────────────

def test_small_positive_kelly_still_buys_shares():
    """A tiny edge must mean a small position, not zero shares forever.

    Half-Kelly of 0.3% on a $100k book is $300, under one $700 SPY share.
    Before the floor that returned 0, so the backtest stopped trading and the
    Kelly estimate (fed only by closed trades) could never recover.
    """
    prof = risk.get_risk_profile('moderate')
    shares = risk.calculate_position_size_kelly(100_000, 700.0, 100_000, 0.003, prof)
    floor_spend = 100_000 * prof['max_position_pct'] * risk.KELLY_FLOOR_FRAC
    assert shares == int(floor_spend / 700.0) > 0


def test_kelly_is_still_capped_at_the_profile_max():
    prof = risk.get_risk_profile('moderate')
    shares = risk.calculate_position_size_kelly(100_000, 100.0, 100_000, 0.9, prof)
    assert shares == int(100_000 * prof['max_position_pct'] / 100.0)


def test_kelly_between_floor_and_cap_is_used_as_is():
    prof = risk.get_risk_profile('moderate')
    f = prof['max_position_pct'] * 0.6
    shares = risk.calculate_position_size_kelly(100_000, 100.0, 100_000, f, prof)
    assert shares == int(100_000 * f / 100.0)
