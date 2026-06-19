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


class TestStopTakeCalculations:
    def test_stop_loss_is_trail_below_entry(self):
        assert risk.calculate_stop_loss(100.0) == round(100.0 * (1 - TRAIL), 2)

    def test_take_profit_is_tp_above_entry(self):
        assert risk.calculate_take_profit(100.0) == round(100.0 * (1 + TP), 2)

    def test_stop_loss_rounds_to_two_decimals(self):
        assert risk.calculate_stop_loss(99.99) == round(99.99 * (1 - TRAIL), 2)

    def test_take_profit_rounds_to_two_decimals(self):
        assert risk.calculate_take_profit(99.99) == round(99.99 * (1 + TP), 2)


class TestCheckStopTake:
    def test_triggers_stop_loss_when_at_threshold(self):
        assert risk.check_stop_take(100.0 * (1 - TRAIL), 100.0) == 'stop_loss'

    def test_triggers_stop_loss_when_below_threshold(self):
        assert risk.check_stop_take(100.0 * (1 - TRAIL) - 1.0, 100.0) == 'stop_loss'

    def test_triggers_take_profit_when_at_threshold(self):
        assert risk.check_stop_take(100.0 * (1 + TP), 100.0) == 'take_profit'

    def test_triggers_take_profit_when_above_threshold(self):
        assert risk.check_stop_take(100.0 * (1 + TP) + 5.0, 100.0) == 'take_profit'

    def test_returns_none_when_between_levels(self):
        assert risk.check_stop_take(100.0, 100.0) is None
        assert risk.check_stop_take(100.0 * (1 - TRAIL) + 0.01, 100.0) is None

    def test_trailing_stop_triggers_from_high_not_entry(self):
        # Entry 100, price ran to 120, then dropped a full trail from the high.
        high = 120.0
        assert risk.check_stop_take(high * (1 - TRAIL), 100.0, high_since_entry=high) == 'stop_loss'

    def test_trailing_stop_does_not_trigger_within_trail(self):
        # 110 high, price only slightly below — within the trail → no stop.
        assert risk.check_stop_take(110.0 * (1 - TRAIL) + 1.0, 100.0, high_since_entry=110.0) is None

    def test_trailing_stop_matches_fixed_stop_at_entry(self):
        # When high equals entry the trailing stop is identical to the entry stop.
        assert risk.check_stop_take(100.0 * (1 - TRAIL), 100.0, high_since_entry=100.0) == 'stop_loss'

    def test_trailing_stop_level_displayed_correctly(self):
        assert risk.calculate_stop_loss(100.0, high_price=120.0) == round(120.0 * (1 - TRAIL), 2)

    def test_trailing_stop_level_falls_back_to_entry(self):
        assert risk.calculate_stop_loss(100.0) == round(100.0 * (1 - TRAIL), 2)


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


class TestDailyTradeLimit:
    def test_limit_not_reached_initially(self):
        risk._daily_trade_count = 0
        risk._last_trade_date = None
        assert risk.check_daily_trade_limit() is True

    def test_limit_reached_after_max_trades(self):
        risk._daily_trade_count = 0
        risk._last_trade_date = None
        for _ in range(risk.MAX_DAILY_TRADES):
            risk.increment_trade_count()
        assert risk.check_daily_trade_limit() is False

    def test_count_increments_correctly(self):
        risk._daily_trade_count = 0
        risk._last_trade_date = None
        risk.increment_trade_count()
        risk.increment_trade_count()
        assert risk.get_daily_trade_count() == 2


class TestCanTrade:
    def test_rejects_when_market_closed(self, monkeypatch):
        monkeypatch.setattr(risk, 'is_market_open', lambda: False)
        monkeypatch.setattr(risk, 'check_daily_trade_limit', lambda: True)
        ok, reason = risk.can_trade(100_000, 50_000)
        assert ok is False
        assert 'closed' in reason.lower()

    def test_rejects_when_daily_limit_hit(self, monkeypatch):
        monkeypatch.setattr(risk, 'is_market_open', lambda: True)
        monkeypatch.setattr(risk, 'check_daily_trade_limit', lambda *a, **kw: False)
        ok, reason = risk.can_trade(100_000, 50_000)
        assert ok is False
        assert 'limit' in reason.lower()

    def test_rejects_when_cash_below_reserve(self, monkeypatch):
        monkeypatch.setattr(risk, 'is_market_open', lambda: True)
        monkeypatch.setattr(risk, 'check_daily_trade_limit', lambda *a, **kw: True)
        # Cash strictly below the reserve floor.
        ok, reason = risk.can_trade(100_000, int(100_000 * RESERVE) - 1)
        assert ok is False
        assert 'reserve' in reason.lower()

    def test_allows_trade_when_all_gates_pass(self, monkeypatch):
        monkeypatch.setattr(risk, 'is_market_open', lambda: True)
        monkeypatch.setattr(risk, 'check_daily_trade_limit', lambda *a, **kw: True)
        ok, reason = risk.can_trade(100_000, 50_000)
        assert ok is True
        assert reason == 'OK'
