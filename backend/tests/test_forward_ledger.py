"""
The committed forward record (tools/forward_ledger.py) and the grading rules it
shares with the live ledger (track_record.py). Offline: predictions and closes
are fixtures.
"""

import os
import sys
from datetime import date, datetime, timezone

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools'))

import forward_ledger as fl
import track_record as tr

# 15:40 ET on Monday 2026-11-02 (EST is UTC-5).
NOW = datetime(2026, 11, 2, 20, 40, tzinfo=timezone.utc)


def pred(ticker, direction='BULLISH', p_up=0.6, price=100.0, available=True):
    return {'available': available, 'ticker': ticker, 'price': price, 'horizon': 5,
            'prediction': {'direction': direction, 'p_up': p_up, 'signal': 'BUY', 'confidence': 0.2}}


def test_due_date_is_horizon_business_days_later():
    assert tr.due_date(date(2026, 11, 2), 5) == pd.Timestamp('2026-11-09')   # Mon -> next Mon
    assert tr.due_date(date(2026, 11, 6), 1) == pd.Timestamp('2026-11-09')   # Fri -> Mon


def test_grade_rules():
    assert tr.grade(100, 103, 'BULLISH') == {'realized_pct': 3.0, 'realized_up': 1, 'correct': 1}
    assert tr.grade(100, 103, 'BEARISH')['correct'] == 0
    assert tr.grade(100, 97, 'BEARISH')['correct'] == 1
    assert tr.grade(100, 103, 'NEUTRAL')['correct'] is None


def test_logs_each_basket_ticker_once_per_day():
    rows = []
    added = fl.log_calls(rows, lambda t: pred(t), NOW, 'v1', basket=['SPY', 'AAPL'])
    assert added == ['SPY', 'AAPL']
    assert rows[0]['created_date'] == '2026-11-02' and rows[0]['due_date'] == '2026-11-09'
    assert rows[0]['model_version'] == 'v1' and rows[0]['status'] == 'pending'
    # A second run the same day adds nothing.
    assert fl.log_calls(rows, lambda t: pred(t), NOW, 'v1', basket=['SPY', 'AAPL']) == []
    assert len(rows) == 2


def test_unavailable_or_failing_predictions_are_skipped():
    def predict(t):
        if t == 'BAD':
            raise RuntimeError('timeout')
        return pred(t, available=(t != 'OFF'))
    rows = []
    assert fl.log_calls(rows, predict, NOW, 'v1', basket=['BAD', 'OFF', 'SPY']) == ['SPY']


def test_grading_waits_for_the_due_date_and_never_regrades():
    rows = []
    fl.log_calls(rows, lambda t: pred(t, price=100.0), NOW, 'v1', basket=['SPY'])
    asked = []

    def close(ticker, due):
        asked.append((ticker, due))
        return 104.0

    assert fl.grade_rows(rows, close, date(2026, 11, 6), NOW) == []          # not due yet
    assert asked == []
    assert fl.grade_rows(rows, close, date(2026, 11, 9), NOW) == ['SPY 2026-11-02']
    assert rows[0]['status'] == 'graded' and rows[0]['correct'] == 1
    assert rows[0]['realized_pct'] == 4.0
    # Graded rows are final, even if the data source would now say otherwise.
    assert fl.grade_rows(rows, lambda t, d: 50.0, date(2026, 11, 20), NOW) == []
    assert rows[0]['price_at_grade'] == 104.0


def test_no_close_yet_leaves_the_row_pending():
    rows = []
    fl.log_calls(rows, lambda t: pred(t), NOW, 'v1', basket=['SPY'])
    assert fl.grade_rows(rows, lambda t, d: None, date(2026, 11, 9), NOW) == []
    assert rows[0]['status'] == 'pending'


def test_csv_round_trip_and_summary_shape(tmp_path):
    rows = []
    fl.log_calls(rows, lambda t: pred(t, direction={'SPY': 'BULLISH', 'QQQ': 'BEARISH'}.get(t, 'NEUTRAL')),
                 NOW, 'v1', basket=['SPY', 'QQQ', 'DIA'])
    fl.grade_rows(rows, lambda t, d: 102.0, date(2026, 11, 9), NOW)
    path = str(tmp_path / 'ledger.csv')
    fl.write_rows(rows, path)
    back = fl.read_rows(path)
    assert [r['ticker'] for r in back] == ['SPY', 'QQQ', 'DIA']

    out = fl.summary_json(back, NOW)
    assert out['available'] and out['source'] == 'git' and out['n_total'] == 3 and out['n_pending'] == 0
    s = out['summary']
    assert s['n_graded'] == 3 and s['n_scored'] == 2        # the neutral call is not scored
    assert s['hit_rate'] == 50.0                            # SPY right, QQQ wrong
    assert isinstance(out['rows'][0]['p_up'], float)


def test_late_runs_still_log_the_day():
    """GitHub starts scheduled runs hours late; any time from 15:15 ET to midnight counts."""
    ny = fl.NY
    at = lambda h, m: datetime(2026, 11, 6, h, m, tzinfo=ny)  # noqa: E731
    assert not fl.can_log(at(14, 59))
    assert fl.can_log(at(15, 15)) and fl.can_log(at(15, 40))
    assert fl.can_log(at(18, 49))           # the 2026-11-06 run, three hours late
    assert fl.can_log(at(23, 59))
    assert not fl.can_log(at(9, 30))        # next morning: that day's close is gone
