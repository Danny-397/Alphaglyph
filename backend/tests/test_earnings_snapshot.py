"""
Unit tests for the earnings snapshot fallback (earnings.py).

Yahoo refuses the live earnings-dates call from datacenter IPs, so production
depends on earnings_snapshot.json. These tests pin the fallback's contract:
live data wins when present, the snapshot fills in when it isn't, callers can
tell which they got, and a missing snapshot degrades instead of raising.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pytest

import earnings


SNAP = {
    'captured_at': '2026-10-25T17:30Z',
    'tickers': {
        'TEST': [
            {'date': '2099-01-30', 'eps_est': 1.5, 'eps_actual': None, 'surprise_pct': None},
            {'date': '2026-07-30', 'eps_est': 1.0, 'eps_actual': 1.1, 'surprise_pct': 10.0},
            {'date': '2026-04-30', 'eps_est': 0.9, 'eps_actual': 0.9, 'surprise_pct': 0.0},
        ],
    },
}


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    path = tmp_path / 'earnings_snapshot.json'
    path.write_text(json.dumps(SNAP), encoding='utf-8')
    monkeypatch.setattr(earnings, 'SNAPSHOT_PATH', str(path))
    monkeypatch.setattr(earnings, '_SNAPSHOT', None)
    monkeypatch.setattr(earnings, '_CACHE', {})
    return path


def test_live_data_wins_over_the_snapshot(snapshot, monkeypatch):
    live = pd.DataFrame({'eps_est': [2.0], 'eps_actual': [2.2], 'surprise_pct': [10.0]},
                        index=pd.to_datetime(['2026-10-01']))
    monkeypatch.setattr(earnings, '_fetch_live_frame', lambda t: live)
    frame = earnings._fetch_earnings_frame('TEST')
    assert frame is live
    assert 'snapshot_date' not in frame.attrs


def test_snapshot_fills_in_when_live_is_blocked(snapshot, monkeypatch):
    monkeypatch.setattr(earnings, '_fetch_live_frame', lambda t: None)
    frame = earnings._fetch_earnings_frame('TEST')
    assert list(frame.columns) == ['eps_est', 'eps_actual', 'surprise_pct']
    assert frame.index.is_monotonic_decreasing                  # newest first, like live
    assert frame['eps_actual'].isna().iloc[0]                   # the future report
    assert frame['surprise_pct'].iloc[1] == pytest.approx(10.0)
    assert frame.attrs['snapshot_date'] == '2026-10-25'


def test_signal_from_snapshot_says_so(snapshot, monkeypatch):
    monkeypatch.setattr(earnings, '_fetch_live_frame', lambda t: None)
    sig = earnings.earnings_signal('TEST')
    assert sig['available'] is True
    assert any('saved snapshot (2026-10-25)' in f for f in sig['factors'])


def test_ticker_missing_from_both_degrades(snapshot, monkeypatch):
    monkeypatch.setattr(earnings, '_fetch_live_frame', lambda t: None)
    assert earnings._fetch_earnings_frame('NOPE') is None
    assert earnings.earnings_signal('NOPE')['available'] is False


def test_unreadable_snapshot_degrades(tmp_path, monkeypatch):
    bad = tmp_path / 'earnings_snapshot.json'
    bad.write_text('{not json', encoding='utf-8')
    monkeypatch.setattr(earnings, 'SNAPSHOT_PATH', str(bad))
    monkeypatch.setattr(earnings, '_SNAPSHOT', None)
    monkeypatch.setattr(earnings, '_fetch_live_frame', lambda t: None)
    assert earnings._fetch_earnings_frame('TEST') is None


def test_committed_snapshot_covers_the_pead_universe():
    """The shipped file must cover every name the PEAD study pools, or the study
    silently shrinks in production."""
    import pead
    with open(os.path.join(os.path.dirname(earnings.__file__), 'earnings_snapshot.json'),
              encoding='utf-8') as fh:
        shipped = json.load(fh)['tickers']
    assert set(pead.DEFAULT_UNIVERSE) <= set(shipped)
