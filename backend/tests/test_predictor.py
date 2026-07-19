"""
Unit tests for the Predictor (earnings.py + predictor.py).

Everything runs offline: yfinance, GDELT sentiment, and the ML runtime are all
monkeypatched, so the tests exercise the scoring/blending logic — never the
network. The two properties that matter most are covered explicitly:

  * graceful degradation — every channel can vanish independently and the
    Predictor still returns a coherent card (or an honest "no signal");
  * honest blending — weights are (prior × confidence) renormalised over the
    channels that are actually available, and channel disagreement lowers the
    combined confidence.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

import earnings as earn
import predictor as pred


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _earnings_frame(surprise_pct, days_since, eps_now=2.0, eps_prev=1.6,
                    next_in_days=45):
    """Build a yfinance-shaped earnings-dates frame (newest first)."""
    today = pd.Timestamp.now().normalize()
    last  = today - pd.Timedelta(days=days_since)
    rows = [
        (today + pd.Timedelta(days=next_in_days), 2.1, np.nan, np.nan),  # future
        (last,                         eps_now / (1 + surprise_pct/100), eps_now, surprise_pct),
        (last - pd.Timedelta(days=91),  1.9, 1.95, 2.6),
        (last - pd.Timedelta(days=182), 1.8, 1.82, 1.1),
        (last - pd.Timedelta(days=273), eps_prev, eps_prev, 0.5),        # 4 back = YoY base
        (last - pd.Timedelta(days=365), 1.5, 1.55, 3.3),
    ]
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {'eps_est':      [r[1] for r in rows],
         'eps_actual':   [r[2] for r in rows],
         'surprise_pct': [r[3] for r in rows]},
        index=idx,
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    earn._CACHE.clear()
    pred._CACHE.clear()
    yield
    earn._CACHE.clear()
    pred._CACHE.clear()


# ── earnings.py ─────────────────────────────────────────────────────────────────

def test_beat_recent_is_bullish(monkeypatch):
    monkeypatch.setattr(earn, '_fetch_earnings_frame',
                        lambda t: _earnings_frame(surprise_pct=8.0, days_since=5))
    sig = earn.earnings_signal('AAA')
    assert sig['available'] is True
    assert sig['score'] > 0                    # a fresh beat leans bullish
    assert sig['surprise_pct'] == 8.0
    assert any('Beat' in f for f in sig['factors'])


def test_miss_recent_is_bearish(monkeypatch):
    monkeypatch.setattr(earn, '_fetch_earnings_frame',
                        lambda t: _earnings_frame(surprise_pct=-9.0, days_since=4))
    sig = earn.earnings_signal('BBB')
    assert sig['score'] < 0
    assert any('Missed' in f for f in sig['factors'])


def test_pead_decays_with_age(monkeypatch):
    """A beat's drift signal should shrink as the surprise ages out."""
    monkeypatch.setattr(earn, '_fetch_earnings_frame',
                        lambda t: _earnings_frame(surprise_pct=8.0, days_since=5))
    fresh = earn.earnings_signal('AAA')['score']
    earn._CACHE.clear()
    monkeypatch.setattr(earn, '_fetch_earnings_frame',
                        lambda t: _earnings_frame(surprise_pct=8.0, days_since=55))
    stale = earn.earnings_signal('AAA')['score']
    assert fresh > stale                       # drift faded (growth component remains)


def test_imminent_report_cuts_confidence(monkeypatch):
    monkeypatch.setattr(earn, '_fetch_earnings_frame',
                        lambda t: _earnings_frame(surprise_pct=6.0, days_since=40, next_in_days=3))
    sig = earn.earnings_signal('CCC')
    assert sig['event_risk'] is True
    assert any('binary event' in f for f in sig['factors'])


def test_no_data_degrades(monkeypatch):
    monkeypatch.setattr(earn, '_fetch_earnings_frame', lambda t: None)
    sig = earn.earnings_signal('ZZZ')
    assert sig['available'] is False
    assert sig['factors'] == []


# ── predictor.py blending ───────────────────────────────────────────────────────

def _stub_channels(monkeypatch, price=None, earnings=None, sentiment=None):
    monkeypatch.setattr(pred, '_price_component',
                        lambda t: price or {'key': 'price', 'label': 'Price',
                                            'available': False, 'factors': []})
    monkeypatch.setattr(pred, '_earnings_component',
                        lambda t: earnings or {'key': 'earnings', 'label': 'Earnings',
                                               'available': False, 'factors': []})
    monkeypatch.setattr(pred, '_sentiment_component',
                        lambda t: sentiment or {'key': 'sentiment', 'label': 'Sentiment',
                                                'available': False, 'factors': []})


def test_all_channels_unavailable(monkeypatch):
    _stub_channels(monkeypatch)
    out = pred.predict('AAA')
    assert out['available'] is False


def test_single_channel_takes_full_weight(monkeypatch):
    _stub_channels(monkeypatch, price={'key': 'price', 'label': 'Price', 'available': True,
                                       'score': 0.6, 'confidence': 0.5, 'source': 'ml',
                                       'distribution': {'q10': -2, 'q50': 1, 'q90': 4},
                                       'factors': []})
    out = pred.predict('AAA')
    price = next(c for c in out['components'] if c['key'] == 'price')
    assert out['available'] is True
    assert price['weight'] == pytest.approx(1.0)      # only channel → all the weight
    assert out['prediction']['p_up'] > 0.5
    assert out['prediction']['direction'] == 'BULLISH'


def test_weights_scale_with_confidence(monkeypatch):
    _stub_channels(
        monkeypatch,
        price={'key': 'price', 'label': 'Price', 'available': True,
               'score': 0.5, 'confidence': 1.0, 'source': 'ml', 'factors': []},
        earnings={'key': 'earnings', 'label': 'Earnings', 'available': True,
                  'score': 0.5, 'confidence': 0.1, 'factors': []},
    )
    out = pred.predict('AAA')
    w = {c['key']: c['weight'] for c in out['components'] if c.get('available')}
    # price prior 0.50×conf 1.0 = 0.50 vs earnings 0.30×0.1 = 0.03 → price dominates.
    assert w['price'] > w['earnings']
    assert w['price'] + w['earnings'] == pytest.approx(1.0)


def test_disagreement_lowers_confidence(monkeypatch):
    agree = dict(price={'key': 'price', 'label': 'P', 'available': True,
                        'score': 0.6, 'confidence': 0.8, 'source': 'ml', 'factors': []},
                 earnings={'key': 'earnings', 'label': 'E', 'available': True,
                           'score': 0.6, 'confidence': 0.8, 'factors': []})
    disagree = dict(price={'key': 'price', 'label': 'P', 'available': True,
                           'score': 0.6, 'confidence': 0.8, 'source': 'ml', 'factors': []},
                    earnings={'key': 'earnings', 'label': 'E', 'available': True,
                              'score': -0.6, 'confidence': 0.8, 'factors': []})
    _stub_channels(monkeypatch, **agree)
    conf_agree = pred.predict('AAA')['prediction']['confidence']
    pred._CACHE.clear()
    _stub_channels(monkeypatch, **disagree)
    conf_disagree = pred.predict('AAA')['prediction']['confidence']
    assert conf_agree > conf_disagree


def test_never_claims_high_confidence(monkeypatch):
    """Even a maximally-confident, unanimous bull case caps its label at MODERATE."""
    strong = {'key': 'price', 'label': 'P', 'available': True,
              'score': 1.0, 'confidence': 1.0, 'source': 'ml', 'factors': []}
    _stub_channels(monkeypatch, price=strong,
                   earnings={'key': 'earnings', 'label': 'E', 'available': True,
                             'score': 1.0, 'confidence': 1.0, 'factors': []})
    out = pred.predict('AAA')
    assert out['prediction']['confidence_label'] in ('MODERATE', 'WEAK', 'LOW')
