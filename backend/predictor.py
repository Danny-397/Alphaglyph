"""
The Predictor — a transparent, multi-signal forward view for one ticker.

This is the engine behind the site's **Predictor** tab. It answers "what's the
lean on this stock over the next few days, and *why*" by combining three
independent, individually-explainable signals:

    1. PRICE / TECHNICAL  — the ML transformer's probability-of-up-move and its
                            q10–q90 return distribution (ml_runtime). If the model
                            isn't loaded it degrades to a transparent technical
                            momentum tilt so the tab still works.
    2. EARNINGS           — post-earnings-announcement drift + YoY EPS growth
                            (earnings.py).
    3. SENTIMENT          — GDELT news tone and its recent momentum (ml_features).

It is deliberately **not** a trained meta-model. It's a documented, weighted
average of the three channels — each channel emits a directional score in
[-1, +1] and a confidence in [0, 1], and the blend weights each channel by
(prior × confidence), renormalised over whatever is actually available. That
honesty is the whole point of AlphaGlyph: every number on the card can be traced
to a component the user can see, and disagreement between channels *lowers* the
combined confidence rather than being hidden.

Nothing here is financial advice, and the price channel alone lands near a coin
flip — the card says so.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np

import earnings as earn
import features as feat
import ml_features as mlf
import ml_runtime

logger = logging.getLogger(__name__)

# Prior weights before confidence-scaling. Price leads (it's the horizon-matched
# signal), earnings is the documented secondary edge, sentiment is the noisiest.
_PRIORS = {'price': 0.50, 'earnings': 0.30, 'sentiment': 0.20}

# Direction thresholds on the blended up-probability — mirror the ML model's own
# conservative buy/sell bands so the Predictor and the ML strategy agree.
_BUY_P, _SELL_P = 0.55, 0.45

_CACHE: dict = {}
_CACHE_TTL = 1800  # 30 min — predictions only move on a new daily bar


# ── Price / technical channel ──────────────────────────────────────────────────

def _price_component(ticker: str) -> dict:
    """
    Prefer the ML transformer's forecast; fall back to a transparent technical
    momentum tilt if the model isn't loaded so the channel is always present.
    """
    pred = None
    try:
        pred = ml_runtime.live_prediction(ticker)
    except Exception as exc:
        logger.warning('predictor: ML prediction failed for %s: %s', ticker, exc)

    if pred and pred.get('available'):
        p_up = float(pred['p_up'])
        q    = pred.get('quantiles') or {}
        # Score from the classifier, centred at 0.5 → [-1, +1].
        score = max(-1.0, min(1.0, (p_up - 0.5) * 2.0))
        # Confidence grows with distance from a coin flip (0.5).
        confidence = min(1.0, abs(p_up - 0.5) * 2.0 + 0.15)
        factors = [
            f"ML transformer puts the odds of an up-move at {p_up*100:.0f}% "
            f"over the next {pred.get('horizon', 5)} trading days.",
            f"Model's median forecast return: {q.get('q50', 0):+.1f}% "
            f"(q10 {q.get('q10', 0):+.1f}% … q90 {q.get('q90', 0):+.1f}%).",
        ]
        return {
            'key': 'price', 'label': 'Price / Technical (ML transformer)',
            'available': True, 'score': score, 'confidence': confidence,
            'source': 'ml', 'p_up': p_up, 'signal': pred.get('signal'),
            'distribution': q, 'vol': pred.get('vol'), 'price': pred.get('price'),
            'factors': factors,
        }

    # ── Fallback: transparent technical momentum ──────────────────────────────
    try:
        df = feat.get_feature_df(ticker, period='6mo')
    except Exception:
        df = None
    if df is None or df.empty:
        return {'key': 'price', 'label': 'Price / Technical',
                'available': False, 'factors': [], 'source': 'none'}

    row = df.iloc[-1]
    trend = float(row['sma20'] / row['sma50'] - 1.0) if row.get('sma50') else 0.0
    rsi   = float(row.get('rsi14', 50.0))
    ret5  = float(row.get('return_5d', 0.0))
    # Each sub-signal saturates into [-1, 1], then average.
    parts = [math.tanh(trend / 0.03), math.tanh((rsi - 50.0) / 25.0), math.tanh(ret5 / 0.05)]
    score = float(np.clip(np.mean(parts), -1.0, 1.0))
    return {
        'key': 'price', 'label': 'Price / Technical (momentum)',
        'available': True, 'score': score, 'confidence': 0.35,
        'source': 'technical', 'price': round(float(row['Close']), 2),
        'factors': [
            f"ML model offline — using a transparent momentum read: "
            f"SMA20 vs SMA50 {trend*100:+.1f}%, RSI {rsi:.0f}, 5-day return {ret5*100:+.1f}%.",
        ],
    }


# ── Sentiment channel ──────────────────────────────────────────────────────────

def _sentiment_component(ticker: str) -> dict:
    """News-tone tilt from GDELT (free, no key). Best-effort — degrades to
    unavailable when GDELT has no coverage or is rate-limited."""
    end = datetime.now().strftime('%Y-%m-%d')
    # Look back ~60 days for a stable tone baseline.
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    try:
        senti = mlf.fetch_sentiment(ticker, start, end)
    except Exception as exc:
        logger.warning('predictor: sentiment failed for %s: %s', ticker, exc)
        senti = None
    if senti is None or senti.empty or 'f_news_tone' not in senti:
        return {'key': 'sentiment', 'label': 'News sentiment (GDELT)',
                'available': False, 'factors': [], 'reason': 'no coverage'}

    tone = senti['f_news_tone'].dropna()
    if tone.empty:
        return {'key': 'sentiment', 'label': 'News sentiment (GDELT)',
                'available': False, 'factors': [], 'reason': 'no coverage'}

    recent = float(tone.tail(7).mean())
    base   = float(tone.mean())
    momentum = recent - base
    # GDELT tone is roughly [-10, +10]; a level of ±3 is already notably skewed.
    level_score = math.tanh(recent / 3.0)
    mom_score   = math.tanh(momentum / 2.0)
    score = float(np.clip(0.6 * level_score + 0.4 * mom_score, -1.0, 1.0))
    # More data points → more trustworthy; still a noisy channel, so cap modest.
    confidence = float(min(0.6, 0.2 + 0.02 * len(tone)))

    mood = 'positive' if recent > 0.5 else 'negative' if recent < -0.5 else 'neutral'
    trend = 'improving' if momentum > 0.3 else 'deteriorating' if momentum < -0.3 else 'steady'
    return {
        'key': 'sentiment', 'label': 'News sentiment (GDELT)',
        'available': True, 'score': score, 'confidence': confidence,
        'tone': round(recent, 2), 'tone_momentum': round(momentum, 2), 'n_points': int(len(tone)),
        'factors': [
            f"News tone over the last week is {mood} ({recent:+.1f} on GDELT's "
            f"−10…+10 scale) and {trend} vs its 60-day average.",
        ],
    }


def _earnings_component(ticker: str) -> dict:
    sig = earn.earnings_signal(ticker)
    if not sig.get('available'):
        return {'key': 'earnings', 'label': 'Earnings (PEAD + growth)',
                'available': False, 'factors': [], 'reason': sig.get('reason')}
    return {
        'key': 'earnings', 'label': 'Earnings (PEAD + growth)',
        'available': True, 'score': sig['score'], 'confidence': sig['confidence'],
        'surprise_pct': sig.get('surprise_pct'), 'eps_yoy_pct': sig.get('eps_yoy_pct'),
        'next_days': sig.get('next_days'), 'event_risk': sig.get('event_risk'),
        'factors': sig.get('factors', []),
    }


# ── Blend ───────────────────────────────────────────────────────────────────────

def _verdict(p_up: float, confidence: float, event_risk: bool) -> tuple[str, str, str]:
    """(direction, signal, confidence_label) from the blended probability."""
    if p_up >= _BUY_P:
        direction, signal = 'BULLISH', 'BUY'
    elif p_up <= _SELL_P:
        direction, signal = 'BEARISH', 'SELL'
    else:
        direction, signal = 'NEUTRAL', 'HOLD'

    if event_risk:
        label = 'LOW — earnings imminent'
    elif confidence >= 0.6:
        label = 'MODERATE'          # we never claim "high" — this is still a coin-flip-ish domain
    elif confidence >= 0.35:
        label = 'WEAK'
    else:
        label = 'LOW'
    return direction, signal, label


def predict(ticker: str) -> dict:
    """
    Full multi-signal prediction for one ticker. Never raises — every channel
    degrades to 'unavailable' independently.
    """
    ticker = (ticker or '').upper().strip()
    if not ticker:
        return {'error': 'ticker required'}

    hit = _CACHE.get(ticker)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]

    components = [
        _price_component(ticker),
        _earnings_component(ticker),
        _sentiment_component(ticker),
    ]
    available = [c for c in components if c.get('available')]

    if not available:
        result = {'ticker': ticker, 'available': False,
                  'reason': 'No signal available for this ticker right now.',
                  'components': components}
        _CACHE[ticker] = (time.time(), result)
        return result

    # Effective weight = prior × confidence, renormalised over available channels.
    raw_w = {c['key']: _PRIORS[c['key']] * max(0.0, c.get('confidence', 0.0)) for c in available}
    total = sum(raw_w.values()) or 1.0
    for c in available:
        c['weight'] = round(raw_w[c['key']] / total, 3)

    combined = sum(c['weight'] * c['score'] for c in available)
    p_up = float(np.clip(0.5 + 0.5 * combined, 0.02, 0.98))

    # Agreement: channels pulling the same way → trust the blend more; a tug of
    # war → discount it. Measured as 1 − normalised spread of the signed scores.
    scores = np.array([c['score'] for c in available])
    weights = np.array([c['weight'] for c in available])
    wmean = float(np.sum(weights * scores))
    dispersion = float(np.sqrt(np.sum(weights * (scores - wmean) ** 2)))
    agreement = max(0.0, 1.0 - dispersion)
    base_conf = float(np.sum(weights * np.array([c['confidence'] for c in available])))
    confidence = round(base_conf * (0.5 + 0.5 * agreement), 3)

    event_risk = any(c.get('event_risk') for c in available)
    direction, signal, conf_label = _verdict(p_up, confidence, event_risk)

    # Pull the ML return distribution through for the range display, if we have it.
    price_c = next((c for c in components if c['key'] == 'price'), {})
    distribution = price_c.get('distribution') if price_c.get('source') == 'ml' else None

    verdict_text = _verdict_text(direction, signal, ticker, p_up, available, agreement, event_risk)

    result = {
        'ticker':       ticker,
        'available':    True,
        'price':        price_c.get('price'),
        'horizon':      int(ml_runtime.get_info().get('horizon') or 5),
        'prediction': {
            'direction':        direction,
            'signal':           signal,
            'p_up':             round(p_up, 4),
            'score':            round(float(combined), 4),
            'confidence':       confidence,
            'confidence_label': conf_label,
            'agreement':        round(agreement, 3),
            'verdict_text':     verdict_text,
        },
        'distribution': distribution,
        'components':   components,
        'disclaimer': (
            'A transparent weighted blend of three signals — not a trained '
            'meta-model and not financial advice. The price channel alone is '
            'close to a coin flip; treat this as a lean, not a forecast.'
        ),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    _CACHE[ticker] = (time.time(), result)
    return result


def _verdict_text(direction, signal, ticker, p_up, available, agreement, event_risk):  # noqa: ARG001
    keys = ', '.join(c['key'] for c in available)
    lean = {'BULLISH': 'leans bullish', 'BEARISH': 'leans bearish',
            'NEUTRAL': 'is roughly balanced'}[direction]
    agree = ('the signals broadly agree' if agreement > 0.6
             else 'the signals disagree, so read this cautiously' if agreement < 0.35
             else 'the signals are mixed')
    risk = (' An earnings report is imminent, which can override any of this in a '
            'single move.' if event_risk else '')
    return (f"{ticker} {lean} over the next few days — a blended "
            f"{p_up*100:.0f}% chance of an up-move across {keys}, where {agree}.{risk}")
