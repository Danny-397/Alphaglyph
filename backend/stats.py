"""
Statistical validation tools for strategy evaluation.

1. Probabilistic Sharpe Ratio (PSR)
   P(SR_true > SR*) corrected for non-normality (skewness, fat tails) and
   finite sample size.  Source: Lopez de Prado (2014), "The Deflated Sharpe Ratio."

2. Deflated Sharpe Ratio (DSR)
   PSR where SR* is the expected maximum Sharpe across N independent strategy
   trials.  Answers: given that we tried N strategies, what is the probability
   the best one is genuinely good rather than the luckiest of the bunch?

3. Fama-French 3-Factor Decomposition
   OLS regression of portfolio excess returns against the market (Mkt-RF),
   size (SMB), and value (HML) factors from Ken French's data library.
   Reports Jensen's alpha, factor loadings, R², and t-statistics.
   Answers: is the strategy generating true alpha, or is it just capturing
   well-known risk premia that a passive factor ETF would give for free?
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

logger = logging.getLogger(__name__)

_EULER_MASCHERONI = 0.5772156649
_TRADING_DAYS     = 252

# Ken French moves these files around occasionally (the old /data_library/ path
# started 404-ing — the real reason the Fama-French test kept showing n/a). Try
# the current /ftp/ location first and keep the old one as a fallback.
_FF3_BASE = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
FF3_URLS = [
    _FF3_BASE + 'ftp/F-F_Research_Data_Factors_daily_CSV.zip',
    _FF3_BASE + 'data_library/F-F_Research_Data_Factors_daily_CSV.zip',
]
FF3_URL = FF3_URLS[0]   # kept for backward compatibility / tests
# Dartmouth blocks the default python-requests user agent from some cloud IPs,
# which is the usual cause of an intermittent "n/a" on the Fama-French test.
_FF3_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (compatible; AlphaGlyph/1.0; '
                   '+https://alphaglyph.org)'),
}
_FF3_TIMEOUT = 20
_FF3_RETRIES = 3
# The factors update once a day, so a parsed copy is good to reuse for hours.
_FF3_TTL_SECONDS = 12 * 3600
_FF3_DISK_PATH   = os.path.join(tempfile.gettempdir(),
                                'alphaglyph_ff3_daily.csv')
# Process-level cache of the *parsed* DataFrame: {'df': DataFrame, 'ts': float}.
_FF3_CACHE: dict = {'df': None, 'ts': 0.0}


# ── Probabilistic Sharpe Ratio ─────────────────────────────────────────────────

def probabilistic_sharpe_ratio(daily_returns: np.ndarray,
                                sr_benchmark_annual: float = 0.0) -> float:
    """
    Return P(SR_true > sr_benchmark_annual) given the observed sample.

    Unlike the naive Sharpe comparison, this correction accounts for:
      - Non-symmetric return distributions (skewness)
      - Fat-tailed distributions (excess kurtosis)
      - Finite sample size (small samples give noisy SR estimates)

    Formula (Lopez de Prado 2014, eq. 1):
        PSR(SR*) = Φ[(SR_hat - SR*) √(T−1) / √(1 − γ₃·SR_hat + (γ₄−1)/4·SR_hat²)]

    All Sharpe values are in per-period (daily) units internally.
    """
    r = np.asarray(daily_returns, dtype=float)
    n = len(r)
    if n < 10:
        return float('nan')

    mu    = r.mean()
    sigma = r.std(ddof=1)
    if sigma < 1e-12:
        return float('nan')

    sr_hat = mu / sigma                                      # daily SR
    sr_b   = sr_benchmark_annual / np.sqrt(_TRADING_DAYS)   # daily benchmark

    skew_r = float(skew(r))
    exkurt = float(kurtosis(r, fisher=True))                 # excess kurtosis

    # Denominator correction for non-normality.
    # Raw kurtosis γ₄ = excess_kurtosis + 3, so (γ₄−1)/4 = (excess_kurtosis+2)/4
    var_correction = 1.0 - skew_r * sr_hat + (exkurt + 2) / 4 * sr_hat ** 2
    if var_correction <= 0:
        return float('nan')

    z = (sr_hat - sr_b) * np.sqrt(n - 1) / np.sqrt(var_correction)
    return float(norm.cdf(z))


# ── Deflated Sharpe Ratio ──────────────────────────────────────────────────────

def deflated_sharpe_ratio(daily_returns: np.ndarray,
                           n_strategies: int = 1) -> dict:
    """
    Deflated Sharpe Ratio: PSR where SR* equals the expected maximum Sharpe
    that would arise by chance if n_strategies independent random strategies
    were backtested and the best one was selected.

    The benchmark SR* scales correctly with sample size:
        SR*_annual = E[max Z_k] × √(252 / T)
    where E[max Z_k] is the expected max of N unit-normal variables and T is
    the number of observed daily returns.

    n_strategies = 1  →  identical to PSR vs SR* = 0 (no multiple testing)
    n_strategies > 1  →  raises the bar to account for selection bias
    """
    r  = np.asarray(daily_returns, dtype=float)
    T  = len(r)
    mu = r.mean()
    s  = r.std(ddof=1)

    _null = {
        'sr_annual': None, 'sr_benchmark': None,
        'psr': None, 'dsr': None,
        'is_significant': False, 'n_strategies': n_strategies,
    }
    if s < 1e-12 or T < 10:
        return _null

    sr_annual = float(mu / s * np.sqrt(_TRADING_DAYS))

    # Expected maximum Sharpe from N independent unit-normal SR estimates
    if n_strategies > 1:
        ez_max = (
            (1 - _EULER_MASCHERONI) * norm.ppf(1 - 1 / n_strategies) +
            _EULER_MASCHERONI * norm.ppf(1 - 1 / (n_strategies * np.e))
        )
    else:
        ez_max = 0.0

    # Scale from unit-normal to annual SR units for the actual sample size
    sr_star_annual = float(ez_max * np.sqrt(_TRADING_DAYS / T))

    psr_val = probabilistic_sharpe_ratio(r, sr_benchmark_annual=0.0)
    dsr_val = probabilistic_sharpe_ratio(r, sr_benchmark_annual=sr_star_annual)

    return {
        'sr_annual':      round(sr_annual,       4),
        'sr_benchmark':   round(sr_star_annual,  4),
        'psr':            round(psr_val, 4) if not np.isnan(psr_val) else None,
        'dsr':            round(dsr_val, 4) if not np.isnan(dsr_val) else None,
        'is_significant': bool(not np.isnan(dsr_val) and dsr_val > 0.95),
        'n_strategies':   n_strategies,
    }


# ── Fama-French 3-Factor data layer ───────────────────────────────────────────

def _fetch_ff3_raw() -> str | None:
    """
    Download and decompress the Ken French daily FF3 CSV. Returns the CSV text,
    or None only if every option fails.

    Robustness: a browser-like User-Agent (Dartmouth blocks the default requests
    UA from some cloud IPs), a few retries with backoff, and — if the network is
    down entirely — a fall back to the last successfully fetched copy on disk.
    """
    last_exc: Exception | None = None
    for attempt in range(_FF3_RETRIES):
        for url in FF3_URLS:
            try:
                import requests
                resp = requests.get(url, timeout=_FF3_TIMEOUT,
                                    headers=_FF3_HEADERS)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = next(n for n in z.namelist()
                                    if n.upper().endswith('.CSV'))
                    text = z.read(csv_name).decode('latin-1')
                try:                               # persist for offline fallback
                    with open(_FF3_DISK_PATH, 'w', encoding='latin-1') as fh:
                        fh.write(text)
                except OSError:
                    pass
                return text
            except Exception as exc:               # network / parse / zip error
                last_exc = exc
        time.sleep(0.6 * (attempt + 1))

    # Every live attempt failed — reuse the last good copy if we have one.
    try:
        with open(_FF3_DISK_PATH, 'r', encoding='latin-1') as fh:
            logger.warning('Fama-French live fetch failed (%s); using cached '
                           'copy from disk.', last_exc)
            return fh.read()
    except OSError:
        logger.warning('Could not fetch Fama-French factors: %s', last_exc)
        return None


def _get_ff3_data() -> pd.DataFrame:
    """
    Return the parsed FF3 factor DataFrame, cached in-process for a few hours so
    repeated backtests don't re-hit the network (the daily factors barely move).
    Returns an empty DataFrame if data is genuinely unavailable.
    """
    fresh = (_FF3_CACHE['df'] is not None
             and (time.time() - _FF3_CACHE['ts']) < _FF3_TTL_SECONDS)
    if fresh:
        return _FF3_CACHE['df']

    raw = _fetch_ff3_raw()
    df  = _parse_ff3_csv(raw) if raw else pd.DataFrame()
    if not df.empty:
        _FF3_CACHE['df'] = df
        _FF3_CACHE['ts'] = time.time()
    return df


def _parse_ff3_csv(text: str) -> pd.DataFrame:
    """
    Parse the Ken French daily-factor CSV into a DataFrame indexed by date,
    with columns [mkt_rf, smb, hml, rf] (all in decimal, not percent).

    The file has multiple sections separated by blank lines; only the first
    (3-factor) block is read — the momentum-factor section is ignored.
    """
    rows         = []
    data_started = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            if data_started:
                break       # end of the first data block
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) < 5:
            continue
        try:
            date   = pd.to_datetime(parts[0], format='%Y%m%d')
            values = [float(p) / 100.0 for p in parts[1:5]]
            rows.append({
                'date':   date,
                'mkt_rf': values[0],
                'smb':    values[1],
                'hml':    values[2],
                'rf':     values[3],
            })
            data_started = True
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('date').sort_index()


# ── Fama-French 3-Factor Decomposition ────────────────────────────────────────

def fama_french_decomposition(
    port_hist: list[dict],
    ff3_data: pd.DataFrame | None = None,
) -> dict:
    """
    Regress daily portfolio excess returns against the Fama-French 3 factors:

        R_p − R_f = α + β_mkt(R_m−R_f) + β_smb·SMB + β_hml·HML + ε

    Parameters
    ----------
    port_hist : list of {date, value} dicts from run_backtest
    ff3_data  : pre-loaded FF3 DataFrame (used in tests to avoid network calls);
                if None, data is downloaded from Ken French's website.

    Returns a dict with:
        alpha_annual   — Jensen's alpha, annualised (%)
        alpha_t_stat   — t-statistic for alpha (|t| > 2 ≈ significant)
        beta_market    — sensitivity to the market
        beta_smb       — size-factor exposure
        beta_hml       — value-factor exposure
        r_squared      — fraction of variance explained by the 3 factors
        t_stats        — per-factor t-statistics
        interpretation — plain-English summary

    Returns {'enabled': False, 'reason': '...'} when data is unavailable.
    """
    if len(port_hist) < 30:
        return {'enabled': False, 'reason': 'Need at least 30 data points.'}

    # Portfolio daily returns
    dates  = pd.to_datetime([p['date'] for p in port_hist])
    values = np.array([p['value'] for p in port_hist], dtype=float)
    rets   = pd.Series(np.diff(values) / values[:-1], index=dates[1:])

    # Load FF3 factors (cached in-process; falls back to last good disk copy)
    if ff3_data is None:
        ff3_data = _get_ff3_data()
        if ff3_data.empty:
            return {'enabled': False,
                    'reason': 'Could not download Fama-French factors (no internet?).'}

    if ff3_data.empty:
        return {'enabled': False, 'reason': 'Failed to parse Fama-French data.'}

    # Align on common dates
    merged = pd.DataFrame({'port': rets}).join(ff3_data, how='inner')
    if len(merged) < 30:
        return {'enabled': False,
                'reason': f'Only {len(merged)} overlapping trading days — need 30+.'}

    excess = merged['port'].values - merged['rf'].values

    # OLS: y = Xβ + ε  where X = [1, Mkt-RF, SMB, HML]
    X = np.column_stack([
        np.ones(len(merged)),
        merged['mkt_rf'].values,
        merged['smb'].values,
        merged['hml'].values,
    ])
    betas, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)

    # Residuals → R² → standard errors → t-statistics
    y_hat  = X @ betas
    resid  = excess - y_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((excess - excess.mean()) ** 2).sum())
    r_sq   = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    n, k   = len(excess), 4
    mse    = ss_res / max(n - k, 1)
    xtx_inv = np.linalg.inv(X.T @ X + np.eye(k) * 1e-12)
    se     = np.sqrt(np.abs(np.diag(mse * xtx_inv)))
    t_stats = betas / (se + 1e-15)

    alpha_annual = float(betas[0]) * _TRADING_DAYS

    return {
        'enabled':        True,
        'n_obs':          n,
        'alpha_daily':    round(float(betas[0]) * 100,  5),   # %
        'alpha_annual':   round(alpha_annual     * 100,  2),   # %
        'alpha_t_stat':   round(float(t_stats[0]),       3),
        'beta_market':    round(float(betas[1]),          4),
        'beta_smb':       round(float(betas[2]),          4),
        'beta_hml':       round(float(betas[3]),          4),
        'r_squared':      round(r_sq,                     4),
        't_stats': {
            'alpha':  round(float(t_stats[0]), 3),
            'market': round(float(t_stats[1]), 3),
            'smb':    round(float(t_stats[2]), 3),
            'hml':    round(float(t_stats[3]), 3),
        },
        'interpretation': _interpret(alpha_annual, betas, t_stats, r_sq),
    }


def _interpret(alpha_annual: float, betas: np.ndarray,
               t_stats: np.ndarray, r_sq: float) -> str:
    """Plain-English summary of the factor decomposition."""
    alpha_pct = alpha_annual * 100
    sig       = 'significant' if abs(float(t_stats[0])) > 2.0 else 'not significant'
    smb_desc  = ('small-cap tilt' if betas[2] > 0.1 else
                 'large-cap tilt' if betas[2] < -0.1 else 'size-neutral')
    hml_desc  = ('value tilt'     if betas[3] > 0.1 else
                 'growth tilt'    if betas[3] < -0.1 else 'style-neutral')
    return (
        f'Annual alpha {alpha_pct:+.2f}% ({sig}, |t|={abs(t_stats[0]):.2f}). '
        f'Market beta {betas[1]:.2f}x. '
        f'SMB {betas[2]:+.2f} ({smb_desc}). '
        f'HML {betas[3]:+.2f} ({hml_desc}). '
        f'R²={r_sq:.3f} — {r_sq*100:.1f}% of variance explained by the 3 factors.'
    )
