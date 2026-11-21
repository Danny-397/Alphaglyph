"""The server enforces its own limits instead of trusting the page that calls it."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import app as app_module  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.limiter, 'enabled', False)
    return app_module.app.test_client()


def test_rate_limit_key_ignores_a_client_written_forwarded_for():
    with app_module.app.test_request_context(headers={
            'X-Forwarded-For': '6.6.6.6', 'CF-Connecting-IP': '1.2.3.4'}):
        assert app_module._client_ip() == '1.2.3.4'
    with app_module.app.test_request_context(headers={'X-Forwarded-For': '6.6.6.6'}):
        assert app_module._client_ip() != '6.6.6.6'


def test_compare_runs_at_most_max_tickers(client, monkeypatch):
    seen = []

    def fake_backtest(strategy, tickers, *args, **kwargs):
        seen.append(len(tickers))
        return {'error': 'stub'}

    monkeypatch.setattr(app_module.backtester, 'run_backtest', fake_backtest)
    r = client.post('/api/compare', json={'tickers': [f'T{i}' for i in range(300)]})
    assert r.status_code == 200
    assert seen and set(seen) == {app_module.MAX_TICKERS}


def test_optimize_clamps_tickers_and_frontier_points(client, monkeypatch):
    seen = {}

    def fake_frontier(tickers, start, end, n_points, shrink=False):
        seen.update(tickers=len(tickers), n_points=n_points)
        return {'ok': True}

    monkeypatch.setattr(app_module.portopt, 'compute_efficient_frontier', fake_frontier)
    r = client.post('/api/portfolio/optimize',
                    json={'tickers': [f'T{i}' for i in range(300)], 'n_points': 10_000_000})
    assert r.status_code == 200
    assert seen == {'tickers': app_module.MAX_TICKERS, 'n_points': 200}
