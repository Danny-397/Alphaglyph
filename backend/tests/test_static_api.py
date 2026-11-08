"""The site reads /api/ml/info and /api/options/presets from saved copies in
frontend/data/api/ so those pages open without waking the Render backend.
If the model or the presets change, the copies must be regenerated with
`python tools/static_api.py fixed`, or the site would show stale data."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('static_api', ROOT / 'tools' / 'static_api.py')
static_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_api)


@pytest.mark.parametrize('name', sorted(static_api.FIXED))
def test_committed_copy_matches_live_answer(name, monkeypatch):
    # Other tests swap in a fake model; load the real files the server ships.
    import ml_runtime
    for attr in ('_session', '_meta', '_load_err'):
        monkeypatch.setattr(ml_runtime, attr, None)
    monkeypatch.setattr(ml_runtime, 'MODEL_PATH', str(ROOT / 'backend' / 'ml_model.onnx'))
    monkeypatch.setattr(ml_runtime, 'META_PATH', str(ROOT / 'backend' / 'ml_model_meta.json'))
    live = static_api.fixed_payloads()[name]
    saved = json.loads((static_api.OUT / name).read_text(encoding='utf-8'))
    assert saved == json.loads(json.dumps(live)), (
        f'frontend/data/api/{name} is stale: run python tools/static_api.py fixed')


def test_screener_tickers_match_frontend_default():
    js = (ROOT / 'frontend' / 'app.js').read_text(encoding='utf-8')
    default = "[" + ", ".join(f"'{t}'" for t in static_api.SCREENER_TICKERS) + "]"
    assert default in js
