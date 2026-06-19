/* ══════════════════════════════════════════════════════════════════════════
   AlphaGlyph — Autonomous Trading Bot — app.js
   Vanilla JS · Chart.js 4 · No frameworks
   ══════════════════════════════════════════════════════════════════════════ */

// ── Config ─────────────────────────────────────────────────────────────────
// RENDER_URL is set in config.js (edit that file before deploying to Vercel).
const _isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1'
const API_BASE = window.RENDER_URL || (_isLocal ? 'http://localhost:5000' : '')

// Owner token for bot controls. The owner unlocks the dashboard once by
// visiting it with ?admin=<token>; it's stored in the browser and the URL is
// cleaned. Sent as a header on every API call (harmless on read endpoints).
const ADMIN_TOKEN = (() => {
  try {
    const url = new URL(location.href)
    const q = url.searchParams.get('admin')
    if (q) {
      localStorage.setItem('ag_admin', q)
      url.searchParams.delete('admin')
      history.replaceState({}, '', url.toString())
    }
    return localStorage.getItem('ag_admin') || ''
  } catch (_) { return '' }
})()

if (!_isLocal && !window.RENDER_URL) {
  console.error(
    '%c⚠ AlphaGlyph: RENDER_URL is not set in config.js.\n' +
    'All API calls will fail. Edit frontend/config.js and set your Render backend URL.',
    'color:#f85149;font-size:14px;font-weight:bold'
  )
}

const STRATEGY_LABELS = {
  adaptive:     'Adaptive (Regime-Based)',
  ma_crossover: 'MA Crossover',
  rsi:          'RSI Mean Reversion',
  macd:         'MACD Momentum',
  ml:           'ML Transformer',
}

// Enable the ML strategy option in a <select> once the backend reports a
// trained model is deployed (until then it stays disabled with a hint).
async function enableMlOption(selectEl) {
  if (!selectEl) return
  const info = await api('/api/ml/info')
  const opt  = selectEl.querySelector('option[value="ml"]')
  if (!opt) return
  if (info && info.loaded) {
    opt.disabled    = false
    opt.textContent = 'ML Transformer'
    opt.title       = `v${info.version} — ${Number(info.n_params).toLocaleString()} params, ` +
                      `test AUC ${info.test_metrics?.auc ?? info.val_metrics?.auc ?? '—'}`
  } else {
    opt.textContent = 'ML Transformer (not trained yet)'
  }
}

const REGIME_COLORS = {
  TRENDING_UP:     '#3fb950',
  TRENDING_DOWN:   '#f85149',
  RANGING:         '#e3b341',
  HIGH_VOLATILITY: '#e3913b',
}

// AlphaGlyph's whole point: the bot shows its work. Every trade is turned into a
// plain-English reason a beginner can follow — what the strategy saw, in what
// market, and why it acted. Used by the live dashboard and the My Bot sandbox.
const REGIME_PHRASE = {
  TRENDING_UP:     'an up-trending market',
  TRENDING_DOWN:   'a down-trending market',
  RANGING:         'a sideways, range-bound market',
  HIGH_VOLATILITY: 'a high-volatility market',
}
const BUY_REASON = {
  ma_crossover: 'the 20-day average crossed above the 50-day — momentum turning up',
  rsi:          'RSI dropped into oversold territory — a mean-reversion bounce setup',
  macd:         'MACD crossed above its signal line on rising volume',
  ml:           'the ML transformer put the odds on the upside',
}
const SELL_REASON = {
  stop_loss:    'price fell back to the trailing stop — locking in the move and capping downside',
  take_profit:  'price hit the take-profit target',
  sell_signal:  'the strategy flipped to a sell signal',
  signal:       'the strategy flipped to a sell signal',
}
function tradeWhy(t) {
  const where = REGIME_PHRASE[t.regime] || 'the current market'
  if (t.action === 'BUY') {
    const base = BUY_REASON[t.strategy] || 'the strategy flagged a buy signal'
    return `Bought because ${base}, in ${where}.`
  }
  const why = SELL_REASON[t.reason] || 'an exit rule triggered'
  let pnl = ''
  if (t.pnl != null) pnl = ` Booked a ${t.pnl >= 0 ? 'gain' : 'loss'} of ${fmt$(Math.abs(t.pnl))}` +
    (t.pnl_pct != null ? ` (${fmtPct(t.pnl_pct)})` : '') + '.'
  return `Sold because ${why}.${pnl}`
}

// ── Connection state (free-tier cold-start UX) ───────────────────────────────
// Render's free instance sleeps after inactivity and takes ~30s to wake. Without
// this the first loads just fail silently and the page looks broken. We detect
// network-level failures (server unreachable) and show a friendly auto-retry
// banner, then clear it the moment a request succeeds.
let _connFails = 0
function setConn(reachable) {
  let b = document.getElementById('conn-banner')
  if (reachable) {
    _connFails = 0
    if (b) b.classList.remove('show')
    return
  }
  if (++_connFails < 2) return          // tolerate a single transient blip
  if (!b) {
    b = document.createElement('div')
    b.id = 'conn-banner'
    b.className = 'conn-banner'
    b.innerHTML = '<span class="conn-dot"></span> Waking the bot up — the free server sleeps after inactivity and takes ~30s to start. Retrying automatically…'
    document.body.appendChild(b)
  }
  b.classList.add('show')
}

// ── Utilities ───────────────────────────────────────────────────────────────
const _sleep = ms => new Promise(r => setTimeout(r, ms))

// Resilient fetch. Free-tier cold starts, brief rate limits, and the market-data
// providers' occasional hiccups are all transient — so instead of surfacing an
// error we back off and retry silently, showing only the friendly "waking up"
// banner. Returns null only after several attempts genuinely fail.
async function api(path, opts = {}, _attempt = 0) {
  const MAX = 5
  try {
    const r = await fetch(API_BASE + path, {
      ...opts,
      headers: {
        'Content-Type': 'application/json',
        ...(ADMIN_TOKEN ? { 'X-Admin-Token': ADMIN_TOKEN } : {}),
        ...(opts.headers || {}),
      },
    })
    setConn(true)                        // we reached the server
    if (r.ok) return r.json()
    // Transient server states (cold start / rate limit) → back off and retry.
    if ((r.status === 429 || r.status >= 500) && _attempt < MAX) {
      await _sleep(Math.min(5000, 500 * 2 ** _attempt))
      return api(path, opts, _attempt + 1)
    }
    return null                          // a real 4xx — caller handles gracefully
  } catch (err) {
    // fetch threw → server asleep/unreachable. Show the waking banner, retry.
    if (err instanceof TypeError) setConn(false)
    if (_attempt < MAX) {
      await _sleep(Math.min(5000, 500 * 2 ** _attempt))
      return api(path, opts, _attempt + 1)
    }
    return null
  }
}

const el     = id => document.getElementById(id)
const fmt$   = n  => n == null ? '—' : '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const fmtPct = (n, d = 2) => n == null ? '—' : (n >= 0 ? '+' : '') + Number(n).toFixed(d) + '%'
const fmtN   = (n, d = 2) => n == null ? '—' : Number(n).toFixed(d)
const clr    = n  => n == null ? '' : n > 0 ? 'positive' : n < 0 ? 'negative' : ''
const today  = () => new Date().toISOString().slice(0, 10)
const daysAgo = d => new Date(Date.now() - d * 86400000).toISOString().slice(0, 10)

// Plain-English explanations of every quant term shown in the UI. Any stat card
// whose label matches a key automatically gets a hoverable ⓘ tooltip — so the
// glossary stays in one place and new stats are covered for free.
const GLOSSARY = {
  'Total Return':     'The strategy’s percent gain or loss over the whole test period.',
  'Final Value':      'What your starting capital grew (or shrank) to by the end.',
  'Sharpe Ratio':     'Return earned per unit of risk. Above 1 is good; above 2 is excellent.',
  'Max Drawdown':     'The largest drop from a peak to a later low — the worst dip you would have sat through.',
  'Win Rate':         'The share of closed trades that ended in a profit.',
  'Benchmark Return': 'What you would have made by simply buying and holding SPY over the same period.',
  'Calmar Ratio':     'Annual return divided by max drawdown — how much reward you got for the risk taken.',
  'Total Trades':     'Number of completed round-trip trades.',
  'Winning Trades':   'Trades that closed for a profit.',
  'Losing Trades':    'Trades that closed for a loss.',
  'Avg Win':          'Average profit on the trades that made money.',
  'Avg Loss':         'Average loss on the trades that lost money.',
  'Best Trade':       'The single most profitable trade.',
  'Worst Trade':      'The single biggest losing trade.',
  'Total Costs':      'Commissions and slippage paid across every trade.',
  'Kelly Fraction':   'The mathematically optimal share of capital to risk per trade, from the historical win rate and payoff.',
  'Kelly %':          'The mathematically optimal share of capital to risk per trade, from the historical win rate and payoff.',
  'vs SPY':           'How the strategy did versus buying and holding SPY — positive means it beat the market.',
  'Gross Return':     'Return before trading costs are subtracted.',
  'Deflated Sharpe':  'The Sharpe ratio corrected for luck and for testing many strategies. Above 95% means the result is very likely real.',
  'Probabilistic Sharpe': 'The probability the true Sharpe ratio is above zero, after accounting for fat tails and sample size.',
  'Annualized Alpha': 'Return that can’t be explained by overall market, size, or value exposure — a proxy for genuine skill.',
}

function infoIcon(label) {
  const tip = GLOSSARY[label]
  if (!tip) return ''
  return ` <i class="info" tabindex="0" data-tip="${tip.replace(/"/g, '&quot;')}" aria-label="What is ${label}?"></i>`
}

function statCard(label, value, cls = '') {
  return `<div class="stat-card"><div class="stat-value ${cls}">${value}</div><div class="stat-label">${label}${infoIcon(label)}</div></div>`
}

function weightBars(weights) {
  return Object.entries(weights)
    .sort((a, b) => b[1] - a[1])
    .map(([ticker, w]) => `
      <div class="weight-row">
        <span class="weight-ticker">${ticker}</span>
        <div class="weight-bar-track"><div class="weight-bar-fill" style="width:${Math.round(w * 100)}%"></div></div>
        <span class="weight-pct">${(w * 100).toFixed(1)}%</span>
      </div>`).join('')
}

Chart.defaults.color       = '#8a978f'
Chart.defaults.borderColor = '#2a352f'
Chart.defaults.font.family = "'JetBrains Mono', monospace"
Chart.defaults.font.size   = 11

function destroyChart(c) { if (c) { try { c.destroy() } catch (_) {} } return null }

function initTabs(root) {
  root.querySelectorAll('.tab-list').forEach(list => {
    const btns   = list.querySelectorAll('.tab-btn')
    const card   = list.closest('.card') || list.parentElement
    btns.forEach(btn => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'))
        card.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'))
        btn.classList.add('active')
        const target = el(btn.dataset.tab)
        if (target) target.classList.add('active')
      })
    })
  })
}

// ════════════════════════════════════════════════════════════════════════════
//  DASHBOARD
// ════════════════════════════════════════════════════════════════════════════
function initDashboard() {
  let seChart = null    // Stock Explorer price chart
  let seSub   = null    // Stock Explorer indicator sub-chart

  // ── Live market regime (stateless — /api/regime) ───────────────────────
  async function loadRegime() {
    const r = await api('/api/regime')
    const badge = el('regime-badge')
    if (!badge || !r || r.error) return
    badge.textContent = (r.label || r.regime || 'UNKNOWN').toUpperCase()
    badge.className   = 'regime-badge ' + (r.regime || '')
    el('regime-description').textContent = r.description || ''
    el('ri-adx').textContent = r.adx     != null ? fmtN(r.adx, 1)          : '—'
    el('ri-vol').textContent = r.vol_30d != null ? fmtN(r.vol_30d, 1) + '%' : '—'
    el('ri-bbw').textContent = r.bb_width != null ? fmtN(r.bb_width, 4)    : '—'
    el('regime-strategy').textContent = STRATEGY_LABELS[r.strategy] || r.strategy || '—'
  }
  loadRegime()
  setInterval(loadRegime, 120000)

  // ── Watchlist → Stock Explorer ticker options ──────────────────────────
  api('/api/watchlist').then(w => {
    if (!Array.isArray(w) || !w.length) return
    const sel = el('se-ticker')
    if (sel) {
      sel.innerHTML = w.map(t => `<option value="${t}">${t}</option>`).join('')
      loadExplorer()
    }
  })

  // ── ML Transformer live forecasts ──────────────────────────────────────
  // Loaded once on open (and on manual refresh), NOT in the 10s poll — the
  // model output only changes on new daily bars and inference is comparatively
  // expensive, so a 30-min server cache backs it.
  async function loadMlInsights() {
    const card  = el('ml-insights')
    const tbody = el('ml-tbody')
    if (!card || !tbody) return
    const info = await api('/api/ml/info')
    if (!info || !info.loaded) { card.hidden = true; return }   // no model → hide
    card.hidden = false
    el('ml-meta').textContent = `v${info.version} · ${Number(info.n_params || 0).toLocaleString()} params` +
      (info.test_metrics?.auc ? ` · test AUC ${info.test_metrics.auc}` : '')
    if (info.horizon) el('ml-horizon').textContent = info.horizon

    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:24px;">Running the transformer <span class="dots"><span></span><span></span><span></span></span></td></tr>`
    const data = await api('/api/ml/predictions')
    const preds = (data && data.predictions || []).filter(p => p.available)
    if (!preds.length) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:24px;">Forecasts momentarily unavailable — try refresh.</td></tr>`
      return
    }

    // Shared domain so every row's distribution bar is comparable.
    const lo = Math.min(...preds.map(p => p.quantiles.q10))
    const hi = Math.max(...preds.map(p => p.quantiles.q90))
    const span = (hi - lo) || 1
    const pos = v => ((v - lo) / span) * 100   // % position within the track
    const zero = pos(0)

    tbody.innerHTML = preds.map(p => {
      const q = p.quantiles
      const pUp = Math.round(p.p_up * 100)
      const upClr = p.p_up >= 0.55 ? 'positive' : p.p_up <= 0.45 ? 'negative' : ''
      const sigCls = p.signal === 'BUY' ? 'badge-buy' : p.signal === 'SELL' ? 'badge-sell' : ''
      const bandClr = q.q50 >= 0 ? 'pos' : 'neg'
      return `<tr data-ticker="${p.ticker}" class="ml-row" title="Open ${p.ticker} in the Stock Explorer">
        <td><strong>${p.ticker}</strong><div class="ml-px">${fmt$(p.price)}</div></td>
        <td>
          <div class="ml-prob"><div class="ml-prob-fill ${upClr}" style="width:${pUp}%"></div></div>
          <div class="ml-prob-val ${upClr}">${pUp}%</div>
        </td>
        <td>
          <div class="ml-dist">
            <div class="ml-dist-zero" style="left:${zero}%"></div>
            <div class="ml-dist-band ${bandClr}" style="left:${pos(q.q10)}%;width:${pos(q.q90) - pos(q.q10)}%"></div>
            <div class="ml-dist-iqr ${bandClr}" style="left:${pos(q.q25)}%;width:${pos(q.q75) - pos(q.q25)}%"></div>
            <div class="ml-dist-med" style="left:${pos(q.q50)}%"></div>
          </div>
          <div class="ml-dist-scale"><span>${fmtPct(q.q10, 1)}</span><span>${fmtPct(q.q90, 1)}</span></div>
        </td>
        <td class="${clr(q.q50)}"><strong>${fmtPct(q.q50, 1)}</strong></td>
        <td><span class="badge ${sigCls}">${p.signal}</span></td>
      </tr>`
    }).join('')

    // Click a forecast row to inspect that stock in the Stock Explorer below.
    tbody.querySelectorAll('tr[data-ticker]').forEach(tr => tr.addEventListener('click', () => {
      const sel = el('se-ticker')
      if (sel && [...sel.options].some(o => o.value === tr.dataset.ticker)) {
        sel.value = tr.dataset.ticker
        loadExplorer()
        el('stock-explorer')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }))
  }

  const mlRefresh = el('ml-refresh')
  if (mlRefresh) mlRefresh.addEventListener('click', loadMlInsights)
  // Kick off the ML panel, then offer the first-visit tour once layout settles.
  loadMlInsights().then(() => setTimeout(maybeStartTour, 500))

  // ── Stock Explorer: see what a strategy sees on any stock ──────────────
  enableMlOption(el('se-strategy'))
  const seTicker = el('se-ticker'), seStrategy = el('se-strategy')
  if (seTicker)   seTicker.addEventListener('change', loadExplorer)
  if (seStrategy) seStrategy.addEventListener('change', loadExplorer)

  async function loadExplorer() {
    const ticker = el('se-ticker')?.value
    const strat  = el('se-strategy')?.value || 'ma_crossover'
    if (!ticker) return
    const empty = el('se-empty')
    const data = await api(`/api/chart?ticker=${encodeURIComponent(ticker)}&strategy=${strat}`)
    if (!data || data.error || !(data.series || []).length) {
      seChart = destroyChart(seChart); seSub = destroyChart(seSub)
      el('se-sub-wrap').hidden = true
      if (empty) { empty.textContent = `No chart data for ${ticker}.`; empty.hidden = false }
      return
    }
    if (empty) empty.hidden = true
    renderExplorer(data)
  }

  function renderExplorer(d) {
    const labels = d.series.map(s => s.date)
    const buyMap = {}, sellMap = {}
    d.signals.forEach(s => { (s.action === 'BUY' ? buyMap : sellMap)[s.date] = s.price })

    const datasets = [{
      label: 'Price', data: d.series.map(s => s.close),
      borderColor: '#e9efeb', borderWidth: 1.6, fill: false, tension: 0.15, pointRadius: 0, order: 3,
    }]
    if (d.strategy === 'ma_crossover') {
      datasets.push({ label: 'SMA 20', data: d.series.map(s => s.sma20 ?? null), borderColor: '#3fb950', borderWidth: 1.3, fill: false, pointRadius: 0, spanGaps: true, order: 2 })
      datasets.push({ label: 'SMA 50', data: d.series.map(s => s.sma50 ?? null), borderColor: '#e3b341', borderWidth: 1.3, fill: false, pointRadius: 0, spanGaps: true, order: 2 })
    }
    datasets.push({ label: '▲ Buy',  data: labels.map(x => buyMap[x]  ?? null), showLine: false, pointStyle: 'triangle', pointRadius: 8, pointBackgroundColor: '#3fb950', pointBorderColor: '#0b0e0c', pointBorderWidth: 1, order: 1 })
    datasets.push({ label: '▼ Sell', data: labels.map(x => sellMap[x] ?? null), showLine: false, pointStyle: 'triangle', rotation: 180, pointRadius: 8, pointBackgroundColor: '#f85149', pointBorderColor: '#0b0e0c', pointBorderWidth: 1, order: 1 })

    seChart = destroyChart(seChart)
    seChart = new Chart(el('se-chart').getContext('2d'), {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true } },
          tooltip: { callbacks: { label: c => c.parsed.y == null ? null : ' ' + c.dataset.label + ': ' + fmt$(c.parsed.y) } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 7, maxRotation: 0 } },
          y: { grid: { color: 'rgba(36,48,42,0.55)' }, ticks: { callback: v => '$' + v } },
        },
      },
    })

    // Indicator sub-chart for oscillators that don't share the price scale.
    const subWrap = el('se-sub-wrap')
    seSub = destroyChart(seSub)
    if (d.strategy === 'rsi') {
      subWrap.hidden = false
      seSub = new Chart(el('se-subchart').getContext('2d'), {
        type: 'line',
        data: { labels, datasets: [
          { label: 'RSI(14)', data: d.series.map(s => s.rsi14 ?? null), borderColor: '#d2a8ff', borderWidth: 1.5, pointRadius: 0, spanGaps: true },
          { label: '70', data: labels.map(() => 70), borderColor: 'rgba(248,81,73,0.5)', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
          { label: '30', data: labels.map(() => 30), borderColor: 'rgba(63,185,80,0.5)', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: { x: { display: false }, y: { min: 0, max: 100, ticks: { stepSize: 50 }, grid: { color: 'rgba(36,48,42,0.4)' } } },
        },
      })
    } else if (d.strategy === 'macd') {
      subWrap.hidden = false
      seSub = new Chart(el('se-subchart').getContext('2d'), {
        data: { labels, datasets: [
          { type: 'bar', label: 'Histogram', data: d.series.map(s => s.macd_hist ?? null), backgroundColor: d.series.map(s => (s.macd_hist >= 0 ? 'rgba(63,185,80,0.5)' : 'rgba(248,81,73,0.5)')) },
          { type: 'line', label: 'MACD', data: d.series.map(s => s.macd_line ?? null), borderColor: '#58a6ff', borderWidth: 1.4, pointRadius: 0, spanGaps: true },
          { type: 'line', label: 'Signal', data: d.series.map(s => s.macd_signal ?? null), borderColor: '#e3b341', borderWidth: 1.4, pointRadius: 0, spanGaps: true },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: { x: { display: false }, y: { grid: { color: 'rgba(36,48,42,0.4)' } } },
        },
      })
    } else {
      subWrap.hidden = true
    }

    const nb = d.signals.filter(s => s.action === 'BUY').length
    const ns = d.signals.filter(s => s.action === 'SELL').length
    el('se-legend').innerHTML =
      `<span style="color:var(--green);">▲ ${nb}</span> · <span style="color:var(--red);">▼ ${ns}</span> signals · 1y`
  }

  // ── First-visit guided tour (coachmarks) ───────────────────────────────
  function maybeStartTour() {
    if (localStorage.getItem('tb_tour_done')) return
    startTour()
  }

  function startTour() {
    const steps = [
      { sel: '#sandbox-live', title: 'A bot trading live',
        body: 'This is a real bot running in your browser — it trades on real prices and explains every move in plain English.' },
      { sel: '#ml-insights', title: 'Watch the AI think',
        body: 'The ML transformer’s live forecasts: the odds of an up-move and a full return distribution for each stock.' },
      { sel: '#stock-explorer', title: 'See what it sees',
        body: 'Pick any stock and strategy to see the price, its indicators, and exactly where it would buy or sell.' },
    ].filter(s => { const e = document.querySelector(s.sel); return e && e.offsetParent !== null })
    if (!steps.length) { localStorage.setItem('tb_tour_done', '1'); return }

    let i = 0, curEl = null
    const tip = document.createElement('div')
    tip.className = 'tour-tip'
    document.body.appendChild(tip)

    function cleanup() {
      if (curEl) curEl.classList.remove('tour-highlight')
      tip.remove()
      localStorage.setItem('tb_tour_done', '1')
    }
    function show() {
      if (curEl) curEl.classList.remove('tour-highlight')
      const st = steps[i]
      const e = document.querySelector(st.sel)
      if (!e) return cleanup()
      curEl = e
      e.classList.add('tour-highlight')
      e.scrollIntoView({ behavior: 'smooth', block: 'center' })
      tip.innerHTML =
        `<div class="tour-tip-title">${st.title}</div>` +
        `<div class="tour-tip-body">${st.body}</div>` +
        `<div class="tour-tip-foot">` +
          `<button class="tour-tip-skip" id="tour-skip">Skip</button>` +
          `<span>${i + 1} / ${steps.length}</span>` +
          `<button class="btn btn-sm btn-primary" id="tour-next" style="margin-left:10px;">${i === steps.length - 1 ? 'Got it' : 'Next'}</button>` +
        `</div>`
      requestAnimationFrame(() => {
        const r = e.getBoundingClientRect()
        const tr = tip.getBoundingClientRect()
        let top = r.bottom + 12
        if (top + tr.height > window.innerHeight - 12) top = Math.max(66, r.top - tr.height - 12)
        const left = Math.min(Math.max(12, r.left), window.innerWidth - tr.width - 12)
        tip.style.top = top + 'px'
        tip.style.left = left + 'px'
      })
      el('tour-next').onclick = () => { i++; (i >= steps.length) ? cleanup() : show() }
      el('tour-skip').onclick = cleanup
    }
    show()
  }
}

// ════════════════════════════════════════════════════════════════════════════
//  BACKTEST
// ════════════════════════════════════════════════════════════════════════════
function initBacktest() {
  let btChart  = null
  let mcChart  = null
  let ff3Chart = null
  let cmpChart = null

  el('bt-start').value = daysAgo(365)
  el('bt-end').value   = today()

  enableMlOption(el('bt-strategy'))

  // ── ML out-of-sample guard ─────────────────────────────────────────────
  // The transformer is trained once (chronological 60/20/20). To show it
  // "actually working", an ML backtest must run only on dates AFTER its
  // train+validation cutoff — data the model never saw. We read that cutoff
  // from the model meta and push the start date out-of-sample automatically.
  let mlCutoff = null   // { train_end, val_end, purge_days }
  api('/api/ml/info').then(info => {
    if (info && info.loaded && info.splits) { mlCutoff = info.splits; applyMlOOS() }
  })

  function applyMlOOS() {
    const note = el('bt-ml-note')
    const isMl = el('bt-strategy').value === 'ml'
    if (!note) return
    if (!isMl || !mlCutoff) { note.hidden = true; return }
    const valEnd = mlCutoff.val_end
    if (el('bt-start').value < valEnd) el('bt-start').value = valEnd
    note.innerHTML =
      `🧠 Trained on data through <strong>${mlCutoff.train_end}</strong>, validated through ` +
      `<strong>${valEnd}</strong>. So you see the model working on data it never trained on, ` +
      `this backtest runs <strong>out-of-sample from ${valEnd}</strong>.`
    note.hidden = false
  }

  el('bt-strategy').addEventListener('change', applyMlOOS)

  // ── Ticker management (free-text input, validated against the backend) ──
  let btTickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'JPM', 'SPY']
  const tickerInput = el('bt-ticker-input')
  const tickerAddBtn = el('bt-ticker-add')
  const tickerErr   = el('bt-ticker-error')

  function renderTickerChips() {
    if (!btTickers.length) {
      el('bt-tickers').innerHTML =
        '<span style="font-size:12px;color:var(--muted);">No tickers added yet.</span>'
      return
    }
    el('bt-tickers').innerHTML = btTickers.map(t =>
      `<span class="ticker-chip selected" data-ticker="${t}">${t}<button class="chip-x" data-ticker="${t}" title="Remove ${t}">×</button></span>`
    ).join('')
    el('bt-tickers').querySelectorAll('.chip-x').forEach(b =>
      b.addEventListener('click', () => {
        btTickers = btTickers.filter(x => x !== b.dataset.ticker)
        renderTickerChips()
      })
    )
  }

  function showTickerErr(msg) { tickerErr.textContent = msg; tickerErr.hidden = false }
  function hideTickerErr()    { tickerErr.hidden = true }

  async function addTicker() {
    const sym = (tickerInput.value || '').trim().toUpperCase()
    hideTickerErr()
    if (!sym) return
    if (btTickers.includes(sym)) { tickerInput.value = ''; return }

    tickerAddBtn.disabled = true
    tickerAddBtn.textContent = '…'
    const res = await api('/api/validate_ticker?symbol=' + encodeURIComponent(sym))
    tickerAddBtn.disabled = false
    tickerAddBtn.textContent = 'Add'

    if (res && res.status === 'valid') {
      btTickers.push(res.symbol)
      renderTickerChips()
      tickerInput.value = ''
      tickerInput.focus()
    } else if (res && res.status === 'rate_limited') {
      showTickerErr('Couldn’t verify "' + sym + '" right now (data provider busy). Try again in a moment.')
    } else {
      showTickerErr('"' + sym + '" doesn’t exist. Enter a valid ticker symbol.')
      tickerInput.select()
    }
  }

  tickerAddBtn.addEventListener('click', addTicker)
  tickerInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); addTicker() }
  })
  renderTickerChips()

  el('bt-risk-btns').querySelectorAll('.risk-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      el('bt-risk-btns').querySelectorAll('.risk-btn').forEach(b => b.classList.remove('active'))
      btn.classList.add('active')
    })
  })

  el('run-btn').addEventListener('click', runBacktest)

  // ── Strategy leaderboard (compare all strategies) ──────────────────────────
  const CMP_COLORS = {
    adaptive: '#3fb950', ma_crossover: '#58a6ff', rsi: '#d2a8ff',
    macd: '#e3913b', ml: '#f778ba',
  }
  const cmpBtn = el('compare-btn')
  if (cmpBtn) cmpBtn.addEventListener('click', runCompare)

  async function runCompare() {
    const tickers = [...btTickers]
    if (!tickers.length) { alert('Add at least one ticker.'); return }
    const riskEl = el('bt-risk-btns').querySelector('.risk-btn.active')

    el('empty-state').hidden       = true
    el('results-container').hidden = true
    el('compare-container').hidden = true
    el('loading-state').hidden     = true
    el('compare-loading').hidden   = false
    cmpBtn.disabled = true
    el('run-btn').disabled = true

    const data = await api('/api/compare', {
      method: 'POST',
      body: JSON.stringify({
        tickers,
        start_date:      el('bt-start').value,
        end_date:        el('bt-end').value,
        initial_capital: parseFloat(el('bt-capital').value) || 100000,
        risk_tolerance:  riskEl ? riskEl.dataset.risk : 'moderate',
      }),
    })

    el('compare-loading').hidden = true
    cmpBtn.disabled = false
    el('run-btn').disabled = false

    if (!data || data.error) {
      el('empty-state').hidden = false
      el('empty-state').querySelector('strong').textContent =
        'Error: ' + (data?.error || 'Comparison failed — try again.')
      return
    }
    renderCompare(data)
    el('compare-container').hidden = false
  }

  function renderCompare(data) {
    const ranked = (data.results || []).filter(r => !r.error)
    const errored = (data.results || []).filter(r => r.error)
    const bench = data.benchmark_return

    el('cmp-meta').textContent =
      `${(data.tickers || []).join(', ')} · ${data.start_date} → ${data.end_date}`

    // ── Leaderboard table ──
    el('cmp-tbody').innerHTML = ranked.map((r, i) => {
      const vs = (r.total_return != null && bench != null) ? r.total_return - bench : null
      const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : (i + 1)
      return `<tr>
        <td>${medal}</td>
        <td><span style="color:${CMP_COLORS[r.strategy] || 'var(--text)'};font-weight:700;">${r.label}</span></td>
        <td class="${clr(r.total_return)}"><strong>${fmtPct(r.total_return)}</strong></td>
        <td class="${clr(vs)}">${vs == null ? '—' : fmtPct(vs)}</td>
        <td class="${r.sharpe_ratio > 1 ? 'positive' : r.sharpe_ratio < 0 ? 'negative' : ''}">${fmtN(r.sharpe_ratio)}</td>
        <td class="negative">${r.max_drawdown != null ? '-' + fmtN(r.max_drawdown) + '%' : '—'}</td>
        <td>${fmtN(r.calmar_ratio)}</td>
        <td>${r.win_rate != null ? r.win_rate + '%' : '—'}</td>
        <td>${r.total_trades ?? '—'}</td>
      </tr>`
    }).join('')

    const errEl = el('cmp-error')
    if (errored.length) {
      errEl.hidden = false
      errEl.textContent = 'Skipped: ' + errored.map(e => e.label).join(', ') +
        ' (no data or model unavailable).'
    } else { errEl.hidden = true }

    // ── Overlay equity chart ──
    cmpChart = destroyChart(cmpChart)
    const base = ranked.find(r => (r.equity_curve || []).length)
    if (!base) return
    const labels = base.equity_curve.map(p => p.date)
    const byDate = curve => {
      const m = {}; (curve || []).forEach(p => { m[p.date] = p.value }); return m
    }
    const datasets = ranked.map(r => {
      const m = byDate(r.equity_curve)
      return {
        label: r.label,
        data: labels.map(d => (m[d] != null ? m[d] : null)),
        borderColor: CMP_COLORS[r.strategy] || '#8a978f',
        borderWidth: 2, fill: false, tension: 0.2, pointRadius: 0, spanGaps: true,
      }
    })
    const spyMap = byDate(data.spy_curve)
    datasets.push({
      label: 'SPY Benchmark', data: labels.map(d => (spyMap[d] != null ? spyMap[d] : null)),
      borderColor: '#e3b341', borderWidth: 1.5, borderDash: [4, 4],
      fill: false, tension: 0.2, pointRadius: 0, spanGaps: true,
    })

    cmpChart = new Chart(el('cmp-chart').getContext('2d'), {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true } },
          tooltip: { callbacks: { label: c => ' ' + c.dataset.label + ': ' + fmt$(c.parsed.y) } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
          y: { grid: { color: 'rgba(36,48,42,0.55)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
        },
      },
    })
  }

  async function runBacktest() {
    const tickers = [...btTickers]
    if (!tickers.length) { alert('Add at least one ticker.'); return }

    const riskEl = el('bt-risk-btns').querySelector('.risk-btn.active')
    const commPct = parseFloat(el('bt-commission').value) || 0.10
    const slipPct = parseFloat(el('bt-slippage').value)  || 0.05

    // Keep ML backtests strictly out-of-sample (after the model's val cutoff).
    let startDate = el('bt-start').value
    if (el('bt-strategy').value === 'ml' && mlCutoff && startDate < mlCutoff.val_end) {
      startDate = mlCutoff.val_end
      el('bt-start').value = startDate
      applyMlOOS()
    }

    const payload = {
      strategy:        el('bt-strategy').value,
      tickers,
      start_date:      startDate,
      end_date:        el('bt-end').value,
      initial_capital: parseFloat(el('bt-capital').value) || 100000,
      walk_forward:    el('bt-walkforward').checked,
      risk_tolerance:  riskEl ? riskEl.dataset.risk : 'moderate',
      commission_pct:  commPct / 100,
      slippage_pct:    slipPct / 100,
      use_markowitz:   el('bt-markowitz').checked,
    }

    el('empty-state').hidden      = true
    el('results-container').hidden = true
    el('compare-container').hidden = true
    el('loading-state').hidden    = false
    el('run-btn').disabled        = true

    const data = await api('/api/backtest', { method: 'POST', body: JSON.stringify(payload) })

    el('loading-state').hidden = true
    el('run-btn').disabled     = false

    if (!data || data.error) {
      el('empty-state').hidden = false
      el('empty-state').querySelector('strong').textContent =
        'Error: ' + (data?.error || 'Request failed — is the backend running?')
      return
    }

    btChart  = destroyChart(btChart)
    mcChart  = destroyChart(mcChart)
    ff3Chart = destroyChart(ff3Chart)

    renderResults(data)
    el('results-container').hidden = false
    initTabs(el('results-container'))
  }

  function renderResults(data) {
    const m = data.metrics || {}

    const wfb = el('wf-banner')
    if (data.walk_forward?.enabled && data.walk_forward.split_date) {
      wfb.hidden = false
      el('wf-split-date').textContent = data.walk_forward.split_date
    } else {
      wfb.hidden = true
    }

    renderSummary(m)

    el('summary-stats').innerHTML = [
      statCard('Total Return', fmtPct(m.total_return), clr(m.total_return)),
      statCard('Final Value',  fmt$(m.final_value)),
      statCard('Sharpe Ratio', fmtN(m.sharpe_ratio), m.sharpe_ratio > 1 ? 'positive' : m.sharpe_ratio < 0 ? 'negative' : ''),
      statCard('Max Drawdown', m.max_drawdown != null ? '-' + fmtN(m.max_drawdown) + '%' : '—', 'negative'),
    ].join('')

    renderBtChart(data)
    renderSecondaryStats(m)
    renderRegimeBreakdown(data.regime_breakdown || {})
    renderMonteCarlo(data.monte_carlo, data.equity_curve)
    renderResearchTab(data)
    renderTradesTable(data.trades || [])
  }

  // Turn the raw backtest metrics into one human-readable paragraph + a verdict
  // a non-expert can act on. Directly answers "why did it (under/over)perform?"
  function renderSummary(m) {
    const box = el('bt-summary')
    if (!box) return
    if (m.total_return == null) { box.hidden = true; return }

    const ret = m.total_return, bench = m.benchmark_return, dd = m.max_drawdown
    const wr = m.win_rate, n = m.total_trades, fv = m.final_value, cap = m.initial_capital
    const verb = ret >= 0 ? 'grew' : 'shrank'

    let s = `Starting from ${fmt$(cap)}, the strategy ${verb} to <strong>${fmt$(fv)}</strong> — a <strong>${fmtPct(ret)}</strong> return`
    if (bench != null) s += `, versus <strong>${fmtPct(bench)}</strong> for simply buying and holding the market (SPY)`
    s += '. '
    if (n != null) {
      s += `It made <strong>${n}</strong> trade${n === 1 ? '' : 's'}`
      if (wr != null) s += ` with a <strong>${fmtN(wr, 0)}% win rate</strong>`
      if (dd != null) s += `, and its deepest peak-to-low dip was just <strong>${fmtN(dd, 1)}%</strong>`
      s += '.'
    }

    let take = ''
    if (bench != null && ret > bench) {
      take = `It beat buy-and-hold over this period${dd != null ? `, with the worst drop held to ${fmtN(dd, 1)}%` : ''} — outperforming with controlled risk.`
    } else if (bench != null) {
      take = `It trailed buy-and-hold here, but ${dd != null ? `with a far smaller drawdown (${fmtN(dd, 1)}%)` : 'with less risk'}. Defensive strategies like this give up upside in strong bull markets and protect capital when markets fall — judge them over a full cycle, not a single year.`
    } else if (ret >= 0) {
      take = 'A positive result — open the Research tab to check whether it is statistically real or just luck.'
    }

    box.innerHTML = s + (take ? `<span class="summary-take">${take}</span>` : '')
    box.hidden = false
  }

  function renderBtChart(data) {
    const curve = data.equity_curve || []
    const spy   = data.spy_curve    || []
    if (!curve.length) return
    const capital = data.metrics?.initial_capital || curve[0]?.value || 100000
    const datasets = [
      {
        label: 'Strategy', data: curve.map(p => p.value),
        borderColor: '#3fb950', borderWidth: 2,
        backgroundColor: 'rgba(63,185,80,0.06)', fill: true, tension: 0.2, pointRadius: 0,
      },
    ]
    if (spy.length) datasets.push({
      label: 'SPY Benchmark', data: spy.map(p => p.value),
      borderColor: '#e3b341', borderWidth: 1.5, borderDash: [4, 4],
      fill: false, tension: 0.2, pointRadius: 0,
    })
    datasets.push({
      label: 'Initial Capital', data: curve.map(() => capital),
      borderColor: '#2a352f', borderWidth: 1, borderDash: [2, 4],
      fill: false, pointRadius: 0,
    })

    btChart = new Chart(el('bt-chart').getContext('2d'), {
      type: 'line',
      data: { labels: curve.map(p => p.date), datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true } },
          tooltip: { callbacks: { label: c => ' ' + c.dataset.label + ': ' + fmt$(c.parsed.y) } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
          y: { grid: { color: 'rgba(36,48,42,0.55)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
        },
      },
    })
  }

  function renderSecondaryStats(m) {
    el('secondary-stats').innerHTML = [
      statCard('Win Rate',      m.win_rate  != null ? m.win_rate + '%' : '—'),
      statCard('Total Trades',  m.total_trades ?? '—'),
      statCard('Avg Win',       fmt$(m.avg_win),       'positive'),
      statCard('Avg Loss',      fmt$(m.avg_loss),      'negative'),
      statCard('Best Trade',    fmt$(m.best_trade),    'positive'),
      statCard('Worst Trade',   fmt$(m.worst_trade),   'negative'),
      statCard('Calmar Ratio',  fmtN(m.calmar_ratio)),
      statCard('Kelly %',       m.kelly_fraction != null ? fmtN(m.kelly_fraction) + '%' : '—'),
      statCard('Gross Return',  fmtPct(m.gross_return)),
      statCard('vs SPY',        fmtPct(m.benchmark_return), clr((m.total_return || 0) - (m.benchmark_return || 0))),
    ].join('')
  }

  function renderRegimeBreakdown(breakdown) {
    const box     = el('regime-breakdown')
    const entries = Object.entries(breakdown)
    if (!entries.length) { box.innerHTML = ''; return }
    box.innerHTML = `
      <div class="section-header" style="margin-top:20px;">Performance by Market Regime</div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr>
          <th>Regime</th><th>Trades</th><th>Win Rate</th>
          <th>Total P&L</th><th>Avg P&L</th><th>Best</th><th>Worst</th>
        </tr></thead>
        <tbody>${entries.map(([r, v]) => `<tr>
          <td><span style="color:${REGIME_COLORS[r] || '#8a978f'};font-weight:700;">${v.label || r}</span></td>
          <td>${v.trade_count}</td>
          <td>${v.win_rate}%</td>
          <td class="${clr(v.total_pnl)}">${fmt$(v.total_pnl)}</td>
          <td class="${clr(v.avg_pnl)}">${fmt$(v.avg_pnl)}</td>
          <td class="positive">${fmt$(v.best_trade)}</td>
          <td class="negative">${fmt$(v.worst_trade)}</td>
        </tr>`).join('')}</tbody>
      </table></div>`
  }

  function renderMonteCarlo(mc, curve) {
    const box = el('mc-tables')
    if (!mc || !mc.enabled) {
      box.innerHTML = '<p style="color:var(--muted);padding:20px;">Need at least 5 data points to run Monte Carlo.</p>'
      return
    }
    const fc = mc.fan_chart
    const actualValues = curve ? curve.map(p => p.value) : []

    mcChart = new Chart(el('mc-chart').getContext('2d'), {
      type: 'line',
      data: {
        labels: fc.dates,
        datasets: [
          { label: 'P95',          data: fc.p95, borderColor: 'transparent', backgroundColor: 'rgba(63,185,80,0.04)', fill: '+1', pointRadius: 0 },
          { label: 'P75',          data: fc.p75, borderColor: 'transparent', backgroundColor: 'rgba(63,185,80,0.08)', fill: '+1', pointRadius: 0 },
          { label: 'P50 (Median)', data: fc.p50, borderColor: 'rgba(63,185,80,0.5)', borderWidth: 1.5, borderDash: [4, 3], backgroundColor: 'rgba(63,185,80,0.08)', fill: '+1', pointRadius: 0 },
          { label: 'P25',          data: fc.p25, borderColor: 'transparent', backgroundColor: 'rgba(63,185,80,0.04)', fill: '+1', pointRadius: 0 },
          { label: 'P5',           data: fc.p5,  borderColor: 'transparent', backgroundColor: 'transparent', fill: false, pointRadius: 0 },
          {
            label: 'Actual Strategy',
            data:  fc.dates.map((d, i) => actualValues[i + 1] ?? null),
            borderColor: '#f0f6f0', borderWidth: 2.5, fill: false, pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 10, usePointStyle: true, filter: l => !['P95','P75','P25','P5'].includes(l.text) } },
          tooltip: { callbacks: { label: c => ' ' + c.dataset.label + ': ' + fmt$(c.parsed.y) } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 6, maxRotation: 0 } },
          y: { grid: { color: 'rgba(36,48,42,0.55)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
        },
      },
    })

    const rd = mc.return_distribution
    const sd = mc.sharpe_distribution
    box.innerHTML = `
      <div class="card">
        <div class="section-header">Return Distribution</div>
        <table class="data-table"><thead><tr><th>Percentile</th><th>Return</th></tr></thead>
        <tbody>
          ${[['P5', rd.p5],['P25', rd.p25],['P50 (Median)', rd.p50],['P75', rd.p75],['P95', rd.p95],['Actual', mc.actual_return_pct]]
            .map(([k, v]) => `<tr><td>${k}</td><td class="${clr(v)}">${fmtPct(v)}</td></tr>`).join('')}
        </tbody></table>
        <p style="margin-top:12px;font-size:12px;color:var(--muted);">
          Actual ranks in the <strong style="color:var(--text);">${fmtN(mc.actual_percentile, 0)}th percentile</strong> of 1,000 paths.
        </p>
      </div>
      <div class="card">
        <div class="section-header">Sharpe Distribution</div>
        <table class="data-table"><thead><tr><th>Percentile</th><th>Sharpe</th></tr></thead>
        <tbody>
          ${[['P5', sd.p5],['P25', sd.p25],['P50 (Median)', sd.p50],['P75', sd.p75],['P95', sd.p95]]
            .map(([k, v]) => `<tr><td>${k}</td><td>${fmtN(v)}</td></tr>`).join('')}
        </tbody></table>
        <p style="margin-top:12px;font-size:12px;color:var(--muted);">
          Sharpe ranks in the <strong style="color:var(--text);">${fmtN(mc.sharpe_percentile, 0)}th percentile</strong>.
        </p>
      </div>`
  }

  function renderResearchTab(data) {
    const mc  = data.monte_carlo      || {}
    const dsr = data.deflated_sharpe  || {}
    const ff3 = data.fama_french      || {}

    const mcPct    = mc.actual_percentile ?? 0
    const test1    = mcPct > 75
    const test2    = dsr.is_significant ?? false
    const test3    = ff3.enabled && Math.abs(ff3.alpha_t_stat || 0) > 2.0

    const passes   = [test1, test2, test3].filter(Boolean).length
    let vClass, vText
    if (passes === 3) { vClass = 'verdict-significant';  vText = '✓  STATISTICALLY SIGNIFICANT' }
    else if (passes >= 1) { vClass = 'verdict-promising'; vText = '~  PROMISING — NEEDS MORE DATA' }
    else               { vClass = 'verdict-inconclusive'; vText = '✗  INCONCLUSIVE — MAY BE NOISE' }

    const testRow = (pass, html) => `
      <div class="verdict-test ${pass ? 'pass' : 'fail'}">
        <span class="test-icon">${pass ? '✓' : '✗'}</span>
        <span class="test-label">${html}</span>
      </div>`

    el('validation-report').innerHTML = `
      <div class="verdict-card ${vClass}">
        <div class="verdict-label">Strategy Validation Report</div>
        <div class="verdict-main">${vText}</div>
        <div class="verdict-tests">
          ${testRow(test1, `Monte Carlo: actual result ranked in the <strong>${fmtN(mcPct, 0)}th percentile</strong> of 1,000 resampled market paths`)}
          ${testRow(test2, `Deflated Sharpe: <strong>${dsr.dsr != null ? (dsr.dsr * 100).toFixed(1) + '% confidence' : 'n/a'}</strong> result is real — corrected for ${dsr.n_strategies || 5} strategies (DSR&nbsp;=&nbsp;${dsr.dsr != null ? fmtN(dsr.dsr, 3) : 'n/a'})`)}
          ${testRow(test3, ff3.enabled
            ? `Fama-French: annual alpha <strong class="${clr(ff3.alpha_annual)}">${fmtPct(ff3.alpha_annual, 2)}/yr</strong> — ${Math.abs(ff3.alpha_t_stat || 0) > 2 ? 'statistically significant' : 'not significant'} (|t|&nbsp;=&nbsp;${fmtN(Math.abs(ff3.alpha_t_stat || 0), 2)})`
            : 'Fama-French: factor data unavailable — connect to the internet and re-run')}
        </div>
      </div>`

    const psrPct = dsr.psr != null ? +(dsr.psr * 100).toFixed(1) : null
    const dsrPct = dsr.dsr != null ? +(dsr.dsr * 100).toFixed(1) : null
    const barColor = v => v >= 95 ? 'green' : v >= 70 ? 'yellow' : 'red'
    const bar = (label, pct) => pct == null ? '' : `
      <div class="psr-bar-wrap">
        <div class="psr-bar-label"><span>${label}</span><span>${pct}%</span></div>
        <div class="psr-bar-track"><div class="psr-bar-fill ${barColor(pct)}" style="width:${pct}%"></div></div>
      </div>`

    el('research-detail-grid').innerHTML = `
      <div class="card">
        <div class="section-header">Deflated Sharpe Ratio</div>
        <p style="font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.6;">
          PSR corrects the Sharpe ratio for <strong>fat tails and skewness</strong>.
          DSR additionally corrects for <strong>multiple testing</strong> — if you tried N strategies and picked the best, the bar is higher.
          Both are expressed as the probability the result is genuinely positive (not luck).
        </p>
        ${psrPct != null ? bar('PSR — P(SR > 0)', psrPct) + bar(`DSR — P(SR > benchmark | ${dsr.n_strategies || 5} strategies)`, dsrPct) : '<p style="color:var(--muted);font-size:13px;">Insufficient data.</p>'}
        ${psrPct != null ? `
          <div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border);font-size:12px;color:var(--muted);">
            Annualised SR: <strong style="color:var(--text);">${fmtN(dsr.sr_annual)}</strong>
            &nbsp;·&nbsp; SR* benchmark: <strong style="color:var(--text);">${fmtN(dsr.sr_benchmark)}</strong>
          </div>` : ''}
      </div>
      <div class="card">
        <div class="section-header">Fama-French 3-Factor Attribution</div>
        ${ff3.enabled ? `
          <p style="font-size:12px;color:var(--muted);margin-bottom:12px;line-height:1.6;">
            Decomposes returns into known <strong>market</strong>, <strong>size (SMB)</strong>, and <strong>value (HML)</strong> risk premia.
            Alpha is what remains — skill that a passive factor ETF cannot replicate.
          </p>
          <div style="margin-bottom:14px;">
            <div style="font-size:26px;font-family:var(--font-mono);font-weight:700;color:${ff3.alpha_annual >= 0 ? 'var(--green)' : 'var(--red)'};">
              ${fmtPct(ff3.alpha_annual, 2)}/yr
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:3px;">
              Jensen's Alpha &nbsp;·&nbsp;
              ${Math.abs(ff3.alpha_t_stat || 0) > 2 ? '<span style="color:var(--green);">✓ significant</span>' : '<span style="color:var(--muted);">✗ not significant</span>'}
              &nbsp;(|t|=${fmtN(Math.abs(ff3.alpha_t_stat || 0), 2)})
              &nbsp;·&nbsp; R²=${fmtN(ff3.r_squared, 3)}
            </div>
          </div>
          <canvas id="ff3-chart" style="max-height:130px;"></canvas>
          <p style="font-size:11px;color:var(--muted);margin-top:12px;line-height:1.6;">${ff3.interpretation || ''}</p>
        ` : '<p style="color:var(--muted);font-size:13px;">Factor data unavailable — requires internet access to download from Ken French\'s data library.</p>'}
      </div>`

    if (ff3.enabled) {
      const ctx = el('ff3-chart')
      if (ctx) {
        ff3Chart = new Chart(ctx.getContext('2d'), {
          type: 'bar',
          data: {
            labels: ['β Market', 'β SMB (Size)', 'β HML (Value)'],
            datasets: [{
              data:            [ff3.beta_market, ff3.beta_smb, ff3.beta_hml],
              backgroundColor: ['rgba(63,185,80,0.55)', 'rgba(210,168,255,0.55)', 'rgba(227,179,65,0.55)'],
              borderColor:     ['#3fb950', '#d2a8ff', '#e3b341'],
              borderWidth:     1, borderRadius: 4,
            }],
          },
          options: {
            indexAxis: 'y', responsive: true,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ' β = ' + fmtN(c.parsed.x, 4) } } },
            scales: {
              x: { grid: { color: 'rgba(36,48,42,0.55)' }, ticks: { callback: v => v.toFixed(2) } },
              y: { grid: { display: false } },
            },
          },
        })
      }
    }

    const mw = data.markowitz_weights
    if (mw && Object.keys(mw).length) {
      el('markowitz-section').innerHTML = `
        <div class="card" style="margin-top:16px;">
          <div class="section-header">Markowitz Position Sizing Used</div>
          <p style="font-size:12px;color:var(--muted);margin-bottom:16px;">
            Position sizes were capped by the mean-variance optimal allocation — allocating more capital to assets with the best risk/return profile.
          </p>
          ${weightBars(mw)}
        </div>`
    } else {
      el('markowitz-section').innerHTML = ''
    }
  }

  function renderTradesTable(trades) {
    const tbody = el('trades-tbody')
    const sells = trades.filter(t => t.action === 'SELL')
    if (!sells.length) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:24px;">No closed trades in this period</td></tr>`
      return
    }
    tbody.innerHTML = sells.slice(0, 200).map(t => `
      <tr>
        <td>${t.date || ''}</td>
        <td><strong>${t.ticker}</strong></td>
        <td><span class="badge badge-sell">SELL</span></td>
        <td>${fmt$(t.price)}</td>
        <td>${t.shares}</td>
        <td class="${clr(t.pnl)}">${fmt$(t.pnl)}</td>
        <td class="${clr(t.pnl_pct)}">${fmtPct(t.pnl_pct)}</td>
        <td style="color:var(--muted);font-size:11px;">${t.reason || '—'}</td>
        <td style="color:${REGIME_COLORS[t.regime] || 'var(--muted)'};font-size:11px;">${t.regime || '—'}</td>
      </tr>`).join('')
  }
}

// ════════════════════════════════════════════════════════════════════════════
//  PORTFOLIO OPTIMIZER
// ════════════════════════════════════════════════════════════════════════════
function initPortfolio() {
  let frontierChart = null

  el('opt-start').value = daysAgo(365)
  el('opt-end').value   = today()

  el('opt-tickers').querySelectorAll('.ticker-chip').forEach(chip => {
    chip.addEventListener('click', () => chip.classList.toggle('selected'))
  })

  el('opt-run-btn').addEventListener('click', runOptimization)

  async function runOptimization() {
    const tickers = [...el('opt-tickers').querySelectorAll('.ticker-chip.selected')]
      .map(c => c.dataset.ticker)
    if (tickers.length < 2) { alert('Select at least 2 tickers.'); return }

    el('opt-run-btn').disabled = true
    el('opt-error').hidden     = true
    el('opt-results').hidden   = true
    el('opt-loading').hidden   = false

    const data = await api('/api/portfolio/optimize', {
      method: 'POST',
      body: JSON.stringify({
        tickers,
        start_date: el('opt-start').value,
        end_date:   el('opt-end').value,
        n_points:   60,
      }),
    })

    el('opt-loading').hidden   = true
    el('opt-run-btn').disabled = false

    if (!data || data.error) {
      el('opt-error').textContent = data?.error || 'Optimization failed — check backend logs'
      el('opt-error').hidden = false
      return
    }

    frontierChart = destroyChart(frontierChart)
    renderFrontierChart(data)
    renderWeights(data)
    renderCorrMatrix(data)
    el('opt-results').hidden = false
  }

  function renderFrontierChart(data) {
    const frontier = data.efficient_frontier || []
    const ms       = data.max_sharpe
    const mv       = data.min_variance
    const assets   = data.individual_assets  || {}

    frontierChart = new Chart(el('frontier-chart').getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Efficient Frontier',
            data:  frontier.map(p => ({ x: p.volatility, y: p.return })),
            borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.12)',
            showLine: true, tension: 0.3, pointRadius: 2.5, borderWidth: 2,
          },
          {
            label: '★ Max-Sharpe',
            data:  [{ x: ms.volatility, y: ms.expected_return }],
            borderColor: '#e3b341', backgroundColor: '#e3b341',
            pointRadius: 10, pointStyle: 'star', pointHoverRadius: 13,
          },
          {
            label: '◆ Min-Variance',
            data:  [{ x: mv.volatility, y: mv.expected_return }],
            borderColor: '#e6edf3', backgroundColor: '#e6edf3',
            pointRadius: 7, pointStyle: 'rectRot', pointHoverRadius: 9,
          },
          {
            label: 'Individual Assets',
            data:  Object.entries(assets).map(([t, a]) => ({ x: a.volatility, y: a.expected_return, label: t })),
            borderColor: '#8a978f', backgroundColor: 'rgba(138,151,143,0.5)',
            pointRadius: 5, pointHoverRadius: 7,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true } },
          tooltip: { callbacks: { label: c => {
            const d = c.raw
            const name = d.label || c.dataset.label
            return ` ${name}: Vol ${fmtN(d.x, 1)}%  Ret ${fmtPct(d.y, 1)}`
          }}},
        },
        scales: {
          x: { title: { display: true, text: 'Annualised Volatility (%)', color: '#8a978f' }, grid: { color: 'rgba(36,48,42,0.55)' }, ticks: { callback: v => v.toFixed(1) + '%' } },
          y: { title: { display: true, text: 'Annualised Return (%)',    color: '#8a978f' }, grid: { color: 'rgba(36,48,42,0.55)' }, ticks: { callback: v => v.toFixed(1) + '%' } },
        },
      },
    })
  }

  function renderWeights(data) {
    const ms = data.max_sharpe
    const mv = data.min_variance

    const statRows = port => `
      <div class="opt-stat-row"><span class="opt-stat-label">Expected Return</span><span class="opt-stat-value ${clr(port.expected_return)}">${fmtPct(port.expected_return, 1)}</span></div>
      <div class="opt-stat-row"><span class="opt-stat-label">Annualised Volatility</span><span class="opt-stat-value">${fmtPct(port.volatility, 1)}</span></div>
      <div class="opt-stat-row"><span class="opt-stat-label">Sharpe Ratio</span><span class="opt-stat-value ${clr(port.sharpe_ratio)}">${fmtN(port.sharpe_ratio, 3)}</span></div>`

    el('max-sharpe-stats').innerHTML  = statRows(ms)
    el('max-sharpe-weights').innerHTML = weightBars(ms.weights)
    el('min-var-stats').innerHTML      = statRows(mv)
    el('min-var-weights').innerHTML    = weightBars(mv.weights)
  }

  function renderCorrMatrix(data) {
    const cov     = data.covariance_matrix
    const tickers = cov.tickers
    const covData = cov.data
    const stds    = tickers.map((_, i) => Math.sqrt(Math.max(covData[i][i], 0)))
    const corr    = tickers.map((_, i) =>
      tickers.map((_, j) =>
        stds[i] * stds[j] > 1e-12 ? covData[i][j] / (stds[i] * stds[j]) : (i === j ? 1 : 0)
      )
    )

    const cellColor = v => v >= 0
      ? `rgba(63,185,80,${0.08 + Math.abs(v) * 0.65})`
      : `rgba(248,81,73,${0.08 + Math.abs(v) * 0.65})`

    const textColor = v => Math.abs(v) > 0.45 ? 'var(--text)' : 'var(--muted)'

    el('corr-matrix').innerHTML = `
      <div class="table-wrap"><table class="corr-table">
        <thead><tr><th></th>${tickers.map(t => `<th>${t}</th>`).join('')}</tr></thead>
        <tbody>${tickers.map((t, i) => `
          <tr><th>${t}</th>${corr[i].map((v, j) => `
            <td style="background:${cellColor(v)};color:${textColor(v)};">
              ${i === j ? '—' : v.toFixed(2)}
            </td>`).join('')}
          </tr>`).join('')}
        </tbody>
      </table></div>`
  }
}

// ════════════════════════════════════════════════════════════════════════════
//  SANDBOX — "Run Your Own Bot"
//  A personal paper-trading bot that lives entirely in the visitor's browser.
//  It reuses the REAL server strategy engine (/api/backtest) so the signals,
//  risk management, regime adaptation, Kelly sizing and ML transformer are
//  identical to the live bot — then plays the result back day-by-day as a live
//  bot the user owns. No server-side per-user state; it can never "stop".
// ════════════════════════════════════════════════════════════════════════════
function initSandbox() {
  const LS_KEY = 'ag_sandbox_v2'
  // On the dashboard this same engine runs an auto-launched demo bot (no
  // builder, no persistence), so visitors land on a bot already trading.
  const IS_DASHBOARD = document.body.dataset.page === 'dashboard'

  const SPEEDS = {
    slow:    { days: 1,    ms: 220 },
    normal:  { days: 1,    ms: 70  },
    fast:    { days: 5,    ms: 40  },
    instant: { days: 1e9,  ms: 0   },
  }

  // ── Run state ──
  let timeline = []          // [{date, value}] — authoritative portfolio value
  let trades   = []          // chronologically sorted BUY/SELL events
  let spyByDate = {}         // date -> SPY buy&hold value (same starting capital)
  let cursor = 0             // index into timeline (trading days elapsed)
  let tradePtr = 0           // next trade to reveal
  let cash = 0, capital = 0
  let positions = {}         // ticker -> {shares, entry, date, strategy}
  let closed = []            // realised pnl per closed trade
  let best = null, worst = null
  let lastRegime = '—', lastSpy = 0
  let stratLabel = '—'
  let speed = 'normal'
  let playing = false, timer = null, chart = null

  // ── Config form ──
  let tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'JPM', 'SPY']
  const configCard  = el('sandbox-config')
  const livePanel   = el('sandbox-live')
  const tickerInput = el('sb-ticker-input')
  const tickerAdd   = el('sb-ticker-add')
  const tickerErr   = el('sb-ticker-error')

  if (IS_DASHBOARD && configCard) configCard.hidden = true   // demo: no builder

  enableMlOption(el('sb-strategy'))

  // First-run 3-step primer (dismissible, remembered)
  const primer = el('sb-primer')
  if (primer && !localStorage.getItem('sb_primer_dismissed')) {
    primer.hidden = false
    const close = el('sb-primer-close')
    if (close) close.addEventListener('click', () => {
      primer.hidden = true
      localStorage.setItem('sb_primer_dismissed', '1')
    })
  }

  // ── One-click preset bots (zero-config launch) ─────────────────────────
  const PRESETS = [
    { icon: '🧠', name: 'AI Transformer Bot', desc: 'Trades on a machine-learning return forecast',
      strategy: 'ml',       risk: 'moderate',     tickers: ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'] },
    { icon: '🌦️', name: 'All-Weather Adaptive', desc: 'Auto-switches strategy to fit the market regime',
      strategy: 'adaptive', risk: 'moderate',     tickers: ['SPY', 'AAPL', 'MSFT', 'NVDA', 'JPM'] },
    { icon: '🚀', name: 'Momentum Chaser', desc: 'Rides strong trends with MACD on rising volume',
      strategy: 'macd',     risk: 'aggressive',   tickers: ['NVDA', 'TSLA', 'AAPL', 'AMD'] },
    { icon: '🪙', name: 'Bargain Hunter', desc: 'Buys oversold dips, trims overbought rips (RSI)',
      strategy: 'rsi',      risk: 'conservative', tickers: ['SPY', 'MSFT', 'JPM', 'KO'] },
  ]

  function renderPresets() {
    const grid = el('preset-grid')
    if (!grid) return
    grid.innerHTML = PRESETS.map((p, i) => `
      <button class="preset-card" data-i="${i}" type="button">
        <span class="preset-icon">${p.icon}</span>
        <span class="preset-name">${p.name}</span>
        <span class="preset-desc">${p.desc}</span>
        <span class="preset-go">Launch →</span>
      </button>`).join('')
    grid.querySelectorAll('.preset-card').forEach(c =>
      c.addEventListener('click', () => applyPreset(PRESETS[+c.dataset.i])))
  }

  function applyPreset(p) {
    const sel = el('sb-strategy')
    const opt = sel.querySelector(`option[value="${p.strategy}"]`)
    if (opt) opt.disabled = false            // ensure ML is selectable even mid-load
    sel.value = p.strategy
    el('sb-capital').value = 100000
    el('sb-window').value = '365'
    el('sb-risk-btns').querySelectorAll('.risk-btn')
      .forEach(b => b.classList.toggle('active', b.dataset.risk === p.risk))
    tickers = [...p.tickers]
    renderChips()
    launch()
  }

  renderPresets()

  function renderChips() {
    const box = el('sb-tickers')
    if (!tickers.length) {
      box.innerHTML = '<span style="font-size:12px;color:var(--muted);">No stocks added yet.</span>'
      return
    }
    box.innerHTML = tickers.map(t =>
      `<span class="ticker-chip selected" data-ticker="${t}">${t}<button class="chip-x" data-ticker="${t}" title="Remove ${t}">×</button></span>`
    ).join('')
    box.querySelectorAll('.chip-x').forEach(b =>
      b.addEventListener('click', () => { tickers = tickers.filter(x => x !== b.dataset.ticker); renderChips() }))
  }

  async function addTicker() {
    const sym = (tickerInput.value || '').trim().toUpperCase()
    tickerErr.hidden = true
    if (!sym) return
    if (tickers.includes(sym)) { tickerInput.value = ''; return }
    tickerAdd.disabled = true; tickerAdd.textContent = '…'
    const res = await api('/api/validate_ticker?symbol=' + encodeURIComponent(sym))
    tickerAdd.disabled = false; tickerAdd.textContent = 'Add'
    if (res && res.status === 'valid') {
      tickers.push(res.symbol); renderChips(); tickerInput.value = ''; tickerInput.focus()
    } else if (res && res.status === 'rate_limited') {
      tickerErr.textContent = 'Couldn’t verify "' + sym + '" right now — try again in a moment.'; tickerErr.hidden = false
    } else {
      tickerErr.textContent = '"' + sym + '" doesn’t exist. Enter a valid ticker symbol.'; tickerErr.hidden = false; tickerInput.select()
    }
  }

  tickerAdd.addEventListener('click', addTicker)
  tickerInput.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); addTicker() } })
  renderChips()

  el('sb-risk-btns').querySelectorAll('.risk-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      el('sb-risk-btns').querySelectorAll('.risk-btn').forEach(b => b.classList.remove('active'))
      btn.classList.add('active')
    })
  })

  function showErr(msg) {
    const e = el('sb-error'); if (e) { e.textContent = msg; e.hidden = false }
    // On the dashboard the builder is hidden, so surface failures in the loader.
    const dl = el('sb-demo-loading')
    if (IS_DASHBOARD && dl) dl.textContent = msg
  }

  // ── Launch: run the real engine, then start playback ──
  el('sb-launch').addEventListener('click', launch)

  async function launch() {
    if (!tickers.length) { showErr('Add at least one stock to trade.'); return }
    capital = Math.max(1000, parseFloat(el('sb-capital').value) || 100000)
    const windowDays = parseInt(el('sb-window').value, 10) || 365
    const riskEl = el('sb-risk-btns').querySelector('.risk-btn.active')
    const strategy = el('sb-strategy').value
    speed = el('sb-speed').value
    stratLabel = STRATEGY_LABELS[strategy] || strategy

    const payload = {
      strategy,
      tickers: [...tickers],
      start_date: daysAgo(windowDays),
      end_date: today(),
      initial_capital: capital,
      risk_tolerance: riskEl ? riskEl.dataset.risk : 'moderate',
      commission_pct: 0.001,
      slippage_pct: 0.0005,
    }

    el('sb-error').hidden = true
    el('sb-loading').hidden = false
    el('sb-launch').disabled = true
    const data = await api('/api/backtest', { method: 'POST', body: JSON.stringify(payload) })
    el('sb-loading').hidden = true
    el('sb-launch').disabled = false

    if (!data || data.error) {
      showErr(data?.error || 'Could not reach the bot — the free server may be waking up (~30s). Try again in a moment.')
      return
    }
    if (!data.equity_curve || !data.equity_curve.length) {
      showErr('Not enough price data for those stocks and window. Try different tickers or a longer history.')
      return
    }
    loadRun({
      timeline:   data.equity_curve,
      trades:     data.trades || [],
      spy:        data.spy_curve || [],
      capital,
      speed,
      stratLabel,
      cursor: 0,
    })
    play()
    save()
  }

  // ── (Re)hydrate a run into memory and show the live panel ──
  function loadRun(run) {
    timeline   = run.timeline
    capital    = run.capital
    speed      = run.speed || 'normal'
    stratLabel = run.stratLabel || '—'
    trades = run.trades.slice().sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0))
    spyByDate = {}
    run.spy.forEach(p => { spyByDate[p.date] = p.value })

    el('sb-strategy-label').textContent = stratLabel
    el('sb-speed').value = speed
    el('sb-speed-live').value = speed

    resetState()
    // fast-forward (no animation) to the saved cursor
    const target = Math.min(run.cursor || 0, timeline.length - 1)
    while (cursor < target) { cursor++; applyTradesUpTo(timeline[cursor].date) }

    configCard.hidden = true
    const presetsEl = el('sandbox-presets')
    if (presetsEl) presetsEl.hidden = true
    const demoLoading = el('sb-demo-loading')
    if (demoLoading) demoLoading.hidden = true
    livePanel.hidden = false
    buildChart()
    renderAll()
    pause()
  }

  function resetState() {
    cursor = 0; tradePtr = 0; cash = capital
    positions = {}; closed = []; best = null; worst = null
    lastRegime = '—'; lastSpy = capital
    clearFeed()
    applyTradesUpTo(timeline[0].date)   // reveal day-zero trades
  }

  // ── Trade application (reconstructs cash & open positions exactly) ──
  function applyTradesUpTo(dateStr) {
    while (tradePtr < trades.length && trades[tradePtr].date <= dateStr) {
      applyTrade(trades[tradePtr]); tradePtr++
    }
  }

  function applyTrade(t) {
    const cost = t.cost || 0
    if (t.action === 'BUY') {
      cash -= t.price * t.shares + cost
      positions[t.ticker] = { shares: t.shares, entry: t.price, date: t.date, strategy: t.strategy }
    } else {
      cash += t.price * t.shares - cost
      delete positions[t.ticker]
      if (t.pnl != null) {
        closed.push(t.pnl)
        if (best  == null || t.pnl > best)  best  = t.pnl
        if (worst == null || t.pnl < worst) worst = t.pnl
      }
    }
    if (t.regime) lastRegime = t.regime
    pushFeed(t)
  }

  // ── Playback loop ──
  function step() {
    const conf = SPEEDS[speed]
    const advance = speed === 'instant' ? timeline.length : conf.days
    for (let k = 0; k < advance && cursor < timeline.length - 1; k++) {
      cursor++
      applyTradesUpTo(timeline[cursor].date)
    }
    renderAll()
    save()
    if (cursor >= timeline.length - 1) finish()
  }

  function play() {
    const v = el('sb-verdict'); if (v) v.hidden = true
    if (cursor >= timeline.length - 1) resetState()   // at the end → replay
    playing = true
    el('sb-playpause').textContent = '⏸ Pause'
    el('sb-dot').className = 'dot dot-green'
    el('sb-state-text').textContent = 'YOUR BOT IS LIVE'
    clearInterval(timer)
    if (speed === 'instant') { step() }
    else { timer = setInterval(step, SPEEDS[speed].ms) }
    renderAll()
  }

  function pause() {
    playing = false
    clearInterval(timer); timer = null
    el('sb-playpause').textContent = '⏵ Resume'
    el('sb-dot').className = 'dot dot-yellow'
    el('sb-state-text').textContent = 'PAUSED'
  }

  function finish() {
    playing = false
    clearInterval(timer); timer = null
    el('sb-dot').className = 'dot dot-green'
    el('sb-state-text').textContent = 'CAUGHT UP TO TODAY'
    el('sb-playpause').textContent = '↻ Replay'
    showVerdict()
  }

  // Plain-English wrap-up of how the bot did — closes the loop satisfyingly.
  function showVerdict() {
    const v = el('sb-verdict'); if (!v) return
    const finalVal = timeline[timeline.length - 1].value
    const ret   = (finalVal - capital) / capital * 100
    const spyRet = lastSpy ? (lastSpy - capital) / capital * 100 : null
    const vs    = spyRet != null ? ret - spyRet : null
    const wins  = closed.filter(p => p > 0).length
    const wr    = closed.length ? Math.round(wins / closed.length * 100) : null
    const sells = trades.filter(t => t.action === 'SELL' && t.pnl != null)
    let bestT = null
    sells.forEach(t => { if (!bestT || t.pnl > bestT.pnl) bestT = t })

    el('sb-verdict-headline').innerHTML =
      `${ret >= 0 ? '📈' : '📉'} ${stratLabel}: ${fmt$(capital)} → <strong>${fmt$(finalVal)}</strong> ` +
      `<span class="${clr(ret)}">(${fmtPct(ret)})</span>`

    let s
    if (!closed.length) {
      s = `Over ${timeline.length} trading days this strategy found no trades that met its rules on ` +
          `these stocks — a real, honest outcome (it stays in cash rather than forcing bad trades). ` +
          `Try another preset, different stocks, or a longer window.`
    } else {
      s = `Over ${timeline.length} trading days your bot made <strong>${closed.length}</strong> ` +
          `completed trade${closed.length === 1 ? '' : 's'}`
      if (wr != null) s += ` with a <strong>${wr}%</strong> win rate`
      s += '. '
      if (vs != null) {
        s += vs >= 0
          ? `It <strong class="positive">beat</strong> buy-and-hold (SPY) by <strong>${fmtPct(vs)}</strong>. `
          : `It <strong class="negative">trailed</strong> buy-and-hold by <strong>${fmtPct(Math.abs(vs))}</strong>. `
      }
      if (bestT) s += `Best call: <strong>${bestT.ticker}</strong> for ${fmt$(bestT.pnl)}.`
    }
    el('sb-verdict-text').innerHTML = s
    v.hidden = false
  }

  // ── Controls ──
  el('sb-playpause').addEventListener('click', () => { playing ? pause() : play() })
  el('sb-restart').addEventListener('click', () => { resetState(); renderAll(); play() })
  el('sb-speed-live').addEventListener('change', e => {
    speed = e.target.value
    if (playing) play()    // restart timer at new cadence
  })
  function reconfigure() {
    if (IS_DASHBOARD) { location.href = 'sandbox.html'; return }   // build on the full page
    pause()
    localStorage.removeItem(LS_KEY)
    const v = el('sb-verdict'); if (v) v.hidden = true
    livePanel.hidden = true
    configCard.hidden = false
    const presetsEl = el('sandbox-presets')
    if (presetsEl) presetsEl.hidden = false
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }
  el('sb-reconfigure').addEventListener('click', reconfigure)
  el('sb-verdict-new').addEventListener('click', reconfigure)
  el('sb-verdict-again').addEventListener('click', () => { resetState(); renderAll(); play() })

  // ── Shareable bot link ─────────────────────────────────────────────────
  el('sb-share').addEventListener('click', async () => {
    const riskEl = el('sb-risk-btns').querySelector('.risk-btn.active')
    const params = new URLSearchParams({
      strategy: el('sb-strategy').value,
      risk:     riskEl ? riskEl.dataset.risk : 'moderate',
      capital:  String(Math.round(capital || 100000)),
      window:   el('sb-window').value,
      tickers:  tickers.join(','),
    })
    const url = `${location.origin}${location.pathname}?${params.toString()}`
    const btn = el('sb-share')
    const restore = () => { btn.textContent = '🔗 Share' }
    try {
      await navigator.clipboard.writeText(url)
      btn.textContent = '✓ Link copied!'
      setTimeout(restore, 1800)
    } catch (_) {
      window.prompt('Copy your bot link:', url)
    }
  })

  function applySharedConfig(sp) {
    const strat = sp.get('strategy')
    const sel = el('sb-strategy')
    const opt = strat && sel.querySelector(`option[value="${strat}"]`)
    if (!opt) return false                       // unknown strategy → ignore
    opt.disabled = false
    sel.value = strat
    const risk = sp.get('risk') || 'moderate'
    el('sb-risk-btns').querySelectorAll('.risk-btn')
      .forEach(b => b.classList.toggle('active', b.dataset.risk === risk))
    const cap = parseFloat(sp.get('capital'))
    if (cap >= 1000) el('sb-capital').value = cap
    const win = sp.get('window')
    if (['180', '365', '730', '1825'].includes(win)) el('sb-window').value = win
    const tk = (sp.get('tickers') || '').split(',').map(s => s.trim().toUpperCase()).filter(Boolean)
    if (tk.length) { tickers = tk.slice(0, 12); renderChips() }
    launch()
    return true
  }

  // ── Rendering ──
  function renderAll() {
    const day  = timeline[cursor]
    const val  = day.value
    const ret  = (val - capital) / capital * 100
    lastSpy = spyByDate[day.date] != null ? spyByDate[day.date] : lastSpy
    const invested = Math.max(0, val - cash)

    el('sb-simdate').textContent = day.date + '  ·  Day ' + (cursor + 1) + ' / ' + timeline.length
    el('sb-value').textContent  = fmt$(val)
    const rEl = el('sb-return'); rEl.textContent = fmtPct(ret); rEl.className = 'hero-stat-value ' + clr(ret)
    el('sb-cash').textContent   = fmt$(cash)
    el('sb-positions-count').textContent = Object.keys(positions).length
    el('sb-invested').textContent = fmt$(invested)

    const spyRet = lastSpy ? (lastSpy - capital) / capital * 100 : 0
    const vs = ret - spyRet
    const vsEl = el('sb-vsspy'); vsEl.textContent = fmtPct(vs); vsEl.className = 'perf-value ' + clr(vs)

    el('sb-trades').textContent  = closed.length
    const wins = closed.filter(p => p > 0).length
    el('sb-winrate').textContent = closed.length ? Math.round(wins / closed.length * 100) + '%' : '—'
    const bEl = el('sb-best');  bEl.textContent  = best  != null ? fmt$(best)  : '—'; bEl.className  = 'perf-value ' + clr(best)
    const wEl = el('sb-worst'); wEl.textContent  = worst != null ? fmt$(worst) : '—'; wEl.className  = 'perf-value ' + clr(worst)

    const rb = el('sb-regime-badge')
    rb.textContent = lastRegime
    rb.className = 'regime-badge ' + (lastRegime !== '—' ? lastRegime : '')

    const pct = timeline.length > 1 ? cursor / (timeline.length - 1) * 100 : 100
    el('sb-progress').style.width = pct + '%'

    renderPositions()
    updateChart()
  }

  function renderPositions() {
    const tbody = el('sb-positions-tbody')
    const rows = Object.entries(positions)
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:24px;">No open positions</td></tr>'
      return
    }
    tbody.innerHTML = rows.map(([tkr, p]) => `
      <tr>
        <td><strong>${tkr}</strong></td>
        <td>${fmtN(p.shares, 0)}</td>
        <td>${fmt$(p.entry)}</td>
        <td style="color:var(--muted);font-size:11px;">${p.date}</td>
        <td style="color:var(--blue);font-size:11px;">${STRATEGY_LABELS[p.strategy] || p.strategy || '—'}</td>
      </tr>`).join('')
  }

  // ── Live trade feed ──
  function clearFeed() {
    el('sb-feed').innerHTML = '<div style="text-align:center;color:var(--muted);padding:24px;font-size:13px;">Your bot’s trades will appear here as it runs…</div>'
  }

  function pushFeed(t) {
    const box = el('sb-feed')
    if (!box.querySelector('.sb-feed-item')) box.innerHTML = ''   // drop the placeholder
    const isBuy = t.action === 'BUY'
    const pnlTxt = (t.action === 'SELL' && t.pnl != null)
      ? `<span class="sb-feed-pnl ${clr(t.pnl)}">${fmt$(t.pnl)} (${fmtPct(t.pnl_pct)})</span>` : ''
    const item = document.createElement('div')
    item.className = 'sb-feed-item'
    item.innerHTML =
      `<div class="sb-feed-row">` +
        `<span class="sb-feed-time">${t.date}</span>` +
        `<span class="badge ${isBuy ? 'badge-buy' : 'badge-sell'}">${t.action}</span>` +
        `<span class="sb-feed-tkr">${t.ticker}</span>` +
        `<span class="sb-feed-trade">${fmtN(t.shares, 0)} @ ${fmt$(t.price)}</span>` +
        pnlTxt +
      `</div>` +
      `<div class="sb-feed-why">${tradeWhy(t)}</div>`
    box.insertBefore(item, box.firstChild)
    while (box.children.length > 120) box.removeChild(box.lastChild)
  }

  // ── Chart ──
  function buildChart() {
    chart = destroyChart(chart)
    chart = new Chart(el('sb-chart').getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Your Bot', data: [], borderColor: '#3fb950', borderWidth: 2,
            backgroundColor: 'rgba(63,185,80,0.07)', fill: true, tension: 0.25, pointRadius: 0 },
          { label: 'Buy & Hold SPY', data: [], borderColor: '#e3b341', borderWidth: 1.5,
            borderDash: [4, 4], fill: false, tension: 0.25, pointRadius: 0, spanGaps: true },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true } },
          tooltip: { callbacks: { label: c => ' ' + c.dataset.label + ': ' + fmt$(c.parsed.y) } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 7, maxRotation: 0 } },
          y: { grid: { color: 'rgba(36,48,42,0.55)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
        },
      },
    })
  }

  function updateChart() {
    if (!chart) return
    const slice = timeline.slice(0, cursor + 1)
    chart.data.labels = slice.map(p => p.date)
    chart.data.datasets[0].data = slice.map(p => p.value)
    chart.data.datasets[1].data = slice.map(p => (spyByDate[p.date] != null ? spyByDate[p.date] : null))
    chart.update('none')
  }

  // ── Persistence (survives reloads — the bot is "always there") ──
  function save() {
    if (IS_DASHBOARD) return   // the dashboard demo is ephemeral — never persist
    try {
      localStorage.setItem(LS_KEY, JSON.stringify({
        timeline, trades, spy: Object.entries(spyByDate).map(([date, value]) => ({ date, value })),
        capital, speed, stratLabel, cursor,
      }))
    } catch (_) { /* storage full / disabled — non-fatal */ }
  }

  function restore() {
    let saved
    try { saved = JSON.parse(localStorage.getItem(LS_KEY) || 'null') } catch (_) { saved = null }
    if (saved && Array.isArray(saved.timeline) && saved.timeline.length) {
      loadRun(saved)
      return true
    }
    return false
  }

  // Dashboard: auto-launch a demo bot so visitors land on one already trading.
  // Full My Bot page: a shared link (?strategy=…) launches that exact bot,
  // otherwise restore a previous run paused where it left off.
  if (IS_DASHBOARD) {
    applyPreset(PRESETS[1])   // "All-Weather Adaptive" — a sensible default demo
  } else {
    const sharedParams = new URLSearchParams(location.search)
    if (sharedParams.get('strategy')) {
      if (!applySharedConfig(sharedParams)) restore()
    } else {
      restore()
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
//  LIVE SIGNAL SCANNER
//  What every strategy + the ML transformer says about your stocks right now.
// ════════════════════════════════════════════════════════════════════════════
function initSignals() {
  const LS_KEY = 'ag_scan_watchlist'
  let tickers = []
  try { tickers = JSON.parse(localStorage.getItem(LS_KEY) || 'null') } catch (_) {}
  if (!Array.isArray(tickers) || !tickers.length)
    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'JPM', 'SPY']

  const tickerInput = el('sig-ticker-input')
  const tickerAdd   = el('sig-ticker-add')
  const tickerErr   = el('sig-ticker-error')

  const sigBadge = s =>
    s === 'BUY'  ? '<span class="badge badge-buy">BUY</span>'  :
    s === 'SELL' ? '<span class="badge badge-sell">SELL</span>' :
    s === 'HOLD' ? '<span class="badge sig-hold">HOLD</span>'   :
                   '<span style="color:var(--muted);">—</span>'

  function save() { try { localStorage.setItem(LS_KEY, JSON.stringify(tickers)) } catch (_) {} }

  function renderChips() {
    const box = el('sig-tickers')
    box.innerHTML = tickers.length
      ? tickers.map(t => `<span class="ticker-chip selected" data-ticker="${t}">${t}<button class="chip-x" data-ticker="${t}" title="Remove ${t}">×</button></span>`).join('')
      : '<span style="font-size:12px;color:var(--muted);">No tickers — add some above.</span>'
    box.querySelectorAll('.chip-x').forEach(b => b.addEventListener('click', () => {
      tickers = tickers.filter(x => x !== b.dataset.ticker); save(); renderChips(); scan()
    }))
  }

  async function addTicker() {
    const sym = (tickerInput.value || '').trim().toUpperCase()
    tickerErr.hidden = true
    if (!sym) return
    if (tickers.includes(sym)) { tickerInput.value = ''; return }
    tickerAdd.disabled = true; tickerAdd.textContent = '…'
    const res = await api('/api/validate_ticker?symbol=' + encodeURIComponent(sym))
    tickerAdd.disabled = false; tickerAdd.textContent = 'Add'
    if (res && res.status === 'valid') {
      tickers.push(res.symbol); save(); renderChips(); tickerInput.value = ''; tickerInput.focus(); scan()
    } else if (res && res.status === 'rate_limited') {
      tickerErr.textContent = 'Couldn’t verify "' + sym + '" right now — try again shortly.'; tickerErr.hidden = false
    } else {
      tickerErr.textContent = '"' + sym + '" doesn’t exist. Enter a valid ticker symbol.'; tickerErr.hidden = false; tickerInput.select()
    }
  }

  tickerAdd.addEventListener('click', addTicker)
  tickerInput.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); addTicker() } })
  el('sig-scan').addEventListener('click', scan)

  async function loadRegime() {
    const r = await api('/api/regime')
    const badge = el('sig-regime-badge')
    if (!r || r.error) { el('sig-regime-desc').textContent = 'Market regime unavailable right now.'; return }
    badge.textContent = (r.label || r.regime || 'UNKNOWN').toUpperCase()
    badge.className = 'regime-badge ' + (r.regime || '')
    el('sig-regime-desc').innerHTML = (r.description || '') +
      (r.strategy ? ` · favors <strong>${STRATEGY_LABELS[r.strategy] || r.strategy}</strong>` : '')
  }

  async function scan() {
    const tbody = el('sig-tbody')
    if (!tickers.length) { tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:24px;">Add a ticker to scan.</td></tr>'; return }
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:24px;">Scanning <span class="dots"><span></span><span></span><span></span></span></td></tr>`
    const data = await api('/api/scan?tickers=' + encodeURIComponent(tickers.join(',')))
    if (!data || !data.scanned) { tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:24px;">Couldn’t scan right now — try Rescan.</td></tr>'; return }

    tbody.innerHTML = data.scanned.map(row => {
      const s = row.signals || {}
      const mlCell = !data.ml_loaded ? '<span style="color:var(--muted);">—</span>'
        : `${sigBadge(s.ml)}${row.p_up != null ? ` <span style="font-size:11px;color:${row.p_up >= 0.5 ? 'var(--green)' : 'var(--red)'};">${Math.round(row.p_up * 100)}%${row.p_up >= 0.5 ? '↑' : '↓'}</span>` : ''}`
      const net = row.buys - row.sells
      const consensus = net > 0
        ? `<span class="positive" style="font-weight:700;">▲ ${row.buys} buy</span>`
        : net < 0 ? `<span class="negative" style="font-weight:700;">▼ ${row.sells} sell</span>`
        : '<span style="color:var(--muted);">— mixed</span>'
      return `<tr>
        <td><strong>${row.ticker}</strong></td>
        <td>${fmt$(row.price)}</td>
        <td>${sigBadge(s.ma_crossover)}</td>
        <td>${sigBadge(s.rsi)}</td>
        <td>${sigBadge(s.macd)}</td>
        <td>${mlCell}</td>
        <td>${consensus}</td>
      </tr>`
    }).join('')

    const now = new Date()
    el('sig-updated').textContent = 'Updated ' + now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  renderChips()
  loadRegime()
  scan()
}

// ── Router ──────────────────────────────────────────────────────────────────
const PAGE = document.body.dataset.page
if (PAGE === 'dashboard') { initSandbox(); initDashboard() }  // demo bot + market panels
if (PAGE === 'backtest')  initBacktest()
if (PAGE === 'portfolio') initPortfolio()
if (PAGE === 'sandbox')   initSandbox()
if (PAGE === 'signals')   initSignals()
