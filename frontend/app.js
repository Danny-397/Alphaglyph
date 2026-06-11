/* ══════════════════════════════════════════════════════════════════════════
   TradeBot Research Platform — app.js
   Vanilla JS · Chart.js 4 · No frameworks
   ══════════════════════════════════════════════════════════════════════════ */

// ── Config ─────────────────────────────────────────────────────────────────
// RENDER_URL is set in config.js (edit that file before deploying to Vercel).
const _isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1'
const API_BASE = window.RENDER_URL || (_isLocal ? 'http://localhost:5000' : '')

if (!_isLocal && !window.RENDER_URL) {
  console.error(
    '%c⚠ TradeBot: RENDER_URL is not set in config.js.\n' +
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

// ── Utilities ───────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  try {
    const r = await fetch(API_BASE + path, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    })
    if (!r.ok) throw new Error('HTTP ' + r.status)
    return r.json()
  } catch (err) {
    console.warn('API error:', path, err.message)
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

Chart.defaults.color       = '#8b949e'
Chart.defaults.borderColor = '#30363d'
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
  let eqChart    = null
  let botRunning = false

  initTabs(document.body)

  // First-run onboarding banner — shown until the visitor dismisses it.
  const onboard = el('onboard')
  if (onboard && !localStorage.getItem('tb_onboard_dismissed')) {
    onboard.hidden = false
    el('onboard-close').addEventListener('click', () => {
      onboard.hidden = true
      localStorage.setItem('tb_onboard_dismissed', '1')
    })
  }

  const startStopBtn   = el('start-stop-btn')
  const strategySelect = el('strategy-select')
  const riskBtns       = el('risk-btns')

  enableMlOption(strategySelect)

  startStopBtn.addEventListener('click', async () => {
    startStopBtn.disabled = true
    if (botRunning) {
      await api('/api/stop', { method: 'POST' })
    } else {
      await api('/api/start', { method: 'POST', body: JSON.stringify({ strategy: strategySelect.value }) })
    }
    await refreshAll()
    startStopBtn.disabled = false
  })

  strategySelect.addEventListener('change', async () => {
    await api('/api/strategy', { method: 'POST', body: JSON.stringify({ strategy: strategySelect.value }) })
  })

  riskBtns.querySelectorAll('.risk-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      riskBtns.querySelectorAll('.risk-btn').forEach(b => b.classList.remove('active'))
      btn.classList.add('active')
      await api('/api/risk_tolerance', { method: 'POST', body: JSON.stringify({ tolerance: btn.dataset.risk }) })
    })
  })

  async function refreshAll() {
    const [status, history, trades, activity] = await Promise.all([
      api('/api/status'),
      api('/api/portfolio/history?limit=500'),
      api('/api/trades?limit=20'),
      api('/api/activity'),
    ])
    if (status)   updateFromStatus(status)
    if (history)  updateEquityChart(history)
    if (trades)   updateTradesFeed(trades)
    if (activity) updateActivityLog(activity)
  }

  function updateFromStatus(s) {
    botRunning = s.is_running

    const dot = el('bot-dot')
    dot.className = 'dot ' + (s.is_running ? 'dot-green' : 'dot-red')
    el('bot-status-text').textContent = s.is_running ? 'BOT RUNNING' : 'BOT STOPPED'
    el('strategy-label').textContent  = STRATEGY_LABELS[s.strategy] || s.strategy || '—'

    const mkt = el('market-status-badge')
    mkt.textContent = s.market_open ? 'MARKET OPEN' : 'MARKET CLOSED'
    mkt.className   = 'badge ' + (s.market_open ? 'badge-green' : 'badge-red')

    startStopBtn.textContent = s.is_running ? 'STOP BOT' : 'START BOT'
    startStopBtn.className   = 'btn ' + (s.is_running ? 'btn-danger' : 'btn-primary')
    startStopBtn.style.width = '100%'
    startStopBtn.style.marginTop = '4px'

    if (strategySelect.value !== s.strategy && s.strategy)
      strategySelect.value = s.strategy

    riskBtns.querySelectorAll('.risk-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.risk === s.risk_tolerance)
    })

    const p = s.portfolio || {}
    const rv = el('hero-value')
    rv.textContent = fmt$(p.portfolio_value)
    rv.className   = 'hero-stat-value'

    const rr = el('hero-return')
    rr.textContent = fmtPct(p.total_return)
    rr.className   = 'hero-stat-value ' + clr(p.total_return)

    el('hero-daily').textContent     = fmt$(p.equity)
    el('hero-positions').textContent = p.active_positions ?? 0

    updatePositions(p.positions || [])
    if (s.regime) updateRegime(s.regime)

    const m  = s.metrics     || {}
    const lv = s.live_metrics || {}
    el('perf-winrate').textContent  = m.win_rate  != null ? m.win_rate + '%' : '—'
    el('perf-sharpe').textContent   = fmtN(lv.sharpe_ratio)
    el('perf-drawdown').textContent = lv.max_drawdown != null ? '-' + fmtN(lv.max_drawdown) + '%' : '—'
    el('perf-kelly').textContent    = s.kelly_fraction != null ? fmtN(s.kelly_fraction) + '%' : '—'
    el('perf-trades').textContent   = m.total_trades ?? '—'
    el('perf-daily').textContent    = (s.daily_trades ?? 0) + ' / ' + (s.max_daily ?? '?')
  }

  function updatePositions(positions) {
    const tbody = el('positions-tbody')
    if (!positions.length) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:24px;">No open positions</td></tr>`
      return
    }
    tbody.innerHTML = positions.map(p => `
      <tr>
        <td><strong>${p.ticker}</strong></td>
        <td>${fmtN(p.shares, 0)}</td>
        <td>${fmt$(p.entry_price)}</td>
        <td>${fmt$(p.current_price)}</td>
        <td class="${clr(p.pnl_pct)}">${fmtPct(p.pnl_pct)}</td>
        <td class="negative">${fmt$(p.stop_loss)}</td>
        <td class="positive">${fmt$(p.take_profit)}</td>
      </tr>`).join('')
  }

  function updateRegime(r) {
    const badge = el('regime-badge')
    badge.textContent = (r.label || r.regime || 'UNKNOWN').toUpperCase()
    badge.className   = 'regime-badge ' + (r.regime || '')
    el('regime-description').textContent = r.description || ''
    el('ri-adx').textContent = r.adx    != null ? fmtN(r.adx, 1)        : '—'
    el('ri-vol').textContent = r.vol_30d != null ? fmtN(r.vol_30d, 1) + '%' : '—'
    el('ri-bbw').textContent = r.bb_width != null ? fmtN(r.bb_width, 4)  : '—'
    el('regime-strategy').textContent = STRATEGY_LABELS[r.strategy] || r.strategy || '—'
  }

  function updateEquityChart(history) {
    if (!history.length) return
    const labels = history.map(p => (p.timestamp || '').slice(0, 10))
    const values = history.map(p => p.portfolio_value)
    const ctx = el('eq-chart').getContext('2d')
    if (eqChart) {
      eqChart.data.labels = labels
      eqChart.data.datasets[0].data = values
      eqChart.update('none')
      return
    }
    eqChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data:            values,
          borderColor:     '#3fb950',
          borderWidth:     2,
          backgroundColor: 'rgba(63,185,80,0.08)',
          fill:            true,
          tension:         0.3,
          pointRadius:     0,
          pointHoverRadius: 4,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ' ' + fmt$(c.parsed.y) } } },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 6, maxRotation: 0 } },
          y: { grid: { color: 'rgba(48,54,61,0.6)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
        },
      },
    })
  }

  function updateTradesFeed(trades) {
    const tbody = el('trades-tbody')
    if (!trades.length) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:24px;">No trades yet — start the bot to begin trading</td></tr>`
      return
    }
    tbody.innerHTML = trades.map(t => {
      const isBuy = t.action === 'BUY'
      return `<tr>
        <td>${(t.timestamp || '').slice(11, 19)}</td>
        <td><strong>${t.ticker}</strong></td>
        <td><span class="badge ${isBuy ? 'badge-buy' : 'badge-sell'}">${t.action}</span></td>
        <td>${fmt$(t.price)}</td>
        <td>${fmtN(t.shares, 0)}</td>
        <td class="${clr(t.pnl)}">${t.pnl != null ? fmt$(t.pnl) : '—'}</td>
        <td>${t.pnl_pct != null ? fmtPct(t.pnl_pct) : '—'}</td>
      </tr>`
    }).join('')
  }

  function updateActivityLog(lines) {
    const box = el('activity-log')
    if (!lines.length) return
    box.innerHTML = lines.map(l => `<div class="log-line">${l}</div>`).join('')
  }

  refreshAll()
  setInterval(refreshAll, 10000)
}

// ════════════════════════════════════════════════════════════════════════════
//  BACKTEST
// ════════════════════════════════════════════════════════════════════════════
function initBacktest() {
  let btChart  = null
  let mcChart  = null
  let ff3Chart = null

  el('bt-start').value = daysAgo(365)
  el('bt-end').value   = today()

  enableMlOption(el('bt-strategy'))

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

  async function runBacktest() {
    const tickers = [...btTickers]
    if (!tickers.length) { alert('Add at least one ticker.'); return }

    const riskEl = el('bt-risk-btns').querySelector('.risk-btn.active')
    const commPct = parseFloat(el('bt-commission').value) || 0.10
    const slipPct = parseFloat(el('bt-slippage').value)  || 0.05
    const payload = {
      strategy:        el('bt-strategy').value,
      tickers,
      start_date:      el('bt-start').value,
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
      borderColor: '#30363d', borderWidth: 1, borderDash: [2, 4],
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
          y: { grid: { color: 'rgba(48,54,61,0.6)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
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
          <td><span style="color:${REGIME_COLORS[r] || '#8b949e'};font-weight:700;">${v.label || r}</span></td>
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
          y: { grid: { color: 'rgba(48,54,61,0.6)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
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
              x: { grid: { color: 'rgba(48,54,61,0.6)' }, ticks: { callback: v => v.toFixed(2) } },
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
            borderColor: '#8b949e', backgroundColor: 'rgba(139,148,158,0.5)',
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
          x: { title: { display: true, text: 'Annualised Volatility (%)', color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.6)' }, ticks: { callback: v => v.toFixed(1) + '%' } },
          y: { title: { display: true, text: 'Annualised Return (%)',    color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.6)' }, ticks: { callback: v => v.toFixed(1) + '%' } },
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

// ── Router ──────────────────────────────────────────────────────────────────
const PAGE = document.body.dataset.page
if (PAGE === 'dashboard') initDashboard()
if (PAGE === 'backtest')  initBacktest()
if (PAGE === 'portfolio') initPortfolio()
