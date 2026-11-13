/* ══════════════════════════════════════════════════════════════════════════
   AlphaGlyph — Strategy Validation Lab — app.js
   Vanilla JS · Chart.js 4 · No frameworks
   ══════════════════════════════════════════════════════════════════════════ */

// ── Config ─────────────────────────────────────────────────────────────────
// RENDER_URL is set in config.js (edit that file before deploying to Vercel).
const _isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1'
const API_BASE = window.RENDER_URL || (_isLocal ? 'http://localhost:5000' : '')

if (!_isLocal && !window.RENDER_URL) {
  console.error(
    '%cAlphaGlyph: RENDER_URL is not set in config.js.\n' +
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
  dip_buyer:    'Dip Buyer (52-Week Value)',
  custom:       'Custom Strategy',
}

// Enable the ML strategy option in a <select> once the backend reports a
// trained model is deployed (until then it stays disabled with a hint).
// The model's metadata changes only on redeploy, so the saved copy is exact
// (backend/tests/test_static_api.py keeps it in step). One request per page.
let _mlInfo = null
function mlInfo() {
  if (!_mlInfo) _mlInfo = savedApi('ml_info.json').then(d => d || api('/api/ml/info'))
  return _mlInfo
}

async function enableMlOption(selectEl) {
  if (!selectEl) return
  const info = await mlInfo()
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
  TRENDING_UP:     '#30d158',
  TRENDING_DOWN:   '#ff453a',
  RANGING:         '#ffd60a',
  HIGH_VOLATILITY: '#ff9f0a',
}

// Every trade is turned into a plain-English reason a beginner can follow — what
// the strategy saw, in which market regime, and why it acted. Used by the
// backtest trades table.
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
  dip_buyer:    'the stock was trading near its 52-week low — a value entry',
  custom:       'your custom buy conditions were met',
}
const SELL_REASON = {
  stop_loss:    'price fell back to the trailing stop — locking in the move and capping downside',
  take_profit:  'price hit the take-profit target',
  sell_signal:  'the strategy flipped to a sell signal',
  signal:       'the strategy flipped to a sell signal',
  recovered:    'it recovered toward its 52-week high — locking in the gain',
}
function tradeWhy(t) {
  const where = REGIME_PHRASE[t.regime] || 'the current market'
  if (t.action === 'BUY') {
    if (t.reason === 'scale_in')
      return `Averaged down — bought more as it fell further toward its 52-week low.`
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
// Render's free instance sleeps after inactivity and takes ~30-75s to wake.
// While it boots, Render holds the request open, so a slow answer is a cold
// start and the "waking" banner covers it. A request that *fails* instead means
// the server answered without CORS headers: a deploy restarting (seconds) or a
// suspended service (Render's 503 page, which lasts until the account's free
// hours reset). After RETRY_BUDGET_MS of failures we stop pretending it is
// waking, say plainly that the backend is offline, and fail fast from then on.
let _connFails = 0
let _offline = false
const RETRY_BUDGET_MS = 20000

function setConn(reachable, offline = false) {
  let b = document.getElementById('conn-banner')
  if (reachable) {
    _connFails = 0
    _offline = false
    if (b) b.classList.remove('show', 'offline')
    return
  }
  if (!offline && ++_connFails < 2) return   // tolerate a single transient blip
  if (!b) {
    b = document.createElement('div')
    b.id = 'conn-banner'
    b.className = 'conn-banner'
    b.setAttribute('role', 'status')
    document.body.appendChild(b)
  }
  if (offline) {
    _offline = true
    b.classList.add('offline')
    b.innerHTML = '<span class="conn-dot"></span><span>The live backend is offline right now, so backtests, ' +
      'predictions and scans can’t run. Please check back later.</span>' +
      '<button type="button" class="btn btn-ghost btn-sm conn-retry">Check again</button>'
    b.querySelector('.conn-retry').addEventListener('click', checkBackend)
  } else if (!b.classList.contains('offline')) {
    b.innerHTML = '<span class="conn-dot"></span> Waking the server up — the free instance sleeps after inactivity and takes ~30s to start. Retrying automatically…'
  }
  b.classList.add('show')
}

// "Check again": one request, and a reload once the backend answers so every
// panel on the page loads its data.
async function checkBackend(e) {
  const btn = e && e.currentTarget
  if (btn) { btn.disabled = true; btn.textContent = 'Checking…' }
  try {
    const r = await fetch(API_BASE + '/health', { cache: 'no-store' })
    if (r.ok) { location.reload(); return }
  } catch (_) { /* still offline */ }
  if (btn) { btn.disabled = false; btn.textContent = 'Still offline' }
}

// ── Saved answers and waking the backend early ──────────────────────────────
// A page that calls the API as it opens makes a cold visitor sit through the
// whole 30-60 s boot. Answers that are the same for everyone are saved as
// static files in /data/api/ (tools/static_api.py; the daily ones refresh after
// each close), so pages read those first and open instantly.
async function savedApi(name) {
  try {
    const r = await fetch('/data/api/' + name, { cache: 'no-cache' })
    if (r.ok) return await r.json()
  } catch (_) { /* fall back to the live API */ }
  return null
}

// The backend's own keep-alive holds it awake on weekdays 09:00-16:30 New York
// time (backend/app.py), so a live call in that window costs no wake-up.
function backendLikelyWarm(now = new Date()) {
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', weekday: 'short', hour: 'numeric', minute: 'numeric', hourCycle: 'h23',
  }).formatToParts(now).map(x => [x.type, x.value]))
  if (p.weekday === 'Sat' || p.weekday === 'Sun') return false
  const m = Number(p.hour) * 60 + Number(p.minute)
  return m >= 9 * 60 && m < 16 * 60 + 30
}

// "Saved Nov 7, 4:41 PM. Refresh for live numbers." under a result shown from a saved copy.
function savedNote(box, iso, what = 'Refresh') {
  if (!box || !iso) return
  const when = new Date(iso.replace(/Z$/, ':00Z'))
  const p = document.createElement('p')
  p.className = 'text-muted saved-note'
  p.style.cssText = 'font-size:11.5px;margin-top:8px;'
  p.textContent = `Saved ${when.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}. ${what} for live numbers.`
  box.appendChild(p)
}

// Wake on intent: the first time a visitor touches a form control, send one
// /health ping so the backend boots while they pick a ticker and settings,
// instead of after they press Run. Once per tab, never for automated browsers
// (a crawler would burn free-tier hours), and only on pages that use the API.
function wakeOnIntent() {
  if (navigator.webdriver) return
  const KEY = 'ag_wake_sent'
  const fire = e => {
    if (!e.target.closest || !e.target.closest('input, select, textarea, button')) return
    document.removeEventListener('focusin', fire, true)
    document.removeEventListener('pointerdown', fire, true)
    try {
      if (sessionStorage.getItem(KEY)) return
      sessionStorage.setItem(KEY, '1')
    } catch (_) { /* private mode: still ping once per page */ }
    fetch(API_BASE + '/health', { cache: 'no-store' }).catch(() => {})
  }
  document.addEventListener('focusin', fire, true)
  document.addEventListener('pointerdown', fire, true)
}

// ── Utilities ───────────────────────────────────────────────────────────────
const _sleep = ms => new Promise(r => setTimeout(r, ms))

// Resilient fetch. Free-tier cold starts, brief rate limits, and the market-data
// providers' occasional hiccups are all transient — so instead of surfacing an
// error we back off and retry silently, showing only the friendly "waking up"
// banner. Network failures are retried for RETRY_BUDGET_MS, then the backend is
// declared offline (see setConn) and later calls make a single attempt.
// Returns null when the request genuinely fails.
async function api(path, opts = {}) {
  const MAX = 5
  const started = Date.now()
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(API_BASE + path, {
        ...opts,
        headers: {
          'Content-Type': 'application/json',
          ...(opts.headers || {}),
        },
      })
      setConn(true)                        // we reached the server
      if (r.ok) return r.json()
      // Transient server states (cold start / rate limit) → back off and retry.
      if ((r.status === 429 || r.status >= 500) && attempt < MAX) {
        await _sleep(Math.min(5000, 500 * 2 ** attempt))
        continue
      }
      return null                          // a real 4xx — caller handles gracefully
    } catch (err) {
      if (!(err instanceof TypeError)) return null
      // fetch threw → the server answered without CORS or not at all.
      if (_offline || Date.now() - started >= RETRY_BUDGET_MS) {
        setConn(false, true)
        return null
      }
      setConn(false)
      await _sleep(Math.min(5000, 500 * 2 ** attempt))
    }
  }
}

// What to tell the reader when a request returned nothing.
const failMsg = (data, fallback) =>
  data?.error ? 'Error: ' + data.error
  : _offline ? 'The backend is offline right now, so this can’t run. Please check back later.'
  : fallback

const el     = id => document.getElementById(id)
const fmt$   = n  => n == null ? '—' : '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const fmtPct = (n, d = 2) => n == null ? '—' : (n >= 0 ? '+' : '') + Number(n).toFixed(d) + '%'
const fmtN   = (n, d = 2) => n == null ? '—' : Number(n).toFixed(d)
const clr    = n  => n == null ? '' : n > 0 ? 'positive' : n < 0 ? 'negative' : ''
// Ordinal suffix: 1→"1st", 2→"2nd", 3→"3rd", 11→"11th", 21→"21st", 51→"51st".
const ordinal = n => {
  if (n == null) return '—'
  const i = Math.round(Number(n)), m100 = i % 100, m10 = i % 10
  const suffix = (m100 >= 11 && m100 <= 13) ? 'th'
    : m10 === 1 ? 'st' : m10 === 2 ? 'nd' : m10 === 3 ? 'rd' : 'th'
  return i + suffix
}
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
  'vs SPY':           'Excess return over buying and holding SPY (strategy return − SPY return). Positive means it beat the market; SPY\'s own return is in the summary line and equity chart.',
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

// Chart.js is loaded only by the pages that actually draw something, so the
// screener -- which has no chart on it -- must not run head-first into these
// globals. An unguarded reference here threw at module scope and took the whole
// of app.js down with it, which on that page meant the watchlist never wired up.
if (typeof Chart !== 'undefined') {
  Chart.defaults.color       = '#86868b'
  Chart.defaults.borderColor = 'rgba(255,255,255,0.08)'
  Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter', 'Segoe UI', system-ui, sans-serif"
  Chart.defaults.font.size   = 11
  Chart.defaults.plugins.legend.labels.boxWidth  = 10
  Chart.defaults.plugins.legend.labels.boxHeight = 10
  Chart.defaults.plugins.legend.labels.usePointStyle = true
  Object.assign(Chart.defaults.plugins.tooltip, {
    backgroundColor: 'rgba(44,44,46,0.94)', borderColor: 'rgba(255,255,255,0.14)', borderWidth: 1,
    titleColor: '#f5f5f7', bodyColor: '#d1d1d6', padding: 10, cornerRadius: 8,
    titleFont: { weight: '600' }, displayColors: true, boxPadding: 4,
  })
}

function destroyChart(c) { if (c) { try { c.destroy() } catch (_) {} } return null }

function initTabs(root) {
  root.querySelectorAll('.tab-list').forEach(list => {
    const btns   = list.querySelectorAll('.tab-btn')
    const card   = list.closest('.card') || list.parentElement
    list.setAttribute('role', 'tablist')
    btns.forEach(btn => {
      btn.setAttribute('role', 'tab')
      btn.setAttribute('aria-selected', btn.classList.contains('active') ? 'true' : 'false')
      btn.addEventListener('click', () => {
        btns.forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false') })
        btn.setAttribute('aria-selected', 'true')
        card.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'))
        btn.classList.add('active')
        const target = el(btn.dataset.tab)
        if (target) target.classList.add('active')
      })
    })
  })
}

// ── Custom strategy rule builder (reusable) ──────────────────────────────────
// Renders editable BUY/SELL condition rows into a container and exposes
// getRules() → { buy: {logic, conditions}, sell: {logic, conditions} }.
const RB_INDICATORS = [
  ['close', 'Price'], ['sma20', 'SMA 20'], ['sma50', 'SMA 50'], ['rsi14', 'RSI (14)'],
  ['macd_line', 'MACD line'], ['macd_signal', 'MACD signal'], ['macd_hist', 'MACD histogram'],
  ['vol_ma20', 'Avg volume (20d)'], ['volume', 'Volume'],
  ['return_1d', '1-day return %'], ['return_5d', '5-day return %'],
  ['range52', '52-week range (0=low,1=high)'],
]
const RB_OPS = [['lt', 'is below'], ['gt', 'is above'], ['cross_up', 'crosses above'], ['cross_dn', 'crosses below']]

function makeRuleBuilder(container) {
  const ind  = RB_INDICATORS.map(([v, l]) => `<option value="${v}">${l}</option>`).join('')
  const ops  = RB_OPS.map(([v, l]) => `<option value="${v}">${l}</option>`).join('')
  const rind = `<option value="__num">a number…</option>` + ind

  container.innerHTML = ['buy', 'sell'].map(side => `
    <div class="rb-section">
      <div class="rb-head"><strong>${side.toUpperCase()}</strong> when
        <select class="rb-logic" data-side="${side}"><option value="all">ALL</option><option value="any">ANY</option></select>
        of these are true:</div>
      <div class="rb-rows" data-side="${side}"></div>
      <button class="btn btn-sm rb-add" data-side="${side}" type="button">+ Add ${side} condition</button>
    </div>`).join('')

  function addRow(side, preset) {
    const rows = container.querySelector(`.rb-rows[data-side="${side}"]`)
    const row = document.createElement('div')
    row.className = 'rb-row'
    row.innerHTML =
      `<select class="rb-left">${ind}</select>` +
      `<select class="rb-op">${ops}</select>` +
      `<select class="rb-right">${rind}</select>` +
      `<input class="rb-num" type="number" step="any" value="30">` +
      `<button class="rb-del" type="button" title="Remove">×</button>`
    rows.appendChild(row)
    const right = row.querySelector('.rb-right'), num = row.querySelector('.rb-num')
    const sync = () => { num.style.display = right.value === '__num' ? '' : 'none' }
    right.addEventListener('change', sync)
    row.querySelector('.rb-del').addEventListener('click', () => row.remove())
    if (preset) {
      row.querySelector('.rb-left').value = preset.left
      row.querySelector('.rb-op').value = preset.op
      if (typeof preset.right === 'number') { right.value = '__num'; num.value = preset.right }
      else right.value = preset.right
    }
    sync()
  }

  container.querySelectorAll('.rb-add').forEach(b => b.addEventListener('click', () => addRow(b.dataset.side)))
  addRow('buy',  { left: 'rsi14', op: 'lt', right: 30 })
  addRow('sell', { left: 'rsi14', op: 'gt', right: 70 })

  function group(side) {
    const logic = container.querySelector(`.rb-logic[data-side="${side}"]`).value
    const conditions = [...container.querySelectorAll(`.rb-rows[data-side="${side}"] .rb-row`)].map(r => {
      const rv = r.querySelector('.rb-right').value
      return {
        left:  r.querySelector('.rb-left').value,
        op:    r.querySelector('.rb-op').value,
        right: rv === '__num' ? parseFloat(r.querySelector('.rb-num').value) : rv,
      }
    }).filter(c => c.left && c.op && (typeof c.right === 'string' || !isNaN(c.right)))
    return { logic, conditions }
  }
  function setRules(rules) {
    for (const side of ['buy', 'sell']) {
      const g = rules?.[side]
      if (!g || !Array.isArray(g.conditions)) continue
      container.querySelector(`.rb-logic[data-side="${side}"]`).value = g.logic === 'any' ? 'any' : 'all'
      container.querySelector(`.rb-rows[data-side="${side}"]`).innerHTML = ''
      g.conditions.forEach(c => addRow(side, c))
    }
  }
  return { getRules: () => ({ buy: group('buy'), sell: group('sell') }), setRules }
}

// ════════════════════════════════════════════════════════════════════════════
//  BACKTEST
// ════════════════════════════════════════════════════════════════════════════
function initBacktest() {
  let btChart  = null
  let mcChart  = null
  let ff3Chart = null
  let cmpChart = null
  let btRules  = null   // custom rule builder (lazy)
  el('bt-start').value = daysAgo(365)
  el('bt-end').value   = today()

  enableMlOption(el('bt-strategy'))

  // ── ML out-of-sample guard ─────────────────────────────────────────────
  // The transformer is trained once (chronological 60/20/20). To show it
  // "actually working", an ML backtest must run only on dates AFTER its
  // train+validation cutoff — data the model never saw. We read that cutoff
  // from the model meta and push the start date out-of-sample automatically.
  let mlCutoff = null   // { train_end, val_end, purge_days }
  mlInfo().then(info => {
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
      `Trained on data through <strong>${mlCutoff.train_end}</strong>, validated through ` +
      `<strong>${valEnd}</strong>. So you see the model working on data it never trained on, ` +
      `this backtest runs <strong>out-of-sample from ${valEnd}</strong>.`
    note.hidden = false
  }

  function applyBtStrategyHints() {
    applyMlOOS()   // handles ML out-of-sample note/dates (and hides note otherwise)
    const strat = el('bt-strategy').value
    if (strat === 'dip_buyer') {
      el('bt-start').value = daysAgo(730)   // 2y so it has dips to buy
      const note = el('bt-ml-note')
      if (note) {
        note.innerHTML = '<strong>Dip Buyer</strong> only buys near 52-week lows — date range set to ' +
          '<strong>2 years</strong> so it has dips to act on.'
        note.hidden = false
      }
    }
    const rb = el('bt-rule-builder')
    if (rb) {
      const isCustom = strat === 'custom'
      rb.hidden = !isCustom
      if (isCustom && !btRules) btRules = makeRuleBuilder(rb)
    }
  }
  el('bt-strategy').addEventListener('change', () => {
    applyBtStrategyHints()
    // Changing the strategy invalidates any result on screen — clear it so the
    // summary/leaderboard can never disagree with the currently-selected strategy.
    el('results-container').hidden = true
    el('compare-container').hidden = true
    if (el('loading-state').hidden && el('compare-loading').hidden) {
      el('empty-state').hidden = false
    }
  })
  applyBtStrategyHints()   // apply hints for the default (recommended) strategy on load

  // ── Ticker management (free-text input, validated against the backend) ──
  let btTickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'PLTR']
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

  // ── Quick-start presets ────────────────────────────────────────────────────
  // A first-time visitor shouldn't have to know what a strategy, ticker list
  // and date range should look like. Each preset sets all three (and resets
  // the advanced toggles) and runs straight away; every field stays editable.
  const BT_PRESETS = {
    dip:      { strategy: 'dip_buyer',    tickers: ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META'], years: 3, cash_in_market: true },
    trend:    { strategy: 'ma_crossover', tickers: ['SPY', 'QQQ'],                                 years: 5 },
    oversold: { strategy: 'rsi',          tickers: ['KO', 'PG', 'JNJ', 'PEP', 'WMT'],               years: 5 },
    chips:    { strategy: 'macd',         tickers: ['NVDA', 'AMD', 'AVGO', 'QCOM'],                 years: 3 },
    regime:   { strategy: 'adaptive',     tickers: ['SPY', 'QQQ', 'IWM'],                           years: 5 },
  }

  function applyPreset(key) {
    const p = BT_PRESETS[key]
    if (!p) return
    el('bt-strategy').value = p.strategy
    applyBtStrategyHints()
    el('bt-start').value = daysAgo(Math.round(p.years * 365.25))
    el('bt-end').value = today()
    btTickers = [...p.tickers]
    renderTickerChips()
    hideTickerErr()
    el('bt-risk-btns').querySelectorAll('.risk-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.risk === 'moderate'))
    el('bt-walkforward').checked   = false
    el('bt-markowitz').checked     = false
    el('bt-range-sizing').checked  = false
    el('bt-cash-in-market').checked = !!p.cash_in_market
    el('bt-presets').querySelectorAll('.preset-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.preset === key))
    runBacktest()
    // On a phone the form sits above the results; bring the result into view.
    if (window.matchMedia('(max-width: 1024px)').matches) {
      el('loading-state').scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  el('bt-presets').querySelectorAll('.preset-btn').forEach(b =>
    b.addEventListener('click', () => applyPreset(b.dataset.preset)))
  // ── Shareable links ────────────────────────────────────────────────────────
  // Every run writes its settings into the address bar, so the URL always
  // reproduces what is on screen; "Copy link" hands that URL to someone else,
  // and opening it fills the form and runs the same backtest.
  const OPT_FLAGS = { wf: 'bt-walkforward', mz: 'bt-markowitz', rs: 'bt-range-sizing', cim: 'bt-cash-in-market' }
  const b64url = s => btoa(unescape(encodeURIComponent(s))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  const unb64url = s => decodeURIComponent(escape(atob(s.replace(/-/g, '+').replace(/_/g, '/'))))

  function settingsToQuery() {
    const q = new URLSearchParams()
    q.set('s', el('bt-strategy').value)
    q.set('t', btTickers.join(','))
    q.set('from', el('bt-start').value)
    q.set('to', el('bt-end').value)
    q.set('cap', el('bt-capital').value)
    q.set('risk', el('bt-risk-btns').querySelector('.risk-btn.active')?.dataset.risk || 'moderate')
    const opts = Object.keys(OPT_FLAGS).filter(k => el(OPT_FLAGS[k]).checked)
    if (opts.length) q.set('opt', opts.join(','))
    q.set('cost', `${el('bt-commission').value},${el('bt-slippage').value}`)
    if (el('bt-strategy').value === 'custom' && btRules) q.set('rules', b64url(JSON.stringify(btRules.getRules())))
    return q.toString()
  }

  function settingsFromQuery(search) {
    const q = new URLSearchParams(search)
    const tickers = (q.get('t') || '').toUpperCase().split(',')
      .map(x => x.trim()).filter(x => /^[A-Z0-9.^-]{1,10}$/.test(x)).slice(0, 20)
    if (!tickers.length) return false
    const strat = q.get('s')
    const opt = [...el('bt-strategy').options].find(o => o.value === strat && !o.disabled)
    if (opt) { el('bt-strategy').value = strat; applyBtStrategyHints() }
    const date = v => /^\d{4}-\d{2}-\d{2}$/.test(v || '') ? v : null
    if (date(q.get('from'))) el('bt-start').value = q.get('from')
    if (date(q.get('to')))   el('bt-end').value = q.get('to')
    const cap = parseFloat(q.get('cap'))
    if (cap >= 1000 && cap <= 1e7) el('bt-capital').value = cap
    const risk = q.get('risk')
    el('bt-risk-btns').querySelectorAll('.risk-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.risk === (['conservative', 'moderate', 'aggressive'].includes(risk) ? risk : 'moderate')))
    const opts = (q.get('opt') || '').split(',')
    Object.entries(OPT_FLAGS).forEach(([k, id]) => { el(id).checked = opts.includes(k) })
    const [com, slip] = (q.get('cost') || '').split(',').map(parseFloat)
    if (com >= 0 && com <= 5) el('bt-commission').value = com
    if (slip >= 0 && slip <= 5) el('bt-slippage').value = slip
    if (strat === 'custom' && q.get('rules') && btRules) {
      try { btRules.setRules(JSON.parse(unb64url(q.get('rules')))) } catch { /* keep the default rules */ }
    }
    btTickers = tickers
    renderTickerChips()
    return true
  }

  function rememberRunInUrl() {
    history.replaceState(null, '', `${location.pathname}?${settingsToQuery()}`)
  }

  const shareBtn = el('bt-share-btn')
  if (shareBtn) shareBtn.addEventListener('click', async () => {
    const url = `${location.origin}${location.pathname}?${settingsToQuery()}`
    try {
      await navigator.clipboard.writeText(url)
      shareBtn.textContent = 'Link copied'
    } catch {
      window.prompt('Copy this link:', url)
    }
    setTimeout(() => { shareBtn.textContent = 'Copy link' }, 1800)
  })

  if (location.search && settingsFromQuery(location.search)) setTimeout(runBacktest, 0)

  // Editing the form by hand means the preset no longer describes it.
  ;['bt-strategy', 'bt-start', 'bt-end'].forEach(id =>
    el(id).addEventListener('input', () =>
      el('bt-presets').querySelectorAll('.preset-btn.active').forEach(b => b.classList.remove('active'))))

  // ── Strategy leaderboard (compare all strategies) ──────────────────────────
  const CMP_COLORS = {
    adaptive: '#30d158', ma_crossover: '#0a84ff', rsi: '#bf5af2',
    macd: '#ff9f0a', ml: '#ff375f',
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
      el('empty-state').querySelector('.empty-title').textContent =
        failMsg(data, 'The comparison failed. Please try again.')
      el('empty-state').querySelector('.empty-hint')?.setAttribute('hidden', '')
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
      return `<tr>
        <td><span class="rank-badge${i === 0 ? ' rank-top' : ''}">${i + 1}</span></td>
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
        borderColor: CMP_COLORS[r.strategy] || '#86868b',
        borderWidth: 2, fill: false, tension: 0.2, pointRadius: 0, spanGaps: true,
      }
    })
    const spyMap = byDate(data.spy_curve)
    datasets.push({
      label: 'SPY Benchmark', data: labels.map(d => (spyMap[d] != null ? spyMap[d] : null)),
      borderColor: '#86868b', borderWidth: 1.5, borderDash: [4, 4],
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
          y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
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
      range_sizing:    el('bt-range-sizing').checked,
      cash_in_market:  el('bt-cash-in-market').checked,
      ...(el('bt-strategy').value === 'custom' && btRules ? { custom_rules: btRules.getRules() } : {}),
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
      el('empty-state').querySelector('.empty-title').textContent =
        failMsg(data, 'The backtest failed. Please try again.')
      el('empty-state').querySelector('.empty-hint')?.setAttribute('hidden', '')
      return
    }

    btChart  = destroyChart(btChart)
    mcChart  = destroyChart(mcChart)
    ff3Chart = destroyChart(ff3Chart)

    renderResults(data)
    el('results-container').hidden = false
    initTabs(el('results-container'))
    rememberRunInUrl()
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

    renderSummary(m, data)

    el('summary-stats').innerHTML = [
      statCard('Total Return', fmtPct(m.total_return), clr(m.total_return)),
      statCard('Final Value',  fmt$(m.final_value)),
      statCard('Sharpe Ratio', fmtN(m.sharpe_ratio), m.sharpe_ratio > 1 ? 'positive' : m.sharpe_ratio < 0 ? 'negative' : ''),
      statCard('Max Drawdown', m.max_drawdown != null ? '-' + fmtN(m.max_drawdown) + '%' : '—', 'negative'),
    ].join('')

    renderBtChart(data)
    renderSecondaryStats(m)
    renderRegimeBreakdown(data.regime_breakdown || {}, data.regime_conditional || {})
    renderMonteCarlo(data.monte_carlo, data.equity_curve)
    renderResearchTab(data)
    renderTradesTable(data.trades || [])
  }

  // Turn the raw backtest metrics into a one-line verdict plus one short
  // paragraph a non-expert can act on. The verdict answers the two questions
  // people actually have: did it beat just buying the index, and is that real?
  function maxDrawdownPct(curve) {
    let peak = -Infinity, worst = 0
    for (const p of curve || []) {
      peak = Math.max(peak, p.value)
      if (peak > 0) worst = Math.max(worst, (peak - p.value) / peak * 100)
    }
    return worst
  }

  function verdictFor(m, data) {
    const ret = m.total_return, bench = m.benchmark_return
    const real = !!(data.skill_test?.is_significant || data.deflated_sharpe?.is_significant)
    if (ret == null) return null
    if (ret <= 0) return { tone: 'red', title: 'Lost money over this period' }
    if (bench == null) return { tone: 'neutral', title: 'Made money. Check the Validation tab to see if it is luck.' }
    if (ret > bench && real) return { tone: 'green', title: 'Beat the market, and the edge looks real' }
    if (ret > bench) return { tone: 'yellow', title: 'Beat the market, but it may be luck' }
    return { tone: 'yellow', title: 'Made money, but less than buying the index' }
  }

  function renderSummary(m, data = {}) {
    const box = el('bt-summary')
    if (!box) return
    if (m.total_return == null) { box.hidden = true; return }

    const ret = m.total_return, bench = m.benchmark_return, dd = m.max_drawdown
    const wr = m.win_rate, n = m.total_trades, fv = m.final_value, cap = m.initial_capital
    const spyDD = data.spy_curve?.length ? maxDrawdownPct(data.spy_curve) : null
    const exposure = data.skill_test?.exposure_pct
    const verb = ret >= 0 ? 'grew' : 'shrank'

    let s = `${fmt$(cap)} ${verb} to <strong>${fmt$(fv)}</strong> (<strong>${fmtPct(ret)}</strong>)`
    if (bench != null) s += `, versus <strong>${fmtPct(bench)}</strong> for buying and holding SPY`
    s += '. '
    if (n != null) {
      s += `It made <strong>${n}</strong> trade${n === 1 ? '' : 's'}`
      if (wr != null) s += ` and won <strong>${fmtN(wr, 0)}%</strong> of them`
      s += '. '
    }
    if (dd != null) {
      s += `Its worst drop was <strong>${fmtN(dd, 1)}%</strong>`
      if (spyDD != null) s += ` (SPY's was ${fmtN(spyDD, 1)}%)`
      s += '.'
    }

    let take = ''
    if (exposure != null && exposure < 60 && bench != null && ret < bench) {
      take = `It was invested only ${fmtN(exposure, 0)}% of the time, which is the main reason it trailed a market that rose ${fmtPct(bench)}.`
    } else if (bench != null && ret < bench && dd != null && spyDD != null && dd < spyDD) {
      take = 'It gave up some upside for a smaller worst drop. Strategies like this are best judged over a full market cycle.'
    } else if (bench != null && ret > bench && !(data.skill_test?.is_significant || data.deflated_sharpe?.is_significant)) {
      take = 'The Validation tab shows how often random trading would have done as well.'
    }

    const v = verdictFor(m, data)
    box.innerHTML =
      (v ? `<div class="verdict-line verdict-${v.tone}"><span class="verdict-dot" aria-hidden="true"></span>${v.title}</div>` : '') +
      `<p class="summary-body">${s}</p>` +
      (take ? `<span class="summary-take">${take}</span>` : '')
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
        borderColor: '#0a84ff', borderWidth: 2,
        backgroundColor: 'rgba(10,132,255,0.10)', fill: true, tension: 0.2, pointRadius: 0,
      },
    ]
    if (spy.length) datasets.push({
      label: 'SPY Benchmark', data: spy.map(p => p.value),
      borderColor: '#86868b', borderWidth: 1.5, borderDash: [4, 4],
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
          y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
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
      statCard('vs SPY',        fmtPct((m.total_return || 0) - (m.benchmark_return || 0)), clr((m.total_return || 0) - (m.benchmark_return || 0))),
    ].join('')
  }

  function renderRegimeBreakdown(breakdown, conditional) {
    const box     = el('regime-breakdown')
    const entries = Object.entries(breakdown)
    const cond    = Object.entries(conditional || {})
      .sort((a, b) => (b[1].days || 0) - (a[1].days || 0))
    if (!entries.length && !cond.length) { box.innerHTML = ''; return }

    // Time-based conditional block: is the edge spread across conditions, or is
    // it just one lucky regime carrying the whole result?
    let condHtml = ''
    if (cond.length) {
      condHtml = `
        <div class="section-header" style="margin-top:28px;">Edge by market regime</div>
        <p class="text-muted" style="font-size:12px;line-height:1.6;margin:-6px 0 12px;">
          Splits every invested <em>day</em> by the market regime it fell in, and compares the strategy's compounded return to buy-and-hold (SPY) <strong>within</strong> that regime. A strategy whose whole edge lives in one regime is far more fragile than one that beats the benchmark across several.
        </p>
        <div class="table-wrap"><table class="data-table">
          <thead><tr>
            <th>Regime</th><th>% of Time</th><th>Strategy</th><th>SPY</th><th>Excess</th><th>Sharpe</th>
          </tr></thead>
          <tbody>${cond.map(([r, v]) => `<tr>
            <td><span style="color:${REGIME_COLORS[r] || '#86868b'};font-weight:700;">${v.label || r}</span></td>
            <td>${v.pct_of_time}%</td>
            <td class="${clr(v.strat_return)}">${fmtPct(v.strat_return)}</td>
            <td>${v.spy_return == null ? '—' : fmtPct(v.spy_return)}</td>
            <td class="${clr(v.excess)}">${v.excess == null ? '—' : fmtPct(v.excess)}</td>
            <td>${fmtN(v.sharpe)}</td>
          </tr>`).join('')}</tbody>
        </table></div>`
    }

    let tradeHtml = ''
    if (entries.length) {
      tradeHtml = `
        <div class="section-header" style="margin-top:28px;">Closed trades by market regime</div>
        <div class="table-wrap"><table class="data-table">
          <thead><tr>
            <th>Regime</th><th>Trades</th><th>Win Rate</th>
            <th>Total P&L</th><th>Avg P&L</th><th>Best</th><th>Worst</th>
          </tr></thead>
          <tbody>${entries.map(([r, v]) => `<tr>
            <td><span style="color:${REGIME_COLORS[r] || '#86868b'};font-weight:700;">${v.label || r}</span></td>
            <td>${v.trade_count}</td>
            <td>${v.win_rate}%</td>
            <td class="${clr(v.total_pnl)}">${fmt$(v.total_pnl)}</td>
            <td class="${clr(v.avg_pnl)}">${fmt$(v.avg_pnl)}</td>
            <td class="positive">${fmt$(v.best_trade)}</td>
            <td class="negative">${fmt$(v.worst_trade)}</td>
          </tr>`).join('')}</tbody>
        </table></div>`
    }

    box.innerHTML = condHtml + tradeHtml
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
          { label: 'P95',          data: fc.p95, borderColor: 'transparent', backgroundColor: 'rgba(10,132,255,0.10)', fill: '+1', pointRadius: 0 },
          { label: 'P75',          data: fc.p75, borderColor: 'transparent', backgroundColor: 'rgba(10,132,255,0.20)', fill: '+1', pointRadius: 0 },
          { label: 'P50 (Median)', data: fc.p50, borderColor: 'rgba(10,132,255,0.9)', borderWidth: 1.5, borderDash: [4, 3], backgroundColor: 'rgba(10,132,255,0.20)', fill: '+1', pointRadius: 0 },
          { label: 'P25',          data: fc.p25, borderColor: 'transparent', backgroundColor: 'rgba(10,132,255,0.10)', fill: '+1', pointRadius: 0 },
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
          y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } },
        },
      },
    })

    const rd = mc.return_distribution
    const sd = mc.sharpe_distribution
    box.innerHTML = `
      <div class="card">
        <div class="section-header">Return distribution</div>
        <table class="data-table"><thead><tr><th>Percentile</th><th>Return</th></tr></thead>
        <tbody>
          ${[['P5', rd.p5],['P25', rd.p25],['P50 (Median)', rd.p50],['P75', rd.p75],['P95', rd.p95],['Actual', mc.actual_return_pct]]
            .map(([k, v]) => `<tr><td>${k}</td><td class="${clr(v)}">${fmtPct(v)}</td></tr>`).join('')}
        </tbody></table>
        <p style="margin-top:12px;font-size:12px;color:var(--muted);">
          The range of outcomes when the strategy's own daily returns are
          resampled. This is a picture of <strong style="color:var(--text);">outcome spread</strong>,
          not a skill test — the actual sits near the middle by construction.
          The <em>Research</em> tab's skill test is what asks whether the timing beat random.
          ${mc.bootstrap_method === 'stationary'
            ? `<br><span style="opacity:0.8;">Stationary block bootstrap (avg block ${mc.avg_block_len} days) — preserves autocorrelation.</span>`
            : ''}
        </p>
      </div>
      <div class="card">
        <div class="section-header">Sharpe distribution</div>
        <table class="data-table"><thead><tr><th>Percentile</th><th>Sharpe</th></tr></thead>
        <tbody>
          ${[['P5', sd.p5],['P25', sd.p25],['P50 (Median)', sd.p50],['P75', sd.p75],['P95', sd.p95]]
            .map(([k, v]) => `<tr><td>${k}</td><td>${fmtN(v)}</td></tr>`).join('')}
        </tbody></table>
        <p style="margin-top:12px;font-size:12px;color:var(--muted);">
          Spread of Sharpe ratios across the ${fmtN(mc.n_simulations || 1000, 0)} resampled paths.
        </p>
      </div>`
  }

  function renderResearchTab(data) {
    const dsr   = data.deflated_sharpe  || {}
    const ff3   = data.fama_french      || {}
    const skill = data.skill_test       || {}

    const skillPct = skill.enabled ? (skill.skill_percentile ?? 0) : null
    const test1    = skill.enabled && skillPct >= 95   // beat 95% of random-timing schedules
    const test2    = dsr.is_significant ?? false
    const test3    = ff3.enabled && Math.abs(ff3.alpha_t_stat || 0) > 2.0

    const passes   = [test1, test2, test3].filter(Boolean).length
    let vClass, vText
    if (passes === 3) { vClass = 'verdict-significant';  vText = 'Statistically significant' }
    else if (passes >= 1) { vClass = 'verdict-promising'; vText = 'Promising, but needs more data' }
    else               { vClass = 'verdict-inconclusive'; vText = 'Inconclusive. This may be noise' }

    const testRow = (pass, html) => `
      <div class="verdict-test ${pass ? 'pass' : 'fail'}">
        <span class="test-icon">${pass ? '✓' : '✗'}</span>
        <span class="test-label">${html}</span>
      </div>`

    el('validation-report').innerHTML = `
      <div class="verdict-card ${vClass}">
        <div class="verdict-label">Strategy validation report</div>
        <div class="verdict-main">${vText}</div>
        <div class="verdict-tests">
          ${testRow(test1, skill.enabled
            ? `Skill test: the strategy's Sharpe beat <strong>${ordinal(skillPct)} percentile</strong> of 1,000 random-timing schedules (matched exposure ${fmtN(skill.exposure_pct, 0)}%) — ${skillPct >= 95 ? 'better than random' : 'not distinguishable from random'}`
            : 'Skill test: needs a benchmark series — run with SPY available')}
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
        <div class="section-header">Skill test vs. random timing</div>
        <p style="font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.6;">
          Compares the strategy's Sharpe to <strong>1,000 "monkey" traders</strong> that
          go long the market on random days, matched to the strategy's own market
          exposure. A high percentile means the <strong>timing</strong> — not just being
          invested — added value. (This is the real skill test; the Monte&nbsp;Carlo tab
          shows the outcome <em>spread</em>, which sits at ~50% by construction.)
        </p>
        ${skill.enabled
          ? bar('Beat random-timing schedules', skillPct) + `
            <div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border);font-size:12px;color:var(--muted);">
              Strategy Sharpe: <strong style="color:var(--text);">${fmtN(skill.strategy_sharpe)}</strong>
              &nbsp;·&nbsp; Random median: <strong style="color:var(--text);">${fmtN(skill.null_sharpe_median)}</strong>
              &nbsp;·&nbsp; Exposure: <strong style="color:var(--text);">${fmtN(skill.exposure_pct, 0)}%</strong>
            </div>
            <p style="font-size:11px;color:var(--muted);margin-top:10px;line-height:1.6;">
              Honest limitation: the null trades the SPY benchmark, so it blends
              market-timing skill with asset selection — a genuine null, not a perfect attribution.
            </p>`
          : '<p style="color:var(--muted);font-size:13px;">Needs a benchmark series (SPY) over the backtest window.</p>'}
      </div>
      <div class="card">
        <div class="section-header">Deflated Sharpe ratio</div>
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
        <div class="section-header">Fama-French 3-factor attribution</div>
        ${ff3.enabled ? `
          <p style="font-size:12px;color:var(--muted);margin-bottom:12px;line-height:1.6;">
            Decomposes returns into known <strong>market</strong>, <strong>size (SMB)</strong>, and <strong>value (HML)</strong> risk premia.
            Alpha is what remains — skill that a passive factor ETF cannot replicate.
          </p>
          <div style="margin-bottom:14px;">
            <div style="font-size:28px;font-weight:600;letter-spacing:-0.02em;font-variant-numeric:tabular-nums;color:${ff3.alpha_annual >= 0 ? 'var(--green)' : 'var(--red)'};">
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
              backgroundColor: ['rgba(48,209,88,0.55)', 'rgba(191,90,242,0.55)', 'rgba(255,214,10,0.55)'],
              borderColor:     ['#30d158', '#bf5af2', '#ffd60a'],
              borderWidth:     1, borderRadius: 4,
            }],
          },
          options: {
            indexAxis: 'y', responsive: true,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ' β = ' + fmtN(c.parsed.x, 4) } } },
            scales: {
              x: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { callback: v => v.toFixed(2) } },
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
          <div class="section-header">Markowitz position sizing</div>
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
      tbody.innerHTML = `<tr><td colspan="9" class="empty-cell">No closed trades in this period</td></tr>`
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
        shrink:     el('opt-shrink').checked,
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
    renderShrinkage(data)
    el('opt-results').hidden = false
  }

  // Show the Ledoit-Wolf shrinkage intensity + resulting concentration, so the
  // effect of the toggle is legible rather than invisible.
  function renderShrinkage(data) {
    let box = el('opt-shrink-note')
    const s = data.shrinkage
    if (!s) { if (box) box.remove(); return }
    if (!box) {
      box = document.createElement('div')
      box.id = 'opt-shrink-note'
      box.className = 'card'
      box.style.cssText = 'margin-bottom:20px;border-color:var(--green);'
      const results = el('opt-results')
      results.insertBefore(box, results.firstChild)
    }
    const pct = Math.round(s.delta * 100)
    box.innerHTML = `<div class="section-header" style="color:var(--green);">Ledoit-Wolf shrinkage applied</div>
      <p style="font-size:12.5px;line-height:1.65;margin:0;color:var(--text);">
        Optimal shrinkage intensity <strong>δ = ${s.delta}</strong> — the covariance was pulled
        <strong>${pct}%</strong> of the way toward the well-conditioned target, ${pct < 10 ? 'a light touch (the sample was already fairly stable)' : pct > 40 ? 'a heavy pull (the raw estimate was noisy)' : 'a moderate correction'}.
        Max-Sharpe concentration (Herfindahl) is now <strong>${s.hhi}</strong> — lower means less over-concentration than naive Markowitz. Toggle the box off and re-run to compare.
      </p>`
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
            borderColor: '#30d158', backgroundColor: 'rgba(48,209,88,0.12)',
            showLine: true, tension: 0.3, pointRadius: 2.5, borderWidth: 2,
          },
          {
            label: '★ Max-Sharpe',
            data:  [{ x: ms.volatility, y: ms.expected_return }],
            borderColor: '#ffd60a', backgroundColor: '#ffd60a',
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
            borderColor: '#86868b', backgroundColor: 'rgba(134,134,139,0.5)',
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
          x: { title: { display: true, text: 'Annualised Volatility (%)', color: '#86868b' }, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { callback: v => v.toFixed(1) + '%' } },
          y: { title: { display: true, text: 'Annualised Return (%)',    color: '#86868b' }, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { callback: v => v.toFixed(1) + '%' } },
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
      ? `rgba(48,209,88,${0.08 + Math.abs(v) * 0.65})`
      : `rgba(255,69,58,${0.08 + Math.abs(v) * 0.65})`

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

  async function loadRegime(r) {
    if (!r) r = await api('/api/regime')
    const badge = el('sig-regime-badge')
    if (!r || r.error) { el('sig-regime-desc').textContent = 'Market regime unavailable right now.'; return }
    badge.textContent = (r.label || r.regime || 'UNKNOWN').toUpperCase()
    badge.className = 'regime-badge ' + (r.regime || '')
    el('sig-regime-desc').innerHTML = (r.description || '') +
      (r.strategy ? ` · favors <strong>${STRATEGY_LABELS[r.strategy] || r.strategy}</strong>` : '')
  }

  async function scan() {
    const tbody = el('sig-tbody')
    if (!tickers.length) { tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">Add a ticker to scan.</td></tr>'; return }
    if (regimeFromSave) { regimeFromSave = false; loadRegime() }   // Rescan refreshes both
    tbody.innerHTML = `<tr><td colspan="7" class="empty-cell">Scanning <span class="dots"><span></span><span></span><span></span></span></td></tr>`
    const data = await api('/api/scan?tickers=' + encodeURIComponent(tickers.join(',')))
    if (!data || !data.scanned) { tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">Couldn’t scan right now — try Rescan.</td></tr>'; return }
    renderScan(data)
    const now = new Date()
    el('sig-updated').textContent = 'Updated ' + now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  function renderScan(data) {
    const tbody = el('sig-tbody')

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
  }

  // Opening the page: the default list's scan saved after the last close shows
  // at once (tools/static_api.py). During market hours the backend is awake
  // anyway, so a live scan replaces it; outside them, Rescan goes live.
  let regimeFromSave = false
  async function firstLoad() {
    renderChips()
    const saved = await savedApi('screener.json')
    const same = saved && saved.tickers && saved.tickers.length === tickers.length &&
      tickers.every(t => saved.tickers.includes(t))
    if (!same || backendLikelyWarm()) { loadRegime(); scan(); return }
    loadRegime(saved.regime); regimeFromSave = true
    const keep = saved.scan.scanned.filter(row => tickers.includes(row.ticker))
    renderScan({ ...saved.scan, scanned: tickers.map(t => keep.find(r => r.ticker === t)).filter(Boolean) })
    const when = new Date(saved.generated_at.replace(/Z$/, ':00Z'))
    el('sig-updated').textContent = 'Saved ' + when.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) + ' · Rescan for live'
  }

  firstLoad()
}

// ── Predictor ────────────────────────────────────────────────────────────────
// A single-ticker forward lean, blended from the ML price forecast, an earnings
// (PEAD + growth) tilt, and news sentiment. Every number traces to a component
// the user can see — the whole card is built to be inspectable, not magic.
function initPredictor() {
  const input = el('pred-input')
  const run   = el('pred-run')
  const errBox = el('pred-error')
  const QUICK = ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'JPM']

  const DIR = {
    BULLISH: { cls: 'dir-bull',    text: 'LEANS BULLISH' },
    BEARISH: { cls: 'dir-bear',    text: 'LEANS BEARISH' },
    NEUTRAL: { cls: 'dir-neutral', text: 'ROUGHLY BALANCED' },
  }

  // A signed contribution bar for a score in [-1, +1], centred at zero.
  function signedBar(score) {
    const s = Math.max(-1, Math.min(1, Number(score) || 0))
    const pct = Math.abs(s) * 50
    const side = s >= 0 ? 'pos' : 'neg'
    const left = s >= 0 ? 50 : 50 - pct
    return `<div class="comp-bar"><span class="comp-bar-mid"></span>` +
           `<span class="comp-bar-fill ${side}" style="left:${left}%;width:${pct}%;"></span></div>`
  }

  function renderQuick() {
    el('pred-quick').innerHTML = QUICK.map(t =>
      `<span class="ticker-chip" data-ticker="${t}">${t}</span>`).join('')
    el('pred-quick').querySelectorAll('.ticker-chip').forEach(c =>
      c.addEventListener('click', () => { input.value = c.dataset.ticker; predict() }))
  }

  function componentCard(c) {
    if (!c.available) {
      const why = c.reason ? ` — ${c.reason}` : ''
      return `<div class="comp comp-off">
        <div class="comp-head"><span class="comp-label">${c.label}</span>
          <span class="comp-weight">unavailable${why}</span></div>
      </div>`
    }
    const weight = c.weight != null ? `${Math.round(c.weight * 100)}% weight` : ''
    const factors = (c.factors || []).map(f => `<li>${f}</li>`).join('')
    return `<div class="comp">
      <div class="comp-head"><span class="comp-label">${c.label}</span>
        <span class="comp-weight">${weight}</span></div>
      ${signedBar(c.score)}
      <ul class="comp-factors">${factors}</ul>
    </div>`
  }

  function rangeCard(d, horizon) {
    if (!d) return ''
    const lo = d.q10, hi = d.q90, mid = d.q50
    const span = (hi - lo) || 1
    const pos = v => Math.max(0, Math.min(100, ((v - lo) / span) * 100))
    return `<div class="card" style="margin-bottom:16px;">
      <div class="section-header">Modeled ${horizon}-day move (ML return distribution)</div>
      <div class="pred-range">
        <span class="pred-range-fill" style="left:${pos(d.q25)}%;width:${pos(d.q75) - pos(d.q25)}%;"></span>
        <span class="pred-range-mid" style="left:${pos(mid)}%;"></span>
      </div>
      <div class="pred-range-labels">
        <span>q10 ${fmtPct(lo, 1)}</span>
        <span><strong>median ${fmtPct(mid, 1)}</strong></span>
        <span>q90 ${fmtPct(hi, 1)}</span>
      </div>
      <p class="text-muted" style="font-size:11px;margin-top:10px;line-height:1.6;">
        The shaded band is the interquartile range (q25–q75); the line is the median. A wide band means the model is
        uncertain — the honest signal is the <em>range</em>, not a single number.
      </p>
    </div>`
  }

  function render(data) {
    const box = el('pred-result')
    box.hidden = false
    if (!data || data.error) {
      box.innerHTML = `<div class="card"><p class="text-muted">Couldn’t run a prediction right now — try again shortly.</p></div>`
      return
    }
    if (!data.available) {
      box.innerHTML = `<div class="card"><div class="section-header">No signal</div>
        <p class="text-muted" style="font-size:13px;line-height:1.6;">${data.reason || 'No signal available for this ticker right now.'}
        Free earnings and news data is thin for some tickers — try a large, liquid name (e.g. AAPL, MSFT, NVDA).</p></div>`
      return
    }

    const p = data.prediction
    const dir = DIR[p.direction] || DIR.NEUTRAL
    const prob = Math.round(p.p_up * 100)
    const fillLeft = p.p_up >= 0.5 ? 50 : prob
    const fillW = Math.abs(prob - 50)
    const fillSide = p.p_up >= 0.5 ? 'pos' : 'neg'

    box.innerHTML = `
      <div class="card pred-verdict" style="margin-bottom:16px;">
        <div class="pred-verdict-head">
          <div>
            <div class="pred-ticker">${data.ticker}${data.price != null ? ` <span class="text-muted" style="font-weight:500;font-size:15px;">${fmt$(data.price)}</span>` : ''}</div>
            <div class="pred-dir ${dir.cls}">${dir.text}</div>
          </div>
          <div class="pred-prob">
            <div class="pred-prob-num ${fillSide}">${prob}%</div>
            <div class="pred-prob-lbl">chance up · next ${data.horizon}d</div>
          </div>
        </div>
        <div class="prob-track"><span class="prob-mid"></span>
          <span class="prob-fill ${fillSide}" style="left:${fillLeft}%;width:${fillW}%;"></span></div>
        <div class="pred-conf">Confidence: <strong>${p.confidence_label}</strong>
          <span class="text-muted"> · ${p.agreement > 0.6 ? 'signals broadly agree' : p.agreement < 0.35 ? 'signals disagree — read cautiously' : 'signals are mixed'}</span></div>
        <p class="pred-verdict-text">${p.verdict_text}</p>
      </div>

      ${rangeCard(data.distribution, data.horizon)}

      <div class="card" style="margin-bottom:16px;">
        <div class="section-header">How the signals blend</div>
        <p class="text-muted" style="font-size:11.5px;margin:-4px 0 14px;line-height:1.6;">
          Each channel emits a score from −1 (bearish) to +1 (bullish); the blend weights each by its confidence and how much a channel like it usually deserves. Bars point right for bullish, left for bearish.
        </p>
        ${(data.components || []).map(componentCard).join('')}
      </div>

      <p class="text-muted" style="font-size:11px;line-height:1.6;padding:0 2px;">⚠ ${data.disclaimer}</p>
    `
  }

  async function predict() {
    const sym = (input.value || '').trim().toUpperCase()
    errBox.hidden = true
    if (!sym) { errBox.textContent = 'Enter a ticker symbol.'; errBox.hidden = false; return }
    if (!/^[A-Z][A-Z.\-]{0,7}$/.test(sym)) { errBox.textContent = '"' + sym + '" isn’t a valid ticker.'; errBox.hidden = false; return }
    run.disabled = true; run.textContent = '…'
    el('pred-loading').hidden = false
    el('pred-result').hidden = true
    const data = await api('/api/predict?ticker=' + encodeURIComponent(sym))
    el('pred-loading').hidden = true
    run.disabled = false; run.textContent = 'Predict'
    render(data)
  }

  // ── Track record / calibration ─────────────────────────────────────────────
  let calChart = null
  function renderCalibration(data) {
    const box = el('cal-result')
    box.hidden = false
    if (!data || !data.available) {
      box.innerHTML = `<p class="text-muted" style="font-size:12.5px;">${(data && data.reason) || 'Track record unavailable right now.'} ${data && data.reason && data.reason.includes('not loaded') ? '' : 'Try again shortly.'}</p>`
      return
    }
    const m = data.metrics
    const bssTag = m.brier_skill > 0.02 ? ['positive', 'real skill']
      : m.brier_skill < -0.02 ? ['negative', 'worse than base rate']
      : ['', '≈ no skill (honest)']
    const edge = ((m.accuracy - data.base_rate) * 100)

    box.innerHTML = `
      <div class="hero-row" style="margin-bottom:16px;">
        <div class="hero-stat"><div class="hero-stat-value">${data.n_predictions.toLocaleString()}</div><div class="hero-stat-label">Out-of-sample predictions</div></div>
        <div class="hero-stat"><div class="hero-stat-value ${bssTag[0]}">${fmtN(m.brier_skill, 3)}</div><div class="hero-stat-label">Brier skill · ${bssTag[1]}</div></div>
        <div class="hero-stat"><div class="hero-stat-value">${Math.round(m.accuracy * 100)}%</div><div class="hero-stat-label">Hit rate vs ${Math.round(data.base_rate * 100)}% base (${edge >= 0 ? '+' : ''}${edge.toFixed(1)} pts)</div></div>
        <div class="hero-stat"><div class="hero-stat-value">${m.model_test_auc ?? '—'}</div><div class="hero-stat-label">Model test AUC (0.5 = coin flip)</div></div>
      </div>
      <div class="chart-wrap" style="height:360px;"><canvas id="cal-chart"></canvas></div>
      <p class="text-muted" style="font-size:11px;line-height:1.6;margin-top:12px;">
        ${data.n_tickers} tickers · ${data.eval_start} → ${data.eval_end} · equal-count (quantile) bins.
        ${data.note}
      </p>`

    const pts  = data.bins.map(b => ({ x: b.mean_pred, y: b.frac_up, n: b.count }))
    const lo   = Math.min(...data.bins.map(b => Math.min(b.mean_pred, b.frac_up)))
    const hi   = Math.max(...data.bins.map(b => Math.max(b.mean_pred, b.frac_up)))
    const pad  = Math.max(0.02, (hi - lo) * 0.15)
    const axLo = Math.max(0, lo - pad), axHi = Math.min(1, hi + pad)
    const maxN = Math.max(...pts.map(p => p.n))

    if (calChart) calChart.destroy()
    calChart = new Chart(el('cal-chart'), {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Perfect calibration', type: 'line',
            data: [{ x: axLo, y: axLo }, { x: axHi, y: axHi }],
            borderColor: 'rgba(134,134,139,0.6)', borderDash: [6, 5], borderWidth: 1.5,
            pointRadius: 0, fill: false },
          { label: 'Predicted vs actual',
            data: pts,
            backgroundColor: 'rgba(10,132,255,0.75)', borderColor: '#0a84ff', borderWidth: 1,
            pointRadius: c => 4 + 8 * (c.raw.n / maxN), pointHoverRadius: c => 6 + 8 * (c.raw.n / maxN) },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { min: axLo, max: axHi, title: { display: true, text: 'Predicted P(up)', color: '#86868b' },
               ticks: { color: '#86868b', callback: v => Math.round(v * 100) + '%' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { min: axLo, max: axHi, title: { display: true, text: 'Actual frequency of up-moves', color: '#86868b' },
               ticks: { color: '#86868b', callback: v => Math.round(v * 100) + '%' }, grid: { color: 'rgba(255,255,255,0.06)' } },
        },
        plugins: {
          legend: { labels: { color: '#86868b', usePointStyle: true, filter: i => i.text !== 'Perfect calibration' } },
          tooltip: { callbacks: { label: c => `pred ${Math.round(c.raw.x * 100)}% → actual ${Math.round(c.raw.y * 100)}% (n=${c.raw.n})` } },
        },
      },
    })
  }

  const calBtn = el('cal-load')
  let calLoaded = false
  calBtn.addEventListener('click', async () => {
    calBtn.disabled = true; calBtn.textContent = calLoaded ? 'Refreshing…' : 'Loading…'
    el('cal-loading').hidden = false
    el('cal-result').hidden = true
    // First open: the copy saved after the last close (instant). Refresh: live.
    const saved = calLoaded ? null : await savedApi('calibration.json')
    const data = saved || await api('/api/predict/calibration')
    el('cal-loading').hidden = true
    calBtn.disabled = false; calBtn.textContent = 'Refresh'
    calLoaded = true
    renderCalibration(data)
    if (saved && saved.available) savedNote(el('cal-result'), saved.snapshot_at)
  })

  // ── Live forward ledger ─────────────────────────────────────────────────────
  const LDIR = { BULLISH: 'positive', BEARISH: 'negative', NEUTRAL: '' }
  function renderLedger(data) {
    const box = el('ledger-result'); box.hidden = false
    if (!data || !data.available) {
      box.innerHTML = `<p class="text-muted" style="font-size:12.5px;">${(data && data.reason) || 'Ledger unavailable right now.'}</p>`
      return
    }
    const rows = data.rows || []
    const s = data.summary || {}
    if (!rows.length) {
      box.innerHTML = `<p class="text-muted" style="font-size:12.5px;line-height:1.65;">
        No calls logged yet. The first ones are recorded on the next trading day near the close, and graded once their 5-day horizon passes.</p>`
      return
    }
    const hitTag = s.hit_rate == null ? ['', 'awaiting first graded call']
      : s.hit_rate > (s.base_up_rate ?? 50) ? ['positive', `vs ${s.base_up_rate}% base up-rate`]
      : ['negative', `vs ${s.base_up_rate}% base up-rate`]
    const bssTag = s.brier_skill == null ? '' : s.brier_skill > 0.02 ? 'positive' : s.brier_skill < -0.02 ? 'negative' : ''

    const outcomeCell = r => {
      if (r.status !== 'graded') return `<span class="text-muted">pending · due ${r.due_date}</span>`
      if (r.correct == null) return `<span class="text-muted">${fmtPct(r.realized_pct, 1)} · n/a</span>`
      const mark = r.correct ? '<span class="positive">✓</span>' : '<span class="negative">✗</span>'
      return `<span class="${clr(r.realized_pct)}">${fmtPct(r.realized_pct, 1)}</span> ${mark}`
    }

    box.innerHTML = `
      <div class="hero-row" style="margin-bottom:16px;">
        <div class="hero-stat"><div class="hero-stat-value">${s.n_scored || 0}</div><div class="hero-stat-label">Graded directional calls (${data.n_pending || 0} pending${data.since ? `, since ${data.since}` : ''})</div></div>
        <div class="hero-stat"><div class="hero-stat-value ${hitTag[0]}">${s.hit_rate == null ? '—' : s.hit_rate + '%'}</div><div class="hero-stat-label">Forward hit rate · ${hitTag[1]}</div></div>
        <div class="hero-stat"><div class="hero-stat-value ${bssTag}">${s.brier_skill == null ? '—' : fmtN(s.brier_skill, 3)}</div><div class="hero-stat-label">Brier skill (>0 = beats base rate)</div></div>
        <div class="hero-stat"><div class="hero-stat-value ${clr(s.avg_realized_pct)}">${s.avg_realized_pct == null ? '—' : fmtPct(s.avg_realized_pct, 2)}</div><div class="hero-stat-label">Avg realized move / call</div></div>
      </div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr><th>Logged</th><th>Ticker</th><th>Call</th><th>P(up)</th><th>Entry</th><th>Outcome</th></tr></thead>
        <tbody>${rows.map(r => `<tr>
          <td class="text-muted" style="font-size:11px;">${r.created_date}</td>
          <td><strong>${r.ticker}</strong></td>
          <td><span class="${LDIR[r.direction] || ''}" style="font-weight:700;font-size:11px;">${r.direction}</span></td>
          <td>${Math.round(r.p_up * 100)}%</td>
          <td>${fmt$(r.price_at_pred)}</td>
          <td>${outcomeCell(r)}</td>
        </tr>`).join('')}</tbody>
      </table></div>
      <p class="text-muted" style="font-size:11px;line-height:1.6;margin-top:12px;">${data.note}</p>`
  }

  const ledgerBtn = el('ledger-load')
  let ledgerLoaded = false
  ledgerBtn.addEventListener('click', async () => {
    ledgerBtn.disabled = true; ledgerBtn.textContent = ledgerLoaded ? 'Refreshing…' : 'Loading…'
    el('ledger-loading').hidden = false
    el('ledger-result').hidden = true
    // The committed record is a static file served with the site, so reading
    // it never has to wake the backend.
    let data = null
    try {
      const res = await fetch('/data/forward_ledger.json', { cache: 'no-cache' })
      if (res.ok) data = await res.json()
    } catch (_) { data = null }
    if (!data) data = await api('/api/predict/ledger')
    el('ledger-loading').hidden = true
    ledgerBtn.disabled = false; ledgerBtn.textContent = 'Refresh'
    ledgerLoaded = true
    renderLedger(data)
  })

  run.addEventListener('click', predict)
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); predict() } })
  renderQuick()
  // Both records are saved files served with the site, so show them on arrival
  // rather than behind a button; the button then refreshes them.
  calBtn.click()
  ledgerBtn.click()
}

// ── Research Lab (PEAD event study + Data-Mining Lab) ────────────────────────
function initResearch() {
  let peadChart = null, dmChart = null

  // ── PEAD event study ───────────────────────────────────────────────────────
  const CURVE_COLOR = { beat: '#30d158', inline: '#86868b', miss: '#ff453a' }

  function renderPead(data) {
    const box = el('pead-result'); box.hidden = false
    if (!data || !data.available) {
      box.innerHTML = `<p class="text-muted" style="font-size:12.5px;">${(data && data.reason) || 'Study unavailable right now.'} — the free earnings feed can be sparse; try again shortly.</p>`
      return
    }
    const s = data.spread
    const sigBadge = s.significant
      ? '<span class="badge badge-buy">SIGNIFICANT</span>'
      : '<span class="badge sig-hold">NOT SIGNIFICANT</span>'
    box.innerHTML = `
      <div class="hero-row" style="margin-bottom:16px;">
        <div class="hero-stat"><div class="hero-stat-value">${data.n_events.toLocaleString()}</div><div class="hero-stat-label">Earnings events · ${data.n_tickers} tickers</div></div>
        <div class="hero-stat"><div class="hero-stat-value ${s.beat_minus_miss_pct >= 0 ? 'positive' : 'negative'}">${fmtPct(s.beat_minus_miss_pct)}</div><div class="hero-stat-label">Beat − miss drift @ ${data.window}d</div></div>
        <div class="hero-stat"><div class="hero-stat-value">${s.t_stat ?? '—'}</div><div class="hero-stat-label">t-stat (p ${s.p_value ?? '—'})</div></div>
        <div class="hero-stat"><div class="hero-stat-value">${s.surprise_drift_corr ?? '—'}</div><div class="hero-stat-label">Surprise → drift corr</div></div>
      </div>
      <div style="margin-bottom:12px;">${sigBadge}</div>
      <div class="chart-wrap" style="height:340px;"><canvas id="pead-chart"></canvas></div>
      <p class="text-muted" style="font-size:11px;line-height:1.6;margin-top:12px;">${data.note}</p>`

    const days = Array.from({ length: data.window }, (_, i) => i + 1)
    if (peadChart) peadChart.destroy()
    peadChart = new Chart(el('pead-chart'), {
      type: 'line',
      data: {
        labels: days,
        datasets: data.curves.map(c => ({
          label: `${c.label} (n=${c.n}, x̄=${c.avg_surprise > 0 ? '+' : ''}${c.avg_surprise}%)`,
          data: c.car, borderColor: CURVE_COLOR[c.key], backgroundColor: CURVE_COLOR[c.key],
          borderWidth: 2, pointRadius: 0, tension: 0.15,
        })),
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Trading days after report', color: '#86868b' },
               ticks: { color: '#86868b' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { title: { display: true, text: 'Cumulative abnormal return (%)', color: '#86868b' },
               ticks: { color: '#86868b', callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.06)' } },
        },
        plugins: { legend: { labels: { color: '#c9d3cd', usePointStyle: true, boxWidth: 8 } } },
      },
    })
  }

  el('pead-run').addEventListener('click', async () => {
    const btn = el('pead-run'); btn.disabled = true; btn.textContent = 'Running…'
    el('pead-loading').hidden = false; el('pead-result').hidden = true
    // First run: the copy saved after the last close (instant). Refresh: live.
    const saved = btn.dataset.ran ? null : await savedApi('pead.json')
    const data = saved || await api('/api/research/pead')
    el('pead-loading').hidden = true
    btn.disabled = false; btn.textContent = 'Refresh'; btn.dataset.ran = '1'
    renderPead(data)
    if (saved && saved.available) savedNote(el('pead-result'), saved.snapshot_at)
  })

  // ── Data-Mining Lab ────────────────────────────────────────────────────────
  const dmN = el('dm-n')
  dmN.addEventListener('input', () => { el('dm-n-label').textContent = dmN.value })

  function verdictTile(label, sig, value) {
    const cls = sig ? 'negative' : 'positive'          // "significant" here is the BAD outcome
    const tag = sig ? 'calls it significant ✗' : 'not significant ✓'
    return `<div class="hero-stat"><div class="hero-stat-value ${cls}">${value ?? '—'}</div>
      <div class="hero-stat-label">${label} — ${tag}</div></div>`
  }

  function renderDm(data) {
    const box = el('dm-result'); box.hidden = false
    if (!data || !data.available) {
      box.innerHTML = `<p class="text-muted" style="font-size:12.5px;">${(data && data.reason) || 'Sweep failed'} — try another ticker.</p>`
      return
    }
    box.innerHTML = `
      <div class="hero-row" style="margin-bottom:16px;">
        <div class="hero-stat"><div class="hero-stat-value">${data.best_sharpe}</div><div class="hero-stat-label">Best-of-${data.n_strategies} Sharpe (pure luck)</div></div>
        ${verdictTile('Naive PSR', data.naive_significant, data.naive_psr)}
        ${verdictTile('Deflated Sharpe', data.deflated_significant, data.deflated_sharpe)}
        <div class="hero-stat"><div class="hero-stat-value">${data.expected_max_sharpe ?? '—'}</div><div class="hero-stat-label">Expected max from noise</div></div>
      </div>
      <div class="chart-wrap" style="height:300px;"><canvas id="dm-chart"></canvas></div>
      <p style="font-size:13px;line-height:1.65;margin-top:14px;color:var(--text);"><strong>${data.verdict}</strong></p>
      <p class="text-muted" style="font-size:11px;line-height:1.6;margin-top:8px;">${data.note}</p>`

    const labels = data.histogram.map(h => h.x)
    const counts = data.histogram.map(h => h.count)
    const vline = (x, color, text) => ({
      type: 'line', xMin: x, xMax: x, borderColor: color, borderWidth: 2, borderDash: [5, 4],
      label: { display: true, content: text, color: color, position: 'start',
               font: { size: 10 }, backgroundColor: 'rgba(44,44,46,0.94)' },
    })
    if (dmChart) dmChart.destroy()
    dmChart = new Chart(el('dm-chart'), {
      type: 'bar',
      data: { labels, datasets: [{ label: 'Random strategies', data: counts,
              backgroundColor: 'rgba(134,134,139,0.55)', borderWidth: 0, barPercentage: 1, categoryPercentage: 1 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { type: 'linear', offset: false, title: { display: true, text: 'Backtested Sharpe ratio', color: '#86868b' },
               ticks: { color: '#86868b', maxTicksLimit: 10 }, grid: { display: false } },
          y: { title: { display: true, text: 'Count', color: '#86868b' }, ticks: { color: '#86868b' }, grid: { color: 'rgba(255,255,255,0.06)' } },
        },
        plugins: {
          legend: { display: false },
          annotation: { annotations: {
            best: vline(data.best_sharpe, '#ffd60a', `best ${data.best_sharpe}`),
            emax: data.expected_max_sharpe != null ? vline(data.expected_max_sharpe, '#86868b', `E[max] ${data.expected_max_sharpe}`) : {},
          } },
        },
      },
    })
  }

  el('dm-run').addEventListener('click', async () => {
    const ticker = (el('dm-ticker').value || 'AAPL').trim().toUpperCase()
    const err = el('dm-error'); err.hidden = true
    if (!/^[A-Z][A-Z.\-]{0,7}$/.test(ticker)) { err.textContent = 'Enter a valid ticker.'; err.hidden = false; return }
    const btn = el('dm-run'); btn.disabled = true; btn.textContent = 'Running…'
    el('dm-loading').hidden = false; el('dm-result').hidden = true
    const data = await api('/api/research/datamine', {
      method: 'POST', body: JSON.stringify({ ticker, n_strategies: Number(dmN.value) }),
    })
    el('dm-loading').hidden = true
    btn.disabled = false; btn.textContent = 'Run sweep'
    renderDm(data)
  })

  // ── CSCV + PBO ───────────────────────────────────────────────────────────────
  let cpcvHist = null, cpcvScatter = null
  function pboTone(pbo) { return pbo >= 0.5 ? 'negative' : pbo >= 0.25 ? '' : 'positive' }

  function renderCpcv(data) {
    const box = el('cpcv-result'); box.hidden = false
    if (!data || !data.available) {
      box.innerHTML = `<p class="text-muted" style="font-size:12.5px;">${(data && data.reason) || 'CSCV unavailable right now.'}</p>`
      return
    }
    const d = data.degradation || {}
    const retention = (d.median_is_sr && d.median_is_sr > 0)
      ? Math.max(0, Math.min(100, (d.median_oos_sr / d.median_is_sr) * 100)) : 0
    // Small IS/OOS grid diagram of the first few splits.
    const grid = (data.splits_diagram || []).map(row =>
      `<div class="cpcv-row">${row.map(c =>
        `<span class="cpcv-cell ${c === 'IS' ? 'cpcv-is' : 'cpcv-oos'}">${c}</span>`).join('')}</div>`).join('')

    box.innerHTML = `
      <div class="hero-row" style="margin-bottom:16px;">
        <div class="hero-stat"><div class="hero-stat-value ${pboTone(data.pbo)}">${Math.round(data.pbo * 100)}%</div><div class="hero-stat-label">Probability of Backtest Overfitting</div></div>
        <div class="hero-stat"><div class="hero-stat-value">${retention.toFixed(0)}%</div><div class="hero-stat-label">Median IS Sharpe surviving OOS</div></div>
        <div class="hero-stat"><div class="hero-stat-value ${data.prob_oos_loss > 0.4 ? 'negative' : ''}">${Math.round(data.prob_oos_loss * 100)}%</div><div class="hero-stat-label">Chosen strategy loses money OOS</div></div>
        <div class="hero-stat"><div class="hero-stat-value">${data.n_strategies}×${data.n_combos}</div><div class="hero-stat-label">Variants × IS/OOS splits</div></div>
      </div>
      <div class="two-col" style="margin-bottom:16px;">
        <div><div class="section-header" style="font-size:13px;">Overfit statistic λ (left of 0 = overfit)</div>
          <div class="chart-wrap" style="height:260px;"><canvas id="cpcv-hist"></canvas></div></div>
        <div><div class="section-header" style="font-size:13px;">In-sample vs out-of-sample Sharpe</div>
          <div class="chart-wrap" style="height:260px;"><canvas id="cpcv-scatter"></canvas></div></div>
      </div>
      <div class="section-header" style="font-size:13px;">CSCV split structure (${data.n_groups} groups, embargo ${data.embargo}d) — first ${(data.splits_diagram || []).length} of ${data.n_combos}</div>
      <div class="cpcv-grid" style="margin-bottom:14px;">${grid}</div>
      <p style="font-size:13px;line-height:1.65;color:var(--text);"><strong>${data.verdict}</strong></p>
      <p class="text-muted" style="font-size:11px;line-height:1.6;margin-top:8px;">${data.ticker} · ${data.n_rows} trading days after warm-up · CSCV (Bailey & López de Prado 2014). A trend/mean-reversion grid, not the site's headline strategies.</p>`

    // λ histogram
    const labels = data.logit_hist.map(h => h.x)
    const counts = data.logit_hist.map(h => h.count)
    const bars = data.logit_hist.map(h => h.x < 0 ? 'rgba(255,69,58,0.55)' : 'rgba(48,209,88,0.55)')
    if (cpcvHist) cpcvHist.destroy()
    cpcvHist = new Chart(el('cpcv-hist'), {
      type: 'bar',
      data: { labels, datasets: [{ data: counts, backgroundColor: bars, borderWidth: 0, barPercentage: 1, categoryPercentage: 1 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { type: 'linear', offset: false, title: { display: true, text: 'λ = logit(OOS rank)', color: '#86868b' },
               ticks: { color: '#86868b', maxTicksLimit: 8 }, grid: { display: false } },
          y: { title: { display: true, text: 'Splits', color: '#86868b' }, ticks: { color: '#86868b' }, grid: { color: 'rgba(255,255,255,0.06)' } },
        },
        plugins: {
          legend: { display: false },
          annotation: { annotations: { zero: {
            type: 'line', xMin: 0, xMax: 0, borderColor: '#ffd60a', borderWidth: 2, borderDash: [5, 4],
            label: { display: true, content: 'overfit ◄ │ ► holds up', color: '#ffd60a', position: 'end', font: { size: 10 }, backgroundColor: 'rgba(44,44,46,0.94)' } } } },
        },
      },
    })

    // IS vs OOS scatter with y=x and the regression line
    const pts = data.scatter.map(p => ({ x: p.is, y: p.oos }))
    const xs = pts.map(p => p.x)
    const xLo = Math.min(...xs), xHi = Math.max(...xs)
    const reg = [{ x: xLo, y: d.intercept + d.slope * xLo }, { x: xHi, y: d.intercept + d.slope * xHi }]
    if (cpcvScatter) cpcvScatter.destroy()
    cpcvScatter = new Chart(el('cpcv-scatter'), {
      type: 'scatter',
      data: { datasets: [
        { label: 'y = x (no decay)', type: 'line', data: [{ x: xLo, y: xLo }, { x: xHi, y: xHi }],
          borderColor: 'rgba(134,134,139,0.6)', borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0, fill: false },
        { label: 'IS→OOS fit', type: 'line', data: reg, borderColor: '#0a84ff', borderWidth: 2, pointRadius: 0, fill: false },
        { label: 'Selected champion', data: pts, backgroundColor: 'rgba(48,209,88,0.55)', borderColor: '#30d158', borderWidth: 1, pointRadius: 3 },
      ] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'In-sample Sharpe', color: '#86868b' }, ticks: { color: '#86868b' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { title: { display: true, text: 'Out-of-sample Sharpe', color: '#86868b' }, ticks: { color: '#86868b' }, grid: { color: 'rgba(255,255,255,0.06)' } },
        },
        plugins: { legend: { labels: { color: '#86868b', usePointStyle: true, boxWidth: 8, filter: i => i.text !== 'y = x (no decay)' } } },
      },
    })
  }

  el('cpcv-run').addEventListener('click', async () => {
    const ticker = (el('cpcv-ticker').value || 'SPY').trim().toUpperCase()
    const err = el('cpcv-error'); err.hidden = true
    if (!/^[A-Z][A-Z.\-]{0,7}$/.test(ticker)) { err.textContent = 'Enter a valid ticker.'; err.hidden = false; return }
    const btn = el('cpcv-run'); btn.disabled = true; btn.textContent = 'Running…'
    el('cpcv-loading').hidden = false; el('cpcv-result').hidden = true
    const data = await api('/api/research/cpcv', { method: 'POST', body: JSON.stringify({ ticker }) })
    el('cpcv-loading').hidden = true
    btn.disabled = false; btn.textContent = 'Run CSCV'
    renderCpcv(data)
  })

  // ── Cost-sensitivity sweep ─────────────────────────────────────────────────
  let costChart = null
  function renderCost(data) {
    const box = el('cost-result'); box.hidden = false
    if (!data || !data.available) {
      box.innerHTML = `<p class="text-muted" style="font-size:12.5px;">${(data && data.reason) || 'Sweep unavailable right now.'}</p>`
      return
    }
    const beBench = data.breakeven_bench_bps
    const beTone = beBench == null ? '' : beBench <= 5 ? 'negative' : beBench <= 20 ? '' : 'positive'
    box.innerHTML = `
      <div class="hero-row" style="margin-bottom:16px;">
        <div class="hero-stat"><div class="hero-stat-value ${beTone}">${beBench == null ? '>100' : beBench}${' '}bp</div><div class="hero-stat-label">Break-even vs buy-and-hold (per side)</div></div>
        <div class="hero-stat"><div class="hero-stat-value">${data.breakeven_zero_bps == null ? '>100' : data.breakeven_zero_bps} bp</div><div class="hero-stat-label">Break-even vs zero return</div></div>
        <div class="hero-stat"><div class="hero-stat-value">${fmtPct(data.benchmark_return)}</div><div class="hero-stat-label">Buy-and-hold (SPY) return</div></div>
      </div>
      <div class="chart-wrap" style="height:320px;"><canvas id="cost-chart"></canvas></div>
      <p style="font-size:13px;line-height:1.65;margin-top:14px;color:var(--text);"><strong>${data.verdict}</strong></p>
      <p class="text-muted" style="font-size:11px;line-height:1.6;margin-top:8px;">Real-world large-cap costs run ~3–10 bp/side. Each point reruns the full backtest at that friction level.</p>`

    const pts = data.points
    const labels = pts.map(p => p.cost_bps)
    const anns = {}
    if (beBench != null) anns.beB = { type: 'line', xMin: beBench, xMax: beBench, borderColor: '#ffd60a', borderWidth: 2, borderDash: [5, 4],
      label: { display: true, content: `edge dies @ ${beBench}bp`, color: '#ffd60a', position: 'start', font: { size: 10 }, backgroundColor: 'rgba(44,44,46,0.94)' } }
    if (costChart) costChart.destroy()
    costChart = new Chart(el('cost-chart'), {
      type: 'line',
      data: { labels, datasets: [
        { label: 'Strategy net return', data: pts.map(p => p.net_return), borderColor: '#0a84ff', backgroundColor: 'rgba(10,132,255,0.10)', borderWidth: 2, fill: true, tension: 0.2, pointRadius: 3 },
        { label: 'Buy & hold (SPY)', data: pts.map(() => data.benchmark_return), borderColor: '#86868b', borderWidth: 1.5, borderDash: [5, 4], fill: false, pointRadius: 0 },
      ] },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { title: { display: true, text: 'Transaction cost per side (basis points)', color: '#86868b' }, ticks: { color: '#86868b' }, grid: { display: false } },
          y: { title: { display: true, text: 'Total return (%)', color: '#86868b' }, ticks: { color: '#86868b', callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.06)' } },
        },
        plugins: {
          legend: { labels: { color: '#86868b', usePointStyle: true, boxWidth: 10 } },
          tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${fmtPct(c.parsed.y)}` } },
          annotation: { annotations: anns },
        },
      },
    })
  }

  el('cost-run').addEventListener('click', async () => {
    const ticker = (el('cost-ticker').value || 'AAPL').trim().toUpperCase()
    const err = el('cost-error'); err.hidden = true
    if (!/^[A-Z][A-Z.\-]{0,7}$/.test(ticker)) { err.textContent = 'Enter a valid ticker.'; err.hidden = false; return }
    const btn = el('cost-run'); btn.disabled = true; btn.textContent = 'Running…'
    el('cost-loading').hidden = false; el('cost-result').hidden = true
    const data = await api('/api/research/cost-sensitivity', {
      method: 'POST', body: JSON.stringify({ strategy: el('cost-strategy').value, tickers: [ticker] }),
    })
    el('cost-loading').hidden = true
    btn.disabled = false; btn.textContent = 'Run sweep'
    renderCost(data)
  })
}

// ── Router ──────────────────────────────────────────────────────────────────
// One page per area, so a visitor lands on the thing they came for instead of
// hunting for it inside a tab strip. Only the current page's init runs; each
// one wires up elements that exist nowhere else, so running the wrong one
// would throw on the first missing node.
const PAGE = document.body.dataset.page
if (PAGE === 'backtest')  initBacktest()
if (PAGE === 'predict')   initPredictor()
if (PAGE === 'screener')  initSignals()
if (PAGE === 'portfolio') initPortfolio()
if (PAGE === 'research')  { initTabs(document.body); initResearch() }
if (['backtest', 'predict', 'screener', 'portfolio', 'research', 'options'].includes(PAGE)) wakeOnIntent()
