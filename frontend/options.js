/* ══════════════════════════════════════════════════════════════════════════
   AlphaGlyph — Options strategy backtester (options.html)
   Uses app.js for the API client, formatters, stat cards, tabs and Chart.js
   defaults. Everything here is specific to this page.
   ══════════════════════════════════════════════════════════════════════════ */

Object.assign(GLOSSARY, {
  'CAGR':             'Compound annual growth rate: the yearly return that would turn the start value into the end value.',
  'Time in Market':   'Share of trading days with a position open.',
  'Positions':        'Number of option positions opened and closed over the test.',
  'Profitable':       'Share of positions that closed for a profit, after costs.',
  'Volatility':       'Annualized standard deviation of daily returns.',
  'Capital at Risk':  'Average share of the account tied up in positions: what each could lose, or the cash it sets aside.',
  'Worst Month':      'The worst calendar month for the account.',
  'Longest Drawdown': 'The longest stretch, in calendar days, spent below a previous high.',
})

const OB_CUSTOM_ALLOC = 10        // capital per position for custom legs, % (the backend's default)
const OB_GROUPS = ['Single leg', 'Two legs', 'Three legs', 'Four legs', 'Cycle']
const OB_STRIKE_MODES = [['atm', 'At the money'], ['pct', '% from price'], ['delta', 'Delta'], ['offset', '% from leg']]
const obState = { presets: [], byKey: {}, legs: [], result: null, charts: {} }

// "Protective collar: a three-leg backtest" etc. from a preset or the custom legs (mirrors describe_strategy).
const OB_WORDS = { 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six' }

function obStrikeWords(leg, legs) {
  const s = leg.strike || { mode: 'atm' }
  const v = Number(s.value)
  if (s.mode === 'atm' || (s.mode === 'pct' && !v)) return 'at the money'
  if (s.mode === 'pct') return `${Math.abs(v)}% ${v > 0 ? 'above' : 'below'} the price`
  if (s.mode === 'delta') return `at ${v.toFixed(2)} delta`
  if (s.mode === 'offset') {
    const ref = legs[s.leg]
    return ref ? `${Math.abs(v)}% of the price ${v > 0 ? 'above' : 'below'} the ${ref.side} ${ref.type}'s strike` : ''
  }
  return ''
}

function obDescribe(name, legs, dte) {
  const kinds = []
  legs.forEach(l => {
    const k = `${l.side} ${l.type === 'stock' ? 'stock' : l.type}`
    if (!kinds.includes(k)) kinds.push(k)
  })
  const n = legs.length
  return {
    headline: `${name}: a ${OB_WORDS[n] || n}-leg backtest (${kinds.join(', ')})`,
    legs: legs.map(l => {
      const q = Number(l.qty || 1)
      if (l.type === 'stock') return `${l.side} ${q * 100} shares`
      return `${l.side} ${q} ${l.type}${q > 1 ? 's' : ''} ${obStrikeWords(l, legs)}, ${l.dte || dte} days to expiry`
    }),
  }
}

// Fetch that keeps the server's error message on a 400/422 and falls back to
// app.js's retrying api() for cold starts and rate limits.
async function obPost(path, body) {
  const opts = { method: 'POST', body: JSON.stringify(body) }
  try {
    const r = await fetch(API_BASE + path, { ...opts, headers: { 'Content-Type': 'application/json' } })
    if (r.ok || r.status === 400 || r.status === 422) { setConn(true); return r.json() }
  } catch (_) { /* cold start: retry below */ }
  return api(path, opts)
}

function obSettings() {
  const num = id => { const v = el(id).value.trim(); return v === '' ? null : Number(v) }
  return {
    ticker: el('ob-ticker').value.trim().toUpperCase() || 'SPY',
    start_date: el('ob-start').value, end_date: el('ob-end').value,
    initial_capital: num('ob-capital'), dte: num('ob-dte'), alloc_pct: num('ob-alloc'),
    exit_dte: num('ob-exit-dte'), profit_take: num('ob-pt'), stop_loss: num('ob-sl'),
    commission: num('ob-commission'), spread_pct: num('ob-spread'), skew: el('ob-skew').checked,
    entry: obEntry(num),
    expiries: el('ob-expiries').value,
    exit_earnings: el('ob-exit-earn').checked,
    exit_delta: num('ob-exit-delta'), stagger_days: num('ob-stagger'),
  }
}

function obEntry(num) {
  const e = { vix_min: num('ob-vix-min'), vix_max: num('ob-vix-max'), ivr_min: num('ob-ivr-min'), ivr_max: num('ob-ivr-max'),
    trend: el('ob-trend').value, earnings: el('ob-earn').value }
  Object.keys(e).forEach(k => { if (e[k] == null || e[k] === 'any') delete e[k] })
  return Object.keys(e).length ? e : null
}

// ── strategy picker and the "what this holds" box ──────────────────────────

function obRenderWhat() {
  obRenderWhatBox()
  obRenderLearn()
}

function obRenderWhatBox() {
  const key = el('ob-strategy').value
  const custom = key === 'custom'
  el('ob-custom').hidden = !custom
  const dte = Number(el('ob-dte').value) || 30
  let name, legs, view, wheel = null
  if (custom) {
    name = el('ob-name').value.trim() || 'Custom strategy'
    legs = obState.legs
    view = ''
  } else {
    const p = obState.byKey[key]
    if (!p) return
    name = p.name; legs = p.legs; view = p.view; wheel = p.wheel || null
  }
  el('ob-stagger').disabled = !!wheel
  if (wheel) el('ob-stagger').value = ''
  if (wheel) {
    const put = obDescribe(name, legs, dte).legs[0].replace(/^short 1 put /, '')
    const call = obDescribe(name, [wheel.call], dte).legs[0].replace(/^short 1 call /, '')
    el('ob-what').innerHTML = `
    <div class="opt-what-head"><span class="badge badge-green">Cycle</span><span class="opt-what-view">${view}</span></div>
    <p class="opt-what-title">The wheel: puts until assigned, then covered calls until called away</p>
    <ul class="opt-what-legs">
      <li><span class="opt-side opt-side-short">short</span>1 put ${put}</li>
      <li><span class="opt-side opt-side-long">then</span>100 shares if the put is assigned</li>
      <li><span class="opt-side opt-side-short">short</span>1 call ${call} on them, until called away</li></ul>`
    return
  }
  if (!legs.length) { el('ob-what').innerHTML = '<p class="check-desc">Add a leg to start.</p>'; return }
  const d = obDescribe(name, legs, dte)
  el('ob-what').innerHTML = `
    <div class="opt-what-head"><span class="badge badge-green">${legs.length} leg${legs.length > 1 ? 's' : ''}</span>
      ${view ? `<span class="opt-what-view">${view}</span>` : ''}</div>
    <p class="opt-what-title">${d.headline.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</p>
    <ul class="opt-what-legs">${legs.map((l, i) =>
      `<li><span class="opt-side opt-side-${l.side}">${l.side}</span>${d.legs[i].replace(/^(long|short) /, '')}</li>`).join('')}</ul>`
}

function obPickStrategy() {
  const key = el('ob-strategy').value
  if (key !== 'custom' && obState.byKey[key]) el('ob-alloc').value = obState.byKey[key].alloc
  // Custom legs size like the backend's default (10% per position). Without this they kept the
  // previous preset's share: the page opens on the collar at 100%, so a strategy imported from
  // Tradeski put the whole account into each position. A share link's own size is applied after this.
  if (key === 'custom') el('ob-alloc').value = OB_CUSTOM_ALLOC
  if (key === 'custom' && !obState.legs.length) {
    obState.legs = [{ type: 'put', side: 'short', qty: 1, strike: { mode: 'delta', value: 0.3 } },
                    { type: 'put', side: 'long', qty: 1, strike: { mode: 'offset', leg: 0, value: -5 } }]
    obRenderLegs()
  }
  obRenderWhat()
}

// ── custom legs editor ─────────────────────────────────────────────────────

function obRenderLegs() {
  const wrap = el('ob-legs')
  wrap.innerHTML = obState.legs.map((l, i) => {
    const opt = (v, cur, label) => `<option value="${v}"${v === cur ? ' selected' : ''}>${label}</option>`
    const s = l.strike || { mode: 'atm' }
    const refOpts = obState.legs.slice(0, i).map((x, j) => x.type === 'stock' ? '' :
      opt(String(j), String(s.leg), `leg ${j + 1}`)).join('')
    return `<div class="opt-leg" data-i="${i}">
      <div class="opt-leg-row">
        <span class="opt-leg-n">${i + 1}</span>
        <select data-f="side" aria-label="Leg ${i + 1} side">${opt('long', l.side, 'Long')}${opt('short', l.side, 'Short')}</select>
        <input type="number" data-f="qty" value="${l.qty}" min="1" max="10" aria-label="Leg ${i + 1} quantity">
        <select data-f="type" aria-label="Leg ${i + 1} type">${opt('call', l.type, 'Call')}${opt('put', l.type, 'Put')}${opt('stock', l.type, '100 shares')}</select>
        <button type="button" class="chip-x" data-remove="${i}" aria-label="Remove leg ${i + 1}">×</button>
      </div>
      ${l.type === 'stock' ? '' : `<div class="opt-leg-row">
        <select data-f="mode" aria-label="Leg ${i + 1} strike">${OB_STRIKE_MODES.map(([v, t]) =>
          (v === 'offset' && !refOpts) ? '' : opt(v, s.mode, t)).join('')}</select>
        ${s.mode === 'atm' ? '' : `<input type="number" data-f="value" step="${s.mode === 'delta' ? '0.05' : '0.5'}"
            value="${s.value ?? ''}" aria-label="Leg ${i + 1} strike value">`}
        ${s.mode === 'offset' ? `<select data-f="leg" aria-label="Leg ${i + 1} offset from">${refOpts}</select>` : ''}
        <input type="number" data-f="dte" min="1" max="365" placeholder="DTE" value="${l.dte ?? ''}" aria-label="Leg ${i + 1} days to expiry (optional)">
      </div>`}
    </div>`
  }).join('')
  el('ob-add-leg').disabled = obState.legs.length >= 6
}

function obLegChange(e) {
  const row = e.target.closest('.opt-leg')
  if (!row) return
  const i = Number(row.dataset.i)
  const l = obState.legs[i]
  const f = e.target.dataset.f
  const v = e.target.value
  if (f === 'side') l.side = v
  if (f === 'qty') l.qty = Math.max(1, Math.min(10, parseInt(v, 10) || 1))
  if (f === 'type') {
    l.type = v
    if (v === 'stock') { delete l.strike; delete l.dte; l.side = 'long' } else if (!l.strike) l.strike = { mode: 'atm' }
  }
  if (f === 'mode') {
    l.strike = { mode: v }
    if (v === 'pct') l.strike.value = 5
    if (v === 'delta') l.strike.value = 0.3
    if (v === 'offset') {
      const ref = obState.legs.findIndex((x, j) => j < i && x.type !== 'stock')
      l.strike = { mode: 'offset', leg: ref, value: 5 }
    }
  }
  if (f === 'value') l.strike.value = v === '' ? null : Number(v)
  if (f === 'leg') l.strike.leg = Number(v)
  if (f === 'dte') { if (v === '') delete l.dte; else l.dte = parseInt(v, 10) }
  if (['type', 'mode'].includes(f)) obRenderLegs()
  obRenderWhat()
}

// ── running ────────────────────────────────────────────────────────────────

function obShow(which) {
  el('ob-empty').hidden = which !== 'empty'
  el('ob-loading').hidden = which !== 'loading'
  el('ob-results').hidden = which !== 'results'
  el('ob-compare-out').hidden = which !== 'compare'
  el('ob-error').hidden = true
}

function obError(msg) {
  obShow('empty')
  el('ob-error').textContent = msg
  el('ob-error').hidden = false
}

async function obRun(presetKey) {
  if (presetKey) { el('ob-strategy').value = presetKey; obPickStrategy() }
  const key = el('ob-strategy').value
  const body = obSettings()
  if (key === 'custom') { body.legs = obState.legs; body.name = el('ob-name').value.trim() || 'Custom strategy' }
  else body.preset = key
  el('ob-loading-title').firstChild.textContent = 'Running options backtest '
  obShow('loading')
  el('ob-run').disabled = el('ob-compare').disabled = true
  obState.lastBody = body
  const data = await obPost('/api/options/backtest', body)
  el('ob-run').disabled = el('ob-compare').disabled = false
  if (!data || data.error) { obError(failMsg(data, 'The backtest could not run. Try again in a moment.')); return }
  obState.result = data
  obRenderResult(data)
}

// How much of the result is the volatility model? (the same backtest with IV nudged, and without skew)
function obSensLine(r) {
  const s = r.sensitivity
  if (!s || !s.variants.length) return ''
  const v = k => s.variants.find(x => x.label === k)
  const lo = v('IV 10% lower'), hi = v('IV 10% higher')
  const wide = s.cagr_range >= 5
  return `<p class="summary-body opt-sens ${wide ? 'opt-sens-wide' : ''}"><strong>How much is the model?</strong>
    With implied volatility 10% lower or higher throughout, the same backtest returns
    ${lo ? fmtPct(lo.cagr, 1) : '—'} or ${hi ? fmtPct(hi.cagr, 1) : '—'} a year${wide
    ? `: a spread of ${s.cagr_range.toFixed(1)} points, so read the headline as a range, not a number.` : '.'}
    Six of Cboe's real strategy indices pin the volatility level to within a few percent.</p>`
}

function obSensTable(r) {
  const s = r.sensitivity
  if (!s || !s.variants.length) return ''
  const m = r.metrics
  const row = (label, x) => `<tr><td>${label}</td><td class="${clr(x.cagr)}">${fmtPct(x.cagr, 1)}</td><td>${fmtN(x.sharpe)}</td>
    <td class="negative">${fmtPct(x.max_drawdown, 1)}</td></tr>`
  return `<p class="card-desc"><strong>Sensitivity to the model.</strong> The same backtest with the volatility model changed:</p>
    <div class="table-wrap"><table class="data-table opt-check-table"><thead><tr><th>Model</th><th>CAGR</th><th>Sharpe</th><th>Max drawdown</th></tr></thead>
    <tbody>${row('As run', m)}${s.variants.map(x => row(x.label, x)).join('')}</tbody></table></div>`
}

const obEsc = v => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

const obMoney0 = n => n == null ? '—' : (n < 0 ? '−$' : '$') + Math.abs(Math.round(n)).toLocaleString('en-US')

function obRenderResult(r) {
  obShow('results')
  const m = r.metrics, b = r.benchmark, s = r.strategy
  el('ob-model-line').textContent = `Option prices are reconstructed with Black-Scholes from each day's close; implied volatility is ${r.model.iv}. Not real quotes.`
  const first = r.equity[0].date, last = r.equity[r.equity.length - 1].date
  const beat = m.total_return - b.total_return
  el('ob-summary').innerHTML = `<p class="summary-body"><strong>${obEsc(s.name)} on ${obEsc(r.ticker)}</strong>, ${first} to ${last}:
    <strong class="${clr(m.total_return)}">${fmtPct(m.total_return, 1)}</strong> (${fmtPct(m.cagr, 1)} a year) against
    ${fmtPct(b.total_return, 1)} for buying and holding ${r.ticker}, ${beat >= 0 ? 'ahead by' : 'behind by'}
    ${Math.abs(beat).toFixed(1)} points. ${m.trades} positions, ${m.win_rate ?? '—'}% of them profitable after costs;
    worst drawdown ${fmtPct(m.max_drawdown, 1)} versus ${fmtPct(b.max_drawdown, 1)}.</p>${obSensLine(r)}`

  el('ob-legs-badge').textContent = `${s.legs_count} leg${s.legs_count > 1 ? 's' : ''}`
  el('ob-headline').textContent = s.headline
  el('ob-leg-list').innerHTML = s.legs.map(t => {
    const side = t.split(' ')[0]
    return `<li><span class="opt-side opt-side-${side}">${side}</span>${t.slice(side.length + 1)}</li>`
  }).join('')
  el('ob-rules').textContent = s.rules

  const ex = r.example
  if (ex) {
    el('ob-example').innerHTML = `<p class="card-desc">Opened ${ex.date} with ${r.ticker} at ${fmt$(ex.price)}:
      ${ex.units} unit${ex.units > 1 ? 's' : ''}, ${obMoney0(ex.capital_per_unit)} of capital each.</p>
      <div class="table-wrap"><table class="data-table opt-ex-table"><thead><tr><th>Leg</th><th>Strike</th><th>Expiry</th><th>Price</th></tr></thead><tbody>
      ${ex.legs.map(l => `<tr><td><span class="opt-side opt-side-${l.side}">${l.side}</span>${l.type === 'stock' ? `${l.qty * 100} shares` : `${l.qty} ${l.type}`}</td>
        <td>${l.strike == null ? '—' : fmt$(l.strike)}</td><td>${l.expiry || '—'}</td><td>${fmt$(l.price)}</td></tr>`).join('')}
      </tbody></table></div>`
    obDrawPayoff(ex)
  }

  el('ob-stats').innerHTML = [
    statCard('Total Return', fmtPct(m.total_return, 1), clr(m.total_return)),
    statCard('CAGR', fmtPct(m.cagr, 1), clr(m.cagr)),
    statCard('Sharpe Ratio', fmtN(m.sharpe), clr(m.sharpe)),
    statCard('Max Drawdown', fmtPct(m.max_drawdown, 1), 'negative'),
    statCard('Probabilistic Sharpe', m.psr == null ? '—' : m.psr.toFixed(1) + '%'),
    statCard('Positions', String(m.trades)),
    statCard('Profitable', m.win_rate == null ? '—' : m.win_rate + '%'),
    statCard('Total Costs', obMoney0(m.total_costs)),
    statCard('Time in Market', m.time_in_market + '%'),
    statCard('Capital at Risk', m.capital_used == null ? '—' : m.capital_used + '%'),
    statCard('Worst Month', m.worst_month == null ? '—' : fmtPct(m.worst_month, 1), clr(m.worst_month)),
    statCard('Longest Drawdown', m.longest_drawdown_days == null ? '—' : m.longest_drawdown_days + ' days'),
  ].join('')
  el('ob-bench-stats').innerHTML = [
    statCard('Benchmark Return', fmtPct(b.total_return, 1), clr(b.total_return)),
    statCard('CAGR', fmtPct(b.cagr, 1), clr(b.cagr)),
    statCard('Sharpe Ratio', fmtN(b.sharpe), clr(b.sharpe)),
    statCard('Max Drawdown', fmtPct(b.max_drawdown, 1), 'negative'),
  ].join('')
  obDrawEquity(r)

  el('ob-trades-tbody').innerHTML = r.trades.length ? r.trades.map(t => `<tr>
    <td>${t.opened}</td><td>${t.closed}</td>
    <td class="opt-pos-cell">${t.legs.map(l => `<span class="opt-side opt-side-${l.side}">${l.side}</span>${l.type === 'stock'
      ? `${l.qty * 100} sh` : `${l.qty} ${fmt$(l.strike).replace('.00', '')} ${l.type}`}`).join('<br>')}</td>
    <td class="${clr(t.pnl)}"><strong>${obMoney0(t.pnl)}</strong></td>
    <td>${t.reason}${t.earnings ? `<div class="fine">held through ${t.earnings === 1 ? 'an earnings report' : t.earnings + ' earnings reports'}</div>` : ''}${t.split ? `<div class="fine">${t.split}</div>` : ''}</td>
    <td>${obDriver(t.attribution, t.pnl)}</td>
    <td>${fmt$(t.underlying_open)} → ${fmt$(t.underlying_close)}</td>
    <td>${t.net_open >= 0 ? 'paid ' + obMoney0(t.net_open) : 'collected ' + obMoney0(-t.net_open)}</td>
    <td>${obMoney0(t.net_close)}</td><td>${obMoney0(t.costs)}</td></tr>`).join('')
    : '<tr><td colspan="10" class="empty-cell">No positions were opened.</td></tr>'
  obDrawHist(r)
  obRenderWhy(r)
  el('ob-share-msg').textContent = ''
  el('ob-sweep-out').hidden = true
  el('ob-sweep-error').hidden = true
  el('ob-sweep-run').textContent = 'Run the robustness test (25 backtests)'

  const ck = obState.check && obState.check.test
  el('ob-model-body').innerHTML = `
    <p class="card-desc"><strong>Prices.</strong> Free historical option quotes don't exist, so every option is priced with Black-Scholes on
      that day's close. Implied volatility is ${r.model.iv}. Term structure: ${(r.model.term || 'a modelled slope.').replace(/^./, c => c.toLowerCase())}
      ${r.model.skew ? 'Out-of-the-money puts are priced richer and calls cheaper, as real chains do.' : 'The skew is turned off.'}
      The rate is the 3-month Treasury bill.</p>
    ${ck ? `<p class="card-desc"><strong>Checked against real returns.</strong> A replica of Cboe's PutWrite index (sell a one-month
      at-the-money put every month) earned ${ck.model.cagr.toFixed(1)}% a year with these prices from ${ck.window[0].slice(0, 4)} on, against
      ${ck.real.cagr.toFixed(1)}% for the real index; monthly returns correlate at ${ck.monthly_corr.toFixed(2)}. The volatility level was
      fitted on ${obState.check.level.fit_window[0].slice(0, 4)}-${obState.check.level.fit_window[1].slice(0, 4)} only.</p>` : ''}
    ${r.model.dividends ? `<p class="card-desc"><strong>Dividends.</strong> ${r.model.dividends}</p>` : ''}
    ${r.model.splits ? `<p class="card-desc"><strong>Splits.</strong> ${r.model.splits}</p>` : ''}
    ${r.model.earnings ? `<p class="card-desc"><strong>Earnings.</strong> ${r.model.earnings}</p>` : ''}
    <p class="card-desc"><strong>Costs.</strong> ${r.model.costs}. Stock legs pay 0.03% slippage. Options held to expiry settle at intrinsic value with no spread.</p>
    <p class="card-desc"><strong>Cash.</strong> Cash not tied up earns the Treasury-bill rate. Credit received stays in the account as collateral.</p>
    ${obSensTable(r)}
    <ul class="opt-limits">${r.model.limits.map(x => `<li>${x}</li>`).join('')}</ul>`
  el('ob-notes').textContent = (r.notes || []).join(' ')
}

// ── Why it made or lost money: the P&L split by Greek ─────────────────────
const OB_ATTR = [
  ['delta', 'Direction (delta)'], ['gamma', 'Size of moves (gamma)'], ['theta', 'Time decay (theta)'],
  ['vega', 'Volatility (vega)'], ['other', 'Other'], ['costs', 'Commissions and spreads'], ['dividends', 'Dividends'],
  ['interest', 'Interest on cash'],
]
const OB_GREEK_NOTE = {
  delta: 'Dollars gained or lost for a 1% rise in the underlying, on each day a position was open.',
  theta: 'Dollars of time decay per calendar day: positive when the position collects it, negative when it pays it.',
  vega: 'Dollars gained or lost for a one-point rise in implied volatility.',
}

function obSigned(v) { return (v >= 0 ? '+' : '−') + obMoney0(Math.abs(v)) }

// The largest part of a position's P&L split that pushed the same way as its result: a winner's
// main driver is its biggest gain, a loser's its biggest loss.
function obDriver(a, pnl) {
  if (!a) return '—'
  const short = { delta: 'Direction', gamma: 'Big moves', theta: 'Time decay', vega: 'Volatility', other: 'Other',
    costs: 'Costs', dividends: 'Dividends' }
  const sign = pnl == null || pnl >= 0 ? 1 : -1
  const keys = Object.keys(short).filter(x => (a[x] || 0) * sign > 0)
  if (!keys.length) return '—'
  const k = keys.reduce((best, x) => Math.abs(a[x]) > Math.abs(a[best]) ? x : best)
  return `${short[k]} <span class="${clr(a[k])}">${obSigned(a[k])}</span>`
}

// Histogram of each position's P&L after costs.
function obDrawHist(r) {
  obState.charts.hist = destroyChart(obState.charts.hist)
  const p = r.trades.map(t => t.pnl).filter(v => v != null)
  el('ob-hist-note').textContent = ''
  if (p.length < 2) return
  const lo = Math.min(...p), hi = Math.max(...p)
  const nb = Math.min(24, Math.max(6, Math.round(Math.sqrt(p.length) * 2)))
  const w = (hi - lo) / nb || 1
  const bins = Array.from({ length: nb }, (_, i) => ({ from: lo + i * w, to: lo + (i + 1) * w, n: 0 }))
  p.forEach(v => { bins[Math.min(nb - 1, Math.floor((v - lo) / w))].n += 1 })
  const sorted = [...p].sort((x, y) => x - y)
  const med = sorted[Math.floor(sorted.length / 2)]
  el('ob-hist-note').textContent = `(median ${obMoney0(med)}, worst ${obMoney0(sorted[0])}, best ${obMoney0(sorted[sorted.length - 1])})`
  obState.charts.hist = new Chart(el('ob-hist'), {
    type: 'bar',
    data: { labels: bins.map(b => obMoney0((b.from + b.to) / 2)), datasets: [{ data: bins.map(b => b.n),
      backgroundColor: bins.map(b => (b.from + b.to) / 2 >= 0 ? 'rgba(48,209,88,0.7)' : 'rgba(255,69,58,0.7)'),
      barPercentage: 1, categoryPercentage: 0.95 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { title: c => `${obMoney0(bins[c[0].dataIndex].from)} to ${obMoney0(bins[c[0].dataIndex].to)}`,
        label: c => `${c.parsed.y} position${c.parsed.y === 1 ? '' : 's'}` } } },
      scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { ticks: { precision: 0 } } } },
  })
}

function obRenderWhy(r) {
  const a = r.attribution
  el('ob-why').classList.toggle('opt-why-empty', !a)
  if (!a) { el('ob-why-lede').textContent = 'Run the backtest again to see the split (this server is older than the feature).'; return }
  const rows = OB_ATTR.filter(([k]) => a[k] != null && !(k === 'dividends' && !a[k]))
  const total = rows.reduce((s, [k]) => s + a[k], 0)
  const opt = rows.filter(([k]) => ['delta', 'gamma', 'theta', 'vega'].includes(k))
  const best = opt.reduce((x, y) => a[y[0]] > a[x[0]] ? y : x)
  const worst = opt.reduce((x, y) => a[y[0]] < a[x[0]] ? y : x)
  el('ob-why-lede').innerHTML = `Every day a position was open, its change in value is split by Greek, using that day's
    exposures. Over the whole test the account ${total >= 0 ? 'made' : 'lost'} <strong class="${clr(total)}">${obMoney0(Math.abs(total))}</strong>.
    The biggest help was <strong>${best[1].replace(/ \(.*\)$/, '').toLowerCase()}</strong> (${obSigned(a[best[0]])}) and the biggest drag
    <strong>${worst[1].replace(/ \(.*\)$/, '').toLowerCase()}</strong> (${obSigned(a[worst[0]])}); interest on idle cash added ${obSigned(a.interest || 0)}.
    The parts add up to the total exactly.`
  obState.charts.attr = destroyChart(obState.charts.attr)
  obState.charts.attr = new Chart(el('ob-attr-chart'), {
    type: 'bar',
    data: { labels: rows.map(([, l]) => l), datasets: [{ data: rows.map(([k]) => a[k]),
      backgroundColor: rows.map(([k]) => a[k] >= 0 ? 'rgba(48,209,88,0.75)' : 'rgba(255,69,58,0.75)'), borderRadius: 4 }] },
    options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      scales: { x: { ticks: { callback: v => (v < 0 ? '−$' : '$') + Math.abs(v / 1000).toFixed(0) + 'k' } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => obSigned(c.parsed.x) } } } },
  })
  obDrawGreek(r, obState.greek || 'delta')
}

function obDrawGreek(r, g) {
  obState.greek = g
  document.querySelectorAll('#ob-greek-seg .seg-btn').forEach(b => b.classList.toggle('active', b.dataset.greek === g))
  el('ob-greek-note').textContent = OB_GREEK_NOTE[g]
  obState.charts.greek = destroyChart(obState.charts.greek)
  if (r.equity[0][g] == null) return
  obState.charts.greek = new Chart(el('ob-greek-chart'), {
    type: 'line',
    data: { labels: r.equity.map(e => e.date), datasets: [{ label: g, data: r.equity.map(e => e[g]), borderColor: '#a371f7',
      backgroundColor: 'rgba(163,113,247,0.10)', borderWidth: 1.5, pointRadius: 0, fill: 'origin', stepped: false }] },
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { ticks: { callback: v => (v < 0 ? '−$' : '$') + Math.abs(v).toLocaleString() } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => obSigned(c.parsed.y) } } } },
  })
}

// ── Is the edge real? A settings sweep with PBO and a Deflated Sharpe ─────
async function obSweep() {
  const body = obState.lastBody
  if (!body) return
  const btn = el('ob-sweep-run')
  btn.disabled = true
  btn.textContent = 'Running 25 backtests…'
  el('ob-sweep-error').hidden = true
  const data = await obPost('/api/options/sweep', body)
  btn.disabled = false
  btn.textContent = 'Run the robustness test again'
  if (!data || data.error) {
    el('ob-sweep-error').textContent = failMsg(data, 'The robustness test could not run. Try again in a moment.')
    el('ob-sweep-error').hidden = false
    return
  }
  obRenderSweep(data)
}

function obRenderSweep(d) {
  el('ob-sweep-out').hidden = false
  el('ob-sweep-verdict').textContent = d.verdict
  const p = d.pbo
  el('ob-sweep-stats').innerHTML = [
    statCard('Overfitting probability (PBO)', p ? (p.pbo * 100).toFixed(0) + '%' : '—', p ? (p.pbo >= 0.5 ? 'negative' : p.pbo < 0.25 ? 'positive' : '') : ''),
    statCard('Best setting: Deflated Sharpe', d.best_dsr == null ? '—' : d.best_dsr.toFixed(0) + '%', d.best_dsr >= 95 ? 'positive' : ''),
    statCard('Sharpe luck alone could reach', d.sharpe_bar == null ? '—' : fmtN(d.sharpe_bar)),
    statCard('Best setting loses out of sample', p ? (p.prob_oos_loss * 100).toFixed(0) + '% of splits' : '—'),
    ...(d.start_dates && d.start_dates.cagr_range != null ? [statCard('Your settings, 5 start dates',
      `${fmtPct(d.start_dates.cagr_low, 1)} to ${fmtPct(d.start_dates.cagr_high, 1)}`, d.start_dates.cagr_range >= 5 ? 'negative' : '')] : []),
  ].join('')
  const sh = d.cells.map(c => c.sharpe)
  const lo = Math.min(...sh), hi = Math.max(...sh)
  const tint = v => {
    if (v >= 0) return `rgba(48,209,88,${(0.12 + 0.6 * (hi > 0 ? v / hi : 0)).toFixed(2)})`
    return `rgba(255,69,58,${(0.12 + 0.6 * (lo < 0 ? v / lo : 0)).toFixed(2)})`
  }
  const head = `<thead><tr><th>Days to expiry</th>${d.exits.map(x => `<th>${x.label}</th>`).join('')}</tr></thead>`
  const body = d.dte.map(dte => `<tr><th scope="row">${dte === d.current_dte ? '▸ ' : ''}${dte} days</th>${d.exits.map(x => {
    const i = d.cells.findIndex(c => c.dte === dte && c.exit === x.code)
    const c = d.cells[i]
    return `<td class="opt-heat-cell${i === d.best ? ' opt-heat-best' : ''}" style="background:${tint(c.sharpe)}"
      title="CAGR ${fmtPct(c.cagr, 1)} · max drawdown ${fmtPct(c.max_drawdown, 1)} · ${c.trades} positions">${fmtN(c.sharpe)}
      <span class="opt-heat-sub">${fmtPct(c.cagr, 1)} a year</span></td>`
  }).join('')}</tr>`).join('')
  el('ob-heat').innerHTML = head + '<tbody>' + body + '</tbody>'
}

function obDrawEquity(r) {
  obState.charts.eq = destroyChart(obState.charts.eq)
  const labels = r.equity.map(e => e.date)
  obState.charts.eq = new Chart(el('ob-chart'), {
    type: 'line',
    data: { labels, datasets: [
      { label: r.strategy.name, data: r.equity.map(e => e.value), borderColor: '#0a84ff', backgroundColor: 'rgba(10,132,255,0.10)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.15 },
      { label: r.benchmark.label, data: r.equity.map(e => e.benchmark), borderColor: '#86868b', borderDash: [5, 4],
        borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.15 },
    ] },
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } } },
      plugins: { tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt$(c.parsed.y)}` } } } },
  })
}

function obDrawPayoff(ex) {
  obState.charts.pay = destroyChart(obState.charts.pay)
  const pts = ex.payoff
  obState.charts.pay = new Chart(el('ob-payoff'), {
    type: 'line',
    data: { labels: pts.map(p => p.x), datasets: [{
      data: pts.map(p => p.pnl), borderWidth: 2, pointRadius: 0, tension: 0,
      segment: { borderColor: c => (c.p0.parsed.y + c.p1.parsed.y) / 2 >= 0 ? '#30d158' : '#ff453a' },
      fill: { target: 'origin', above: 'rgba(48,209,88,0.12)', below: 'rgba(255,69,58,0.12)' },
    }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { title: c => `${obState.result.ticker} at ${fmt$(Number(c[0].label))}`, label: c => obMoney0(c.parsed.y) } } },
      scales: { x: { ticks: { maxTicksLimit: 6, callback: function (v) { return '$' + Math.round(this.getLabelForValue(v)) } } },
                y: { ticks: { callback: v => obMoney0(v) } } } },
  })
}

// ── compare all ────────────────────────────────────────────────────────────

async function obCompare() {
  el('ob-loading-title').firstChild.textContent = 'Comparing every strategy '
  obShow('loading')
  el('ob-run').disabled = el('ob-compare').disabled = true
  const data = await obPost('/api/options/compare', obSettings())
  el('ob-run').disabled = el('ob-compare').disabled = false
  if (!data || data.error) { obError(failMsg(data, 'The comparison could not run. Try again in a moment.')); return }
  obShow('compare')
  const b = data.benchmark
  el('ob-cmp-model').textContent = 'Option prices are reconstructed with Black-Scholes from each day’s close and a volatility estimate, not real quotes.'
  el('ob-cmp-meta').textContent = `${data.ticker} · ${data.start_date} to ${data.end_date} · ${data.dte} days to expiry · buy and hold ${fmtPct(b.total_return, 1)}`
  obState.cmp = data
  obRenderCmpRows()
  const df = data.deflation
  el('ob-cmp-deflation').textContent = df && df.sharpe_bar != null
    ? `Deflated Sharpe: ${df.n} strategies were tried here, and the luckiest of ${df.n} with no edge at all would show a Sharpe near
       ${df.sharpe_bar.toFixed(2)} over this window by chance. Each row's figure is the probability its Sharpe beats that bar; 95% is
       the usual line. It tests the selection, not the model prices.`
    : ''
  obState.charts.cmp = destroyChart(obState.charts.cmp)
  const top = obCmpSorted().slice(0, 5)
  const palette = ['#30d158', '#bf5af2', '#ffd60a', '#0a84ff', '#ff9f0a']
  const n = Math.max(...top.map(r => r.equity.length))
  obState.charts.cmp = new Chart(el('ob-cmp-chart'), {
    type: 'line',
    data: { labels: Array.from({ length: n }, (_, i) => i), datasets: top.map((r, i) => ({
      label: r.name, data: r.equity, borderColor: palette[i], borderWidth: 1.8, pointRadius: 0, tension: 0.15 })) },
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      scales: { x: { display: false }, y: { ticks: { callback: v => '$' + (v / 1000).toFixed(0) + 'k' } } },
      plugins: { tooltip: { callbacks: { title: () => '', label: c => `${c.dataset.label}: ${fmt$(c.parsed.y)}` } } } },
  })
}

function obCmpSorted() {
  const d = obState.cmp
  if (!d) return []
  const key = obState.cmpSort || 'sharpe'
  return [...d.results].sort((x, y) => (y[key] ?? -1e9) - (x[key] ?? -1e9))
}

function obRenderCmpRows() {
  el('ob-cmp-tbody').innerHTML = obCmpSorted().map((r, i) => `<tr class="opt-cmp-row" data-key="${r.key}" tabindex="0">
    <td>${i + 1}</td><td><strong>${r.name}</strong></td>
    <td><span class="badge">${r.key === 'wheel' ? 'cycle' : r.legs_count}</span></td>
    <td class="${clr(r.total_return)}">${fmtPct(r.total_return, 1)}</td><td class="${clr(r.cagr)}">${fmtPct(r.cagr, 1)}</td>
    <td>${fmtN(r.sharpe)}</td><td class="negative">${fmtPct(r.max_drawdown, 1)}</td>
    <td class="${clr(r.worst_month)}">${r.worst_month == null ? '—' : fmtPct(r.worst_month, 1)}</td>
    <td>${r.capital_used == null ? '—' : r.capital_used + '%'}</td>
    <td>${r.win_rate == null ? '—' : r.win_rate + '%'}</td><td>${r.trades}</td>
    <td class="${r.dsr != null && r.dsr >= 95 ? 'positive' : ''}">${r.dsr == null ? '—' : r.dsr.toFixed(0) + '%'}</td></tr>`).join('')
}

// One-click setups: what most traders run.
const OB_QUICK = {
  hold: { dte: 30, exit_dte: '', pt: '', sl: '' },
  managed: { dte: 45, exit_dte: 21, pt: 50, sl: '' },
}

// ── init ───────────────────────────────────────────────────────────────────

// How close the model gets to Cboe's real PutWrite index (tools/options_validation.py).
// The model priced against real closing quotes archived by Tradeski (tools/options_quotes_check.py).
async function obLoadQuotes() {
  let d
  try {
    const res = await fetch('/data/options_quotes_check.json', { cache: 'no-cache' })
    if (!res.ok) return
    d = await res.json()
  } catch (e) { return }
  const o = d.overall
  if (!o) return
  const pts = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + ' pts'
  const nd = d.days.length
  el('ob-quotes-badge').textContent = `${o.n.toLocaleString('en-US')} real contracts`
  const etf = d.by_report.find(g => g.label.startsWith('index'))
  const stk = d.by_report.filter(g => g.label.startsWith('single'))
  const stkMiss = stk.length ? Math.max(...stk.map(g => g.iv_mae)) : null
  el('ob-quotes-lede').textContent = `The same model, pricing every out-of-the-money option quoted at the close for
    ${d.symbols.length} underlyings on ${nd === 1 ? d.days[0] : `${nd} trading days, ${d.days[0]} to ${d.days[nd - 1]}`}, against the
    midpoint of the real bid and ask. ${etf ? `On the index funds its volatility is typically ${etf.iv_mae.toFixed(1)} points off.` : ''}
    ${stkMiss != null ? `On single stocks it is typically ${stkMiss.toFixed(1)} points off: a stock's own implied volatility often
    sits far from what its recent moves suggest, so read single-stock backtests as rougher than SPY's.` : ''}
    The archive grows every trading day.`
  el('ob-quotes-stats').innerHTML = [
    statCard('Model minus market IV', pts(o.iv_bias)),
    statCard('Typical IV miss', o.iv_mae.toFixed(1) + ' pts'),
    statCard('Typical price miss', o.price_mae.toFixed(0) + '%'),
    statCard('Inside the bid/ask', o.inside_spread.toFixed(0) + '%'),
  ].join('')
  el('ob-quotes-sym').innerHTML = d.by_symbol.slice().sort((a, b) => b.n - a.n).map(s => `<tr>
    <td>${obEsc(s.symbol)}${s.note ? ` <span class="text-muted" title="${obEsc(s.note)}">*</span>` : ''}</td>
    <td>${s.atm ? s.atm.model.toFixed(1) + '%' : '—'}</td><td>${s.atm ? s.atm.market.toFixed(1) + '%' : '—'}</td>
    <td class="${Math.abs(s.iv_bias) >= 5 ? 'negative' : ''}">${pts(s.iv_bias)}</td><td>${s.iv_mae.toFixed(1)} pts</td>
    <td>${s.n.toLocaleString('en-US')}</td></tr>`).join('')
  const grp = [...d.by_moneyness, ...d.by_dte, ...d.by_report]
  el('ob-quotes-grp').innerHTML = grp.map(g => `<tr><td>${obEsc(g.label)}</td><td>${pts(g.iv_bias)}</td>
    <td>${g.iv_mae.toFixed(1)} pts</td><td>${fmtPct(g.price_bias, 0)}</td><td>${g.n.toLocaleString('en-US')}</td></tr>`).join('')
  el('ob-quotes-fine').textContent = `${d.filters} Errors are the model's implied volatility for each strike and expiry minus the
    one the midpoint implies, with the same rate and dividend yield; out-of-the-money options are cheap, so a small volatility
    miss is a large share of the price. Quotes: ${d.source}. Updated ${d.generated}.`
  el('ob-quotes').hidden = false
}

async function obLoadCheck() {
  let d
  try {
    const res = await fetch('/data/options_validation.json', { cache: 'no-cache' })
    if (!res.ok) return
    d = await res.json()
  } catch (e) { return }
  obState.check = d
  const t = d.test, f = d.fit
  const y = s => s.slice(0, 4)
  const fitYears = `${y(f.window[0])}-${y(f.window[1])}`
  const idx = d.indices && d.indices.length ? d.indices : null
  el('ob-check-lede').innerHTML = idx
    ? `There are no free historical option quotes to check against, but Cboe publishes the real returns of option
      strategies as indices. Each one below is run here on SPY with the model's prices and compared with the real index.
      The model's settings were fitted on ${fitYears} only: the volatility level against the PutWrite index and the skew
      against the put-protection and 30-delta buy-write indices. Everything shown is ${y(t.window[0])} on, which the
      fit never saw; the other indices weren't used at all. Click a row to chart it.`
    : `Cboe's <strong>PutWrite index</strong> sells a one-month at-the-money S&amp;P 500 put every month. Run the same
      strategy here on SPY and the two can be compared. The volatility level was fitted on ${fitYears}; everything
      below is ${y(t.window[0])} on.`
  const pct = v => v.toFixed(1) + '%'
  const gap = v => v == null ? '—' : `<span class="${Math.abs(v) <= 1 ? 'positive' : Math.abs(v) <= 2 ? '' : 'negative'}">${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}</span>`
  const rows = idx || [{ code: 'PUT', name: 'PutWrite', what: 'sell a one-month at-the-money put', ...t, chart: d.chart }]
  el('ob-check-idx').hidden = !idx
  if (idx) {
    el('ob-check-idx-tbody').innerHTML = rows.map((r, i) => `<tr class="opt-check-row" data-i="${i}" tabindex="0">
      <td><strong>${r.code}</strong> <span class="text-muted">${r.name}</span></td><td>${r.what}</td>
      <td>${pct(r.model.cagr)}</td><td>${pct(r.real.cagr)}</td><td>${gap(r.cagr_gap)}</td>
      <td>${gap(r.cagr_gap_before_skew_fit)}</td><td>${r.monthly_corr.toFixed(2)}</td>
      <td class="text-muted">${r.weekly_schedule ? r.weekly_schedule.monthly_corr.toFixed(2) : '—'}</td></tr>`).join('')
  }
  const show = i => {
    const r = rows[i]
    document.querySelectorAll('.opt-check-row').forEach(tr => tr.classList.toggle('active', Number(tr.dataset.i) === i))
    const row = (k, a, b) => `<tr><td>${k}</td><td>${pct(a)}</td><td>${pct(b)}</td></tr>`
    el('ob-check-tbody').innerHTML = [
      row('Return a year', r.model.cagr, r.real.cagr),
      row('Volatility (monthly, annualised)', r.model.monthly_vol, r.real.monthly_vol),
      row('Worst drawdown', r.model.max_drawdown, r.real.max_drawdown),
      row('Worst month', r.model.worst_month, r.real.worst_month),
    ].join('') + `<tr><td>Monthly correlation</td><td colspan="2">${r.monthly_corr.toFixed(2)} over ${r.months} months</td></tr>`
    el('ob-check-which').textContent = `${r.code}: ${r.what}`
    obState.charts.check = destroyChart(obState.charts.check)
    obState.charts.check = new Chart(el('ob-check-chart'), {
      type: 'line',
      data: { labels: d.chart.months.slice(0, r.chart.model.length), datasets: [
        { label: `Model replica (SPY)`, data: r.chart.model, borderColor: '#a371f7', borderWidth: 2, pointRadius: 0, tension: 0.15 },
        { label: `Real Cboe ${r.code} index`, data: r.chart.real, borderColor: '#86868b', borderDash: [5, 4], borderWidth: 1.5,
          pointRadius: 0, tension: 0.15 },
      ] },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
        scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { ticks: { callback: v => v.toFixed(0) } } },
        plugins: { tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}` } } } },
    })
  }
  if (idx) {
    el('ob-check-idx-tbody').addEventListener('click', e => { const tr = e.target.closest('[data-i]'); if (tr) show(Number(tr.dataset.i)) })
    el('ob-check-idx-tbody').addEventListener('keydown', e => {
      const tr = e.target.closest('[data-i]'); if (tr && e.key === 'Enter') show(Number(tr.dataset.i))
    })
  }
  show(0)
  el('ob-check-fine').textContent = `Gaps are the model's annual return minus the real index's, in points; "before" is the same
    check with the skew used before this fit. Replicas roll on the third Friday, as the indices do; rolled every four to five
    weeks instead, they track far less closely (the last column), because in a crash the day an option was sold decides most
    of the loss. The worst months still differ, mainly because the indices settle at Cboe's opening quotation on the third
    Friday and the replicas at SPY's close. Updated ${d.generated}.`
  el('ob-check').hidden = false
}

// A strategy handed over in the URL (from Tradeski's planner): fill the form, say how it was
// translated, and run it when asked to.
function obApplyImport() {
  if (typeof obParseImport !== 'function') return
  const imp = obParseImport(location.search)
  const set = obParseSettings(location.search)
  const preset = set.preset && obState.byKey[set.preset] ? set.preset : null
  if (!imp && !preset) return
  const box = el('ob-import')
  if (imp && imp.error) {
    box.innerHTML = `<strong>That link could not be imported.</strong> ${obEsc(imp.error)}`
    box.hidden = false
    return
  }
  if (imp) {
    el('ob-name').value = imp.name
    obState.legs = imp.legs
    el('ob-strategy').value = 'custom'
  } else {
    el('ob-strategy').value = preset
  }
  obPickStrategy()
  // Every other setting the link carries (a shared backtest), over the strategy's defaults.
  const put = (id, v) => { if (v != null) el(id).value = v }
  put('ob-ticker', set.ticker || (imp && imp.ticker))
  put('ob-dte', set.dte != null ? set.dte : imp && imp.dte)
  put('ob-start', set.start_date); put('ob-end', set.end_date)
  put('ob-capital', set.initial_capital); put('ob-alloc', set.alloc_pct)
  put('ob-exit-dte', set.exit_dte); put('ob-pt', set.profit_take); put('ob-sl', set.stop_loss)
  put('ob-commission', set.commission); put('ob-spread', set.spread_pct)
  put('ob-exit-delta', set.exit_delta); put('ob-stagger', set.stagger_days)
  if (set.skew === false) el('ob-skew').checked = false
  if (set.expiries) el('ob-expiries').value = set.expiries
  const e = set.entry || {}
  put('ob-vix-min', e.vix_min); put('ob-vix-max', e.vix_max); put('ob-ivr-min', e.ivr_min); put('ob-ivr-max', e.ivr_max)
  if (e.trend) el('ob-trend').value = e.trend
  if (e.earnings) el('ob-earn').value = e.earnings
  if (set.exit_earnings) el('ob-exit-earn').checked = true
  if (set.entry) el('ob-entry').open = true
  if (set.exit_dte != null || set.profit_take != null || set.stop_loss != null || set.exit_earnings ||
      set.exit_delta != null) el('ob-exits').open = true
  if (imp) obRenderLegs()
  obRenderWhat()
  if (imp) {
    box.innerHTML = `<p>${obEsc(obImportSummary(imp))}</p>` +
      (imp.notes.length ? `<ul>${imp.notes.map(n => `<li>${obEsc(n)}</li>`).join('')}</ul>` : '')
  } else {
    box.innerHTML = `<p>${obEsc(`Opened from a shared link: ${obState.byKey[preset].name} on ${el('ob-ticker').value}, with the settings it was run with.`)}</p>`
  }
  box.hidden = false
  if ((imp && imp.run) || set.run) obRun()
}

// ── Share and export the result on screen ─────────────────────────────────────
function obShareUrl() {
  return `${location.origin}${location.pathname}?${obShareQuery(obState.lastBody)}`
}

async function obCopyLink() {
  if (!obState.lastBody) return
  const url = obShareUrl()
  const msg = el('ob-share-msg')
  try {
    await navigator.clipboard.writeText(url)
    msg.textContent = 'Link copied. Anyone who opens it reruns this exact backtest.'
  } catch (e) {
    msg.textContent = url                      // no clipboard access: show it to copy by hand
  }
}

function obDownload(kind) {
  const r = obState.result
  if (!r) return
  const csv = kind === 'trades' ? obCsv(r.trades, OB_TRADE_COLS) : obCsv(r.equity, OB_EQUITY_COLS)
  const name = `${r.ticker}-${(r.strategy.name || 'options').toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${kind}.csv`
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  a.download = name
  document.body.appendChild(a)
  a.click()
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove() }, 0)
}

async function initOptionsBacktest() {
  obLoadCheck()
  obLoadQuotes()
  el('ob-start').value = daysAgo(5 * 365)
  el('ob-end').value = today()
  initTabs(document.body)
  const sel = el('ob-strategy')
  sel.innerHTML = '<option>Loading strategies…</option>'
  const data = (await savedApi('options_presets.json')) || await api('/api/options/presets')
  obState.presets = (data && data.presets) || []
  obState.presets.forEach(p => { obState.byKey[p.key] = p })
  sel.innerHTML = OB_GROUPS.map(g => {
    const items = obState.presets.filter(p => p.group === g)
    return items.length ? `<optgroup label="${g}">${items.map(p =>
      `<option value="${p.key}">${p.name}</option>`).join('')}</optgroup>` : ''
  }).join('') + '<optgroup label="Your own"><option value="custom">Custom (build the legs)</option></optgroup>'
  sel.value = obState.byKey.collar ? 'collar' : sel.options[0].value
  obBuildGallery()
  obPickStrategy()
  obApplyImport()

  sel.addEventListener('change', obPickStrategy)
  el('ob-dte').addEventListener('input', obRenderWhat)
  el('ob-name').addEventListener('input', obRenderWhat)
  el('ob-legs').addEventListener('change', obLegChange)
  el('ob-legs').addEventListener('click', e => {
    const b = e.target.closest('[data-remove]')
    if (!b) return
    const i = Number(b.dataset.remove)
    obState.legs.splice(i, 1)
    // Offsets pointing at a removed or later leg fall back to at the money.
    obState.legs.forEach((l, j) => {
      if (l.strike && l.strike.mode === 'offset') {
        if (l.strike.leg === i || l.strike.leg >= j) l.strike = { mode: 'atm' }
        else if (l.strike.leg > i) l.strike.leg -= 1
      }
    })
    obRenderLegs(); obRenderWhat()
  })
  el('ob-add-leg').addEventListener('click', () => {
    if (obState.legs.length >= 6) return
    obState.legs.push({ type: 'call', side: 'long', qty: 1, strike: { mode: 'atm' } })
    obRenderLegs(); obRenderWhat()
  })
  el('ob-greek-seg').addEventListener('click', e => {
    const b = e.target.closest('[data-greek]')
    if (b && obState.result) obDrawGreek(obState.result, b.dataset.greek)
  })
  el('ob-sweep-run').addEventListener('click', obSweep)
  document.querySelector('.opt-quick').addEventListener('click', e => {
    const b = e.target.closest('[data-quick]')
    if (!b) return
    const q = OB_QUICK[b.dataset.quick]
    el('ob-dte').value = q.dte; el('ob-exit-dte').value = q.exit_dte; el('ob-pt').value = q.pt; el('ob-sl').value = q.sl
    if (b.dataset.quick === 'managed') el('ob-exits').open = true
    obRenderWhat()
  })
  el('ob-cmp-sort').addEventListener('click', e => {
    const b = e.target.closest('[data-sort]')
    if (!b || !obState.cmp) return
    obState.cmpSort = b.dataset.sort
    el('ob-cmp-sort').querySelectorAll('.seg-btn').forEach(x => x.classList.toggle('active', x === b))
    obRenderCmpRows()
  })
  el('ob-share-link').addEventListener('click', obCopyLink)
  el('ob-csv-trades').addEventListener('click', () => obDownload('trades'))
  el('ob-csv-equity').addEventListener('click', () => obDownload('equity'))
  el('ob-run').addEventListener('click', () => obRun())
  el('ob-compare').addEventListener('click', obCompare)
  el('ob-cmp-tbody').addEventListener('click', e => {
    const row = e.target.closest('[data-key]')
    if (row) obRun(row.dataset.key)
  })
  el('ob-cmp-tbody').addEventListener('keydown', e => {
    const row = e.target.closest('[data-key]')
    if (row && e.key === 'Enter') obRun(row.dataset.key)
  })
}

if (document.body.dataset.page === 'options') initOptionsBacktest()

// ══ Strategy gallery and explainer ═══════════════════════════════════════
// The page is for learning what each structure does, so strategies are picked
// from cards that show their shape, and the space where results will appear
// first explains the chosen one in plain words.

const OB_CATS = [
  ['all', 'All'], ['bullish', 'Bullish'], ['bearish', 'Bearish'], ['income', 'Income'],
  ['range', 'Range-bound'], ['move', 'Big move'], ['protect', 'Protection'],
]

const OB_LEARN = {
  long_call:        { cat: 'bullish', idea: "Pay a premium for the right to buy the stock at today's price.",
    wins: 'The stock rises by more than the premium before expiry.',
    risk: "If it doesn't rise enough, the whole premium is lost, and time decay costs you every day." },
  long_put:         { cat: 'bearish', idea: "Pay a premium for the right to sell the stock at today's price.",
    wins: 'The stock falls by more than the premium before expiry.',
    risk: "If it doesn't fall enough, the whole premium is lost, and time decay costs you every day." },
  short_put:        { cat: 'income', idea: "Get paid to promise to buy the stock a little below today's price, with the cash set aside.",
    wins: 'The stock stays flat or rises: the put expires and you keep the premium.',
    risk: 'In a sharp fall you buy the shares at the strike, well above where they trade.' },
  short_call:       { cat: 'income', idea: "Get paid to promise to sell shares you don't own, above today's price.",
    wins: 'The stock stays flat or falls: you keep the premium.',
    risk: 'A strong rally. The loss has no ceiling, which is why this one reserves extra capital.' },
  covered_call:     { cat: 'income', idea: 'Own 100 shares and sell a call above the price for income.',
    wins: "Flat or gently rising markets, where the premium adds to the stock's return.",
    risk: "You keep all of the stock's downside but give up its gains above the strike." },
  protective_put:   { cat: 'protect', idea: 'Own 100 shares and buy a put below the price as insurance.',
    wins: 'A strong rally, or a crash the put protects you from.',
    risk: 'The insurance costs something every month, a steady drag in calm markets.' },
  bull_call_spread: { cat: 'bullish', idea: 'Buy a call and sell a higher one, so the bet costs less.',
    wins: 'A moderate rise up to the upper strike.',
    risk: "The premium is lost if the stock doesn't rise, and the gain is capped at the upper strike." },
  bear_put_spread:  { cat: 'bearish', idea: 'Buy a put and sell a lower one, so the bet costs less.',
    wins: 'A moderate fall down to the lower strike.',
    risk: "The premium is lost if the stock doesn't fall, and the gain is capped at the lower strike." },
  bull_put_spread:  { cat: 'bullish', idea: "Sell a put and buy a lower one as a floor. You're paid up front.",
    wins: 'The stock stays above the short strike.',
    risk: 'A fall through both strikes loses the gap between them, less the premium.' },
  bear_call_spread: { cat: 'bearish', idea: "Sell a call and buy a higher one as a ceiling. You're paid up front.",
    wins: 'The stock stays below the short strike.',
    risk: 'A rally through both strikes loses the gap between them, less the premium.' },
  long_straddle:    { cat: 'move', idea: 'Buy a call and a put at the same strike: a bet on movement, not direction.',
    wins: 'A big move either way, larger than the two premiums.',
    risk: 'A quiet market, where both options lose value every day.' },
  short_straddle:   { cat: 'range', idea: 'Sell a call and a put at the same strike and collect both premiums.',
    wins: 'The price stays pinned near the strike.',
    risk: 'A big move either way; on the upside the loss has no ceiling.' },
  long_strangle:    { cat: 'move', idea: 'Buy a call above and a put below the price. Cheaper than a straddle.',
    wins: 'A move big enough to get past either strike by more than the cost.',
    risk: 'Needs a bigger move than a straddle; a quiet market loses both premiums.' },
  short_strangle:   { cat: 'range', idea: 'Sell a call above and a put below the price, for a wide range.',
    wins: 'The price stays between the two strikes.',
    risk: 'A breakout past either strike, with no limit on the upside.' },
  calendar_call:    { cat: 'range', idea: 'Sell a near-month call and buy a later one at the same strike.',
    wins: 'The price stays near the strike while the near option decays faster.',
    risk: 'A big move either way, which erases the difference between the two.' },
  collar:           { cat: 'protect', idea: 'Own 100 shares, buy a put below and sell a call above to pay for it.',
    wins: 'Any market where you want a known band of outcomes.',
    risk: 'Gains above the call are given up in exchange for the protection.' },
  call_butterfly:   { cat: 'range', idea: 'Three strikes: a cheap bet that the price ends near the middle one.',
    wins: "The price finishes close to today's.",
    risk: 'The small premium is lost if the price ends outside the outer strikes.' },
  iron_condor:      { cat: 'range', idea: 'Sell a put spread below and a call spread above the price.',
    wins: 'The price stays inside the range between the short strikes.',
    risk: 'A breakout past either side loses the width of that spread, less the premium.' },
  iron_butterfly:   { cat: 'range', idea: 'Sell a put and a call at the price and buy protection on both sides.',
    wins: 'The price stays very close to where it started.',
    risk: 'A move out to either wing loses the width, less the premium.' },
  wheel:            { cat: 'income', idea: "Sell puts until you're assigned shares, then sell calls on them until they're called away.",
    wins: 'Steady or rising markets, collecting premium at every step.',
    risk: 'You hold the shares through a decline, with only the premiums as a cushion.' },
}
const OB_CUSTOM_LEARN = { idea: 'Your own combination of up to six legs: calls, puts and shares.',
  wins: 'Read the shape: green is where the position profits at the first expiry.',
  risk: 'Red is where it loses. A line still sloping at the edge keeps going.' }

// ── payoff at the first expiry, priced with Black-Scholes ─────────────────
// A teaching picture, not the backtest's pricing: flat 20% volatility, a
// 4% rate, the price at 100. The backtest itself prices from each day's data.
const OB_VOL = 0.2, OB_RATE = 0.04

function obNorm(x) {                        // standard normal CDF (Abramowitz-Stegun 7.1.26)
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2)
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
    t * Math.exp(-x * x / 2)
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2
}

function obBS(type, S, K, T) {
  if (T <= 0) return Math.max(type === 'call' ? S - K : K - S, 0)
  const sq = OB_VOL * Math.sqrt(T)
  const d1 = (Math.log(S / K) + (OB_RATE + OB_VOL * OB_VOL / 2) * T) / sq, d2 = d1 - sq
  return type === 'call' ? S * obNorm(d1) - K * Math.exp(-OB_RATE * T) * obNorm(d2)
                         : K * Math.exp(-OB_RATE * T) * obNorm(-d2) - S * obNorm(-d1)
}

function obStrikeFor(leg, ks, T) {
  const s = leg.strike || { mode: 'atm' }, v = Number(s.value) || 0
  if (s.mode === 'pct') return 100 * (1 + v / 100)
  if (s.mode === 'offset') return (ks[s.leg] ?? 100) + v
  if (s.mode === 'delta' && v > 0 && v < 1) {   // the strike whose delta matches, by bisection
    let lo = 40, hi = 250
    for (let i = 0; i < 60; i++) {
      const K = (lo + hi) / 2
      const d1 = (Math.log(100 / K) + (OB_RATE + OB_VOL * OB_VOL / 2) * T) / (OB_VOL * Math.sqrt(T))
      const delta = leg.type === 'call' ? obNorm(d1) : 1 - obNorm(d1)   // put delta, as a positive number
      if (leg.type === 'call' ? delta > v : delta < v) lo = K; else hi = K
    }
    return (lo + hi) / 2
  }
  return 100
}

function obPayoff(legs, dte) {
  const opts = legs.filter(l => l.type !== 'stock')
  const front = opts.length ? Math.min(...opts.map(l => l.dte || dte)) : dte
  const ks = [], prem = []
  legs.forEach((l, i) => {
    if (l.type === 'stock') { ks.push(null); prem.push(100); return }
    const T = (l.dte || dte) / 365
    ks.push(obStrikeFor(l, ks, T))
    prem.push(obBS(l.type, 100, ks[i], T))
  })
  const xs = [], ys = []
  for (let S = 70; S <= 130.001; S += 0.75) {
    let pnl = 0
    legs.forEach((l, i) => {
      const sign = (l.side === 'short' ? -1 : 1) * Number(l.qty || 1)
      const val = l.type === 'stock' ? S : obBS(l.type, S, ks[i], ((l.dte || dte) - front) / 365)
      pnl += sign * (val - prem[i])
    })
    xs.push(S); ys.push(pnl)
  }
  return { xs, ys, strikes: ks.filter(k => k != null) }
}

let obSvgId = 0
function obPayoffSVG(legs, dte, big) {
  if (!legs || !legs.length) return ''
  const { xs, ys, strikes } = obPayoff(legs, dte)
  const W = big ? 560 : 120, H = big ? 200 : 52, pad = big ? 12 : 3
  const m = Math.max(...ys.map(Math.abs), 0.01)
  const X = x => pad + (x - 70) / 60 * (W - 2 * pad)
  const Y = y => H / 2 - y / m * (H / 2 - pad)
  const pts = xs.map((x, i) => `${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join(' ')
  const area = `${X(70)},${H / 2} ${pts} ${X(130)},${H / 2}`
  const id = `obp${++obSvgId}`
  const marks = !big ? '' :
    strikes.filter(k => k > 70 && k < 130).map(k =>
      `<line x1="${X(k)}" x2="${X(k)}" y1="${pad}" y2="${H - pad}" class="opt-pay-strike"/>`).join('') +
    `<line x1="${X(100)}" x2="${X(100)}" y1="${pad}" y2="${H - pad}" class="opt-pay-now"/>`
  return `<svg class="opt-pay${big ? ' opt-pay-big' : ''}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
    <defs><clipPath id="${id}u"><rect x="0" y="0" width="${W}" height="${H / 2}"/></clipPath>
      <clipPath id="${id}d"><rect x="0" y="${H / 2}" width="${W}" height="${H / 2}"/></clipPath></defs>
    ${marks}
    <line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" class="opt-pay-zero"/>
    <g clip-path="url(#${id}u)"><polygon points="${area}" class="opt-pay-up-fill"/><polyline points="${pts}" class="opt-pay-up"/></g>
    <g clip-path="url(#${id}d)"><polygon points="${area}" class="opt-pay-dn-fill"/><polyline points="${pts}" class="opt-pay-dn"/></g>
  </svg>`
}

function obLegsFor(key) {
  if (key === 'custom') return obState.legs
  const p = obState.byKey[key]
  return p ? p.legs : []
}

function obBuildGallery() {
  const wrap = el('ob-cards'), bar = el('ob-filter')
  if (!wrap || !bar) return
  bar.innerHTML = OB_CATS.map(([k, t], i) =>
    `<button type="button" class="seg-btn${i ? '' : ' active'}" data-cat="${k}" aria-pressed="${!i}">${t}</button>`).join('')
  wrap.innerHTML = obState.presets.map(p => `<button type="button" class="opt-card" data-key="${p.key}"
      data-cat="${(OB_LEARN[p.key] || {}).cat || ''}" aria-pressed="false">
      ${obPayoffSVG(p.legs, 30, false)}
      <span class="opt-card-name">${p.name}</span>
      <span class="opt-card-view">${p.view}</span>
    </button>`).join('') +
    `<button type="button" class="opt-card opt-card-own" data-key="custom" data-cat="" aria-pressed="false">
      <span class="opt-card-plus" aria-hidden="true">+</span>
      <span class="opt-card-name">Build your own</span>
      <span class="opt-card-view">Up to six legs</span>
    </button>`
  bar.addEventListener('click', e => {
    const b = e.target.closest('[data-cat]')
    if (!b) return
    bar.querySelectorAll('.seg-btn').forEach(x => {
      x.classList.toggle('active', x === b)
      x.setAttribute('aria-pressed', String(x === b))
    })
    wrap.querySelectorAll('.opt-card').forEach(c => {
      c.hidden = !(b.dataset.cat === 'all' || c.dataset.cat === b.dataset.cat || c.dataset.key === 'custom')
    })
  })
  wrap.addEventListener('click', e => {
    const c = e.target.closest('.opt-card')
    if (!c) return
    el('ob-strategy').value = c.dataset.key
    obPickStrategy()
    obShow('empty')                          // a result on screen no longer matches the strategy
    el('ob-tool').scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}

function obRenderLearn() {
  const key = el('ob-strategy').value
  const p = obState.byKey[key]
  const custom = key === 'custom'
  if (!custom && !p) return
  const L = custom ? OB_CUSTOM_LEARN : (OB_LEARN[key] || {})
  const name = custom ? (el('ob-name').value.trim() || 'Your own strategy') : p.name
  el('ob-learn-name').textContent = name
  el('ob-learn-view').textContent = custom ? 'Custom' : p.view
  el('ob-learn-idea').textContent = L.idea || ''
  el('ob-learn-wins').textContent = L.wins || ''
  el('ob-learn-risk').textContent = L.risk || ''
  el('ob-learn-chart').innerHTML = obPayoffSVG(obLegsFor(key), Number(el('ob-dte').value) || 30, true)
  el('ob-picked').textContent = name
  document.querySelectorAll('.opt-card').forEach(c => {
    const on = c.dataset.key === key
    c.classList.toggle('active', on)
    c.setAttribute('aria-pressed', String(on))
  })
}
