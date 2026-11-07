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
})

const OB_GROUPS = ['Single leg', 'Two legs', 'Three legs', 'Four legs']
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
  }
}

// ── strategy picker and the "what this holds" box ──────────────────────────

function obRenderWhat() {
  const key = el('ob-strategy').value
  const custom = key === 'custom'
  el('ob-custom').hidden = !custom
  const dte = Number(el('ob-dte').value) || 30
  let name, legs, view
  if (custom) {
    name = el('ob-name').value.trim() || 'Custom strategy'
    legs = obState.legs
    view = ''
  } else {
    const p = obState.byKey[key]
    if (!p) return
    name = p.name; legs = p.legs; view = p.view
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
  const data = await obPost('/api/options/backtest', body)
  el('ob-run').disabled = el('ob-compare').disabled = false
  if (!data || data.error) { obError(failMsg(data, 'The backtest could not run. Try again in a moment.')); return }
  obState.result = data
  obRenderResult(data)
}

const obMoney0 = n => n == null ? '—' : (n < 0 ? '−$' : '$') + Math.abs(Math.round(n)).toLocaleString('en-US')

function obRenderResult(r) {
  obShow('results')
  const m = r.metrics, b = r.benchmark, s = r.strategy
  el('ob-model-line').textContent = `Option prices are reconstructed with Black-Scholes from each day's close; implied volatility is ${r.model.iv}. Not real quotes.`
  const first = r.equity[0].date, last = r.equity[r.equity.length - 1].date
  const beat = m.total_return - b.total_return
  el('ob-summary').innerHTML = `<p class="summary-body"><strong>${s.name} on ${r.ticker}</strong>, ${first} to ${last}:
    <strong class="${clr(m.total_return)}">${fmtPct(m.total_return, 1)}</strong> (${fmtPct(m.cagr, 1)} a year) against
    ${fmtPct(b.total_return, 1)} for buying and holding ${r.ticker}, ${beat >= 0 ? 'ahead by' : 'behind by'}
    ${Math.abs(beat).toFixed(1)} points. ${m.trades} positions, ${m.win_rate ?? '—'}% of them profitable after costs;
    worst drawdown ${fmtPct(m.max_drawdown, 1)} versus ${fmtPct(b.max_drawdown, 1)}.</p>`

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
    <td>${fmt$(t.underlying_open)} → ${fmt$(t.underlying_close)}</td>
    <td>${t.net_open >= 0 ? 'paid ' + obMoney0(t.net_open) : 'collected ' + obMoney0(-t.net_open)}</td>
    <td>${obMoney0(t.net_close)}</td><td>${obMoney0(t.costs)}</td>
    <td class="${clr(t.pnl)}">${obMoney0(t.pnl)}</td><td>${t.reason}</td></tr>`).join('')
    : '<tr><td colspan="9" class="empty-cell">No positions were opened.</td></tr>'

  el('ob-model-body').innerHTML = `
    <p class="card-desc"><strong>Prices.</strong> Free historical option quotes don't exist, so every option is priced with Black-Scholes on
      that day's close. Implied volatility is ${r.model.iv}; longer expiries carry a term structure (higher than 30-day IV in calm markets,
      lower in panics)${r.model.skew ? ', and out-of-the-money puts are priced richer and calls cheaper, as real chains do' : ', with the skew turned off'}.
      The rate is the 3-month Treasury bill.</p>
    <p class="card-desc"><strong>Costs.</strong> ${r.model.costs}. Stock legs pay 0.03% slippage. Options held to expiry settle at intrinsic value with no spread.</p>
    <p class="card-desc"><strong>Cash.</strong> Cash not tied up earns the Treasury-bill rate. Credit received stays in the account as collateral.</p>
    <ul class="opt-limits">${r.model.limits.map(x => `<li>${x}</li>`).join('')}</ul>`
  el('ob-notes').textContent = (r.notes || []).join(' ')
}

function obDrawEquity(r) {
  obState.charts.eq = destroyChart(obState.charts.eq)
  const labels = r.equity.map(e => e.date)
  obState.charts.eq = new Chart(el('ob-chart'), {
    type: 'line',
    data: { labels, datasets: [
      { label: r.strategy.name, data: r.equity.map(e => e.value), borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.08)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.15 },
      { label: r.benchmark.label, data: r.equity.map(e => e.benchmark), borderColor: '#9ba49f', borderDash: [5, 4],
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
      segment: { borderColor: c => (c.p0.parsed.y + c.p1.parsed.y) / 2 >= 0 ? '#3fb950' : '#ff6b61' },
      fill: { target: 'origin', above: 'rgba(63,185,80,0.12)', below: 'rgba(255,107,97,0.12)' },
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
  el('ob-cmp-tbody').innerHTML = data.results.map((r, i) => `<tr class="opt-cmp-row" data-key="${r.key}" tabindex="0">
    <td>${i + 1}</td><td><strong>${r.name}</strong></td>
    <td><span class="badge">${r.legs_count}</span></td>
    <td class="${clr(r.total_return)}">${fmtPct(r.total_return, 1)}</td><td class="${clr(r.cagr)}">${fmtPct(r.cagr, 1)}</td>
    <td>${fmtN(r.sharpe)}</td><td class="negative">${fmtPct(r.max_drawdown, 1)}</td>
    <td>${r.win_rate == null ? '—' : r.win_rate + '%'}</td><td>${r.trades}</td></tr>`).join('')
  obState.charts.cmp = destroyChart(obState.charts.cmp)
  const top = data.results.slice(0, 5)
  const palette = ['#3fb950', '#c4a5f5', '#e3b341', '#58a6ff', '#e3913b']
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

// ── init ───────────────────────────────────────────────────────────────────

async function initOptionsBacktest() {
  el('ob-start').value = daysAgo(5 * 365)
  el('ob-end').value = today()
  initTabs(document.body)
  const sel = el('ob-strategy')
  sel.innerHTML = '<option>Loading strategies…</option>'
  const data = await api('/api/options/presets')
  obState.presets = (data && data.presets) || []
  obState.presets.forEach(p => { obState.byKey[p.key] = p })
  sel.innerHTML = OB_GROUPS.map(g => {
    const items = obState.presets.filter(p => p.group === g)
    return items.length ? `<optgroup label="${g}">${items.map(p =>
      `<option value="${p.key}">${p.name}</option>`).join('')}</optgroup>` : ''
  }).join('') + '<optgroup label="Your own"><option value="custom">Custom (build the legs)</option></optgroup>'
  sel.value = obState.byKey.collar ? 'collar' : sel.options[0].value
  obPickStrategy()

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
