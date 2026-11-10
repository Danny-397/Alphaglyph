/* Strategies handed over from another app (Tradeski's options planner) in the page URL.

   options.html?ticker=AAPL&dte=45&name=Bull%20call%20spread&legs=long_call_1_5|short_call_1_10&from=tradeski&run=1

   Each leg is side_type_qty_pct: pct is the strike as a percent from the price on the day it
   was planned, so the backtest can roll the same shape through history. Stock legs have no pct.
   Pure functions, unit-tested in tests/js/options_import.test.cjs. */

const OB_IMPORT_SOURCES = { tradeski: 'Tradeski' }

function obParseImport(search) {
  const q = new URLSearchParams(search || '')
  const raw = q.get('legs')
  if (!raw) return null
  const notes = []
  const legs = []
  for (const part of raw.split('|').slice(0, 6)) {
    const [side, type, qtyS, pctS] = part.split('_')
    if (!['long', 'short'].includes(side) || !['call', 'put', 'stock'].includes(type)) {
      notes.push(`Skipped a leg that could not be read ("${part.slice(0, 30)}").`)
      continue
    }
    let qty = parseInt(qtyS, 10)
    if (!Number.isFinite(qty) || qty < 1) qty = 1
    if (qty > 10) { notes.push(`A ${type} leg's quantity was cut from ${qty} to 10, the most a backtest leg allows.`); qty = 10 }
    if (type === 'stock') {
      if (side === 'short') { notes.push('Short stock is not supported here, so that leg was left out.'); continue }
      legs.push({ type, side, qty })
      continue
    }
    if (pctS && pctS.includes(':')) {          // a share link: delta, offset or pct, with an optional per-leg expiry
      const parts = pctS.split(':')
      const mode = parts[0]
      const dPart = parts.find(x => /^d\d+$/.test(x))
      const leg = { type, side, qty }
      if (mode === 'atm') leg.strike = { mode: 'atm' }
      else if (mode === 'delta' && Number(parts[1]) >= 0.02 && Number(parts[1]) <= 0.98) leg.strike = { mode, value: Number(parts[1]) }
      else if (mode === 'offset' && Number.isInteger(Number(parts[1])) && Number.isFinite(Number(parts[2])) && Number(parts[2]) !== 0) {
        leg.strike = { mode, leg: Number(parts[1]), value: Number(parts[2]) }
      } else if (mode === 'pct' && Number.isFinite(Number(parts[1]))) leg.strike = { mode, value: Math.max(-50, Math.min(50, Number(parts[1]))) }
      else { notes.push(`A strike that could not be read ("${pctS.slice(0, 20)}") was set at the money.`); leg.strike = { mode: 'atm' } }
      if (dPart) leg.dte = Math.min(365, Math.max(1, Number(dPart.slice(1))))
      legs.push(leg)
      continue
    }
    let pct = Number(pctS)
    if (!Number.isFinite(pct)) pct = 0
    if (Math.abs(pct) > 50) { notes.push(`A strike ${pct.toFixed(1)}% from the price was capped at 50%.`); pct = Math.sign(pct) * 50 }
    pct = Math.round(pct * 10) / 10
    legs.push({ type, side, qty, strike: pct === 0 ? { mode: 'atm' } : { mode: 'pct', value: pct } })
  }
  if (raw.split('|').length > 6) notes.push('Only the first six legs were kept.')
  if (!legs.some(l => l.type !== 'stock')) return { error: 'The link has no option legs to backtest.', notes }

  const tickerRaw = (q.get('ticker') || '').trim().toUpperCase()
  const ticker = /^[A-Z][A-Z0-9.\-^]{0,9}$/.test(tickerRaw) ? tickerRaw : 'SPY'
  let dte = parseInt(q.get('dte'), 10)
  if (!Number.isFinite(dte)) dte = 30
  if (dte < 3) { notes.push(`The expiry was ${dte} days away; backtests need at least 3, so 3 is used.`); dte = 3 }
  if (dte > 180) { notes.push(`The expiry was ${dte} days away; backtests go up to 180, so 180 is used.`); dte = 180 }
  const name = (q.get('name') || 'Imported strategy').trim().slice(0, 60) || 'Imported strategy'
  const from = OB_IMPORT_SOURCES[q.get('from')] || null
  return { ticker, dte, name, legs, from, run: q.get('run') === '1', notes }
}

// One line saying how the handed-over strikes are used.
function obImportSummary(imp) {
  const src = imp.from ? `from ${imp.from}` : 'from a link'
  return `Imported ${src}: ${imp.name} on ${imp.ticker}, ${imp.dte} days to expiry. Strikes are kept as a percent from the price ` +
    'on the day you planned it, so every position in the backtest uses the same shape at that day\'s price.'
}

// ── Share links: every setting of a backtest in the URL ─────────────────────
// Field in the request body -> query parameter, and how to read it back.
const OB_SHARE_FIELDS = [
  ['start_date', 'start', 'date'], ['end_date', 'end', 'date'], ['initial_capital', 'capital', 'num'],
  ['alloc_pct', 'alloc', 'num'], ['exit_dte', 'exit_dte', 'num'], ['profit_take', 'pt', 'num'],
  ['stop_loss', 'sl', 'num'], ['commission', 'commission', 'num'], ['spread_pct', 'spread', 'num'],
]
const OB_ENTRY_FIELDS = ['vix_min', 'vix_max', 'ivr_min', 'ivr_max']

// The query string for a backtest request body (preset key or custom legs).
function obShareQuery(body) {
  const q = new URLSearchParams()
  q.set('ticker', body.ticker || 'SPY')
  if (body.preset) q.set('preset', body.preset)
  else if (body.legs) {
    q.set('legs', body.legs.map(obLegToken).join('|'))
    if (body.name) q.set('name', body.name)
  }
  if (body.dte != null) q.set('dte', String(body.dte))
  for (const [field, param] of OB_SHARE_FIELDS) if (body[field] != null && body[field] !== '') q.set(param, String(body[field]))
  if (body.skew === false) q.set('skew', '0')
  if (body.expiries && body.expiries !== 'weekly') q.set('expiries', body.expiries)
  const e = body.entry || {}
  for (const k of OB_ENTRY_FIELDS) if (e[k] != null) q.set(k, String(e[k]))
  if (e.trend && e.trend !== 'any') q.set('trend', e.trend)
  if (e.earnings && e.earnings !== 'any') q.set('earnings', e.earnings)
  if (body.exit_earnings) q.set('exit_earnings', '1')
  q.set('run', '1')
  return q.toString()
}

// side_type_qty_pct for percent and at-the-money strikes; other strike modes travel as JSON.
function obLegToken(l) {
  if (l.type === 'stock') return `${l.side}_stock_${l.qty}`
  const s = l.strike || { mode: 'atm' }
  if (s.mode === 'atm' && !l.dte) return `${l.side}_${l.type}_${l.qty}_0`
  if (s.mode === 'pct' && !l.dte) return `${l.side}_${l.type}_${l.qty}_${s.value}`
  if (s.mode === 'atm') return `${l.side}_${l.type}_${l.qty}_atm:0:d${l.dte}`
  return `${l.side}_${l.type}_${l.qty}_${s.mode}:${s.mode === 'offset' ? s.leg + ':' : ''}${s.value}${l.dte ? ':d' + l.dte : ''}`
}

// The settings in a share link (everything but the strategy, which obParseImport reads).
function obParseSettings(search) {
  const q = new URLSearchParams(search || '')
  const out = {}
  const ok = v => v != null && v !== ''
  for (const [field, param, kind] of OB_SHARE_FIELDS) {
    const v = q.get(param)
    if (!ok(v)) continue
    if (kind === 'date') { if (/^\d{4}-\d{2}-\d{2}$/.test(v)) out[field] = v }
    else if (Number.isFinite(Number(v))) out[field] = Number(v)
  }
  if (q.get('skew') === '0') out.skew = false
  if (q.get('expiries') === 'monthly') out.expiries = 'monthly'
  const entry = {}
  for (const k of OB_ENTRY_FIELDS) { const v = q.get(k); if (ok(v) && Number.isFinite(Number(v))) entry[k] = Number(v) }
  if (['above', 'below'].includes(q.get('trend'))) entry.trend = q.get('trend')
  if (['avoid', 'only'].includes(q.get('earnings'))) entry.earnings = q.get('earnings')
  if (q.get('exit_earnings') === '1') out.exit_earnings = true
  if (Object.keys(entry).length) out.entry = entry
  const preset = q.get('preset')
  if (preset && /^[a-z_]{2,40}$/.test(preset)) out.preset = preset
  const t = (q.get('ticker') || '').trim().toUpperCase()
  if (/^[A-Z][A-Z0-9.\-^]{0,9}$/.test(t)) out.ticker = t
  const dte = parseInt(q.get('dte'), 10)
  if (Number.isFinite(dte)) out.dte = Math.min(180, Math.max(3, dte))
  out.run = q.get('run') === '1'
  return out
}

// ── CSV export ────────────────────────────────────────────────────────────────
function obCsvCell(v) {
  if (v == null) return ''
  const s = String(v)
  // Quote anything with a separator, quote or newline; neutralise spreadsheet formulas.
  const safe = /^[=+\-@\t\r]/.test(s) && !/^-?\d/.test(s) ? "'" + s : s
  return /[",\n\r]/.test(safe) ? '"' + safe.replace(/"/g, '""') + '"' : safe
}

function obCsv(rows, cols) {
  const lines = [cols.map(c => obCsvCell(c[0])).join(','), ...rows.map(r => cols.map(c => obCsvCell(c[1](r))).join(','))]
  return lines.join('\r\n') + '\r\n'
}

const OB_TRADE_COLS = [
  ['opened', t => t.opened], ['closed', t => t.closed], ['why it closed', t => t.reason], ['units', t => t.units],
  ['position', t => t.legs.map(l => l.type === 'stock' ? `${l.side} ${l.qty * 100} shares`
    : `${l.side} ${l.qty} ${l.strike} ${l.type} ${l.expiry}`).join('; ')],
  ['underlying at open', t => t.underlying_open], ['underlying at close', t => t.underlying_close],
  ['net paid at open', t => t.net_open], ['value at close', t => t.net_close], ['costs', t => t.costs],
  ['dividends', t => t.dividends ?? 0], ['pnl', t => t.pnl], ['pnl % of capital', t => t.pnl_pct],
  ...['delta', 'gamma', 'theta', 'vega', 'other'].map(k => [`pnl from ${k}`, t => t.attribution ? t.attribution[k] : null]),
]
const OB_EQUITY_COLS = [
  ['date', e => e.date], ['account value', e => e.value], ['buy and hold', e => e.benchmark], ['implied vol %', e => e.iv],
  ['delta $ per 1%', e => e.delta], ['theta $ per day', e => e.theta], ['vega $ per vol point', e => e.vega],
]

if (typeof module !== 'undefined') {
  module.exports = { obParseImport, obImportSummary, obShareQuery, obParseSettings, obLegToken, obCsv, obCsvCell,
    OB_TRADE_COLS, OB_EQUITY_COLS }
}
