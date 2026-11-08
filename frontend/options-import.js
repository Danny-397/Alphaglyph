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

if (typeof module !== 'undefined') module.exports = { obParseImport, obImportSummary }
