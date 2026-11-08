// node --test tests/js/
const test = require('node:test')
const assert = require('node:assert/strict')
const { obParseImport, obImportSummary } = require('../../frontend/options-import.js')

test('a Tradeski link becomes custom legs with percent strikes', () => {
  const imp = obParseImport('?ticker=aapl&dte=45&name=Bull%20call%20spread&legs=long_call_1_5|short_call_1_10&from=tradeski&run=1')
  assert.equal(imp.ticker, 'AAPL')
  assert.equal(imp.dte, 45)
  assert.equal(imp.name, 'Bull call spread')
  assert.equal(imp.from, 'Tradeski')
  assert.equal(imp.run, true)
  assert.deepEqual(imp.legs, [
    { type: 'call', side: 'long', qty: 1, strike: { mode: 'pct', value: 5 } },
    { type: 'call', side: 'short', qty: 1, strike: { mode: 'pct', value: 10 } },
  ])
  assert.deepEqual(imp.notes, [])
  assert.match(obImportSummary(imp), /^Imported from Tradeski: Bull call spread on AAPL, 45 days/)
})

test('stock legs, at-the-money strikes and negative percents', () => {
  const imp = obParseImport('?ticker=MSFT&dte=30&legs=long_stock_1|long_put_1_-5.04|short_call_1_0')
  assert.deepEqual(imp.legs, [
    { type: 'stock', side: 'long', qty: 1 },
    { type: 'put', side: 'long', qty: 1, strike: { mode: 'pct', value: -5 } },
    { type: 'call', side: 'short', qty: 1, strike: { mode: 'atm' } },
  ])
  assert.equal(imp.from, null)
  assert.equal(imp.run, false)
})

test('out-of-range values are clamped and explained', () => {
  const imp = obParseImport('?ticker=SPY&dte=400&legs=short_put_25_-70|short_stock_1|bogus|long_call_1_2')
  assert.equal(imp.dte, 180)
  assert.equal(imp.legs.length, 2)
  assert.deepEqual(imp.legs[0], { type: 'put', side: 'short', qty: 10, strike: { mode: 'pct', value: -50 } })
  assert.equal(imp.notes.length, 5)
  assert.ok(imp.notes.some(n => n.includes('Short stock')))
  assert.ok(imp.notes.some(n => n.includes('400 days')))
})

test('no legs, no option legs, or a bad ticker', () => {
  assert.equal(obParseImport('?ticker=SPY'), null)
  assert.equal(obParseImport(''), null)
  assert.match(obParseImport('?legs=long_stock_1').error, /no option legs/)
  assert.equal(obParseImport('?ticker=<script>&legs=long_call_1_0').ticker, 'SPY')
  assert.equal(obParseImport('?ticker=BRK.B&legs=long_call_1_0').ticker, 'BRK.B')
})

test('only six legs are kept', () => {
  const legs = Array.from({ length: 8 }, (_, i) => `long_call_1_${i}`).join('|')
  const imp = obParseImport(`?legs=${legs}`)
  assert.equal(imp.legs.length, 6)
  assert.ok(imp.notes.some(n => n.includes('first six')))
})
