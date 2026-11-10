// node --test tests/js/*.test.cjs
const test = require('node:test')
const assert = require('node:assert/strict')
const m = require('../../frontend/options-import.js')

const body = {
  ticker: 'QQQ', preset: 'iron_condor', start_date: '2021-01-04', end_date: '2025-12-31', initial_capital: 50000,
  dte: 45, alloc_pct: 15, exit_dte: 7, profit_take: 50, stop_loss: null, commission: 0.5, spread_pct: null,
  skew: false, expiries: 'monthly', entry: { vix_min: 18, trend: 'above' },
}

test('a share link round-trips every setting of a preset backtest', () => {
  const s = m.obParseSettings('?' + m.obShareQuery(body))
  assert.equal(s.ticker, 'QQQ')
  assert.equal(s.preset, 'iron_condor')
  assert.equal(s.dte, 45)
  assert.equal(s.start_date, '2021-01-04')
  assert.equal(s.end_date, '2025-12-31')
  assert.equal(s.initial_capital, 50000)
  assert.equal(s.alloc_pct, 15)
  assert.equal(s.exit_dte, 7)
  assert.equal(s.profit_take, 50)
  assert.equal('stop_loss' in s, false)
  assert.equal(s.commission, 0.5)
  assert.equal(s.skew, false)
  assert.equal(s.expiries, 'monthly')
  assert.deepEqual(s.entry, { vix_min: 18, trend: 'above' })
  assert.equal(s.run, true)
  assert.equal(m.obParseImport('?' + m.obShareQuery(body)), null)   // a preset link carries no legs
})

test('custom legs of every strike mode survive the round trip', () => {
  const legs = [
    { type: 'stock', side: 'long', qty: 1 },
    { type: 'put', side: 'short', qty: 2, strike: { mode: 'delta', value: 0.3 } },
    { type: 'put', side: 'long', qty: 2, strike: { mode: 'offset', leg: 1, value: -5 } },
    { type: 'call', side: 'short', qty: 1, strike: { mode: 'pct', value: 7.5 }, dte: 60 },
    { type: 'call', side: 'long', qty: 1, strike: { mode: 'atm' } },
  ]
  const q = m.obShareQuery({ ticker: 'SPY', legs, name: 'My spread, "v2"', dte: 30 })
  const imp = m.obParseImport('?' + q)
  assert.equal(imp.name, 'My spread, "v2"')
  assert.deepEqual(imp.legs, legs)
  assert.deepEqual(imp.notes, [])
})

test('hostile or malformed settings are dropped', () => {
  const s = m.obParseSettings('?ticker=<b>&preset=../../etc&start=yesterday&alloc=lots&trend=sideways&expiries=daily&dte=900')
  assert.equal(s.ticker, undefined)
  assert.equal(s.preset, undefined)
  assert.equal(s.start_date, undefined)
  assert.equal(s.alloc_pct, undefined)
  assert.equal(s.entry, undefined)
  assert.equal(s.expiries, undefined)
  assert.equal(s.dte, 180)
})

test('CSV cells are quoted and spreadsheet formulas neutralised', () => {
  assert.equal(m.obCsvCell(null), '')
  assert.equal(m.obCsvCell(-12.5), '-12.5')                        // a negative number stays a number
  assert.equal(m.obCsvCell('a,b'), '"a,b"')
  assert.equal(m.obCsvCell('say "hi"'), '"say ""hi"""')
  assert.equal(m.obCsvCell('two\nlines'), '"two\nlines"')
  assert.equal(m.obCsvCell('=HYPERLINK("x")'), '"\'=HYPERLINK(""x"")"')
  assert.equal(m.obCsvCell('@SUM(A1)'), "'@SUM(A1)")
})

test('trade and daily CSVs have a header and one row each', () => {
  const trade = { opened: '2024-01-02', closed: '2024-02-02', reason: 'expiry', units: 2,
    legs: [{ side: 'short', type: 'put', qty: 2, strike: 470, expiry: '2024-02-02' }, { side: 'long', type: 'stock', qty: 1 }],
    underlying_open: 475, underlying_close: 480, net_open: -800, net_close: 0, costs: 3.1, dividends: 0, pnl: 796.9,
    pnl_pct: 0.8, attribution: { delta: 100, gamma: -50, theta: 760, vega: -10, other: 0 } }
  const csv = m.obCsv([trade], m.OB_TRADE_COLS).split('\r\n')
  assert.equal(csv.length, 3)                                       // header, row, trailing newline
  assert.match(csv[0], /^opened,closed,why it closed,units,position/)
  assert.match(csv[1], /^2024-01-02,2024-02-02,expiry,2,short 2 470 put 2024-02-02; long 100 shares,475,480,-800,0,3.1,0,796.9,0.8,100,-50,760,-10,0$/)
  const eq = m.obCsv([{ date: '2024-01-02', value: 100000, benchmark: 100000, iv: 14.2, delta: 0, theta: 0, vega: 0 }], m.OB_EQUITY_COLS)
  assert.equal(eq.split('\r\n')[1], '2024-01-02,100000,100000,14.2,0,0,0')
})

test('at-the-money legs with their own expiry keep it', () => {
  const legs = [{ type: 'call', side: 'short', qty: 1, strike: { mode: 'atm' } },
    { type: 'call', side: 'long', qty: 1, strike: { mode: 'atm' }, dte: 60 }]
  assert.deepEqual(m.obParseImport('?' + m.obShareQuery({ ticker: 'SPY', legs, name: 'Calendar' })).legs, legs)
})

test('the earnings rules round-trip', () => {
  const s = m.obParseSettings('?' + m.obShareQuery({ ...body, entry: { earnings: 'avoid' }, exit_earnings: true }))
  assert.deepEqual(s.entry, { earnings: 'avoid' })
  assert.equal(s.exit_earnings, true)
  const plain = m.obParseSettings('?' + m.obShareQuery({ ...body, entry: { earnings: 'any' }, exit_earnings: false }))
  assert.equal('exit_earnings' in plain, false)
  assert.equal('entry' in plain, false)
})
