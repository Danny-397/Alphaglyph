/* Backtest page: an animated example of a finished backtest, shown in the
   results area before the first run so a new visitor can see what they will
   get. The numbers are made up (a seeded random walk) and the figure says so;
   every figure shown is computed from that series, so they agree with the
   chart. Nothing here calls the backend.

   Timeline (seconds): 0-2.8 the equity lines draw and the numbers run with
   them, trade dots land as the line passes; 3.0 the verdict; 3.4-4.2 the
   three tests tick in. Reduced motion draws the finished state at once. */
(function () {
  'use strict'
  var fig = document.getElementById('bt-demo')
  var canvas = document.getElementById('bt-demo-chart')
  if (!fig || !canvas || !canvas.getContext) return

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  var DRAW = 2.8, VERDICT = 3.0, TEST0 = 3.4, TEST_GAP = 0.4
  var END = TEST0 + 2 * TEST_GAP + 0.4

  // ── the example series: seeded, so it is the same for every visitor ──
  function rng(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed)
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
      return ((t ^ t >>> 14) >>> 0) / 4294967296
    }
  }
  function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()) }

  var N = 252, CAP = 100000
  var r = rng(20261637)
  var spy = [CAP], strat = [CAP]
  for (var i = 1; i < N; i++) {
    var mkt = 0.00035 + 0.0095 * gauss(r)
    spy.push(spy[i - 1] * (1 + mkt))
    strat.push(strat[i - 1] * (1 + 0.00025 + 0.8 * mkt + 0.0062 * gauss(r)))
  }

  // trades: buy at the low and sell at the high of each 30-day block
  var trades = []
  for (var b = 0; b + 30 <= N; b += 32) {
    var lo = b, hi = b
    for (var k = b; k < b + 30; k++) { if (strat[k] < strat[lo]) lo = k; if (strat[k] > strat[hi]) hi = k }
    if (hi > lo + 3) { trades.push({ i: lo, buy: true }); trades.push({ i: hi, buy: false }) }
  }

  function stats(upto) {
    var last = Math.max(1, upto)
    var ret = strat[last] / CAP - 1, peak = CAP, dd = 0, rs = []
    for (var j = 1; j <= last; j++) {
      peak = Math.max(peak, strat[j]); dd = Math.max(dd, 1 - strat[j] / peak)
      rs.push(strat[j] / strat[j - 1] - 1)
    }
    var m = rs.reduce(function (a, x) { return a + x }, 0) / rs.length
    var sd = Math.sqrt(rs.reduce(function (a, x) { return a + (x - m) * (x - m) }, 0) / Math.max(1, rs.length - 1))
    return { ret: ret, value: strat[last], sharpe: sd ? (m / sd) * Math.sqrt(252) : 0, dd: dd,
             spy: spy[last] / CAP - 1 }
  }

  var MINUS = '−'
  function pct(v) { return (v >= 0 ? '+' : MINUS) + Math.abs(v * 100).toFixed(1) + '%' }
  function money(v) { return '$' + Math.round(v).toLocaleString('en-US') }
  var out = {
    ret: document.getElementById('bt-demo-ret'), value: document.getElementById('bt-demo-value'),
    sharpe: document.getElementById('bt-demo-sharpe'), dd: document.getElementById('bt-demo-dd'),
  }
  function writeStats(s) {
    out.ret.textContent = pct(s.ret)
    out.ret.className = 'stat-value ' + (s.ret >= 0 ? 'positive' : 'negative')
    out.value.textContent = money(s.value)
    out.sharpe.textContent = s.sharpe.toFixed(2)
    out.dd.textContent = MINUS + (s.dd * 100).toFixed(1) + '%'
  }

  // ── drawing ──
  var css = getComputedStyle(document.documentElement)
  function v(name, fb) { return (css.getPropertyValue(name) || '').trim() || fb }
  var GREEN = v('--accent', '#30d158'), GREY = v('--faint', '#7f8d85'), TEXT = v('--text', '#e6ece8'),
      GRID = 'rgba(214,236,224,0.07)', BG = v('--surface', '#141c18')

  function draw(t) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2)
    var W = canvas.clientWidth, H = canvas.clientHeight
    if (!W || !H) return
    if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr) }
    var ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, W, H)

    var lo = Infinity, hi = -Infinity
    for (var j = 0; j < N; j++) { lo = Math.min(lo, spy[j], strat[j]); hi = Math.max(hi, spy[j], strat[j]) }
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad
    var L = 52, R = 10, T = 10, B = 24
    var X = function (k) { return L + (k / (N - 1)) * (W - L - R) }
    var Y = function (val) { return T + (1 - (val - lo) / (hi - lo)) * (H - T - B) }

    // grid + axis labels
    ctx.font = '11px Geist, Inter, system-ui, sans-serif'
    ctx.fillStyle = GREY; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    var step = 10000, first = Math.ceil(lo / step) * step
    for (var g = first; g <= hi; g += step) {
      ctx.strokeStyle = GRID; ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(L, Y(g)); ctx.lineTo(W - R, Y(g)); ctx.stroke()
      ctx.fillText('$' + Math.round(g / 1000) + 'k', L - 8, Y(g))
    }
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
    ctx.fillText('Jan', L, H - 6)
    ctx.textAlign = 'right'; ctx.fillText('Dec', W - R, H - 6)

    var p = Math.max(0, Math.min(1, t / DRAW))
    var e = 1 - Math.pow(1 - p, 2)
    var head = Math.max(1, Math.round(e * (N - 1)))

    function line(arr, color, width) {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'
      ctx.beginPath(); ctx.moveTo(X(0), Y(arr[0]))
      for (var q = 1; q <= head; q++) ctx.lineTo(X(q), Y(arr[q]))
      ctx.stroke()
    }
    line(spy, GREY, 1.5)
    line(strat, GREEN, 2.25)

    // trade dots land as the line passes them
    trades.forEach(function (tr) {
      if (tr.i > head) return
      var age = (head - tr.i) / (N * 0.06)
      var s = Math.min(1, age)
      var rad = (tr.buy ? 4 : 4) * (0.4 + 0.6 * s)
      ctx.globalAlpha = Math.min(1, age * 1.5)
      ctx.beginPath(); ctx.arc(X(tr.i), Y(strat[tr.i]), rad, 0, Math.PI * 2)
      if (tr.buy) { ctx.fillStyle = GREEN; ctx.fill() }
      else { ctx.fillStyle = BG; ctx.fill(); ctx.strokeStyle = TEXT; ctx.lineWidth = 1.5; ctx.stroke() }
      ctx.globalAlpha = 1
    })

    // the running head of the strategy line
    if (p < 1) {
      ctx.fillStyle = GREEN
      ctx.beginPath(); ctx.arc(X(head), Y(strat[head]), 3.5, 0, Math.PI * 2); ctx.fill()
    }
    writeStats(stats(head))
  }

  // ── the verdict and the captions follow the chart ──
  var verdict = document.getElementById('bt-demo-verdict')
  var tests = fig.querySelectorAll('.bt-demo-test')
  var steps = fig.querySelectorAll('.bt-demo-steps li')
  function stage(t) {
    verdict.classList.toggle('is-on', t >= VERDICT)
    tests.forEach(function (el, n) { el.classList.toggle('is-on', t >= TEST0 + n * TEST_GAP) })
    var active = t < DRAW * 0.45 ? 0 : t < VERDICT ? 1 : 2
    steps.forEach(function (el, n) {
      el.classList.toggle('is-active', t < END && n === active)
      el.classList.toggle('is-done', t >= END || n < active)
    })
  }

  var now = END, raf = 0
  function frame(t) { now = t; draw(t); stage(t) }
  function play() {
    cancelAnimationFrame(raf)
    if (reduce) { frame(END); return }
    var t0 = performance.now()
    ;(function tick(ts) {
      var t = Math.min(END, (ts - t0) / 1000)
      frame(t)
      if (t < END) raf = requestAnimationFrame(tick)
    })(t0)
  }

  document.getElementById('bt-demo-replay').addEventListener('click', play)
  var rz
  window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(function () { draw(now) }, 120) })

  if (reduce || !('IntersectionObserver' in window)) { frame(END) }
  else {
    frame(0)
    var io = new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting) { io.disconnect(); play() }
    }, { threshold: 0.35 })
    io.observe(canvas)
  }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { draw(now) })
})()
