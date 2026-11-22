/* Progress bars for every loading state.

   The backend answers each request in one piece and reports no progress, so
   the percentage here is an estimate from how long that job usually takes
   (data-expect, in seconds). It rises roughly in step with the clock to 90% at the
   expected time, then creeps, and never reaches 100% on its own: the result arriving is
   what ends it. It never moves backwards.

   Each .progress starts when its loading box becomes visible and resets
   when it is hidden again, so no call site has to start or stop it. While
   the free-tier server is waking (app.js shows .conn-banner), the estimate
   is stretched and the line under the bar says so. */
(function () {
  'use strict'
  var COLD_EXTRA = 45          // seconds a cold Render start typically adds
  var state = new WeakMap()    // .progress -> { t0, p, cold }

  function visible(el) { return !!(el.offsetParent || el.getClientRects().length) }
  function coldNow() { return !!document.querySelector('.conn-banner.show') }

  function fmtLeft(s) {
    if (s >= 90) return 'about ' + Math.round(s / 60) + ' min left'
    if (s >= 10) return 'about ' + Math.round(s / 5) * 5 + ' s left'
    return 'a few seconds left'
  }

  function paint(el, st, now) {
    var expect = parseFloat(el.dataset.expect) || 10
    if (coldNow()) st.cold = true
    if (st.cold) expect += COLD_EXTRA
    var t = (now - st.t0) / 1000
    // Close to proportional to elapsed time, 90% at the expected time,
    // then a slow approach to 99%.
    var target = t <= expect
      ? 0.90 * (1 - Math.pow(1 - t / expect, 1.6))
      : 0.90 + 0.09 * (1 - Math.exp(-(t - expect) / (expect * 1.5)))
    st.p = Math.max(st.p, Math.min(0.99, target))
    var pct = Math.floor(st.p * 100)

    el.querySelector('.progress-fill').style.transform = 'scaleX(' + st.p.toFixed(4) + ')'
    el.querySelector('.progress-pct').textContent = pct + '%'
    el.setAttribute('aria-valuenow', String(pct))
    var eta = el.querySelector('.progress-eta')
    if (eta) {
      eta.textContent = coldNow()
        ? 'Waking the free-tier server, this first run can take up to a minute'
        : t < expect ? fmtLeft(expect - t) : 'Taking a little longer than usual'
    }
  }

  function reset(el) {
    el.querySelector('.progress-fill').style.transform = 'scaleX(0)'
    el.querySelector('.progress-pct').textContent = '0%'
    el.setAttribute('aria-valuenow', '0')
    var eta = el.querySelector('.progress-eta')
    if (eta) eta.textContent = ''
  }

  var ticking = false
  function tick(now) {
    var any = false
    document.querySelectorAll('.progress').forEach(function (el) {
      var st = state.get(el)
      if (visible(el)) {
        if (!st) { st = { t0: now, p: 0, cold: false }; state.set(el, st) }
        paint(el, st, now)
        any = true
      } else if (st) {
        state.delete(el)
        reset(el)
      }
    })
    ticking = any
    if (any) requestAnimationFrame(tick)
  }
  function kick() { if (!ticking) { ticking = true; requestAnimationFrame(tick) } }

  // A loading box appearing is an attribute or child change somewhere on the page.
  new MutationObserver(kick).observe(document.body, {
    subtree: true, childList: true, attributes: true, attributeFilter: ['hidden', 'class', 'style'],
  })
  kick()
})()
