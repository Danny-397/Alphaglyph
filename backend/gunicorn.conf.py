# Gunicorn production configuration for AlphaGlyph
#
# workers = 1 on purpose.
# The rate limiter uses in-process memory storage and the keep-warm self-ping
# runs in a single background thread — one worker keeps both consistent and is
# plenty for this stateless API (threads handle concurrent requests).
#
# Render start command: gunicorn app:app
# (gunicorn auto-discovers this file when it's in the working directory)

import os

workers   = 1      # one process only — see note above
threads   = 4      # handle concurrent API requests within the single process
timeout   = 120    # backtests can take 60-90s with yfinance downloads
keepalive = 5
# Render passes the assigned port via $PORT — must be respected or the
# health check fails and the service never comes up.
bind      = f'0.0.0.0:{os.getenv("PORT", "5000")}'
accesslog = '-'   # log to stdout (captured by Render)
errorlog  = '-'
loglevel  = 'info'
