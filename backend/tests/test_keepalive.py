"""Tests for the self-keepalive window in app.py.

WHY THIS MATTERS MORE THAN IT LOOKS
Render's 750 free instance-hours are per workspace, shared by every free
service, and exhausting them suspends all of them. An unwindowed self-ping
holds this service awake ~730 h/month -- the whole allowance -- which has
already taken the sibling backends down together once. These tests pin the
window that keeps the monthly cost near 165 h instead.

The window is defined in America/New_York, so the tests build aware datetimes
in that zone; a naive UTC comparison would drift by an hour twice a year.
"""

import os
from datetime import datetime
from unittest.mock import patch

import pytest
import pytz

from app import (
    KEEPALIVE_END_MIN,
    KEEPALIVE_START_MIN,
    _within_keepalive_window,
)

ET = pytz.timezone('America/New_York')


def et(year, month, day, hour, minute=0):
    """An aware America/New_York datetime with the right DST offset."""
    return ET.localize(datetime(year, month, day, hour, minute))


# 2026-10-20 is a Tuesday (EDT); 2026-01-20 a Tuesday (EST);
# 2026-10-24/25 a Saturday/Sunday.
@pytest.mark.parametrize("when,expected", [
    (et(2026, 10, 20, 8, 59), False),  # just before the warm-up margin
    (et(2026, 10, 20, 9, 0),  True),   # window opens
    (et(2026, 10, 20, 9, 30), True),   # market open
    (et(2026, 10, 20, 12, 0), True),   # midday
    (et(2026, 10, 20, 16, 0), True),   # market close
    (et(2026, 10, 20, 16, 29), True),  # last minute inside
    (et(2026, 10, 20, 16, 30), False),  # window shuts
    (et(2026, 10, 20, 23, 0), False),  # overnight — the expensive case
    (et(2026, 10, 20, 3, 0),  False),  # small hours
])
def test_window_boundaries_on_a_weekday(when, expected):
    assert _within_keepalive_window(when) is expected


@pytest.mark.parametrize("when", [
    et(2026, 10, 24, 12, 0),   # Saturday midday
    et(2026, 10, 25, 12, 0),   # Sunday midday
    et(2026, 10, 24, 9, 30),   # Saturday at what would be the open
])
def test_weekends_are_always_closed(when):
    assert _within_keepalive_window(when) is False


def test_window_is_wall_clock_in_both_dst_halves():
    """09:30 ET must be inside the window in EDT and in EST alike.

    The window is compared in local wall-clock time, so the summer/winter
    offset change must not move it. A UTC-based comparison would fail one of
    these two.
    """
    summer = et(2026, 7, 21, 9, 30)   # EDT, UTC-4
    winter = et(2026, 1, 20, 9, 30)   # EST, UTC-5
    assert summer.utcoffset() != winter.utcoffset(), "fixture must span DST"
    assert _within_keepalive_window(summer) is True
    assert _within_keepalive_window(winter) is True


def test_override_forces_always_on():
    """KEEPALIVE_ALWAYS is the escape hatch for a paid instance."""
    overnight = et(2026, 10, 20, 23, 0)
    with patch.dict(os.environ, {'KEEPALIVE_ALWAYS': '1'}):
        assert _within_keepalive_window(overnight) is True
    with patch.dict(os.environ, {'KEEPALIVE_ALWAYS': 'true'}):
        assert _within_keepalive_window(overnight) is True


def test_override_off_by_default_and_ignores_junk():
    overnight = et(2026, 10, 20, 23, 0)
    for value in ('', '0', 'false', 'no', 'maybe'):
        with patch.dict(os.environ, {'KEEPALIVE_ALWAYS': value}):
            assert _within_keepalive_window(overnight) is False


def test_window_costs_well_under_the_free_allowance():
    """The whole point: the window has to fit inside 750 h/month.

    ~21.7 weekdays a month at the window's width, with headroom left for the
    other three free backends.
    """
    hours_per_day = (KEEPALIVE_END_MIN - KEEPALIVE_START_MIN) / 60
    monthly = hours_per_day * 21.7
    assert hours_per_day == pytest.approx(7.5)
    assert monthly < 200, f"{monthly:.0f} h/month is too close to the cap"
    assert 750 - monthly > 500, "too little left for the sibling services"


def test_defaults_to_now_when_called_with_no_argument():
    """The ping loop calls this with no argument."""
    assert isinstance(_within_keepalive_window(), bool)
