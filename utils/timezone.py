"""
timezone.py — One time zone for every timestamp the tool writes.

Log lines, run and log file names and the estimated finish time of a training
run use ``TIMEZONE`` (Europe/Berlin: CET in winter, CEST in summer), no matter
how the machine is set up — cloud machines and Docker containers usually run
in UTC. Change ``TIMEZONE`` here to move the whole tool to another zone.

The zone comes from the IANA tz database: on Linux the system's tzdata (in the
Docker image via the ``tzdata`` package), elsewhere the ``tzdata`` pip package.
Without either, the system time zone is used and a warning is printed.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIMEZONE = "Europe/Berlin"

try:
    LOCAL_TZ: Optional[tzinfo] = ZoneInfo(TIMEZONE)
except ZoneInfoNotFoundError:
    LOCAL_TZ = None     # datetime.now(None) -> system time zone
    print(f"Warning: time zone {TIMEZONE} not found (no tz database; "
          "pip install tzdata) — timestamps use the system time zone", file=sys.stderr)


def now() -> datetime:
    """Current time in ``TIMEZONE``."""
    return datetime.now(LOCAL_TZ)


def from_timestamp(timestamp: float) -> datetime:
    """POSIX timestamp (``time.time()``, file mtime) in ``TIMEZONE``."""
    return datetime.fromtimestamp(timestamp, LOCAL_TZ)


class Formatter(logging.Formatter):
    """``logging.Formatter`` whose ``%(asctime)s`` is in ``TIMEZONE``."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        moment = from_timestamp(record.created)
        return moment.strftime(datefmt or "%Y-%m-%d %H:%M:%S")
