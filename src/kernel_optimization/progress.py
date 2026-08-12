"""Small dependency-free progress reporting for long optimization runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from typing import Any, Dict, Optional, TextIO


class ProgressReporter:
    """Emit concise, flush-on-write progress lines suitable for remote shells."""

    def __init__(self, stream: Optional[TextIO] = None, enabled: bool = True) -> None:
        self.stream = stream or sys.stderr
        self.enabled = enabled

    def emit(self, event: str, message: str, **fields: Any) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        details = ""
        if fields:
            details = " " + json.dumps(fields, sort_keys=True, default=str)
        print(
            "[%s] %-18s %s%s" % (timestamp, event, message, details),
            file=self.stream,
            flush=True,
        )


class NullProgressReporter(ProgressReporter):
    """No-op reporter used by the library unless a CLI opts into output."""

    def __init__(self) -> None:
        super().__init__(enabled=False)

