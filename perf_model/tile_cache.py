from __future__ import annotations

from collections import OrderedDict
from typing import Any


class TileLRU:
    """Tile-granularity L2 cache model used by the wave simulator."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.used_bytes = 0
        self.cache: OrderedDict[tuple[Any, ...], int] = OrderedDict()

    def access(self, key: tuple[Any, ...], size: int) -> bool:
        # Keys represent logical A/B tiles rather than raw addresses. This captures
        # reuse across CTAs without needing a full memory trace.
        if key in self.cache:
            self.cache.move_to_end(key)
            return True

        while self.cache and self.used_bytes + size > self.capacity_bytes:
            _, evicted_size = self.cache.popitem(last=False)
            self.used_bytes -= evicted_size

        if size <= self.capacity_bytes:
            self.cache[key] = size
            self.used_bytes += size
        return False
