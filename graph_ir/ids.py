from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


def slugify(text: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "node"


def stable_id(node_type: str, raw_id: str, scene_name: str, scope: str | None = None) -> str:
    scoped = scope or ""
    key = "::".join([scene_name, node_type, scoped, str(raw_id)])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{slugify(node_type)}:{slugify(raw_id)}:{digest}"


@dataclass
class StableIdRegistry:
    scene_name: str
    _cache: dict[tuple[str, str, str], str] = field(default_factory=dict)

    def get(self, node_type: str, raw_id: str, scope: str | None = None) -> str:
        key = (node_type, scope or "", str(raw_id))
        if key not in self._cache:
            self._cache[key] = stable_id(node_type=node_type, raw_id=raw_id, scene_name=self.scene_name, scope=scope)
        return self._cache[key]
