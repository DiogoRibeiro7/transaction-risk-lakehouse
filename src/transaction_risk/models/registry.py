"""Local JSON Lines model registry."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "models/registry.jsonl"


@dataclass(frozen=True)
class ModelMetric:
    """A single named evaluation metric for a registered model."""

    name: str
    value: float


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Metadata describing one registered model version."""

    version: int
    model_path: str
    model_type: str
    feature_table_path: str
    threshold: float
    registered_at: str
    metrics: list[ModelMetric] = field(default_factory=list)
    notes: str | None = None

    def to_json(self) -> str:
        """Serialize the entry as a single JSON line."""
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict) -> ModelRegistryEntry:
        """Build an entry from a parsed registry record."""
        metrics = [ModelMetric(**metric) for metric in payload.get("metrics", [])]
        return cls(
            version=int(payload["version"]),
            model_path=str(payload["model_path"]),
            model_type=str(payload["model_type"]),
            feature_table_path=str(payload["feature_table_path"]),
            threshold=float(payload["threshold"]),
            registered_at=str(payload["registered_at"]),
            metrics=metrics,
            notes=payload.get("notes"),
        )


def load_registry(registry_path: str | Path) -> list[ModelRegistryEntry]:
    """Load all registry entries, oldest first."""
    path = Path(registry_path)
    if not path.exists():
        return []

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            entries.append(ModelRegistryEntry.from_dict(json.loads(stripped)))
    return entries


def register_model(
    model_path: str | Path,
    registry_path: str | Path,
    metrics: dict[str, float],
    feature_table_path: str | Path,
    model_type: str,
    threshold: float,
    notes: str | None = None,
) -> ModelRegistryEntry:
    """Append a new model version entry to the registry file."""
    existing_entries = load_registry(registry_path)
    next_version = max((entry.version for entry in existing_entries), default=0) + 1

    entry = ModelRegistryEntry(
        version=next_version,
        model_path=str(model_path),
        model_type=model_type,
        feature_table_path=str(feature_table_path),
        threshold=float(threshold),
        registered_at=datetime.now(timezone.utc).isoformat(),
        metrics=[ModelMetric(name=name, value=float(value)) for name, value in metrics.items()],
        notes=notes,
    )

    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as registry_file:
        registry_file.write(entry.to_json() + "\n")
    logger.info("Registered model version %d at %s", entry.version, registry_path)
    return entry


def get_latest_model(registry_path: str | Path) -> ModelRegistryEntry | None:
    """Return the registry entry with the highest version, or None when empty."""
    entries = load_registry(registry_path)
    if not entries:
        return None
    return max(entries, key=lambda entry: entry.version)
