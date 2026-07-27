"""One schema for every experiment result.

Before this module, each experiment hand-rolled its own `json.dump`: three
different filename conventions and three different `config` key sets, none of
which recorded the git commit or the full parameter set. Stage 5 has to do a
Pareto analysis over 14 configurations x 3 models x 4 ε_total values, so reading
those results uniformly matters more than the convenience of writing them ad hoc.

Writing goes through `write()`; reading goes through `load_all()`. Because the
parameter snapshot comes from `ExperimentConfig.to_dict()`, an experiment cannot
forget to record a parameter -- it records all of them or none.

Schema (SCHEMA_VERSION 1):

    {
      "schema": 1,
      "name": "stage2_temperature_sweep",
      "created_at": "2026-07-25T12:34:56+00:00",
      "git_commit": "cec883d" | null,
      "git_dirty": true | false,
      "params": { ...every ExperimentConfig field... },
      "metrics": { ...experiment-defined summary numbers... },
      "per_item": [ ... optional per-query rows ... ]
    }

`metrics` and `per_item` are deliberately free-form: what a run measures is the
experiment's business, but how it is identified and parameterised is not.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths

SCHEMA_VERSION = 1


def _git(*args: str) -> str | None:
    """Run a git command in the repo, returning None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_commit() -> str | None:
    return _git("rev-parse", "--short", "HEAD")


def git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


@dataclass(frozen=True)
class RunRecord:
    """A loaded result file."""

    name: str
    created_at: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    per_item: list[dict[str, Any]]
    git_commit: str | None = None
    git_dirty: bool | None = None
    schema: int = SCHEMA_VERSION
    path: Path | None = None

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def metric(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)


def write(
    name: str,
    config: Any,
    metrics: dict[str, Any],
    per_item: list[dict[str, Any]] | None = None,
    *,
    filename: str | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Write one run to `<results>/<filename or name>.json` and return its path.

    config: an ExperimentConfig (anything exposing `to_dict()`), or a plain dict.
    metrics: the run's summary numbers.
    per_item: optional per-query detail rows.
    """
    params = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    payload = {
        "schema": SCHEMA_VERSION,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "params": params,
        "metrics": metrics,
        "per_item": per_item or [],
    }
    directory = Path(results_dir) if results_dir else paths.results_dir()
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{filename or name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def load(path: Path | str) -> RunRecord:
    """Load one result file written by `write()`."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if "schema" not in raw:
        raise ValueError(
            f"{path.name} predates the RunRecord schema (see results/archive/README.md); "
            "re-run it rather than parsing it here."
        )
    return RunRecord(
        name=raw["name"],
        created_at=raw["created_at"],
        params=raw.get("params", {}),
        metrics=raw.get("metrics", {}),
        per_item=raw.get("per_item", []),
        git_commit=raw.get("git_commit"),
        git_dirty=raw.get("git_dirty"),
        schema=raw["schema"],
        path=path,
    )


def load_all(results_dir: Path | None = None) -> list[RunRecord]:
    """Load every schema-carrying result file, newest first.

    Files without a schema (the pre-restructure archive) are skipped rather than
    guessed at, so Stage 5 never silently mixes reconstructed data with recorded
    data. `results/archive/` is skipped entirely.
    """
    directory = Path(results_dir) if results_dir else paths.RESULTS_DIR
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(load(path))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda r: r.created_at, reverse=True)
