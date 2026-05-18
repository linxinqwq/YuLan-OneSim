from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def envs_dir() -> Path:
    return project_root() / "src" / "envs"


def default_manifest_path() -> Path:
    return project_root() / "src" / "orchestrator" / "simulator_manifest.json"
