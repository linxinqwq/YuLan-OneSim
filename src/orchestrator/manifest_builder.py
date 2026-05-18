from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .paths import default_manifest_path, envs_dir, project_root


def build_manifest(envs_root: Path | None = None) -> List[Dict[str, Any]]:
    root = envs_root or envs_dir()
    repo = project_root()
    entries: List[Dict[str, Any]] = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_info = scene_dir / "scene_info.json"
        simenv = scene_dir / "code" / "SimEnv.py"
        if not scene_info.exists() or not simenv.exists():
            continue
        raw = json.loads(scene_info.read_text(encoding="utf-8"))
        sid = raw.get("scene_name") or scene_dir.name
        odd = raw.get("odd_protocol") or {}
        ov = odd.get("overview") or {}
        desc = " ".join(str(ov.get(k, "")) for k in ("system_goal", "agent_types", "environment_description") if ov.get(k))
        if not desc:
            desc = f"YuLan scenario {sid}"
        metrics = [
            {"id": m.get("id"), "name": m.get("name"), "description": m.get("description")}
            for m in (raw.get("metrics") or [])
        ]
        entries.append(
            {
                "id": sid,
                "name": f"{raw.get('domain', 'General')}: {sid}",
                "category": raw.get("domain", "General"),
                "description": desc,
                "inputs_hint": list((raw.get("agent_types") or {}).keys()),
                "outputs_metrics": metrics,
                "scene_info_path": str(scene_info.relative_to(repo)).replace("\\", "/"),
                "simenv_path": str(simenv.relative_to(repo)).replace("\\", "/"),
            }
        )
    entries.sort(key=lambda x: x["id"])
    return entries


def write_manifest(out_path: Path | None = None) -> Path:
    path = out_path or default_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default=str(default_manifest_path()))
    args = p.parse_args()
    out = write_manifest(Path(args.output))
    n = len(json.loads(out.read_text(encoding="utf-8")))
    print(f"Wrote {out} ({n} simulators)")


if __name__ == "__main__":
    main()
