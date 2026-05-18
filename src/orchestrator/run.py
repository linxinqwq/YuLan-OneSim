from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .decomposer import decompose_requirement, heuristic_rumor_subtasks
from .glue_generator import generate_glue_code
from .glue_templates import DEFAULT_RUMOR_CHAIN
from .manifest_builder import write_manifest
from .matcher import match_subtasks_to_simulators
from .paths import default_manifest_path, project_root


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        write_manifest(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline(
    user_requirement: str,
    manifest: List[Dict[str, Any]],
    *,
    simple_rumor: bool = False,
    fixed_simulators: List[str] | None = None,
    plan_path: Path | None = None,
) -> Dict[str, Any]:
    plan = heuristic_rumor_subtasks() if simple_rumor else decompose_requirement(user_requirement)
    subtasks = plan["subtasks"]

    if fixed_simulators is None and simple_rumor:
        fixed_simulators = list(DEFAULT_RUMOR_CHAIN)

    if fixed_simulators:
        mmap = {m["id"]: m for m in manifest}
        matches = []
        for i, sid in enumerate(fixed_simulators):
            st = subtasks[i] if i < len(subtasks) else {
                "id": f"t{i+1}",
                "title": f"子任务{i+1}",
                "description": "",
            }
            sim = mmap.get(sid)
            matches.append(
                {
                    "subtask": st,
                    "matched_simulator": sim,
                    "match_score": 1.0 if sim else 0.0,
                    "needs_generated_stub": sim is None,
                }
            )
    else:
        matches = match_subtasks_to_simulators(subtasks, manifest)

    code = generate_glue_code(user_requirement, matches, manifest)
    report = {
        "subtasks": subtasks,
        "matches": [
            {
                "subtask": m["subtask"],
                "simulator_id": (m.get("matched_simulator") or {}).get("id"),
                "simulator_path": (m.get("matched_simulator") or {}).get("simenv_path"),
                "match_score": m.get("match_score"),
                "needs_generated_stub": m.get("needs_generated_stub"),
            }
            for m in matches
        ],
        "code": code,
    }
    if plan_path:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"code": code, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser(description="YuLan 智能编排器（谣言传播与恐慌扩散）")
    ap.add_argument("--requirement", type=str, default="")
    ap.add_argument("--requirement-file", type=str, default="")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--manifest", default=str(default_manifest_path()))
    ap.add_argument("--plan-json", default="")
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument(
        "--simple-rumor",
        action="store_true",
        help="固定谣言传播子任务骨架，并默认绑定三场景",
    )
    ap.add_argument("--fixed-simulators", nargs="*", default=None)
    ap.add_argument("--run-after", action="store_true")
    args = ap.parse_args()

    req = args.requirement
    if args.requirement_file:
        req = Path(args.requirement_file).read_text(encoding="utf-8")
    if not req.strip():
        ap.error("需要 --requirement 或 --requirement-file")

    mp = Path(args.manifest)
    if args.rebuild_manifest or not mp.exists():
        write_manifest(mp)
    manifest = load_manifest(mp)

    result = run_pipeline(
        req,
        manifest,
        simple_rumor=args.simple_rumor,
        fixed_simulators=args.fixed_simulators,
        plan_path=Path(args.plan_json) if args.plan_json else None,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result["code"], encoding="utf-8")
    print("已写入:", out.resolve())

    if args.run_after:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root() / "src")
        subprocess.run([sys.executable, str(out.resolve())], cwd=str(project_root()), env=env)


if __name__ == "__main__":
    main()
