"""一键演示：谣言传播与恐慌扩散 — 编排、运行、生成论文插图。"""
from __future__ import annotations

import os
import subprocess
import sys

from .glue_templates import DEFAULT_RUMOR_CHAIN
from .paths import project_root
from .run import load_manifest, run_pipeline


def main() -> None:
    root = project_root()
    req_path = root / "examples" / "requirement_rumor_propagation.txt"
    req = req_path.read_text(encoding="utf-8") if req_path.exists() else "谣言传播与恐慌扩散仿真"

    print("=" * 60)
    print("智能编排器演示 — 谣言传播与恐慌扩散")
    print("=" * 60)

    mp = root / "src" / "orchestrator" / "simulator_manifest.json"
    manifest = load_manifest(mp)
    result = run_pipeline(
        req,
        manifest,
        simple_rumor=True,
        fixed_simulators=list(DEFAULT_RUMOR_CHAIN),
        plan_path=root / "output" / "orchestration_plan.json",
    )

    out_py = root / "output" / "generated_rumor_simulator.py"
    out_py.parent.mkdir(parents=True, exist_ok=True)
    out_py.write_text(result["code"], encoding="utf-8")
    print("生成:", out_py)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run([sys.executable, str(out_py)], cwd=str(root), env=env, check=False)

    fig_dir = root / "docs" / "thesis_figures"
    if fig_dir.exists():
        print("\n论文插图:", fig_dir.resolve())
        for f in sorted(fig_dir.glob("fig_3_*.png")):
            print(" ", f.name)


if __name__ == "__main__":
    main()
