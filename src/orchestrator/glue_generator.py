from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .glue_templates import (
    DEFAULT_RUMOR_CHAIN,
    is_rumor_propagation_requirement,
    rumor_propagation_glue_code,
)
from .llm_client import chat_text, default_model, no_llm_mode
from .paths import project_root


def read_text_limited(rel_path: str, max_chars: int = 6000) -> str:
    p = project_root() / rel_path.replace("\\", "/")
    if not p.exists():
        return f"# missing: {rel_path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + ("\n# ... truncated" if len(text) > max_chars else "")


def extract_python_from_markdown(raw: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw)
    return m.group(1).strip() if m else raw.strip()


def build_glue_prompt(
    user_requirement: str,
    matches: List[Dict[str, Any]],
) -> str:
    blocks = []
    for m in matches:
        st = m["subtask"]
        line = f"### {st.get('id')} {st.get('title')}\n{st.get('description','')}\n"
        if m.get("needs_generated_stub"):
            line += "（需生成简易代码：初始化/统计/辟谣逻辑）\n"
        else:
            sim = m["matched_simulator"]
            line += f"场景ID: {sim['id']}\n{read_text_limited(sim.get('simenv_path',''))[:1500]}\n"
        blocks.append(line)
    return f"""
根据用户需求编写可运行 Python 3.10+ 脚本。
类名 RumorPropagationSimulator；社交网络谣言传播+恐慌感染+辟谣；不 import onesim。
输出每轮：信谣者数量、恐慌均值、转发次数。
matplotlib 保存到 docs/thesis_figures/：
  fig_3_1_believers_over_time.png
  fig_3_2_mean_panic_over_time.png
  fig_3_3_forward_count_over_time.png

用户需求：
{user_requirement}

{chr(10).join(blocks)}

只输出 ```python ... ``` 代码块。
"""


def generate_glue_code(
    user_requirement: str,
    matches: List[Dict[str, Any]],
    manifest: List[Dict[str, Any]],
) -> str:
    ids = [
        m["matched_simulator"]["id"]
        for m in matches
        if m.get("matched_simulator")
    ]
    if not ids:
        ids = list(DEFAULT_RUMOR_CHAIN)
    req_short = user_requirement.replace('"', "'")[:500]

    if no_llm_mode() or is_rumor_propagation_requirement(user_requirement):
        return rumor_propagation_glue_code(ids, req_short)

    raw = chat_text(default_model(), "只输出 Python 代码块", build_glue_prompt(user_requirement, matches))
    return extract_python_from_markdown(raw)
