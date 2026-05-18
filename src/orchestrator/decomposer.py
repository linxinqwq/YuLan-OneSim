from __future__ import annotations

import json
from typing import Any, Dict, List

from .llm_client import chat_json, default_model, no_llm_mode


def heuristic_rumor_subtasks() -> Dict[str, Any]:
    """论文目标需求：谣言传播与恐慌扩散。"""
    return {
        "subtasks": [
            {
                "id": "t1",
                "title": "社交网络初始化",
                "description": "创建用户节点与关注/好友关系，确定初始信谣者。",
            },
            {
                "id": "t2",
                "title": "信息传播与级联转发",
                "description": "信谣者向邻居转发谣言；邻居根据观察到的转发量决定是否跟随转发（信息级联）。",
            },
            {
                "id": "t3",
                "title": "恐慌情绪感染",
                "description": "用户与信谣者接触时恐慌值向信谣者靠近，情绪升高提高相信与转发概率。",
            },
            {
                "id": "t4",
                "title": "谣言扩散动力学",
                "description": "记录每轮新增信谣者，刻画扩散过程的时序演化（创新扩散式 S 形过程）。",
            },
            {
                "id": "t5",
                "title": "辟谣干预与统计输出",
                "description": "在指定轮次广播辟谣，部分信谣者恢复；汇总信谣者数、恐慌均值、转发次数并绘图。",
            },
        ]
    }


def _fallback_subtasks(user_requirement: str) -> List[Dict[str, Any]]:
    rumor_keys = ("谣言", "信谣", "转发", "恐慌", "辟谣", "传播", "社交网络")
    if sum(1 for k in rumor_keys if k in user_requirement) >= 2:
        return heuristic_rumor_subtasks()["subtasks"]
    return [
        {"id": "t1", "title": "场景初始化", "description": user_requirement[:400]},
        {"id": "t2", "title": "主体交互", "description": "模拟主体之间的规则化交互"},
        {"id": "t3", "title": "统计与可视化", "description": "记录指标并输出图表"},
    ]


def decompose_requirement(user_requirement: str) -> Dict[str, Any]:
    if no_llm_mode():
        return {"subtasks": _fallback_subtasks(user_requirement)}
    model = default_model()
    system = (
        "你是仿真需求分析专家。将用户的复杂仿真需求拆解为 3～8 个相对独立的子任务。"
        '输出严格 JSON：{"subtasks":[{"id":"t1","title":"...","description":"..."}]}'
    )
    raw = chat_json(model, system, f"用户需求：\n{user_requirement.strip()}")
    data = json.loads(raw)
    if "subtasks" not in data:
        raise ValueError("拆解结果缺少 subtasks")
    return data
