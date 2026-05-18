"""
谣言传播与恐慌扩散 — 组合仿真（编排器生成）
匹配 YuLan 场景 ID: ["emotional_contagion_model", "information_cascade_and_silence", "diffusion_of_innovations"]
需求摘要: 请生成一个社交网络谣言传播与恐慌扩散的组合仿真器。网络中有若干用户，最初少数用户接触未经证实的消息并成为信谣者；信谣者以一定概率向邻居转发，传播概率受恐慌情绪影响；恐慌情绪在接触信谣者时升高；若干轮后官方广播辟谣，部分信谣者转为不信谣者。需要输出每轮信谣者数量、恐慌情绪均值、转发次数，并绘制随时间变化的曲线。

"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

MANIFEST_CHAIN = ["emotional_contagion_model", "information_cascade_and_silence", "diffusion_of_innovations"]


def build_random_graph(n: int, avg_degree: float, rng: np.random.RandomState) -> List[List[int]]:
    """社交网络邻接表（无向）。"""
    p = min(avg_degree / max(n - 1, 1), 1.0)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < p:
                adj[i].append(j)
                adj[j].append(i)
    for i in range(n):
        if not adj[i]:
            j = (i + 1) % n
            adj[i].append(j)
            adj[j].append(i)
    return adj


def adapt_cascade_forward(
    believer: np.ndarray,
    adj: List[List[int]],
    base_prob: float,
    panic: np.ndarray,
    trust: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    information_cascade_and_silence：观察邻居中信谣者比例，决定是否转发。
    返回本轮新成为信谣者的掩码及转发次数。
    """
    n = len(believer)
    new_believer = believer.copy()
    forwards = 0
    for i in range(n):
        if believer[i]:
            continue
        neighbors = adj[i]
        if not neighbors:
            continue
        share = float(np.mean([believer[j] for j in neighbors]))
        prob = base_prob * (0.5 + share) * (0.6 + 0.4 * panic[i]) * trust[i]
        prob = float(np.clip(prob, 0.0, 0.95))
        if np.random.rand() < prob:
            new_believer[i] = True
            forwards += 1
    return new_believer, forwards


def adapt_emotional_contagion(
    panic: np.ndarray,
    believer: np.ndarray,
    adj: List[List[int]],
    contagion_coef: float,
) -> np.ndarray:
    """emotional_contagion_model：恐慌情绪向邻居传染。"""
    n = len(panic)
    updated = panic.copy()
    for i in range(n):
        infected_neighbors = [j for j in adj[i] if believer[j]]
        if not infected_neighbors:
            continue
        target = float(np.mean([panic[j] for j in infected_neighbors]))
        updated[i] += contagion_coef * (target - panic[i])
    return np.clip(updated, 0.0, 1.0)


def adapt_diffusion_boost(
    believer: np.ndarray,
    base_prob: float,
    adoption_density: float,
) -> float:
    """diffusion_of_innovations：采纳密度越高，边际传播越强（S 形扩散语义）。"""
    return float(np.clip(base_prob * (1.0 + 0.8 * adoption_density), 0.05, 0.99))


class RumorPropagationSimulator:
    """高层组合仿真器：谣言传播 + 恐慌感染 + 辟谣干预。"""

    def __init__(
        self,
        n_users: int = 100,
        n_rounds: int = 30,
        initial_believers: int = 1,
        base_spread_prob: float = 0.3,
        avg_degree: float = 4.0,
        contagion_coef: float = 0.2,
        debunk_round: int = 10,
        debunk_recovery_prob: float = 0.35,
    ) -> None:
        self.n_users = n_users
        self.n_rounds = n_rounds
        self.initial_believers = initial_believers
        self.base_spread_prob = base_spread_prob
        self.avg_degree = avg_degree
        self.contagion_coef = contagion_coef
        self.debunk_round = debunk_round
        self.debunk_recovery_prob = debunk_recovery_prob
        self.manifest_chain = list(MANIFEST_CHAIN)

    def run(self, seed: int = 42) -> Dict[str, Any]:
        rng = np.random.RandomState(seed)
        n = self.n_users
        adj = build_random_graph(n, self.avg_degree, rng)

        believer = np.zeros(n, dtype=bool)
        init_ids = rng.choice(n, size=min(self.initial_believers, n), replace=False)
        believer[init_ids] = True

        panic = rng.uniform(0.1, 0.3, size=n)
        trust = rng.uniform(0.7, 1.0, size=n)

        hist = {
            "round": [],
            "num_believers": [],
            "mean_panic": [],
            "forward_count": [],
        }

        for t in range(1, self.n_rounds + 1):
            density = float(believer.mean())
            spread_p = adapt_diffusion_boost(believer, self.base_spread_prob, density)

            believer, forwards = adapt_cascade_forward(
                believer, adj, spread_p, panic, trust
            )
            panic = adapt_emotional_contagion(panic, believer, adj, self.contagion_coef)

            if t >= self.debunk_round:
                recover = (rng.rand(n) < self.debunk_recovery_prob) & believer
                believer[recover] = False
                panic[recover] *= 0.5

            hist["round"].append(t)
            hist["num_believers"].append(int(believer.sum()))
            hist["mean_panic"].append(float(panic.mean()))
            hist["forward_count"].append(forwards)

        return {
            "manifest_chain": self.manifest_chain,
            "final_num_believers": int(believer.sum()),
            "peak_num_believers": int(max(hist["num_believers"])),
            "debunk_round": self.debunk_round,
            "history": hist,
        }

    def plot(self, result: Dict[str, Any], out: Path) -> List[Path]:
        if not HAS_MPL:
            return []
        out.mkdir(parents=True, exist_ok=True)
        h = result["history"]
        paths: List[Path] = []

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h["round"], h["num_believers"], "o-", color="#dc2626", linewidth=2)
        ax.axvline(result["debunk_round"], color="#6b7280", linestyle="--", label="辟谣介入")
        ax.set_xlabel("仿真轮次")
        ax.set_ylabel("信谣者数量")
        ax.set_title("信谣者数量随时间变化")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p1 = out / "fig_3_1_believers_over_time.png"
        fig.tight_layout()
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        paths.append(p1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h["round"], h["mean_panic"], "s-", color="#ea580c", linewidth=2)
        ax.axvline(result["debunk_round"], color="#6b7280", linestyle="--", label="辟谣介入")
        ax.set_xlabel("仿真轮次")
        ax.set_ylabel("恐慌情绪均值")
        ax.set_ylim(0, 1)
        ax.set_title("恐慌情绪均值随时间变化")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p2 = out / "fig_3_2_mean_panic_over_time.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        paths.append(p2)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(h["round"], h["forward_count"], color="#2563eb", alpha=0.85)
        ax.axvline(result["debunk_round"], color="#6b7280", linestyle="--", label="辟谣介入")
        ax.set_xlabel("仿真轮次")
        ax.set_ylabel("本轮转发次数")
        ax.set_title("转发行为数量随时间变化")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        p3 = out / "fig_3_3_forward_count_over_time.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        paths.append(p3)

        return paths


if __name__ == "__main__":
    out_dir = Path("docs/thesis_figures")
    sim = RumorPropagationSimulator()
    res = sim.run()
    summary = {k: v for k, v in res.items() if k != "history"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    saved = sim.plot(res, out_dir)
    if saved:
        print("论文插图已保存:")
        for p in saved:
            print(" ", p.resolve())
