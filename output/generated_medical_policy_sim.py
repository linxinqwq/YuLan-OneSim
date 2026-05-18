"""医保政策组合仿真（编排器生成） 场景链: ["health_inequality", "rational_choice_theory", "customer_satisfaction_and_loyalty_model"]"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CHAIN = ["health_inequality", "rational_choice_theory", "customer_satisfaction_and_loyalty_model"]

def adapt_supply(reimb: np.ndarray, boost: float) -> np.ndarray:
    return np.clip(reimb * boost, 0.05, 1.0)  # health_inequality

def adapt_flow(supply: np.ndarray, self_pay: np.ndarray, pref: np.ndarray) -> np.ndarray:
    u = np.maximum(supply * pref / (1.0 + self_pay), 1e-6)  # rational_choice_theory
    return u / u.sum()

def adapt_satisfaction(prev: np.ndarray, q: np.ndarray, share: np.ndarray) -> np.ndarray:
  # customer_satisfaction_and_loyalty_model
    return np.clip(0.7 * prev + 0.3 * q * np.clip(1 - 0.3 * share, 0.4, 1), 0, 1)

class MedicalPolicySimulator:
    def __init__(self, n_hospitals: int = 5, n_rounds: int = 20) -> None:
        self.n_hospitals, self.n_rounds = n_hospitals, n_rounds
        self.chain = list(CHAIN)

    def run(self, seed: int = 42) -> Dict[str, Any]:
        rng = np.random.RandomState(seed)
        n = self.n_hospitals
        reimb = rng.uniform(0.4, 0.95, n)
        self_pay = rng.uniform(0.15, 0.75, n)
        pref = rng.uniform(0.8, 1.2, n)
        quality = rng.uniform(0.5, 0.9, n)
        sat = rng.uniform(0.45, 0.65, n)
        rev_base = rng.uniform(5e5, 1.2e6, n)
        hist_rev, hist_sat, rounds = [], [], []
        flow = np.ones(n) / n
        for t in range(self.n_rounds):
            supply = adapt_supply(reimb, 1.15 if t >= 3 else 1.0)
            flow = adapt_flow(supply, self_pay, pref)
            sat = adapt_satisfaction(sat, quality, flow)
            rev = float((rev_base * (0.85 + 0.35 * flow) * (0.9 + 0.2 * sat)).sum())
            rounds.append(t + 1)
            hist_rev.append(rev)
            hist_sat.append(float(sat.mean()))
        return {
            "manifest_chain": self.chain,
            "patient_flow_distribution": flow.round(4).tolist(),
            "mean_satisfaction": float(sat.mean()),
            "hospital_revenue_index": hist_rev[-1],
            "revenue_change_ratio": hist_rev[-1] / max(hist_rev[0], 1.0),
            "history": {"round": rounds, "total_revenue": hist_rev, "mean_satisfaction": hist_sat},
        }

    def plot(self, result: Dict[str, Any], out: Path) -> None:
        if not HAS_MPL:
            return
        out.mkdir(parents=True, exist_ok=True)
        h = result["history"]
        labels = [f"医院{i+1}" for i in range(len(result["patient_flow_distribution"]))]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h["round"], h["total_revenue"], "o-", color="#2563eb")
        ax.set_title("医院收入变化"); ax.set_xlabel("轮次"); ax.grid(alpha=0.3)
        fig.savefig(out / "fig_3_1_revenue_over_time.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, result["patient_flow_distribution"], color="#059669")
        ax.set_title("患者流量分布")
        fig.savefig(out / "fig_3_2_patient_flow.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h["round"], h["mean_satisfaction"], "s-", color="#d97706")
        ax.set_ylim(0, 1); ax.set_title("患者满意度变化"); ax.grid(alpha=0.3)
        fig.savefig(out / "fig_3_3_satisfaction.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    out = Path("docs/thesis_figures")
    sim = MedicalPolicySimulator()
    res = sim.run()
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False, indent=2))
    sim.plot(res, out)
    if HAS_MPL:
        print("论文插图已保存:", out.resolve())
