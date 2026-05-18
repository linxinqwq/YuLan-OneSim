# 智能编排器 — 谣言传播与恐慌扩散

## 生成论文插图

```powershell
cd D:\yulan\YuLan-OneSim
.\scripts\generate_thesis_figures.ps1
```

输出：
- `docs/thesis_figures/fig_3_1_believers_over_time.png`
- `docs/thesis_figures/fig_3_2_mean_panic_over_time.png`
- `docs/thesis_figures/fig_3_3_forward_count_over_time.png`

## 一键演示

```powershell
$env:PYTHONPATH = "src"
$env:ORCHESTRATOR_NO_LLM = "1"
python -m orchestrator.demo
```

## 固定三场景

- `emotional_contagion_model`
- `information_cascade_and_silence`
- `diffusion_of_innovations`

论文实现章节：`docs/论文第3章_编排器实现（可粘贴Word）.txt`
