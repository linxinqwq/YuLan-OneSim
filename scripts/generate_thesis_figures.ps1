# 生成论文插图：谣言传播与恐慌扩散（3 张 PNG）
Set-Location $PSScriptRoot\..

$env:PYTHONPATH = "src"
$env:ORCHESTRATOR_NO_LLM = "1"

Write-Host ">>> 1. 更新仿真器清单" -ForegroundColor Cyan
python -m orchestrator.manifest_builder

Write-Host ">>> 2. 编排并运行谣言传播组合仿真" -ForegroundColor Cyan
python -m orchestrator.demo

Write-Host ""
Write-Host "插图目录: docs\thesis_figures\" -ForegroundColor Green
Write-Host "  fig_3_1_believers_over_time.png    图3-1 信谣者数量"
Write-Host "  fig_3_2_mean_panic_over_time.png    图3-2 恐慌情绪均值"
Write-Host "  fig_3_3_forward_count_over_time.png 图3-3 转发行为数量"
Write-Host ""
Write-Host "Word: 插入 -> 图片 -> 选择上述文件" -ForegroundColor Yellow
