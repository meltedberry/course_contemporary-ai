# 搜索算法实验

报告：report.tex（XeLaTeX 源码）、report.pdf（5 页）。

复现环境：Python 3.14.6；算法只使用标准库，图表依赖 matplotlib、numpy。

在本目录运行：

```sh
python experiment.py
python benchmark.py
python strong_heuristic.py
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

results.json 保存四题结果；benchmark_results.json 保存随机地图逐图数据、种子、耗时和随机树实验结果。performance.pdf 与 search_regions.pdf 是报告引用的矢量图，PNG 为预览。

地图规模为 15、25、35、45，障碍率 25%，每规模按固定种子序列取前 20 张可达地图。每图每算法运行 7 次，取中位耗时。节点统计包含终点，博弈节点统计包含根。普通 A* 的同 f 平局规则为小 g 优先，优化版本为大 g 优先。所有地图逐一断言两种 A* 的代价与 Dijkstra 一致；随机树逐一断言 Alpha-beta 与 Minimax 的返回值一致。

地形感知启发式另见 strong_heuristic.py；预处理代价与在线搜索分别统计。报告位于 report/main.tex 与 report/report.pdf。
