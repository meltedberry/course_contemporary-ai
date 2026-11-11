# 实验二：搜索算法

实现 A*、Minimax、Alpha-beta 与宝藏分配，并提供配对性能实验。

## 运行

Python 3.14.6 为实测环境。算法仅使用标准库；图表依赖如下安装：

```sh
pip install -r requirements.txt
python experiment.py
python benchmark.py
python strong_heuristic.py
```

在本目录运行。results.json 为四题与空白网格对照；benchmark_results.json 保存随机地图种子、逐图节点数和耗时，以及随机博弈树结果；strong_results.json 保存地形感知启发式实验。

地图规模为 15、25、35、45，障碍率 25%，每规模取固定种子序列前 20 张可达地图，每图每算法重复 7 次，取耗时中位数。节点统计包含终点，博弈统计包含根。逐图核验 A* 与 Dijkstra 的代价一致，逐树核验 Alpha-beta 与 Minimax 返回值一致。
