# 实验一：文本分类

使用 TF-IDF 比较朴素贝叶斯、逻辑回归、线性核 SVC 和单隐藏层 MLP，共 14 组配置。随机种子为 42，训练/验证按类别分层划分为 80%/20%，按验证 macro-F1、accuracy 的顺序选型。

## 仓库文件

- `experiment.py`：实验脚本，包含数据检查、模型与特征对照、MLP 逐轮训练、三折验证、绘图及最终预测。
- `requirements.txt`：实际运行的依赖版本，使用 Python 3.14.6。
- `data/`：课程提供的训练数据 `train_data.csv` 和无标签测试数据 `test_data_unlabeled.csv`。

本仓库提供代码、依赖、说明和课程原始数据。报告和预测文件不上传；结果由脚本在本地生成。

## 运行步骤

下载或克隆仓库后，在 `实验一/` 目录运行：

```powershell
python -m pip install -r requirements.txt
python experiment.py
```

也可以指定数据目录：

```powershell
python experiment.py --data-dir "D:\course-data"
```

运行后生成：

- `results/experiments.csv`：14 组配置的指标与耗时。
- `results/loss.csv`、`results/loss.pdf`：MLP 的逐轮损失与损失曲线。
- `results/validation.csv`：验证集预测。
- `results/summary.json`：数据检查、配置、逐类指标、三折结果和运行环境。
- `prediction.csv`：2457 行测试预测，无表头、无索引，顺序与测试数据一致。

已有结果可再次核对：

```powershell
python experiment.py --check
```

`--check` 用于核对一次完整运行后生成的文件。训练耗时随设备负载变化。

## 实验设置与主要发现

词表和 IDF 仅在训练子集拟合。基础 TF-IDF 最多 10000 维，`min_df=2`、`sublinear_tf=True`，使用平滑 IDF 和 L2 归一化，保留邮件头与引文。

NB 比较 `alpha=0.1/1/10`；LR 和线性核 SVC 比较 `C=0.1/1/10`。MLP 使用 64 个 ReLU 隐藏单元、Adam、学习率 0.001、batch size 128，比较 `alpha=0.0001/0.01`。特征对照分别尝试原始词频、word(1,2) 和英文停用词过滤。

MLP 每次 `partial_fit` 更新一轮，按验证交叉熵选择轮数；连续 8 轮没有至少 0.0001 的改进时停止。最终重训和三折验证沿用相同更新方式，固定为选定的 40 轮。BLAS 线程数固定为 2。

所选 MLP 的验证 accuracy 为 0.932836，macro-F1 为 0.933268；停用词 LR 的 macro-F1 为 0.928958。固定方案三折平均 accuracy 为 0.926303，样本标准差为 0.005598。三折用于观察不同划分下的波动，未用于重新调参。测试集无标签，不计算测试准确率。

