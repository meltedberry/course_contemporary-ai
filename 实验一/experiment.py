"""实验一：运行全部对照、绘制损失曲线并生成 prediction.csv。

阅读顺序：先看 run() 的实验流程，再查看各个辅助函数。
运行 python experiment.py 做完整实验；加 --check 只检查已有结果。
"""
# 一、导入工具：读取数据、训练模型、计算指标、画图和保存记录。
import argparse
import copy
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 直接保存图片，无需弹出绘图窗口。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

# 路径以脚本所在目录为准；固定种子，使数据划分和模型初始化可复现。
ROOT = Path(__file__).resolve().parent
SEED = 42


def sha256(path):
    """计算文件指纹，检查后续使用的数据或预测文件有没有变化。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(data_dir):
    """读取两份 CSV，检查字段、缺失值、标签和重复文本，并返回检查记录。"""
    train = pd.read_csv(data_dir / "train_data.csv")
    test = pd.read_csv(data_dir / "test_data_unlabeled.csv")
    assert list(train.columns) == ["text", "target"] and list(test.columns) == ["text"]
    assert not train.isna().any().any() and not test.isna().any().any(), "数据存在空值"
    assert not train.text.str.strip().eq("").any() and not test.text.str.strip().eq("").any()
    assert np.array_equal(np.sort(train.target.unique()), np.arange(10)), "标签必须为 0--9"
    # 记录各类样本数，以及训练集、测试集内部和两者之间的完全重复文本。
    audit = dict(train_samples=len(train), test_samples=len(test),
                 class_counts={str(k): int(v) for k, v in train.target.value_counts().sort_index().items()},
                 train_duplicates=int(train.text.duplicated().sum()),
                 test_duplicates=int(test.text.duplicated().sum()),
                 train_test_overlap=len(set(train.text) & set(test.text)),
                 train_sha256=sha256(data_dir / "train_data.csv"),
                 test_sha256=sha256(data_dir / "test_data_unlabeled.csv"))
    assert audit["train_duplicates"] == audit["test_duplicates"] == audit["train_test_overlap"] == 0
    return train, test, audit


def vectorizer(**changes):
    """构造基础 TF-IDF；changes 用于覆盖某些参数，进行特征对照。"""
    # 最多保留 10000 个词，忽略仅出现于一篇文档的词，使用对数词频。
    return TfidfVectorizer(**(dict(max_features=10000, min_df=2, sublinear_tf=True,
                                   dtype=np.float32, norm="l2") | changes))


def scores(y, predicted):
    """accuracy 看总体正确比例，macro-F1 对十个类别的 F1 等权平均。"""
    return dict(accuracy=float(accuracy_score(y, predicted)),
                macro_f1=float(f1_score(y, predicted, average="macro")))


def fit_mlp(x, y, alpha, epochs, x_val=None, y_val=None):
    """有验证集时记录损失并早停；否则按选定轮数重训，更新方式完全相同。"""
    # 二、MLP 训练：64 个隐藏单元；alpha 控制 L2 正则化强度。
    model = MLPClassifier(hidden_layer_sizes=(64,), alpha=alpha, solver="adam",
                          batch_size=128, learning_rate_init=0.001, random_state=SEED)
    history, best_model, best_loss, patience_loss, stale = [], None, float("inf"), float("inf"), 0
    best_epoch = epochs
    for epoch in range(1, epochs + 1):
        # partial_fit 在现有参数上继续训练一轮，不会每轮重新初始化模型。
        model.partial_fit(x, y, classes=np.arange(10))
        if x_val is None:
            # 三折检查和最终重训不做早停，直接训练已选定的轮数。
            continue
        # 用预测概率计算交叉熵，作为损失曲线；这里不包含 L2 惩罚项。
        train_loss = float(log_loss(y, model.predict_proba(x), labels=np.arange(10)))
        val_loss = float(log_loss(y_val, model.predict_proba(x_val), labels=np.arange(10)))
        history.append(dict(alpha=alpha, epoch=epoch, train_loss=train_loss, validation_loss=val_loss))
        if val_loss < best_loss:
            # 保存验证损失最低的完整模型副本，避免后续训练覆盖其参数。
            best_loss, best_model, best_epoch = val_loss, copy.deepcopy(model), epoch
        # “最低点快照”与“早停计数”分开：任何新低都保存；
        # 只有累计超过 0.0001 的改善才清零计数，连续 8 轮无此改善则停止。
        if val_loss < patience_loss - 1e-4:
            patience_loss, stale = val_loss, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  MLP alpha={alpha:g}, epoch={epoch}, train={train_loss:.5f}, val={val_loss:.5f}", flush=True)
        if stale >= 8:
            break
    return (best_model if x_val is not None else model), best_epoch, epoch, history


def run(data_dir):
    """三、完整流程：固定划分 → 模型/特征比较 → 选型 → 三折检查 → 全量重训。"""
    # 1. 检查数据并建立结果目录。x 是文本，y 是类别编号。
    train, test, audit = load_data(data_dir)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    x, y = train.text, train.target.to_numpy()
    # tr、va 是训练/验证行号；分层抽样尽量保持各类别比例一致。
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    base = vectorizer()
    # a、b 分别是训练、验证特征。只对训练文本 fit，避免验证信息进入词表和 IDF。
    a, b = base.fit_transform(x.iloc[tr]), base.transform(x.iloc[va])
    # 依次保存指标表、模型及向量化器、损失历史、验证预测。
    rows, models, histories, predictions = [], {}, [], {}
    print(f"数据检查通过：训练 {len(tr)}，验证 {len(va)}，测试 {len(test)}", flush=True)

    def record(name, model, vec, aa, bb, seconds, best_epoch=None, epochs_run=None):
        """统一评估一个配置，保存其指标与预测，便于后面公平比较。"""
        predicted = model.predict(bb)
        row = dict(name=name, **scores(y[va], predicted),
                   train_accuracy=float(accuracy_score(y[tr], model.predict(aa))),
                   features=aa.shape[1], fit_seconds=seconds,
                   best_epoch=best_epoch, epochs_run=epochs_run)
        rows.append(row)
        models[name] = (model, vec)
        predictions[name] = predicted
        print(f"{name}: F1={row['macro_f1']:.6f}, accuracy={row['accuracy']:.6f}, {seconds:.2f}s", flush=True)

    # 2. 比较 9 组传统模型。共用划分与特征，每组只改变一个模型参数。
    # NB 调平滑系数 alpha；LR、线性核 SVC 调约束参数 C。
    candidates = [(f"NB alpha={alpha:g}", MultinomialNB(alpha=alpha)) for alpha in (0.1, 1, 10)]
    candidates += [(f"LR C={c:g}", LogisticRegression(C=c, max_iter=1000, random_state=SEED)) for c in (0.1, 1, 10)]
    candidates += [(f"SVM C={c:g}", SVC(kernel="linear", C=c, random_state=SEED)) for c in (0.1, 1, 10)]
    for name, model in candidates:
        started = time.perf_counter()
        model.fit(a, y[tr])
        record(name, model, base, a, b, time.perf_counter() - started)

    # 3. 比较两组 MLP 正则化系数，记录每轮损失，并恢复各自的最佳轮次。
    for alpha in (0.0001, 0.01):
        started = time.perf_counter()
        model, best_epoch, epochs_run, history = fit_mlp(a, y[tr], alpha, 60, b, y[va])
        record(f"MLP alpha={alpha:g}", model, base, a, b, time.perf_counter() - started, best_epoch, epochs_run)
        histories.extend(history)

    # 4. 固定最佳 LR 的 C，再做三组特征对照；与前面合计 14 组配置。
    # 三组分别从基础设置出发；word(1,2) 同时提高维数上限，并非只改词组长度。
    best_lr = max((r for r in rows if r["name"].startswith("LR ")), key=lambda r: (r["macro_f1"], r["accuracy"]))
    best_c = models[best_lr["name"]][0].C
    feature_options = [("LR raw-TF", dict(sublinear_tf=False)),
                       ("LR word(1,2)", dict(ngram_range=(1, 2), max_features=20000)),
                       ("LR English stopwords", dict(stop_words="english"))]
    for name, changes in feature_options:
        vec = vectorizer(**changes)
        aa, bb = vec.fit_transform(x.iloc[tr]), vec.transform(x.iloc[va])
        model = LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)
        started = time.perf_counter()
        model.fit(aa, y[tr])
        record(name, model, vec, aa, bb, time.perf_counter() - started)

    # 5. 按验证 macro-F1 选配置；相同时再比较 accuracy。
    best = max(rows, key=lambda r: (r["macro_f1"], r["accuracy"]))
    chosen_model, chosen_vectorizer = models[best["name"]]
    print(f"所选配置：{best['name']}；开始三折稳定性检查", flush=True)

    def refit(aa, yy):
        """用选定配置重新训练一个模型，不继续使用原验证实验的模型参数。"""
        if isinstance(chosen_model, MLPClassifier):
            return fit_mlp(aa, yy, chosen_model.alpha, best["best_epoch"])[0]
        return clone(chosen_model).fit(aa, yy)

    # 6. 三折检查：每折独立拟合词表和分类器，检查固定方案的波动，不再调参。
    # 此前选型已经用过部分标签，因此这里的均值不作为独立的无偏评估。
    folds = []
    for fold, (cv_tr, cv_va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, y), 1):
        vec = clone(chosen_vectorizer)
        aa, bb = vec.fit_transform(x.iloc[cv_tr]), vec.transform(x.iloc[cv_va])
        model = refit(aa, y[cv_tr])
        folds.append(dict(fold=fold, **scores(y[cv_va], model.predict(bb))))
        print(f"  Fold {fold}: {folds[-1]}", flush=True)

    # 7. 方案锁定后，用全部有标签数据重训；测试文本仅在此处转换和预测。
    vec = clone(chosen_vectorizer)
    final_model = refit(vec.fit_transform(x), y)
    final_predictions = final_model.predict(vec.transform(test.text))
    # 测试文本未排序，预测保持原顺序；提交 CSV 不写列名和行索引。
    pd.DataFrame(final_predictions).to_csv(ROOT / "prediction.csv", header=False, index=False)
    # 8. 保存对照表、损失历史和验证预测。row_index 保留原数据的零起始行号。
    pd.DataFrame(rows).to_csv(out / "experiments.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "loss.csv", index=False)
    pd.DataFrame(dict(row_index=va, actual=y[va], predicted=predictions[best["name"]],
                      lr_stopwords=predictions["LR English stopwords"])).to_csv(out / "validation.csv", index=False)
    # 汇总数据指纹、选定配置、逐类表现、混淆矩阵和环境版本，方便追溯结果。
    summary = dict(seed=SEED, train_size=len(tr), validation_size=len(va), audit=audit, selected=best,
                   cv=folds, final_epochs=best["best_epoch"], prediction_sha256=sha256(ROOT / "prediction.csv"),
                   classification_report=classification_report(y[va], predictions[best["name"]], output_dict=True),
                   confusion_matrix=confusion_matrix(y[va], predictions[best["name"]]).tolist(),
                   environment=dict(python=platform.python_version(), blas_threads=2,
                       **{name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl")}))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_loss(pd.DataFrame(histories), out)
    check(data_dir)


def plot_loss(history, out):
    """四、画损失图：左图比较两组训练过程，右图放大 alpha=0.0001 的后期变化。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.45), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    for (alpha, group), color in zip(history.groupby("alpha"), ["#2166ac", "#d6604d"]):
        axes[0].plot(group.epoch, group.train_loss, color=color, label=f"alpha={alpha:g} train")
        axes[0].plot(group.epoch, group.validation_loss, color=color, linestyle="--", label=f"alpha={alpha:g} val")
    axes[0].set(xlabel="Epoch", ylabel="Mean cross-entropy", title="Training process")
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    # 右图只显示第 25 轮之后的验证损失，并标出整段历史中的最低点。
    group = history[history.alpha == 0.0001]
    tail, best = group[group.epoch >= 25], group.loc[group.validation_loss.idxmin()]
    axes[1].plot(tail.epoch, tail.validation_loss, color="#2166ac", marker=".")
    axes[1].scatter(best.epoch, best.validation_loss, color="#d6604d", s=24, zorder=3)
    # 文字位置使用坐标轴比例，与数据坐标分开设置，避免标注压住曲线。
    axes[1].annotate(f"Epoch {int(best.epoch)}: {best.validation_loss:.5f}",
                     xy=(best.epoch, best.validation_loss), xytext=(0.96, 0.68), textcoords="axes fraction",
                     ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=2), fontsize=7,
                     arrowprops=dict(arrowstyle="->", color="#555555", shrinkA=4, shrinkB=5))
    axes[1].set(xlabel="Epoch", ylabel="Validation cross-entropy", title="Late epochs (alpha=0.0001)")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(out / "loss.pdf", bbox_inches="tight")
    plt.close(fig)


def check(data_dir):
    """五、检查已有输出是否互相一致；需要先完成一次完整实验，不会重新训练。"""
    train, test, audit = load_data(data_dir)
    out = ROOT / "results"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    # 对比样本统计和文件指纹，确认检查的是生成这些结果时使用的数据。
    assert summary["audit"] == audit, "数据文件与本次实验不一致"
    rows = pd.read_csv(out / "experiments.csv")
    val = pd.read_csv(out / "validation.csv")
    # 重建相同划分，确认验证集行号、真实标签与保存文件一一对应。
    _, va = train_test_split(np.arange(len(train)), test_size=0.2, stratify=train.target, random_state=SEED)
    assert len(rows) == 14 and np.array_equal(val.row_index, va)
    assert np.array_equal(val.actual, train.target.iloc[va])
    # 从保存的预测重新计算所选模型和停用词 LR 的指标，与记录交叉核对。
    for metric, value in scores(val.actual, val.predicted).items():
        assert np.isclose(value, summary["selected"][metric], rtol=0, atol=1e-12)
    stop = rows[rows.name == "LR English stopwords"].iloc[0]
    assert np.isclose(f1_score(val.actual, val.lr_stopwords, average="macro"), stop.macro_f1, rtol=0, atol=1e-12)
    assert rows.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]["name"] == summary["selected"]["name"]
    assert np.array_equal(confusion_matrix(val.actual, val.predicted), summary["confusion_matrix"])
    # 核对 MLP 的选定轮数确实对应最低验证损失。
    loss = pd.read_csv(out / "loss.csv")
    if summary["selected"]["name"].startswith("MLP"):
        alpha = float(summary["selected"]["name"].split("=")[1])
        group = loss[loss.alpha == alpha]
        assert int(group.loc[group.validation_loss.idxmin(), "epoch"]) == summary["final_epochs"]
    # 核对提交行数、标签范围和文件指纹；这些检查不能替代测试集准确率评估。
    lines = (ROOT / "prediction.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(test) and all(line in {str(i) for i in range(10)} for line in lines)
    assert sha256(ROOT / "prediction.csv") == summary["prediction_sha256"]
    print(f"检查通过：数据无空值或重复，{len(rows)} 组结果一致，prediction.csv 为 {len(test)} 行合法标签。", flush=True)


if __name__ == "__main__":
    # 六、命令行入口：默认运行完整实验，--check 改为只检查已保存结果。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--check", action="store_true", help="仅核对已保存结果，不重新训练")
    args = parser.parse_args()
    # 限制底层计算库的线程数，避免线程过多，也便于复现运行环境。
    with threadpool_limits(limits=2):
        check(args.data_dir) if args.check else run(args.data_dir)
