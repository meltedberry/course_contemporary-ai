"""实验一：模型与特征对照、损失绘图及预测输出。"""
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

ROOT = Path(__file__).resolve().parent
SEED = 42


def sha256(path):
    """计算文件的 SHA-256 指纹。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(data_dir):
    """读取数据，检查字段、缺失值、标签及重复文本。"""
    train = pd.read_csv(data_dir / "train_data.csv")
    test = pd.read_csv(data_dir / "test_data_unlabeled.csv")
    assert list(train.columns) == ["text", "target"] and list(test.columns) == ["text"]
    assert not train.isna().any().any() and not test.isna().any().any(), "数据存在空值"
    assert not train.text.str.strip().eq("").any() and not test.text.str.strip().eq("").any()
    assert np.array_equal(np.sort(train.target.unique()), np.arange(10)), "标签必须为 0--9"
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
    """构造 TF-IDF；changes 用于修改特征参数。"""
    return TfidfVectorizer(**(dict(max_features=10000, min_df=2, sublinear_tf=True,
                                   dtype=np.float32, norm="l2") | changes))


def scores(y, predicted):
    """计算准确率与 macro-F1。"""
    return dict(accuracy=float(accuracy_score(y, predicted)),
                macro_f1=float(f1_score(y, predicted, average="macro")))


def fit_mlp(x, y, alpha, epochs, x_val=None, y_val=None):
    """逐轮训练 MLP；有验证集时记录损失、早停并恢复最佳模型。"""
    model = MLPClassifier(hidden_layer_sizes=(64,), alpha=alpha, solver="adam",
                          batch_size=128, learning_rate_init=0.001, random_state=SEED)
    history, best_model, best_loss, patience_loss, stale = [], None, float("inf"), float("inf"), 0
    best_epoch = epochs
    for epoch in range(1, epochs + 1):
        model.partial_fit(x, y, classes=np.arange(10))
        if x_val is None:
            continue
        train_loss = float(log_loss(y, model.predict_proba(x), labels=np.arange(10)))
        val_loss = float(log_loss(y_val, model.predict_proba(x_val), labels=np.arange(10)))
        history.append(dict(alpha=alpha, epoch=epoch, train_loss=train_loss, validation_loss=val_loss))
        if val_loss < best_loss:
            best_loss, best_model, best_epoch = val_loss, copy.deepcopy(model), epoch
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
    """完整实验流程，按下面八个步骤执行。"""
    # 1. 读取数据，固定训练/验证划分，并构建基础 TF-IDF。
    train, test, audit = load_data(data_dir)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    x, y = train.text, train.target.to_numpy()
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    base = vectorizer()
    a, b = base.fit_transform(x.iloc[tr]), base.transform(x.iloc[va])
    rows, models, histories, predictions = [], {}, [], {}
    print(f"数据检查通过：训练 {len(tr)}，验证 {len(va)}，测试 {len(test)}", flush=True)

    def record(name, model, vec, aa, bb, seconds, best_epoch=None, epochs_run=None):
        """保存单组配置的指标、模型和验证预测。"""
        predicted = model.predict(bb)
        row = dict(name=name, **scores(y[va], predicted),
                   train_accuracy=float(accuracy_score(y[tr], model.predict(aa))),
                   features=aa.shape[1], fit_seconds=seconds,
                   best_epoch=best_epoch, epochs_run=epochs_run)
        rows.append(row)
        models[name] = (model, vec)
        predictions[name] = predicted
        print(f"{name}: F1={row['macro_f1']:.6f}, accuracy={row['accuracy']:.6f}, {seconds:.2f}s", flush=True)

    # 2. 比较 NB、LR 和线性核 SVC 的九组参数。
    candidates = [(f"NB alpha={alpha:g}", MultinomialNB(alpha=alpha)) for alpha in (0.1, 1, 10)]
    candidates += [(f"LR C={c:g}", LogisticRegression(C=c, max_iter=1000, random_state=SEED)) for c in (0.1, 1, 10)]
    candidates += [(f"SVM C={c:g}", SVC(kernel="linear", C=c, random_state=SEED)) for c in (0.1, 1, 10)]
    for name, model in candidates:
        started = time.perf_counter()
        model.fit(a, y[tr])
        record(name, model, base, a, b, time.perf_counter() - started)

    # 3. 比较两组 MLP，记录逐轮损失。
    for alpha in (0.0001, 0.01):
        started = time.perf_counter()
        model, best_epoch, epochs_run, history = fit_mlp(a, y[tr], alpha, 60, b, y[va])
        record(f"MLP alpha={alpha:g}", model, base, a, b, time.perf_counter() - started, best_epoch, epochs_run)
        histories.extend(history)

    # 4. 固定最佳 LR 参数，比较词频、词组和停用词设置。
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

    # 5. 优先按验证 macro-F1 选型，其次比较准确率。
    best = max(rows, key=lambda r: (r["macro_f1"], r["accuracy"]))
    chosen_model, chosen_vectorizer = models[best["name"]]
    print(f"所选配置：{best['name']}；开始三折稳定性检查", flush=True)

    def refit(aa, yy):
        """按选定配置重新训练模型。"""
        if isinstance(chosen_model, MLPClassifier):
            return fit_mlp(aa, yy, chosen_model.alpha, best["best_epoch"])[0]
        return clone(chosen_model).fit(aa, yy)

    # 6. 对选定配置做三折检查，每折独立拟合词表和模型。
    folds = []
    for fold, (cv_tr, cv_va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, y), 1):
        vec = clone(chosen_vectorizer)
        aa, bb = vec.fit_transform(x.iloc[cv_tr]), vec.transform(x.iloc[cv_va])
        model = refit(aa, y[cv_tr])
        folds.append(dict(fold=fold, **scores(y[cv_va], model.predict(bb))))
        print(f"  Fold {fold}: {folds[-1]}", flush=True)

    # 7. 用全部有标签数据重训，按测试数据原顺序输出预测。
    vec = clone(chosen_vectorizer)
    final_model = refit(vec.fit_transform(x), y)
    final_predictions = final_model.predict(vec.transform(test.text))
    pd.DataFrame(final_predictions).to_csv(ROOT / "prediction.csv", header=False, index=False)
    # 8. 保存指标、损失、验证预测及汇总记录，随后绘图和检查。
    pd.DataFrame(rows).to_csv(out / "experiments.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "loss.csv", index=False)
    pd.DataFrame(dict(row_index=va, actual=y[va], predicted=predictions[best["name"]],
                      lr_stopwords=predictions["LR English stopwords"])).to_csv(out / "validation.csv", index=False)
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
    """绘制两组损失曲线与所选配置的后期放大图。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.45), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    for (alpha, group), color in zip(history.groupby("alpha"), ["#2166ac", "#d6604d"]):
        axes[0].plot(group.epoch, group.train_loss, color=color, label=f"alpha={alpha:g} train")
        axes[0].plot(group.epoch, group.validation_loss, color=color, linestyle="--", label=f"alpha={alpha:g} val")
    axes[0].set(xlabel="Epoch", ylabel="Mean cross-entropy", title="Training process")
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    group = history[history.alpha == 0.0001]
    tail, best = group[group.epoch >= 25], group.loc[group.validation_loss.idxmin()]
    axes[1].plot(tail.epoch, tail.validation_loss, color="#2166ac", marker=".")
    axes[1].scatter(best.epoch, best.validation_loss, color="#d6604d", s=24, zorder=3)
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
    """核对已有结果及预测格式，不重新训练。"""
    train, test, audit = load_data(data_dir)
    out = ROOT / "results"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["audit"] == audit, "数据文件与本次实验不一致"
    rows = pd.read_csv(out / "experiments.csv")
    val = pd.read_csv(out / "validation.csv")
    _, va = train_test_split(np.arange(len(train)), test_size=0.2, stratify=train.target, random_state=SEED)
    assert len(rows) == 14 and np.array_equal(val.row_index, va)
    assert np.array_equal(val.actual, train.target.iloc[va])
    for metric, value in scores(val.actual, val.predicted).items():
        assert np.isclose(value, summary["selected"][metric], rtol=0, atol=1e-12)
    stop = rows[rows.name == "LR English stopwords"].iloc[0]
    assert np.isclose(f1_score(val.actual, val.lr_stopwords, average="macro"), stop.macro_f1, rtol=0, atol=1e-12)
    assert rows.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]["name"] == summary["selected"]["name"]
    assert np.array_equal(confusion_matrix(val.actual, val.predicted), summary["confusion_matrix"])
    loss = pd.read_csv(out / "loss.csv")
    if summary["selected"]["name"].startswith("MLP"):
        alpha = float(summary["selected"]["name"].split("=")[1])
        group = loss[loss.alpha == alpha]
        assert int(group.loc[group.validation_loss.idxmin(), "epoch"]) == summary["final_epochs"]
    lines = (ROOT / "prediction.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(test) and all(line in {str(i) for i in range(10)} for line in lines)
    assert sha256(ROOT / "prediction.csv") == summary["prediction_sha256"]
    print(f"检查通过：数据无空值或重复，{len(rows)} 组结果一致，prediction.csv 为 {len(test)} 行合法标签。", flush=True)


# 命令行入口：默认运行完整实验，--check 仅核对已有结果。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--check", action="store_true", help="仅核对已保存结果，不重新训练")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        check(args.data_dir) if args.check else run(args.data_dir)
