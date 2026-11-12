"""Plot the local cascade timing and regression outputs for a course report."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.array(
        [
            [
                (float(row["window_start_s"]) + float(row["window_end_s"])) / 2,
                float(row["label_g"]),
                float(row["prediction_g"]),
            ]
            for row in rows
        ]
    )


def configure_font():
    for path in (Path("/mnt/c/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyh.ttc")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"] = 11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cascade", type=Path, default=Path("outputs/cascade-20261112"))
    parser.add_argument(
        "--prediction", type=Path, default=Path("outputs/cascade-prediction-20261112")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/cascade-visualization-20261112")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_font()
    cpu = load_json(args.cascade / "cpu/summary.json")
    spark = load_json(args.cascade / "spark/summary.json")
    metrics = {
        name: load_json(args.prediction / name / "metrics.json") for name in ("gbt", "ridge")
    }
    curves = {
        name: read_predictions(args.prediction / name / "predictions.csv") for name in metrics
    }
    np.testing.assert_allclose(curves["gbt"][:, :2], curves["ridge"][:, :2])
    for name, curve in curves.items():
        error = curve[:, 2] - curve[:, 1]
        np.testing.assert_allclose(np.sqrt(np.mean(error**2)), metrics[name]["rmse"])
        np.testing.assert_allclose(np.mean(np.abs(error)), metrics[name]["mae"])

    values = [cpu["elapsed_seconds"], spark["elapsed_seconds"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(values, [1, 0], s=115, c=["#438d73", "#2d6189"], zorder=3)
    for x, y in zip(values, [1, 0]):
        ax.annotate(f"{x:.2f} s", (x, y), xytext=(10, 0), textcoords="offset points", va="center")
    ax.set_yticks([1, 0], ["CPU 顺序（1 线程）", "Spark 流水（3 任务槽 + 合并进程）"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0, max(values) * 1.28)
    ax.set_xlabel("总耗时（秒）")
    ax.set_title("三试次 Cascade 分解耗时（单次运行）", loc="left")
    ax.grid(axis="x", alpha=0.18)
    fig.text(
        0.12, 0.01, "WSL 单机；两组并行资源不同，未设置同额容器限制", fontsize=9, color="#555555"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.output / "cascade_time.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, name in zip(axes, ("gbt", "ridge")):
        data = curves[name]
        ax.plot(data[:, 0], data[:, 1], color="#2d6189", lw=2, label="真实窗口均值")
        ax.plot(data[:, 0], data[:, 2], color="#d47745", lw=1.2, label="预测")
        ax.set_title(
            f"MU 扩展特征 + {name.upper()}   RMSE {metrics[name]['rmse']:.2f} g", loc="left"
        )
        ax.set_ylabel("中指屈曲力（g）")
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].set_xlabel("测试试次起点后的时间（秒）")
    fig.text(
        0.08,
        0.01,
        "离线探索：模板池包含测试试次肌电，且投影按试次适配",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.output / "cascade_prediction_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(2)
    for offset, metric, color in ((-0.18, "rmse", "#2d6189"), (0.18, "mae", "#d47745")):
        numbers = [metrics[name][metric] for name in ("gbt", "ridge")]
        bars = ax.barh(y + offset, numbers, height=0.30, label=metric.upper(), color=color)
        ax.bar_label(bars, labels=[f"{number:.2f}" for number in numbers], padding=4)
    ax.set_yticks(y, ["MU 扩展特征 + GBT", "MU 扩展特征 + Ridge"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(metrics[name]["rmse"] for name in metrics) * 1.25)
    ax.set_xlabel("预测误差（g，越低越好）")
    ax.set_title("测试试次回归误差（单次探索）", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.15)
    fig.text(
        0.12, 0.01, "模板池包含测试试次肌电；结果不能视为严格独立测试", fontsize=9, color="#555555"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.output / "cascade_prediction_errors.png", dpi=180)
    plt.close(fig)
    print(f"Saved three figures in {args.output}")


if __name__ == "__main__":
    main()
