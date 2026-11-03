"""
plotting.py
------------
Reads every results/<model_name>.json that pipeline.py has produced
(whichever architectures you've actually run -- this never fabricates
numbers for a model you haven't run) and generates:

  1. Per-model training curves (baseline + fine-tune phases: train loss
     and test accuracy vs epoch).
  2. A cross-model accuracy comparison bar chart (fp32 baseline vs
     post-quantization-pre-finetune vs post-finetune-mixed-precision-LVQ).
  3. A cross-model MODEL SIZE comparison bar chart (fp32 MB vs
     mixed-precision+LVQ MB), with compression ratio annotated.
  4. An accuracy-vs-size scatter (the efficiency frontier).
  5. A markdown + CSV summary table combining all of the above.

Run with:  python plotting.py
Output goes to plots/ (PNG figures) and results_summary.md / .csv.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_logger import RunLog, list_available_results

PLOTS_DIR = "plots"


def _epochs_and_series(epoch_log, key):
    return [e["epoch"] for e in epoch_log], [e[key] for e in epoch_log]


def plot_training_curves(log: RunLog, out_dir: str = PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for phase, epoch_log, color in [
        ("baseline (fp32)", log.baseline_epoch_log, "tab:blue"),
        ("fine-tune (mixed-prec + LVQ)", log.finetune_epoch_log, "tab:orange"),
    ]:
        if not epoch_log:
            continue
        xs, loss = _epochs_and_series(epoch_log, "train_loss")
        _, acc = _epochs_and_series(epoch_log, "test_acc")
        axes[0].plot(xs, loss, marker="o", ms=3, label=phase, color=color)
        axes[1].plot(xs, acc, marker="o", ms=3, label=phase, color=color)

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title(f"{log.model_name}: training loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test accuracy")
    axes[1].set_title(f"{log.model_name}: test accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{log.model_name}_training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bitwidth_search(log: RunLog, out_dir: str = PLOTS_DIR):
    trace = log.bitwidth_search_trace
    if not trace:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trace["avg_bits"], trace["cos_dist"], marker="o", ms=4, color="tab:purple")
    idx = trace["elbow_index"]
    ax.scatter([trace["avg_bits"][idx]], [trace["cos_dist"][idx]], color="red", zorder=5,
               label=f"elbow (chosen): {trace['avg_bits'][idx]:.2f} bits")
    ax.set_xlabel("average bitwidth (compression cost)")
    ax.set_ylabel("cosine distance to sensitivity vector")
    ax.set_title(f"{log.model_name}: mixed-precision bitwidth search")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{log.model_name}_bitwidth_search.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_accuracy_comparison(logs, out_dir: str = PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    names = [l.model_name for l in logs]
    fp32 = [l.acc_fp32 or 0 for l in logs]
    pre_ft = [l.acc_post_quant_post_bn_recal or 0 for l in logs]
    post_ft = [l.acc_final or 0 for l in logs]

    x = range(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width for i in x], fp32, width, label="fp32 baseline", color="tab:blue")
    ax.bar(list(x), pre_ft, width, label="post-quant (pre-finetune)", color="tab:red")
    ax.bar([i + width for i in x], post_ft, width, label="post-finetune (mixed-prec + LVQ)", color="tab:green")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy comparison across architectures")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "accuracy_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_size_comparison(logs, out_dir: str = PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    names = [l.model_name for l in logs]
    fp32_mb = [l.size_fp32_mb or 0 for l in logs]
    quant_mb = [l.size_quant_mb or 0 for l in logs]

    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - width / 2 for i in x], fp32_mb, width, label="fp32", color="tab:gray")
    b2 = ax.bar([i + width / 2 for i in x], quant_mb, width, label="mixed-precision + LVQ", color="tab:green")
    for i, l in enumerate(logs):
        if l.compression_ratio:
            ax.annotate(f"{l.compression_ratio:.1f}x", (i, max(fp32_mb[i], quant_mb[i]) * 1.03),
                        ha="center", fontsize=9, color="tab:green", fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("model size (MB)")
    ax.set_title("Model size comparison across architectures")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "model_size_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_efficiency_frontier(logs, out_dir: str = PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for l in logs:
        if l.size_quant_mb and l.acc_final is not None:
            ax.scatter(l.size_quant_mb, l.acc_final, s=90, label=f"{l.model_name} (mixed-prec+LVQ)")
        if l.size_fp32_mb and l.acc_fp32 is not None:
            ax.scatter(l.size_fp32_mb, l.acc_fp32, s=90, marker="x", label=f"{l.model_name} (fp32)")
    ax.set_xlabel("model size (MB)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs. model size")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "efficiency_frontier.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_summary_table(logs, md_path="results_summary.md", csv_path="results_summary.csv"):
    rows = []
    for l in logs:
        rows.append({
            "model": l.model_name,
            "params_avg_bits": f"{l.avg_weight_bits:.2f}" if l.avg_weight_bits else "-",
            "acc_fp32": f"{l.acc_fp32:.4f}" if l.acc_fp32 is not None else "-",
            "acc_post_quant_pre_bn": f"{l.acc_post_quant_pre_bn_recal:.4f}" if l.acc_post_quant_pre_bn_recal is not None else "-",
            "acc_post_quant_post_bn": f"{l.acc_post_quant_post_bn_recal:.4f}" if l.acc_post_quant_post_bn_recal is not None else "-",
            "acc_final": f"{l.acc_final:.4f}" if l.acc_final is not None else "-",
            "size_fp32_mb": f"{l.size_fp32_mb:.3f}" if l.size_fp32_mb is not None else "-",
            "size_quant_mb": f"{l.size_quant_mb:.3f}" if l.size_quant_mb is not None else "-",
            "compression_ratio": f"{l.compression_ratio:.2f}x" if l.compression_ratio is not None else "-",
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    headers = ["Model", "Avg bits", "fp32 acc", "Post-quant (pre-BN)", "Post-quant (post-BN)",
               "Final (post-finetune)", "fp32 size (MB)", "Quant size (MB)", "Compression"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"], r["params_avg_bits"], r["acc_fp32"], r["acc_post_quant_pre_bn"],
            r["acc_post_quant_post_bn"], r["acc_final"], r["size_fp32_mb"], r["size_quant_mb"],
            r["compression_ratio"],
        ]) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return md_path, csv_path, rows


def main():
    model_names = list_available_results()
    if not model_names:
        print("No results found in results/. Run pipeline.py --model <name> first.")
        return

    logs = [RunLog.load(name) for name in model_names]
    print(f"found results for: {', '.join(model_names)}")

    for log in logs:
        p1 = plot_training_curves(log)
        p2 = plot_bitwidth_search(log)
        print(f"  [{log.model_name}] wrote {p1}" + (f", {p2}" if p2 else ""))

    if len(logs) >= 1:
        p3 = plot_accuracy_comparison(logs)
        p4 = plot_size_comparison(logs)
        p5 = plot_efficiency_frontier(logs)
        print(f"  wrote {p3}\n  wrote {p4}\n  wrote {p5}")

    md_path, csv_path, rows = write_summary_table(logs)
    print(f"  wrote {md_path}\n  wrote {csv_path}")
    print()
    with open(md_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
