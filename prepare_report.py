"""Regenerate report tables and figures from the recorded experiments."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "report_experiments"


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as source:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(source)]


def best_row(rows):
    return max(rows, key=lambda row: row["dev_acc"])


def command(name, value):
    return "\\newcommand{\\" + name + "}{" + str(value) + "}\n"


def row_command(name, rows):
    return "\\newcommand{\\" + name + "}{%\n" + "".join(
        " & ".join(row) + r" \\" + "\n" for row in rows
    ) + "}\n"


def main():
    original = {
        "Glove_DAN": read_metrics(ROOT / "Glove_DAN_metrics.csv"),
        "Embeddings_DAN": read_metrics(ROOT / "Embeddings_DAN_metrics.csv"),
    }
    names = ["Glove_DAN", "Embeddings_DAN"] + [
        f"SUBWORDDAN_vocab{size}" for size in (1000, 3000, 5000)
    ]
    metrics = {name: read_metrics(EXPERIMENTS / f"{name}_metrics.csv") for name in names}
    summaries = {name: json.loads((EXPERIMENTS / f"{name}_summary.json").read_text(encoding="utf-8"))
                 for name in names}
    environment = json.loads((EXPERIMENTS / "environment.json").read_text(encoding="utf-8"))

    numbers = command("PythonVersion", environment["python"])
    numbers += command("TorchVersion", environment["torch"])
    part_one_rows = []
    for name, label, prefix in (("Glove_DAN", "Frozen GloVe", "G"),
                                 ("Embeddings_DAN", "Random embeddings", "R")):
        rows = original[name]
        best, final = best_row(rows), rows[-1]
        part_one_rows.append([label, f"{100 * final['train_acc']:.2f}",
                              f"{100 * final['dev_acc']:.2f}",
                              f"{100 * best['dev_acc']:.2f}", str(int(best["epoch"]))])
        numbers += command(prefix + "FullBest", f"{100 * best['dev_acc']:.2f}")
        numbers += command(prefix + "FullFinal", f"{100 * final['dev_acc']:.2f}")
        numbers += command(prefix + "FullTrain", f"{100 * final['train_acc']:.2f}")
    numbers += row_command("PartOneRows", part_one_rows)

    controlled_rows = []
    for name in names:
        best, final = best_row(metrics[name]), metrics[name][-1]
        summary = summaries[name]
        label = "Frozen GloVe" if name == "Glove_DAN" else "Random words" if name == "Embeddings_DAN" else "BPE"
        vocab = summary.get("actual_vocab_size", 14923)
        controlled_rows.append([label, f"{vocab:,}", f"{100 * best['dev_acc']:.2f}",
                                str(int(best["epoch"])), f"{100 * final['dev_acc']:.2f}",
                                f"{summary['training_and_evaluation_seconds']:.2f}"])
    numbers += row_command("ControlledRows", controlled_rows)

    efficiency_rows = []
    bpe_names = names[2:]
    for name in bpe_names:
        summary = summaries[name]
        params = summary["actual_vocab_size"] * 300 + 30302
        efficiency_rows.append([f"{summary['actual_vocab_size']:,}", f"{summary['merge_count']:,}",
                                f"{summary['train_mean_subwords']:.2f}",
                                f"{summary['bpe_training_seconds']:.2f}",
                                f"{summary['data_preparation_seconds']:.2f}", f"{params:,}"])
    numbers += row_command("EfficiencyRows", efficiency_rows)
    winner = max(bpe_names, key=lambda name: best_row(metrics[name])["dev_acc"])
    tied_sizes = [f"{summaries[name]['actual_vocab_size']:,}" for name in bpe_names
                  if best_row(metrics[name])["dev_acc"] == best_row(metrics[winner])["dev_acc"]]
    numbers += command("BestBPEVocab", " and ".join(tied_sizes))
    numbers += command("BestBPEAccuracy", f"{100 * best_row(metrics[winner])['dev_acc']:.2f}")
    numbers += command("ControlledRandomBest", f"{100 * best_row(metrics['Embeddings_DAN'])['dev_acc']:.2f}")
    numbers += command("ControlledGloveBest", f"{100 * best_row(metrics['Glove_DAN'])['dev_acc']:.2f}")
    numbers += command("SmallBPELength", f"{summaries[bpe_names[0]]['train_mean_subwords']:.2f}")
    numbers += command("LargeBPELength", f"{summaries[bpe_names[-1]]['train_mean_subwords']:.2f}")
    (EXPERIMENTS / "report_numbers.tex").write_text(numbers, encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})
    figures = EXPERIMENTS / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), layout="constrained")
    for name, label, color in (("Glove_DAN", "GloVe", "#2166ac"),
                                ("Embeddings_DAN", "Random", "#b35806")):
        rows = original[name]
        epochs = [row["epoch"] for row in rows]
        for split, style in (("train", "-"), ("dev", "--")):
            axes[0].plot(epochs, [row[f"{split}_loss"] for row in rows], style,
                         color=color, label=f"{label} {split}")
            axes[1].plot(epochs, [100 * row[f"{split}_acc"] for row in rows], style,
                         color=color, label=f"{label} {split}")
    axes[0].set(title="Word-level cross-entropy", ylabel="Loss")
    axes[1].set(title="Word-level accuracy", ylabel="Accuracy (%)", ylim=(45, 102))
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(figures / "word_curves.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), layout="constrained")
    for name, color in zip(bpe_names, ("#2166ac", "#b35806", "#1b7837")):
        rows = metrics[name]
        epochs = [row["epoch"] for row in rows]
        label = f"V = {summaries[name]['actual_vocab_size']:,}"
        axes[0].plot(epochs, [row["dev_loss"] for row in rows], marker="o", markersize=3,
                     color=color, label=label)
        axes[1].plot(epochs, [100 * row["dev_acc"] for row in rows], marker="o", markersize=3,
                     color=color, label=label)
    axes[0].set(title="BPE development cross-entropy", ylabel="Loss")
    axes[1].set(title="BPE development accuracy", ylabel="Accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(figures / "bpe_curves.png", dpi=300)
    plt.close(fig)
    print("Regenerated report_numbers.tex and both report figures from recorded results.")


if __name__ == "__main__":
    main()
