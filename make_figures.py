#!/usr/bin/env python3
"""Generate the figures in reports/figures from files under artifacts/.

Reads recorded metrics, sampling records, training logs and external comparison
summaries. Does not recompute predictions, metrics or intervals.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ROOT / "reports" / "figures"

COLOR = {"A": "#6b7280", "B": "#2563eb", "C": "#d97706"}
NAME = {"A": "A · zero-shot", "B": "B · natural sampling", "C": "C · mixed 50%"}

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(group, split="test"):
    return load(ART / group / f"{split}_metrics.json")


def fig_main_results():
    subsets = [
        ("mixed", "mixed\nMER"),
        ("zh", "zh\nCER"),
        ("en", "en\nWER"),
        ("overall", "overall\nMER"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.26
    for i, group in enumerate("ABC"):
        values = [metrics(group)[key]["percent"] for key, _ in subsets]
        xs = [j + (i - 1) * width for j in range(len(subsets))]
        bars = ax.bar(xs, values, width, label=NAME[group], color=COLOR[group], zorder=3)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1.5)
    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels([label for _, label in subsets])
    ax.set_ylabel("error rate (%)")
    ax.set_ylim(0, 55)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.set_title("ASCEND test set (n=1,315): lower is better", fontsize=9, pad=22)
    fig.tight_layout()
    fig.savefig(OUT / "f1_main_results.png")
    plt.close(fig)


def fig_cross_corpus_delta():
    rows = []
    test_b, test_c = metrics("B")["mixed"]["percent"], metrics("C")["mixed"]["percent"]
    val_b = metrics("B", "validation")["mixed"]["percent"]
    val_c = metrics("C", "validation")["mixed"]["percent"]
    rows.append(("ASCEND test · mixed MER (single run)", test_c - test_b, None))
    rows.append(("ASCEND validation · mixed MER (single run)", val_c - val_b, None))
    for dataset, label in [
        ("cs_dialogue", "CS-Dialogue · mixed MER"),
        ("fleurs_zh", "FLEURS zh · CER"),
        ("fleurs_en", "FLEURS en · WER"),
    ]:
        comp = load(ART / "external_eval" / dataset / "comparison.json")["comparisons"]["C_minus_B"]
        rows.append((label, comp["delta_pp"], comp.get("ci95_pp")))

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ys = list(range(len(rows)))[::-1]
    for y, (label, delta, ci) in zip(ys, rows):
        if ci:
            ax.plot([ci[0], ci[1]], [y, y], color="#9ca3af", lw=2, solid_capstyle="round", zorder=2)
        ax.plot(delta, y, "o", ms=6, color="#111827", zorder=3)
        ax.text(4.15, y, f"{delta:+.3f}", va="center", fontsize=7.5)
    ax.axvline(0, color="#6b7280", lw=1, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows], fontsize=8)
    ax.set_xlabel("C − B error-rate difference (pp; negative = C better)", fontsize=8.5)
    ax.set_xlim(-3.2, 5.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("In-sample gain does not transfer across corpora", fontsize=9, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "f2_cross_corpus_delta.png")
    plt.close(fig)


def fig_sampling():
    langs = ["mixed", "zh", "en"]
    colors = {"mixed": "#d97706", "zh": "#2563eb", "en": "#6b7280"}
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    for i, group in enumerate("BC"):
        rec = load(ART / group / "sampling.json")
        counts = rec["language_counts"]
        total = sum(counts.values())
        left = 0
        for lang in langs:
            share = 100 * counts[lang] / total
            ax.barh(i, counts[lang], left=left, height=0.5, color=colors[lang], zorder=3,
                    label=lang if i == 0 else None)
            if share >= 8:
                ax.text(left + counts[lang] / 2, i, f"{share:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if lang != "en" else "#f9fafb")
            left += counts[lang]
        ax.text(left + 60, i, f"{total:,} draws · {rec['unique_examples']:,} unique",
                va="center", fontsize=7.5, color="#374151")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([NAME["B"], NAME["C"]], fontsize=8)
    ax.set_xlabel("sampled examples over 800 optimizer updates")
    ax.set_xlim(0, 8200)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.set_title("Language mix actually sampled (manipulation check)", fontsize=9, pad=22)
    fig.tight_layout()
    fig.savefig(OUT / "f3_sampling_mix.png")
    plt.close(fig)


def load_loss(group):
    steps, losses = [], []
    for line in (ART / group / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            steps.append(rec["step"])
            losses.append(rec["loss"])
    return steps, losses


def ema(values, alpha=0.08):
    out, acc = [], None
    for value in values:
        acc = value if acc is None else (1 - alpha) * acc + alpha * value
        out.append(acc)
    return out


def fig_training():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 2.4]})
    steps, losses = load_loss("pilot")
    ax1.plot(steps, losses, color="#9ca3af", lw=0.8, alpha=0.55)
    ax1.plot(steps, ema(losses), color="#111827", lw=1.4)
    ax1.set_title("pilot · 50 steps (thin = raw, thick = EMA)", fontsize=8.5)
    ax1.set_xlabel("optimizer update")
    ax1.set_ylabel("training loss")

    for group, marks in [("B", [(68, "CUDA error"), (500, "sync diagnostic off")]), ("C", [])]:
        steps, losses = load_loss(group)
        ax2.plot(steps, losses, color=COLOR[group], lw=0.7, alpha=0.3)
        ax2.plot(steps, ema(losses), color=COLOR[group], lw=1.5, label=NAME[group])
        for step, text in marks:
            ax2.axvline(step, color=COLOR[group], ls=":", lw=1)
            ax2.annotate(text, xy=(step, 1.92), fontsize=6.5, color=COLOR[group],
                         rotation=90, ha="right", va="top")
    ax2.set_title("B and C · 800 steps (thin = raw, thick = EMA)", fontsize=8.5)
    ax2.set_xlabel("optimizer update")
    ax2.set_ylim(0.2, 1.95)
    handles = [
        plt.Line2D([], [], color="#9ca3af", lw=1.4, label="pilot · 50 steps"),
        plt.Line2D([], [], color=COLOR["B"], lw=1.5, label=NAME["B"]),
        plt.Line2D([], [], color=COLOR["C"], lw=1.5, label=NAME["C"]),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "f4_training_trajectory.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_main_results, fig_cross_corpus_delta, fig_sampling, fig_training):
        fn()
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
