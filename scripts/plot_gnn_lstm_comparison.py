"""Plot paired strict held-out results for GNN-PPO versus LSTM-PPO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"GNN-PPO": "#2878B5", "LSTM-PPO": "#D95F02"}


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if not n:
        return 0.0, 0.0
    p = successes / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)


def save(fig, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_primary(report, output_dir: Path):
    methods = ("GNN-PPO", "LSTM-PPO")
    colors = [COLORS[m] for m in methods]
    summaries = report["summary"]
    ci = report["nested_training_scenario_ci95"]
    rates = [summaries[m]["success_rate"] for m in methods]
    lower = [rates[i] - ci[m]["ci95"][0] for i, m in enumerate(methods)]
    upper = [ci[m]["ci95"][1] - rates[i] for i, m in enumerate(methods)]
    hop_means = []
    hop_sd = []
    for method in methods:
        vals = [report["per_training_seed"][method][str(s)]["successful_mean_hops"]
                for s in sorted(int(x) for x in report["per_training_seed"][method])]
        vals = np.asarray([v for v in vals if v is not None], dtype=float)
        hop_means.append(float(vals.mean()) if len(vals) else 0.0)
        hop_sd.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    fig, axes = plt.subplots(2, 1, figsize=(3.48, 4.5))
    x = np.arange(2)
    axes[0].bar(x, rates, color=colors, edgecolor="black", linewidth=.45, width=.58)
    axes[0].errorbar(x, rates, yerr=[lower, upper], fmt="none", ecolor="black",
                     capsize=4, linewidth=1)
    axes[0].set_xticks(x, methods)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Route completion")
    axes[0].set_title("Completion (nested 95% CI)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100*y:.0f}%"))
    axes[1].bar(x, hop_means, yerr=hop_sd, color=colors, edgecolor="black",
                linewidth=.45, width=.58, capsize=4)
    axes[1].set_xticks(x, methods)
    axes[1].set_ylabel("Successful-route hops")
    axes[1].set_title("Mean checkpoint average (SD)")
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=.3)
        ax.set_axisbelow(True)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=8)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    fig.tight_layout()
    save(fig, output_dir / "01_gnn_lstm_primary")


def plot_conditions(report, output_dir: Path):
    episodes = report["episodes"]
    learned = [r for r in episodes if r["method"] in COLORS]
    conditions = sorted({(r["season"], int(r["hour"])) for r in learned},
                        key=lambda x: (x[0], x[1]))
    labels = [f"{season.title()}\n{hour:02d}:00" for season, hour in conditions]
    fig, ax = plt.subplots(figsize=(3.48, 2.45))
    x = np.arange(len(conditions))
    width = .34
    for i, method in enumerate(("GNN-PPO", "LSTM-PPO")):
        means, lows, highs = [], [], []
        for season, hour in conditions:
            seed_rates = []
            for training_seed in sorted({int(r["training_seed"]) for r in learned}):
                rows = [r for r in learned if r["method"] == method and
                        int(r["training_seed"]) == training_seed and r["season"] == season and
                        int(r["hour"]) == hour]
                if rows:
                    seed_rates.append(sum(bool(r["success"]) for r in rows) / len(rows))
            means.append(float(np.mean(seed_rates)) if seed_rates else 0.0)
            sd = float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0
            lows.append(sd)
            highs.append(sd)
        pos = x + (i-.5)*width
        ax.bar(pos, means, width, label=method, color=COLORS[method],
               edgecolor="black", linewidth=.4)
        ax.errorbar(pos, means, yerr=[lows, highs], fmt="none", ecolor="black",
                    capsize=2.5, linewidth=.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Route completion")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100*y:.0f}%"))
    ax.set_title("Completion by season/time")
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.18), fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.grid(axis="y", linestyle="--", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, output_dir / "02_gnn_lstm_by_condition")


def plot_latency(report, output_dir: Path):
    latency = report.get("inference_latency", {})
    methods = ("GNN-PPO", "LSTM-PPO")
    metric = "p95_ms"
    by_method = {m: [] for m in methods}
    for key, values in latency.items():
        arch = key.split("|", 1)[0]
        method = f"{arch}-PPO"
        if method in by_method and values.get(metric) is not None:
            by_method[method].append(float(values[metric]))
    means = [float(np.mean(by_method[m])) if by_method[m] else 0.0 for m in methods]
    errors = [float(np.std(by_method[m], ddof=1)) if len(by_method[m]) > 1 else 0.0
              for m in methods]
    fig, ax = plt.subplots(figsize=(3.48, 2.4))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=errors, color=[COLORS[m] for m in methods],
           edgecolor="black", linewidth=.45, width=.58, capsize=4)
    ax.set_xticks(x, methods)
    ax.set_ylabel("p95 actor-call latency (ms)")
    ax.set_title("CPU actor-call p95 latency")
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.grid(axis="y", linestyle="--", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, output_dir / "04_gnn_lstm_latency")


def plot_seed_variability(report, output_dir: Path):
    seeds = sorted(int(x) for x in report["per_training_seed"]["GNN-PPO"])
    labels = [str(seed)[-2:] for seed in seeds]
    fig, ax = plt.subplots(figsize=(3.48, 2.4))
    x = np.arange(len(seeds))
    for method in ("GNN-PPO", "LSTM-PPO"):
        rates = [report["per_training_seed"][method][str(seed)]["success_rate"] for seed in seeds]
        ax.plot(x, rates, marker="o", linewidth=1.4, markersize=4,
                color=COLORS[method], label=method)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Training seed suffix")
    ax.set_ylabel("Held-out route completion")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100*y:.0f}%"))
    ax.set_ylim(0, 1.05)
    ax.set_title("Completion across training seeds")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.grid(axis="y", linestyle="--", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, output_dir / "03_gnn_lstm_seed_variability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("experiments/final_gnn_lstm_holdout_20261007.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/final_evaluation"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise SystemExit("Comparison report is not complete")
    plot_primary(report, args.output_dir)
    plot_conditions(report, args.output_dir)
    plot_seed_variability(report, args.output_dir)
    plot_latency(report, args.output_dir)


if __name__ == "__main__":
    main()
