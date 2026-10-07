"""Create four publication plots from the frozen held-out/stress evaluation JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Random": "#7f7f7f", "Dijkstra-km": "#d95f02", "BFS-hop": "#e6ab02",
          "Max-SKR": "#1f78b4", "Weighted-SP": "#a6761d",
          "GNN-PPO": "#33a02c", "MLP-PPO": "#6a3d9a"}
DISPLAY = {"Random": "Random", "Dijkstra-km": "Dijkstra", "BFS-hop": "BFS",
           "Max-SKR": "Max-SKR", "Weighted-SP": "Weighted-SP",
           "GNN-PPO": "GNN-PPO", "MLP-PPO": "MLP-PPO"}


def save(fig, path: Path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_success(report, output):
    methods = ["Random", "Dijkstra-km", "BFS-hop", "Max-SKR", "Weighted-SP", "GNN-PPO", "MLP-PPO"]
    means, errors = [], []
    for method in methods:
        if method in report["baseline_summary"]:
            means.append(report["baseline_summary"][method]["success_rate"])
            errors.append(0.0)
        else:
            row = report["training_seed_uncertainty_success_rate"][method]
            means.append(row["mean"])
            errors.append(row["std_sample"])
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=errors, capsize=4, color=[COLORS[m] for m in methods],
           edgecolor="black", linewidth=.55)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Held-out route completion rate")
    ax.set_xticks(x, [DISPLAY[m] for m in methods], rotation=18, ha="right")
    ax.set_xlabel("Routing policy")
    ax.grid(axis="y", linestyle="--", alpha=.3)
    ax.set_title("Paired held-out route completion (mean ± training-seed SD)")
    save(fig, output / "01_heldout_success_rate")


def plot_hops(report, output):
    methods = ["Random", "Dijkstra-km", "BFS-hop", "Max-SKR", "Weighted-SP", "GNN-PPO", "MLP-PPO"]
    means, errors = [], []
    for method in methods:
        if method in report["baseline_summary"]:
            value = report["baseline_summary"][method]["successful_mean_hops"]
            means.append(float(value) if value is not None else np.nan)
            errors.append(0.0)
        else:
            per_seed = [v["successful_mean_hops"] for k, v in report["summary_by_method_training_seed"].items()
                        if k.startswith(method + "|") and v["successful_mean_hops"] is not None]
            means.append(float(np.mean(per_seed)) if per_seed else np.nan)
            errors.append(float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=errors, capsize=4, color=[COLORS[m] for m in methods],
           edgecolor="black", linewidth=.55)
    for index, value in enumerate(means):
        if np.isnan(value):
            ax.annotate("N/A", (x[index], 0), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean hops, successful episodes only")
    ax.set_xticks(x, [DISPLAY[m] for m in methods], rotation=18, ha="right")
    ax.set_xlabel("Routing policy"); ax.grid(axis="y", linestyle="--", alpha=.3)
    ax.set_title("Held-out successful-route length (mean ± training-seed SD)")
    save(fig, output / "02_successful_route_hops")


def plot_conditions(report, output):
    methods = ["BFS-hop", "Weighted-SP", "GNN-PPO", "MLP-PPO"]
    condition_order = [(s, h) for s in ("normal", "summer", "winter") for h in (2, 22)]
    data = {m: [] for m in methods}
    for method in methods:
        for season, hour in condition_order:
            key = f"{method}|{season}|{hour:02d}h"
            data[method].append(report["summary_by_method_condition"][key]["success_rate"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, hour, indices in ((axes[0], 2, [0, 2, 4]), (axes[1], 22, [1, 3, 5])):
        x = np.arange(3)
        for method in methods:
            ax.plot(x, [data[method][i] for i in indices], marker="o", linewidth=1.7,
                    label=DISPLAY[method], color=COLORS[method])
        ax.set_xticks(x, ["Normal", "Summer", "Winter"])
        ax.set_title(f"{hour:02d}:00")
        ax.set_ylim(-.03, 1.03); ax.grid(True, linestyle="--", alpha=.3)
        ax.set_xlabel("Held-out weather condition")
    axes[0].set_ylabel("Route completion rate")
    axes[1].legend(frameon=True, fontsize=8, loc="best")
    fig.suptitle("Held-out completion by weather and time condition")
    fig.tight_layout()
    save(fig, output / "03_success_by_condition")


def plot_outage(report, output):
    methods = ["BFS-hop", "Weighted-SP", "GNN-PPO", "MLP-PPO"]
    levels = report["fiber_outage_stress_levels"]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method in methods:
        vals, deviations = [], []
        for level in levels:
            if method in ("GNN-PPO", "MLP-PPO"):
                per_seed = [v["success_rate"] for k, v in
                    report["fiber_outage_stress_summary_by_training_seed"].items()
                    if k.startswith(method + "|") and k.split("|")[1] == f"outage={level:.2f}"]
                vals.append(float(np.mean(per_seed)) if per_seed else np.nan)
                deviations.append(float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0)
            else:
                vals.append(report["fiber_outage_stress_summary"][f"{method}|outage={level:.2f}"]["success_rate"])
                deviations.append(0.0)
        ax.errorbar(np.array(levels)*100, vals, yerr=deviations, marker="o", linewidth=1.8,
                capsize=3, label=DISPLAY[method], color=COLORS[method])
    ax.set_xticks(np.array(levels)*100, [f"{100*x:.0f}%" for x in levels])
    ax.set_ylim(-.03, 1.03); ax.set_xlabel("Fiber-link outage probability")
    ax.set_ylabel("Route completion rate"); ax.grid(True, linestyle="--", alpha=.3)
    ax.set_title("Paired fiber-outage stress sweep (learned-policy mean ± seed SD)")
    ax.legend(frameon=True)
    save(fig, output / "04_fiber_outage_stress")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "experiments" / "final_holdout_20261007.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures" / "final_evaluation")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or "fiber_outage_stress_summary" not in report:
        raise RuntimeError("Complete paired and stress evaluation data are required before plotting.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_success(report, args.output_dir)
    plot_hops(report, args.output_dir)
    plot_conditions(report, args.output_dir)
    plot_outage(report, args.output_dir)
    (args.output_dir / "README.md").write_text(
        f"# Final evaluation figures\n\nGenerated from `{args.input.as_posix()}`. "
        "Each figure is exported as vector PDF and 400 dpi PNG. Learned-policy bars show the "
        "across-training-seed standard deviation; failure episodes remain included in completion rates.\n",
        encoding="utf-8")
    print(f"Saved four figures in {args.output_dir}")


if __name__ == "__main__":
    main()
