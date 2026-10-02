"""Add paired GNN-vs-BFS outcome and route-length analysis to the matrix report."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def exact_mcnemar_p(gnn_only: int, baseline_only: int) -> float:
    discordant = gnn_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i)
               for i in range(min(gnn_only, baseline_only) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] /
                        "paper" / "supplementary" / "pair_season_matrix_20261002.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1313)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        parser.error("at least 1000 bootstrap replicates are required")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("paired analysis requires a complete episode report")
    episodes = report["episodes"]
    identity = lambda row: (row["source"], row["destination"], row["season"],
                            row["hour"], row["replicate"], row["seed"])
    indexed = {}
    for row in episodes:
        key = (row["method"], *identity(row))
        if key in indexed:
            raise ValueError(f"duplicate method/episode record: {key}")
        indexed[key] = row

    gnn_rows = [row for row in episodes if row["method"] == "GNN-PPO"]
    paired = []
    for gnn in gnn_rows:
        baseline = indexed.get(("BFS-hop", *identity(gnn)))
        if baseline is None:
            raise ValueError(f"missing paired BFS episode for {identity(gnn)}")
        paired.append((gnn, baseline))
    if len(paired) != len(gnn_rows):
        raise ValueError("not every GNN episode has one paired BFS episode")

    gnn_only = sum(g["success"] and not b["success"] for g, b in paired)
    bfs_only = sum(not g["success"] and b["success"] for g, b in paired)
    both_success = [(g, b) for g, b in paired if g["success"] and b["success"]]
    hop_groups: dict[str, list[float]] = defaultdict(list)
    for gnn, bfs in both_success:
        cluster = f"{gnn['source']}->{gnn['destination']}"
        hop_groups[cluster].append(float(gnn["hops"] - bfs["hops"]))
    hop_deltas = [value for values in hop_groups.values() for value in values]
    rng = np.random.default_rng(args.seed)
    clusters = list(hop_groups)
    boot_means = np.empty(args.bootstrap_replicates, dtype=float)
    for i in range(args.bootstrap_replicates):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [value for cluster in sampled_clusters for value in hop_groups[cluster]]
        boot_means[i] = np.mean(sample)

    report.setdefault("paired_analysis", {})["GNN-PPO_vs_BFS-hop"] = {
        "paired_episodes": len(paired),
        "gnn_only_successes": gnn_only,
        "bfs_only_successes": bfs_only,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(gnn_only, bfs_only),
        "both_successful_episodes": len(both_success),
        "hop_difference_definition": "GNN-PPO hops minus BFS-hop hops; positive favors fewer BFS hops",
        "mean_paired_hop_difference": float(np.mean(hop_deltas)) if hop_deltas else None,
        "median_paired_hop_difference": float(np.median(hop_deltas)) if hop_deltas else None,
        "hop_difference_95pct_pair_cluster_bootstrap_ci": [
            float(value) for value in np.percentile(boot_means, [2.5, 97.5])],
        "bootstrap": {"clusters": "ordered endpoint pair", "cluster_count": len(clusters),
            "replicates": args.bootstrap_replicates, "seed": args.seed},
        "interpretation": "BFS succeeded in every paired episode; among shared successes, the GNN used more hops on average. The bootstrap resamples endpoint-pair clusters, not individual episodes.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["paired_analysis"]["GNN-PPO_vs_BFS-hop"], indent=2))


if __name__ == "__main__":
    main()
