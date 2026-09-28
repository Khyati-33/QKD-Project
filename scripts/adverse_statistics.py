"""Bootstrap and paired sign statistics for adverse-condition comparisons."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "adverse_conditions_comparison.json"
OUTPUT = ROOT / "experiments" / "adverse_statistics.json"


def bootstrap(values: list[float], rng: random.Random, draws: int = 5000) -> tuple[float, float, float]:
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[len(means) // 2], means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def paired_sign_pvalue(differences: list[float]) -> float:
    nonzero = [d for d in differences if abs(d) > 1e-12]
    if not nonzero:
        return 1.0
    positive = sum(d > 0 for d in nonzero)
    n = len(nonzero)
    tail = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * min(tail, 1.0 - tail + math.comb(n, positive) / (2 ** n)))


def main() -> None:
    report = json.loads(INPUT.read_text(encoding="utf8"))
    output = {"comparison": "GNN-200ep minus BFS-hop", "scenarios": {}}
    for scenario, methods in report["scenarios"].items():
        gnn = methods["GNN-200ep"]["per_season"]["monsoon"]["seed_rewards"]
        bfs = methods["BFS-hop"]["per_season"]["monsoon"]["seed_rewards"]
        rewards = [a["reward"] - b["reward"] for a, b in zip(gnn, bfs)]
        success = [float(a["success"]) - float(b["success"]) for a, b in zip(gnn, bfs)]
        rng = random.Random(20260928 + len(scenario))
        median, low, high = bootstrap(rewards, rng)
        output["scenarios"][scenario] = {"paired_reward_differences": rewards,
            "reward_difference_bootstrap_median": median,
            "reward_difference_bootstrap_95ci": [low, high],
            "success_difference": success,
            "paired_sign_pvalue_reward": paired_sign_pvalue(rewards)}
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
