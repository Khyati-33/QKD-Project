from baselines import BaselinePolicy
from evaluation import evaluate_policy


def test_repeated_seasons_accumulate_and_report_seed_variation_and_validity():
    result = evaluate_policy(BaselinePolicy("Dijkstra-km"),
                             seasons=("normal", "normal"),
                             eval_seeds_per_season=1,
                             env_kwargs={"max_steps": 30}, seed_base=123)
    season = result["per_season"]["normal"]
    assert season["episodes"] == 2
    assert len({row["seed"] for row in season["seed_rewards"]}) == 2
    assert "reward_std" in season
    assert result["qber_skr_validity"]["checked_edges"] > 0
    assert result["qber_skr_validity"]["invalid_edges"] == 0
    assert "sample_path_revisits" in result
    assert "zero_success_epochs" in result
from baselines import BaselinePolicy
from evaluation import evaluate_policy


def test_randomized_endpoint_evaluation_records_each_pair():
    summary = evaluate_policy(
        BaselinePolicy("BFS-hop"), seasons=("normal",), eval_seeds_per_season=4,
        env_kwargs={"randomize_endpoints": True}, seed_base=123)
    pairs = summary["per_season"]["normal"]["seed_rewards"]
    assert len(pairs) == 4
    assert all(row["source"] != row["destination"] for row in pairs)
    assert all(row["source"] and row["destination"] for row in pairs)
