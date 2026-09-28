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
