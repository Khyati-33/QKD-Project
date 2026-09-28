import numpy as np
import torch

from models import GNNActorCritic, LSTMActorCritic, stack_observations
from qkd_attention import (all_masked_rows, apply_hard_mask,
                           QKDAttentionHead, make_batched_distribution, make_single_distribution,
                           NoValidActionError)
from qkd_env import QKDRoutingEnv
from ppo import RolloutBuffer, RunningRewardNormalizer, ppo_update


def test_true_negative_infinity_and_batch_all_masked_guard():
    logits = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    masked = apply_hard_mask(logits, mask)
    assert torch.isneginf(masked[0, 1])
    assert torch.isneginf(masked[1]).all()
    assert all_masked_rows(masked).tolist() == [False, True]
    dist = make_batched_distribution(masked)
    assert torch.isfinite(dist.probs).all()
    assert torch.allclose(dist.probs[1], torch.full((3,), 1 / 3))
    try:
        make_single_distribution(masked[1])
    except NoValidActionError:
        pass
    else:
        raise AssertionError("single-transition all-masked row must use explicit fallback")


def test_attention_score_uses_all_eight_candidate_state_features():
    head = QKDAttentionHead(hidden_dim=8, edge_feature_dim=9)
    with torch.no_grad():
        for parameter in head.query.parameters():
            parameter.zero_()
        head.key[0].weight.zero_()
        head.key[0].bias.zero_()
        head.key[0].weight[:, 8:] = 0.1
        head.score.weight.fill_(1.0 / 8.0)
    current = torch.zeros((1, 8))
    destination = torch.zeros((1, 8))
    neighbor = torch.zeros((1, 1, 8))
    features = torch.zeros((1, 1, 9))
    valid = torch.ones((1, 1), dtype=torch.bool)
    baseline = head(current, destination, neighbor, features, valid)
    for feature_index in range(9):
        changed = features.clone()
        changed[0, 0, feature_index] = 0.25
        score = head(current, destination, neighbor, changed, valid)
        assert not torch.allclose(score, baseline)


def test_model_forward_shapes_synthetic_batch():
    env = QKDRoutingEnv()
    observations = [env.reset(seed=s)[0] for s in (3, 4)]
    batch = stack_observations(observations)
    for model_type in (LSTMActorCritic, GNNActorCritic):
        model = (model_type(hidden_dim=16, dropedge_probability=0.05)
                 if model_type is GNNActorCritic else model_type(hidden_dim=16))
        logits, values = model(batch)
        assert logits.shape == (2, env.max_neighbors)
        assert values.shape == (2,)
        assert torch.isfinite(values).all()
        assert (torch.isfinite(logits) | torch.isneginf(logits)).all()


def test_running_reward_normalizer_is_finite_and_resumable():
    normalizer = RunningRewardNormalizer(gamma=0.9)
    values = [normalizer.normalize(reward, done=(i == 2))
              for i, reward in enumerate((100.0, -1.0, 20.0))]
    assert np.isfinite(values).all()
    state = normalizer.state_dict()
    restored = RunningRewardNormalizer()
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_ppo_update_handles_an_all_masked_transition():
    env = QKDRoutingEnv()
    first, _ = env.reset(seed=103)
    second, _ = env.reset(seed=104)
    first["edge_valid_mask"][:] = 0
    model = LSTMActorCritic(hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    buffer = RolloutBuffer()
    buffer.add(first, 0, 0.0, 0.0, -1.0, True, False)
    buffer.add(second, int(np.flatnonzero(second["edge_valid_mask"])[0]),
               0.0, 0.0, 1.0, True, True)
    metrics = ppo_update(model, optimizer, buffer, np.asarray([-1.0, 1.0], dtype=np.float32),
                         np.asarray([-1.0, 1.0], dtype=np.float32), epochs=1, batch_size=2)
    assert all(np.isfinite(value) for value in metrics.values())
