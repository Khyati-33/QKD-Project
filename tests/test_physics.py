import random

import pytest

from physics import (FIBER_OUTAGE_PROB, QBER_HARD, SEASONS, fiber_qber,
                     edge_is_valid, fso_viability_probability, sample_fiber_outage,
                     sample_link_state, temperature_at_time)


def test_fiber_qber_and_outage():
    values = [fiber_qber(78, t / 10) for t in range(240)]
    assert min(values) >= 0.01053 and max(values) <= 0.01057
    assert max(values) < QBER_HARD
    assert temperature_at_time(6) == pytest.approx(296.0)
    assert temperature_at_time(18) == pytest.approx(290.0)
    # Ambient diurnal temperature must not drive cryogenic SNSPD dark counts.
    assert fiber_qber(78, 6) == pytest.approx(fiber_qber(78, 18))
    rng = random.Random(123)
    n = 100_000
    observed = sum(sample_fiber_outage(rng) for _ in range(n)) / n
    assert abs(observed - FIBER_OUTAGE_PROB) < 0.002


def test_fso_seasonal_viability_ranges():
    targets = {2: (0.99, 1.0), 5: (0.70, 1.0), 10: (0.49, 0.93),
               15: (0.42, 0.82), 30: (0.29, 0.60), 78: (0.0, 0.10)}
    for distance, (lo, hi) in targets.items():
        values = [fso_viability_probability(distance, season) for season in SEASONS]
        assert min(values) >= lo and max(values) <= hi
    assert len({fso_viability_probability(30, season) for season in SEASONS}) == 4
    for distance in targets:
        for season_index, season in enumerate(SEASONS):
            p = fso_viability_probability(distance, season)
            rng = random.Random(100 + 13 * distance + season_index)
            trials = 2_000
            observed = sum(not sample_link_state("fso", distance, season, rng=rng)["outage"]
                           for _ in range(trials)) / trials
            assert observed == pytest.approx(p, abs=0.035)


def test_fso_physics_changes_by_hour_and_season():
    # Reuse the same RNG seed to keep the availability draw identical; the
    # sampled atmospheric conditions and QBER should still change by hour.
    night = sample_link_state("fso", 10, "monsoon", 2.0, random.Random(91))
    day = sample_link_state("fso", 10, "monsoon", 13.0, random.Random(91))
    winter = sample_link_state("fso", 10, "winter", 13.0, random.Random(91))
    assert night["qber"] != pytest.approx(day["qber"])
    assert night["skr"] != pytest.approx(day["skr"])
    assert day["qber"] != pytest.approx(winter["qber"])
    assert day["qber"] < QBER_HARD and day["skr"] > 0.0
    assert night["conditions"]["sun_elevation"] == 0.0
    assert day["conditions"]["sun_elevation"] > 0.9


def test_shared_validity_boundary_is_strict():
    assert not edge_is_valid(QBER_HARD, 1.0)
    assert edge_is_valid(QBER_HARD - 1e-8, 1.0)
    assert not edge_is_valid(0.01, 0.0)
