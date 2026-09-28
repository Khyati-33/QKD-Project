import random

import pytest

from quantum_protocol import (DecoyBB84Profile, DetectorNoiseProfile,
                              detector_click_probability,
                              finite_key_decoy_bb84)
from uncertainty import UncertainParameter


def test_detector_noise_probability_is_bounded():
    profile = DetectorNoiseProfile()
    assert 0.0 < detector_click_probability(0.5, profile) <= 1.0
    assert detector_click_probability(0.0, profile) > 0.0


def test_finite_key_estimator_rejects_bad_channel():
    result = finite_key_decoy_bb84(
        transmission=1e-12, qber=0.2,
        source=DecoyBB84Profile(pulses=100_000), rng=random.Random(2))
    assert result["secure_key_rate"] == pytest.approx(0.0)
    assert result["single_photon_yield_lower"] >= 0.0


def test_finite_key_estimator_is_reproducible():
    profile = DecoyBB84Profile(pulses=100_000)
    left = finite_key_decoy_bb84(transmission=0.2, qber=0.02,
                                 source=profile, rng=random.Random(9))
    right = finite_key_decoy_bb84(transmission=0.2, qber=0.02,
                                  source=profile, rng=random.Random(9))
    assert left["secure_key_rate"] == right["secure_key_rate"]


def test_uncertainty_parameter_is_traceable_and_bounded():
    parameter = UncertainParameter("detector_efficiency", 0.6, 0.05,
                                   source="profile", status="assumption")
    assert parameter.sample(random.Random(3)) > 0.0
    low, high = parameter.bounds()
    assert low < parameter.nominal < high
