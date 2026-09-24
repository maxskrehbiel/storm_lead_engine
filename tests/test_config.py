"""Configuration validation."""

import pytest

from storm_lead_engine.config import (
    BBox,
    ConfidenceWeights,
    EngineConfig,
    HailTiers,
    LeadWeights,
)


def test_bbox_parse_and_contains() -> None:
    b = BBox.parse("-1.5,-0.5,0.5,1.5")
    assert b.as_tuple() == (-1.5, -0.5, 0.5, 1.5)
    assert b.contains(0.0, 0.0)
    assert not b.contains(2.0, 0.0)
    assert b.center == pytest.approx((0.5, -0.5))


@pytest.mark.parametrize("text", ["1,2,3", "-100,40,-101,41", "-100,41,-99,40"])
def test_bbox_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        BBox.parse(text)


def test_tiers_must_increase() -> None:
    assert HailTiers().as_tuple() == (1.0, 1.5, 2.0)
    with pytest.raises(ValueError):
        HailTiers(severe=1.5, damaging=1.0, extreme=2.0)


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        LeadWeights(hail=0.5)
    with pytest.raises(ValueError, match="sum to 1"):
        ConfidenceWeights(severe=0.5)


def test_engine_config_defaults_are_consistent() -> None:
    cfg = EngineConfig()
    assert cfg.weights.min_lead_hail_in == cfg.tiers.severe
    assert cfg.history.neighborhood == 1
