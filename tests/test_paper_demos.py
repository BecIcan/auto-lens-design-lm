"""论文演示处方与独立审计入口的轻量契约测试。"""

from pathlib import Path

import pytest
import yaml

from eadld.desktop.backend import CASE_MODELS, CASE_PRESETS


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("case", ["singlet", "triplet", "four_element"])
def test_selected_zone_count_matches_final_prescription(case):
    path = ROOT / CASE_MODELS[case]["final_design"]
    args = yaml.safe_load(path.read_text(encoding="utf-8"))["model"][
        "lens_parameterization"
    ]["init_args"]
    active_zones = sum(zone[3] > 0 for zone in args["z"][0])

    assert active_zones == CASE_PRESETS[case]["zone_count"]


def test_four_element_replay_seed_uses_selected_topology():
    path = ROOT / CASE_MODELS["four_element"]["design"]
    args = yaml.safe_load(path.read_text(encoding="utf-8"))["model"][
        "lens_parameterization"
    ]["init_args"]

    assert len(args["z"][0]) == CASE_PRESETS["four_element"]["zone_count"]
