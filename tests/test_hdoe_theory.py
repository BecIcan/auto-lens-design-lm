"""HDOE 论文公式回归测试。

锁定 `W(r_i)=i M lambda_0`、`alpha(lambda)` 和
`eta_m=sinc^2(alpha-m)`，防止实现与论文定义漂移。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from eadld.optimization.hdoe import (
    harmonic_zone_count,
    harmonic_zone_edges,
    ideal_opd,
    normalized_phase_depth,
    scalar_blaze_efficiency,
)


def test_visible_demo_zone_counts_match_exact_formula():
    assert harmonic_zone_count(25.0, 100.0, 480, 550.0) == 12
    assert harmonic_zone_count(25.0, 100.0, 627, 550.0) == 9


def test_internal_zone_edges_invert_exact_parent_opd():
    order = 480
    wavelength_nm = 550.0
    edges = harmonic_zone_edges(25.0, 100.0, order, wavelength_nm)
    period = order * wavelength_nm * 1e-6

    # 每个内部边界必须精确落在整数个 M*lambda_0 光程周期上。
    np.testing.assert_allclose(
        ideal_opd(edges[:-1], 100.0),
        np.arange(1, edges.size) * period,
        rtol=2e-14,
        atol=2e-14,
    )
    assert edges[-1] == 25.0


def test_scalar_blaze_screen_is_unity_at_integer_normalized_depth():
    alpha = normalized_phase_depth(
        diffraction_order=4,
        design_wavelength_nm=550.0,
        wavelengths_nm=[550.0, 1100.0],
        delta_n=[0.5, 0.5],
        design_delta_n=0.5,
    )
    nearest_order, efficiency = scalar_blaze_efficiency(alpha)

    np.testing.assert_array_equal(nearest_order, [4, 2])
    np.testing.assert_allclose(efficiency, 1.0, atol=1e-15, rtol=0)


def test_theory_relations_reject_nonphysical_inputs():
    with pytest.raises(ValueError, match="positive integer"):
        harmonic_zone_count(25.0, 100.0, 0, 550.0)
    with pytest.raises(ValueError, match="must match"):
        normalized_phase_depth(4, 550.0, [500.0, 600.0], [0.5], 0.5)


def test_singlet_generator_uses_exact_spherical_opd_edges(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo_singlet.py", "5.6", "1"])
    path = Path("configs/singlet_achromat/demo_singlet.py")
    spec = importlib.util.spec_from_file_location("exact_singlet_demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    expected = harmonic_zone_edges(
        demo.RADIUS, demo.EFL, 29, demo.DESIGN
    )
    np.testing.assert_allclose(demo.zone_radii(29), expected, rtol=0, atol=1e-14)


def test_near_flat_design_seed_controls_only_curvature(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["demo_singlet.py", "5.6", "1"])
    path = Path("configs/singlet_achromat/demo_singlet.py")
    spec = importlib.util.spec_from_file_location("seeded_singlet_demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    demo.write_near_flat_start(29, first, seed=3)
    demo.write_near_flat_start(29, second, seed=7)
    first_args = yaml.safe_load(first.read_text())["model"]["lens_parameterization"]["init_args"]
    second_args = yaml.safe_load(second.read_text())["model"]["lens_parameterization"]["init_args"]
    assert first_args["c"] != second_args["c"]
    for key in ("s", "nd", "vd", "dpgf", "d", "z"):
        assert first_args[key] == second_args[key]
