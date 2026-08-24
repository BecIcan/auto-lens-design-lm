"""原生桌面控制台的输入与配置映射测试。"""

from pathlib import Path

import numpy as np
import pytest

from eadld.desktop.backend import (
    CASE_MODELS,
    CASE_PRESETS,
    DEFAULTS,
    DEMO_DEFAULTS,
    RunManager,
    build_command,
    build_overlay,
    validate_parameters,
)
from eadld.utils.visualization import (
    compute_mtf_slices,
    diffraction_limited_mtf,
    mtf_frequency_limit,
)


def test_desktop_inputs_map_to_optical_specification(tmp_path):
    params = validate_parameters(DEFAULTS | {"visual_every": 200, "steps": 20})
    overlay = build_overlay(params, tmp_path)
    ray = overlay["model"]["ray_initialization"]["init_args"]

    assert ray["aperture"] == pytest.approx(100 / 2.8)
    assert ray["wavelengths"] == [486.1, 550.0, 656.3]
    assert ray["wavelength_weights"] == [1.0, 1.0, 1.0]
    assert (
        overlay["model"]["lens_parameterization"]["init_args"][
            "nominal_wavelength"
        ]
        == 550.0
    )
    assert params["visual_every"] == 20
    assert overlay["trainer"]["max_steps"] == 20
    callback = overlay["trainer"]["callbacks"][-1]
    assert callback["class_path"] == "eadld.desktop.backend.DesktopVisualizationCallback"
    analysis = overlay["trainer"]["callbacks"][-2]
    assert analysis["class_path"] == "eadld.desktop.backend.DesktopAnalysisCallback"
    assert analysis["init_args"]["every_n_steps"] == 20
    assert callback["init_args"]["every_n_steps"] == 20
    assert overlay["model"]["optics_simulator"]["init_args"] == {
        "shape": [129, 129],
        "psf_abs_size": 0.04,
        "psf_grid_shape": [1, 5],
    }


def test_desktop_rejects_invalid_f_number():
    with pytest.raises(ValueError, match="f_number"):
        validate_parameters(DEFAULTS | {"f_number": 0.5})


def test_desktop_rejects_zero_wavelength_weight_sum():
    with pytest.raises(ValueError, match="至少一个波长权重"):
        validate_parameters(DEFAULTS | {"wavelength_weights": [0, 0, 0]})


def test_desktop_rejects_unknown_demo_case():
    with pytest.raises(ValueError, match="demo_case"):
        validate_parameters(DEFAULTS | {"demo_case": "toy"})


def test_desktop_primary_wavelength_controls_reference(tmp_path):
    params = validate_parameters(
        DEFAULTS
        | {
            "wavelengths": [450.0, 550.0, 650.0],
            "wavelength_weights": [0.5, 1.0, 0.25],
            "primary_wavelength": 2,
        }
    )
    overlay = build_overlay(params, tmp_path)

    assert overlay["model"]["ray_initialization"]["init_args"][
        "wavelength_weights"
    ] == [0.5, 1.0, 0.25]
    assert overlay["model"]["lens_parameterization"]["init_args"][
        "nominal_wavelength"
    ] == 650.0


@pytest.mark.parametrize("case", ["singlet", "triplet", "four_element"])
def test_desktop_cases_use_promoted_designs(case, tmp_path):
    params = validate_parameters(CASE_PRESETS[case])
    overlay = build_overlay(params, tmp_path)
    design = Path(__file__).parents[1] / CASE_MODELS[case]["design"]
    final_design = Path(__file__).parents[1] / CASE_MODELS[case]["final_design"]

    assert design.exists()
    assert final_design.exists()
    assert "eisoptx" not in design.read_text(encoding="utf-8")
    assert "eisoptx" not in final_design.read_text(encoding="utf-8")
    assert overlay["model"]["ray_initialization"]["init_args"]["hfov"] == params[
        "half_field"
    ]


def test_singlet_keeps_recorded_m30_phase_constraint(tmp_path):
    overlay = build_overlay(validate_parameters(CASE_PRESETS["singlet"]), tmp_path)
    phase = [
        residual
        for residual in overlay["model"]["residuals"]
        if residual["class_path"].endswith("HDOEPhaseResiduals")
    ]

    assert len(phase) == 1
    assert phase[0]["init_args"]["diffraction_order"] == 30
    assert phase[0]["init_args"]["design_wavelength"] == 550.0
    assert overlay["model"]["optimization_pupil_quadrature"] is False


def test_only_newer_triplet_history_uses_pupil_quadrature(tmp_path):
    enabled = {}
    for case, preset in CASE_PRESETS.items():
        params = validate_parameters(preset)
        enabled[case] = build_overlay(params, tmp_path)["model"][
            "optimization_pupil_quadrature"
        ]

    assert enabled == {"singlet": False, "triplet": True, "four_element": False}


def test_demo_uses_paper_field_grid_and_three_wavelengths():
    assert DEMO_DEFAULTS["demo_case"] == "four_element"
    assert DEMO_DEFAULTS["f_number"] == 2.0
    assert DEMO_DEFAULTS["half_field"] == 15.88
    assert DEMO_DEFAULTS["n_fields"] == 11
    assert DEMO_DEFAULTS["wavelength_weights"] == [1.0, 1.0, 1.0]
    assert DEMO_DEFAULTS["steps"] == 750


def test_four_element_keeps_sensor_and_track_constraints(tmp_path):
    params = validate_parameters(CASE_PRESETS["four_element"])
    residuals = build_overlay(params, tmp_path)["model"]["residuals"]
    by_name = {item["class_path"].rsplit(".", 1)[-1]: item for item in residuals}

    image_height = by_name["ImageHeightResiduals"]["init_args"]
    assert image_height["target"] == pytest.approx(
        params["target_efl"] * np.tan(np.deg2rad(params["half_field"]))
    )
    assert by_name["TotalTrackLengthResiduals"]["init_args"]["target"] == 39.3


@pytest.mark.parametrize("case", ["singlet", "triplet", "four_element"])
def test_every_paper_demo_refreshes_each_step(case):
    assert CASE_PRESETS[case]["visual_every"] == 1


def test_paper_demo_selected_zone_counts_are_exposed():
    assert {case: preset["zone_count"] for case, preset in CASE_PRESETS.items()} == {
        "singlet": 12,
        "triplet": 41,
        "four_element": 19,
    }


def test_command_uses_selected_promoted_design(tmp_path):
    overlay = Path(__file__).parents[1] / "outputs/desktop/test/desktop.yml"
    design = Path(__file__).parents[1] / CASE_MODELS["four_element"]["design"]

    command = build_command(overlay, design)

    assert CASE_MODELS["four_element"]["design"] in command
    assert design.name == "four_element_stage4_seed.yml"


def test_mtf_helper_normalizes_dc():
    psf = np.zeros((9, 9))
    psf[4, 4] = 1.0
    frequency, sagittal, _, tangential = compute_mtf_slices(psf, 0.001)

    assert sagittal[0] == pytest.approx(1.0)
    assert tangential[0] == pytest.approx(1.0)
    assert np.allclose(sagittal, 1.0)
    assert np.all(frequency >= 0)


def test_diffraction_mtf_uses_each_wavelength_cutoff():
    frequency = np.array([0.0, 50.0, 80.0, 100.0])
    blue = diffraction_limited_mtf(frequency, 486.1, 22.0)
    red = diffraction_limited_mtf(frequency, 656.3, 22.0)

    assert blue[0] == pytest.approx(1.0)
    assert red[0] == pytest.approx(1.0)
    assert blue[2] > 0.0
    assert red[2] == 0.0


def test_mtf_frequency_stops_at_psf_nyquist_limit():
    wavelengths = [486.1, 550.0, 656.3]

    assert mtf_frequency_limit(wavelengths, 2.0, 0.001) == pytest.approx(500.0)
    assert mtf_frequency_limit(wavelengths, 8.0, 0.0001) == pytest.approx(
        1.0 / (486.1e-6 * 8.0)
    )


def test_desktop_command_defaults_to_paper_triplet_chain(tmp_path):
    config = Path("outputs") / "desktop" / "test" / "desktop.yml"
    command = build_command(Path(__file__).parents[1] / config)

    assert command[-1] == config.as_posix()
    assert "configs/multi_element/defaults.yml" in command
    assert "configs/paper_demos/designs/triplet_prerefine.yml" in command
    assert "configs/multi_element/stage_zone_3p_fold_cooke_f28_annular_m90.yml" not in command


def test_desktop_manager_starts_idle():
    assert RunManager().snapshot()["state"] == "idle"
