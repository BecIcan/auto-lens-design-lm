import pytest
import torch
from matplotlib.figure import Figure

from eadld.initialization import DesignSpec, LensSeed, run_generation_audit
from eadld.initialization.api import (
    _initializer,
    _mechanical_gate,
    _relocate_stop_in_air_gap,
)
from eadld.initialization.codev_seq import (
    lens_to_spherical_prescription,
    parse_codev_seq,
    prescription_to_lens,
    write_codev_seq,
)
from eadld.utils.visualization import generate_spot_plot, plot_layout


class PublicTestBackend:
    def generate(self, spec):
        return [
            LensSeed(
                candidate_id="public-test",
                lens_sequence="s-aRa-aRa-aRa-",
                spacings_mm=(1.56274209619, 5.46374401399, 10.665431266, 11.975087064, 39.3610263407, 15.0024144163, 54.9176367768),
                curvatures_per_mm=(0.000625821571703, -0.0379287868671, -0.00106417392786, 0.0543700762226, 0.0226606901903, -0.00529313208104),
                refractive_indices_d=(1.620411, 1.717362, 1.620411),
                abbe_numbers=(60.292614, 29.517426, 60.292614),
                asphere_coefficients=(
                    (0.0, -6.36036744916e-05, 2.12294745603e-07, -5.38363660686e-11, -2.77368651596e-13),
                    (0.0, -1.67251318477e-05, 1.11181187259e-07, 1.73816745142e-10, -5.01420185265e-13),
                    (0.0, 6.14570829518e-05, -1.0845611976e-07, -1.09900156959e-10, 2.12537519473e-13),
                    (0.0, 2.54496705935e-05, -3.62035953094e-07, 1.71209881533e-10, 5.65862570868e-13),
                    (0.0, -7.02502841652e-06, -1.06712019709e-08, -1.19151990785e-10, 2.14482049152e-13),
                    (0.0, -3.13195689906e-06, -4.32667647315e-08, 5.1575749036e-11, -9.851430484e-16),
                ),
            )
        ]

    def public_metadata(self):
        return {"name": "public-test", "weights_in_repository": False}


def test_private_backend_public_physics_contract(tmp_path):
    output = tmp_path / "audit"
    manifest = run_generation_audit(
        DesignSpec(100.0, 2.8, 5.0, (486.1, 550.0, 656.3), 3, 1),
        PublicTestBackend(),
        output,
    )
    assert manifest["runtime_contract"]["optimizer_invocations"] == 0
    assert manifest["runtime_contract"]["paraxial_solves"] == 0
    assert manifest["candidates"][0]["passed"]
    assert "curvatures_per_mm" not in (output / "manifest.json").read_text(encoding="utf-8")
    assert (output / "layout.png").stat().st_size > 10_000
    assert (output / "spots.png").stat().st_size > 10_000


def test_rejects_backend_candidate_with_wrong_element_count(tmp_path):
    with pytest.raises(ValueError, match="实际片数为 3.*请求的 4 片"):
        run_generation_audit(
            DesignSpec(100.0, 2.8, 5.0, (486.1, 550.0, 656.3), 4, 1),
            PublicTestBackend(),
            tmp_path / "wrong-elements",
        )


def test_stop_on_glass_surface_does_not_consume_element_closure():
    lens = LensSeed(
        candidate_id="stop-on-surface",
        lens_sequence="R-Rs-RR-",
        spacings_mm=(1.4, 1.5, 0.4, 1.4, 1.1, 0.6, 12.0),
        curvatures_per_mm=(0.13, 0.03, -0.06, 0.14, 0.04, -0.15, -0.07),
        refractive_indices_d=(1.6, 1.7, 1.5, 1.8),
        abbe_numbers=(60.0, 40.0, 70.0, 30.0),
    ).to_lens()
    geometry = list(lens.return_geometry())
    stop_index = next(index for index, row in enumerate(geometry) if row[0] == "s")

    assert geometry[stop_index][3] is False
    assert geometry[stop_index + 1][0] == "r"
    assert geometry[stop_index + 1][3] is True


def test_layout_closes_glass_across_colocated_stop():
    lens = LensSeed(
        candidate_id="stop-on-surface",
        lens_sequence="R-Rs-RR-",
        spacings_mm=(1.4, 1.5, 0.4, 1.4, 1.1, 0.6, 12.0),
        curvatures_per_mm=(0.13, 0.03, -0.06, 0.14, 0.04, -0.15, -0.07),
        refractive_indices_d=(1.6, 1.7, 1.5, 1.8),
        abbe_numbers=(60.0, 40.0, 70.0, 30.0),
    ).to_lens()
    geometry = list(lens.return_geometry())
    stop_index = next(index for index, row in enumerate(geometry) if row[0] == "s")
    half_diameter = 2.0
    diameter_scaler = 17 / 16
    display_radius = half_diameter * diameter_scaler

    figure = Figure()
    ax = figure.subplots()
    plot_layout(
        ax,
        lens,
        torch.full((len(geometry),), 2 * half_diameter, dtype=torch.float64),
        diameter_scaler=diameter_scaler,
    )

    surface_lines = [line for line in ax.lines if len(line.get_xdata()) > 2]
    surface_before_stop = surface_lines[stop_index - 1]
    surface_after_stop = surface_lines[stop_index]
    expected_edges = (
        float(surface_before_stop.get_xdata()[-1]),
        float(surface_after_stop.get_xdata()[-1]),
    )
    top_closure_edges = [
        tuple(float(value) for value in line.get_xdata())
        for line in ax.lines
        if len(line.get_xdata()) == 2
        and all(abs(float(value) - display_radius) < 1e-9 for value in line.get_ydata())
    ]
    assert any(
        all(abs(actual - expected) < 1e-9 for actual, expected in zip(edges, expected_edges))
        for edges in top_closure_edges
    )
    stop_segments = ax.collections[0].get_segments()
    assert all(
        min(abs(float(point[1])) for point in segment) >= display_radius - 1e-9
        for segment in stop_segments
    )


def test_relocates_stop_beyond_full_surface_sag_without_moving_lenses():
    seed = LensSeed(
        candidate_id="embedded-stop",
        lens_sequence="R-R-R-s-R-",
        spacings_mm=(
            1.78488216697,
            0.175000023746,
            4.75579062743,
            0.349872470138,
            3.33471878139,
            0.727690595597,
            6.77851515138,
            4.20388333912,
            15.0508724837,
        ),
        curvatures_per_mm=(
            1 / 20.7945375797,
            1 / 177.582282996,
            1 / 13.9121964271,
            1 / 21.121100958,
            1 / 42.6196982284,
            1 / 9.01153158804,
            1 / 37.7640779125,
            -1 / 163.435932687,
        ),
        refractive_indices_d=(1.618, 1.832408, 1.80491, 1.885671),
        abbe_numbers=(63.3897, 42.9324, 23.8088, 38.2187),
        nominal_wavelength_nm=545.5,
    )
    initializer = _initializer(
        DesignSpec(35.0, 2.8, 6.17, (435.0, 545.5, 656.0), 4)
    )
    before = _mechanical_gate(seed.to_lens(), initializer)
    repaired = _relocate_stop_in_air_gap(seed, initializer)

    assert not before["stop_clearance"]["passed"]
    assert repaired is not None
    assert repaired.curvatures_per_mm == seed.curvatures_per_mm
    assert sum(repaired.spacings_mm[5:7]) == pytest.approx(
        sum(seed.spacings_mm[5:7])
    )
    after = _mechanical_gate(repaired.to_lens(), initializer)
    assert after["stop_clearance"]["passed"]


def test_accepts_surface_mounted_stop_without_moving_optical_surfaces():
    seed = LensSeed(
        candidate_id="collocated-stop",
        lens_sequence="R-sR-",
        spacings_mm=(1.2, 2.0, 1.1, 8.0),
        curvatures_per_mm=(0.02, -0.02, 0.015, -0.015),
        refractive_indices_d=(1.6, 1.7),
        abbe_numbers=(60.0, 45.0),
    )
    initializer = _initializer(DesignSpec(35.0, 4.0, 3.0, (550.0,), 2))
    optical_z_before = [
        float(item[1]) for item in seed.to_lens().return_geometry() if item[0] == "r"
    ]

    repaired = _relocate_stop_in_air_gap(seed, initializer)

    assert repaired is not None
    assert repaired.lens_sequence == seed.lens_sequence
    assert repaired.spacings_mm == seed.spacings_mm
    assert repaired.curvatures_per_mm == seed.curvatures_per_mm
    optical_z_after = [
        float(item[1])
        for item in repaired.to_lens().return_geometry()
        if item[0] == "r"
    ]
    assert optical_z_after == pytest.approx(optical_z_before)
    stop = _mechanical_gate(repaired.to_lens(), initializer)["stop_clearance"]
    assert stop["surface_mounted"]
    assert stop["passed"]


def test_mechanical_gate_maps_split_stop_gap_to_both_spacing_indices():
    seed = LensSeed(
        candidate_id="split-stop",
        lens_sequence="R-s-R-",
        spacings_mm=(1.2, 0.8, 0.4, 1.1, 8.0),
        curvatures_per_mm=(0.02, -0.02, 0.015, -0.015),
        refractive_indices_d=(1.6, 1.7),
        abbe_numbers=(60.0, 45.0),
    )
    initializer = _initializer(DesignSpec(35.0, 4.0, 3.0, (550.0,), 2))

    gate = _mechanical_gate(seed.to_lens(), initializer)

    assert gate["clearances"][1]["spacing_indices"] == (1, 2)


def test_spot_plot_contains_only_tightly_framed_point_clouds():
    xy = torch.linspace(-0.004, 0.004, 72, dtype=torch.float64).reshape(2, 3, 4, 3)
    valid = torch.ones((3, 4, 3), dtype=torch.bool)
    figure = generate_spot_plot(xy, valid, (435.0, 545.5, 656.0), 6.17)

    assert figure._suptitle is None
    assert len(figure.axes) == 3
    assert all(not ax.axison for ax in figure.axes)
    assert all(ax.get_legend() is None for ax in figure.axes)
    assert all(not ax.lines for ax in figure.axes)
    assert all(len(ax.texts) == 1 for ax in figure.axes)
    assert all("° · RMS " in ax.texts[0].get_text() for ax in figure.axes)


def test_spherical_lens_exports_as_reloadable_codev_seq(tmp_path):
    lens = LensSeed(
        candidate_id="seq-export",
        lens_sequence="R-Rs-RR-",
        spacings_mm=(1.4, 1.5, 0.4, 1.4, 1.1, 0.6, 12.0),
        curvatures_per_mm=(0.13, 0.03, -0.06, 0.14, 0.04, -0.15, -0.07),
        refractive_indices_d=(1.6, 1.7, 1.5, 1.8),
        abbe_numbers=(60.0, 40.0, 70.0, 30.0),
    ).to_lens()
    prescription = lens_to_spherical_prescription(
        lens,
        title="SEQ export test",
        epd_mm=5.0,
        wavelengths_nm=(435.0, 545.5, 656.0),
        field_angles_deg=(0.0, 4.0, 6.0),
    )
    path = write_codev_seq(prescription, tmp_path / "candidate.seq")
    parsed = parse_codev_seq(path)
    reloaded = prescription_to_lens(parsed)

    assert parsed.wavelengths_nm == (656.0, 545.5, 435.0)
    assert reloaded.sequence.sequence == lens.sequence.sequence
    assert torch.allclose(reloaded.s, lens.s)
    assert torch.allclose(reloaded.c, lens.c)
    assert torch.allclose(reloaded.nd, lens.nd, atol=1e-6)
    assert torch.allclose(reloaded.vd, lens.vd, atol=1e-4)
