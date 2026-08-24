"""Compare an annular singlet against spherical and aspheric controls.

add by cjy.  The three variants share the specification exactly -- same EFL,
aperture, field, wavelengths and glass -- so the only difference is how much
freedom the surfaces have:

    spherical   s-R-     2 curvatures                        (the base)
    aspheric    s-aRa-   2 curvatures + 4 polynomial terms each
    annular     sRz-     both faces flat, all power from the zones,
                         (delta_A1, A2) per zone, Zoff quantised to M*lambda0

The glass is frozen in all three.  That matters: a single glass puts a hard
floor under the polychromatic spot at EFL / (2 * F# * Vd) regardless of surface
shape, so the comparison is only meaningful once you read the monochromatic
design-wavelength column as well.

    python configs/singlet_achromat/compare.py [f_number] [half_field_deg]
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# demo_singlet reads the specification from argv at import time, so the annular
# variant automatically follows the same f number and field as the controls.
sys.argv = [sys.argv[0]] + sys.argv[1:3]
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("demo_singlet", HERE / "demo_singlet.py")
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

from eisoptx.modeling import optics  # noqa: E402
from eisoptx.modeling import ray_tracing as rt  # noqa: E402

torch.set_default_dtype(torch.float64)

CONTROL_STEPS = 200
N_ASPHERE = 5  # conic + r^4..r^10; the conic stays frozen


def control_design(path, sequence, aspheric):
    curvature = 1 / (2 * (ds.GLASS["nd"] - 1) * ds.EFL)  # equiconvex at the target EFL
    config = {
        "model": {
            "lens_parameterization": {
                "class_path": "eisoptx.optimization.parameterization.LensParameterization",
                "init_args": {
                    "lens_sequence": sequence,
                    "s": [0.5, ds.THICKNESS, ds.EFL],
                    "c": [curvature, -curvature],
                    "nd": [ds.GLASS["nd"]],
                    "vd": [ds.GLASS["vd"]],
                    "a": [[0.0] * N_ASPHERE] * 2 if aspheric else [],
                    "d": [],
                    "m": [],
                    "z": [],
                    "nominal_wavelength": ds.DESIGN,
                    "target_efl": ds.EFL,
                    "solve_type": "focal_length",
                    "solve_idx": 1,
                    "paraxial_image_solve": True,
                    "total_track_length_solve": None,
                    "qc_vars": False,
                    "scale_factor": ds.RADIUS,
                    "bezier_aspherics": False,
                    "glass_file": None,
                    "misc_surface_model": None,
                },
            }
        }
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def control_stage(path, aspheric):
    freeze = {
        "s": [False, False, True],  # image distance comes from the paraxial solve
        "c": False,
        "g": True,  # same glass as the annular variant
        "m": True,
        "d": True,
    }
    if aspheric:
        freeze["a"] = [[True] + [False] * (N_ASPHERE - 1)]
    config = {
        "model": {
            "ray_initialization": {"init_args": {"pupil_sampling_mode": "skew_uniform"}},
            "lens_parameterization": {"init_args": {"freeze": freeze}},
            "residuals": [
                {"class_path": "eisoptx.optimization.residuals.TransverseRayAberrationResiduals",
                 "init_args": {"weight": 1.0}},
                {"class_path": "eisoptx.optimization.residuals.RayPathResiduals",
                 "init_args": {"weight": 10.0, "min_cutoff": 0.5, "max_cutoff": 40.0,
                               "other_max_cutoffs": [[-1, float("inf")]],
                               "min_cutoff_refractive": 1.5, "max_cutoff_refractive": 15.0,
                               "min_cutoff_refractive_relative": 0.0,
                               "max_cutoff_refractive_relative": 100.0}},
            ],
            "lens_optimizer": {
                "class_path": "eisoptx.optimization.optimizers.LMOptimizer",
                "init_args": {"lm_parameter": 1.0, "damped_term_min": 1.0e-6,
                              "tolerance": 1.0, "lam_increase_factor": 4.0,
                              "lam_decrease_factor": 2.0, "lam_eps": 1.0e-10, "beta": 0.95},
            },
        },
        "trainer": {
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": None,
            "callbacks": [
                {"class_path": "eisoptx.utils.callbacks.ConfigFileCallback",
                 "init_args": {"every_n_steps": 50}},
                "eisoptx.main.CustomProgressBar",
            ],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def run_control(design, stage, sequence):
    root = ROOT / "logs" / "singlet_achromat" / sequence
    before = {p.name for p in root.glob("version_*")} if root.exists() else set()
    result = subprocess.run(
        [sys.executable, "-m", "eisoptx.main", "fit",
         "-c", "configs/singlet_achromat/defaults.yml",
         "-c", str(design.relative_to(ROOT)).replace("\\", "/"),
         "-c", str(stage.relative_to(ROOT)).replace("\\", "/"),
         f"--model.ray_initialization.init_args.aperture={2 * ds.RADIUS}",
         f"--model.ray_initialization.init_args.hfov={ds.FIELD_DEG}",
         "--model.ray_initialization.init_args.pupil_sampling_kwargs="
         + json.dumps({"n_r": 192, "n_theta": 64}),
         f"--trainer.max_steps={CONTROL_STEPS}",
         f"--data.init_args.n_samples={CONTROL_STEPS}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    created = [p for p in root.glob("version_*") if p.name not in before]
    return sorted((created[0] / "lens_parameters").glob("*.yml"))[-1]


def control_lens(path, sequence, aspheric):
    p = yaml.safe_load(Path(path).read_text())
    kw = {"dtype": torch.float64}
    column = lambda k: torch.tensor(  # noqa: E731
        [float(v) for v in (p[k] if not isinstance(p[k][0], list) else [r[0] for r in p[k]])],
        **kw).reshape(-1, 1)
    a = torch.tensor(p["a"], **kw) if aspheric else torch.empty((0, 0), **kw)
    return optics.Lens(
        sequence=sequence, s=column("s"), c=column("c"), nd=column("nd"),
        vd=column("vd"), dpgf=column("dpgf"),
        a=a.reshape(a.shape[0], 1, a.shape[-1]) if aspheric else torch.empty((0, 1, 0), **kw),
        d=torch.empty((0, 1, 0), **kw), m=torch.empty((0, 1, 0), **kw),
        z=torch.empty((0, 1, 0, 4), **kw), w0=ds.DESIGN,
    )


def refractive_relief(lens, aspheric):
    """Peak-to-valley sag summed over both faces -- how deep the part has to be cut."""
    rho = torch.linspace(0.0, ds.RADIUS**2, 4001)
    total = 0.0
    for i in range(2):
        a = lens.a[i] if aspheric else None
        sag, _ = rt.evaluate_aspherical_profile(
            rho.reshape(-1, 1), lens.c.reshape(-1)[i:i + 1],
            a.reshape(1, -1) if a is not None else None)
        total += float(sag.max() - sag.min())
    return total


def annular_relief(lens):
    zones = lens.z[0, 0].detach()
    rho = torch.linspace(0.0, ds.RADIUS**2, 4001).reshape(-1, 1)
    sag, inside = rt.evaluate_zonal_profile(rho, lens.c.reshape(-1)[1:2], zones.reshape(1, -1, 4))
    sag = sag[inside]
    return float(sag.max() - sag.min())


def mono(lens, wavelength, kwargs, mode):
    saved = ds.WAVELENGTHS
    ds.WAVELENGTHS = [wavelength]
    try:
        return ds.spot_radii(lens, kwargs, mode)[0]
    finally:
        ds.WAVELENGTHS = saved


def main():
    print(f"Specification: EFL {ds.EFL} mm, f/{ds.F_NUMBER:g} (EPD {2 * ds.RADIUS:.1f} mm), "
          f"+/-{ds.FIELD_DEG} deg, {ds.WAVELENGTHS} nm,\n"
          f"glass fixed nd={ds.GLASS['nd']} Vd={ds.GLASS['vd']}.  "
          f"Single-glass colour floor EFL/(2 F# Vd) = "
          f"{ds.EFL / (2 * ds.F_NUMBER * ds.GLASS['vd']) * 1e3:.0f} um.\n")

    results = []
    for label, sequence, aspheric in (("spherical  (base)", "s-R-", False),
                                      ("aspheric", "s-aRa-", True)):
        design = HERE / "designs" / f"control_{sequence.strip('-')}.yml"
        stage = HERE / f"stage_{sequence.strip('-')}.yml"
        control_design(design, sequence, aspheric)
        control_stage(stage, aspheric)
        lens = control_lens(run_control(design, stage, sequence), sequence, aspheric)
        kwargs = {"n_r": 384, "n_theta": 64}
        spots, valid = ds.spot_radii(lens, kwargs, "skew_uniform")
        results.append((label, spots, mono(lens, ds.DESIGN, kwargs, "skew_uniform"),
                        refractive_relief(lens, aspheric), valid, ""))

    best = None
    for order in ds.ORDERS:
        edges = ds.zone_radii(order)
        if min(ds.widths(edges)) < ds.MIN_ZONE_WIDTH or len(edges) > ds.MAX_ZONES:
            continue
        design = HERE / "designs" / "generated.yml"
        ds.write_flat_plate(order, design)
        lens = ds.lens_from_parameters(ds.run_lm(design, len(edges))[0])
        kwargs = ds.eval_sampling(len(edges))
        spots, valid = ds.spot_radii(lens, kwargs)
        score = sum(spots) / len(spots)
        if best is None or score < best[0]:
            best = (score, order, spots, mono(lens, ds.DESIGN, kwargs, "skew_uniform_zonal"),
                    annular_relief(lens), valid, len(edges), min(ds.widths(edges)))
    results.append((f"annular  (M={best[1]}, {best[6]} zones)", best[2], best[3],
                    best[4], best[5], f"narrowest zone {best[7] * 1e3:.0f} um"))

    head = f"{'variant':<26} {'poly 0/0.5/1 deg':>26} {'mono 550 nm':>26} {'relief':>9} {'valid':>7}"
    print(head)
    print("-" * len(head))
    for label, spots, m550, relief, valid, note in results:
        print(f"{label:<26} " + "  ".join(f"{v:7.2f}" for v in spots) + "  "
              + "  ".join(f"{v:7.2f}" for v in m550)
              + f" {relief * 1e3:7.1f}um {valid:7.4f}" + (f"   {note}" if note else ""))


if __name__ == "__main__":
    main()
