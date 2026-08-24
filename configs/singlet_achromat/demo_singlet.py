"""Generate an annular singlet from a reproducible near-flat starting point.

The starting-point convention follows AutoLens: every refracting interface has
a deterministic weak curvature (|c| < 1e-3 / mm), while all higher-order and
zonal shape coefficients are zero.  The fold order M fixes only the topology --
how many zones exist and where their boundaries sit.  LM must build the optical
power and the M-lambda branch offsets; no MATLAB-generated relief is loaded.

The evaluation metric is the polychromatic RMS spot radius.  Being geometric it
carries no analysis-window or phase-wrapping pitfalls, and it already covers
colour: the residual takes its centroid across rays *and* wavelengths, so axial
and lateral colour both enter.

The material is the named catalog glass OHARA S-BSL 7.  Its catalog nd, vd and
dPgF values are written explicitly so the run never relies on a proxy material.

    python configs/singlet_achromat/demo_singlet.py [f_number] [half_field_deg]
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eadld.modeling import optics  # noqa: E402
from eadld.modeling import ray_initialization as ri  # noqa: E402
from eadld.optimization.hdoe import harmonic_zone_edges  # noqa: E402

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- specification
# F number comes from the command line.  Everything scales with it --
# aperture, zone count, pupil sampling and the size of the LM Jacobian -- so the
# sampling is derived from the zone count rather than fixed (see sampling()).
EFL = 100.0
F_NUMBER = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
WAVELENGTHS, DESIGN = [486.1, 550.0, 656.3], 550.0
FIELD_DEG = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
N_FIELDS = 3
GLASS = {
    "name": "S-BSL 7",
    "nd": 1.51633,
    "vd": 64.140373,
    "dpgf": -0.0024,
}
THICKNESS = 3.0
SEED = 42
CURVATURE_SCALE = 1e-3
MIN_ZONE_WIDTH = 0.020  # mm
# the LM Jacobian is [n_residuals, 2 * n_zones] and the sampling
# grows with the zone count too, so cost goes as the square.  144 zones at f/5.6
# filled a 24 GB card; 96 is the largest that stays comfortable.  Nothing is
# lost by capping it -- the spot is flat across M (28.19-28.23 um at f/8,
# 57.2-57.3 at f/4), because a geometric tracer sees the annular surface as
# purely refractive.  M buys manufacturability, not performance.
MAX_ZONES = 96
# The earlier f/8 result used M=30 and twelve zones together with a front DOE.
# Reuse that proven topology, but not its analytically shaped prescription:
# DOE and annular coefficients below still start exactly at zero.
ORDERS = (30,)
LM_STEPS = 300

GENERATED_DESIGN = ROOT / "outputs" / "singlet_achromat" / "generated.yml"
RADIUS = EFL / (2 * F_NUMBER)
LAM0 = DESIGN * 1e-6


def zone_radii(order):
    """Exact spherical-OPD zone boundaries for this fold order."""
    edges = harmonic_zone_edges(RADIUS, EFL, order, DESIGN).tolist()
    # the aperture almost never lands on a fold boundary, so the
    # outermost zone is a partial one of arbitrary width.  At f/8 that sliver
    # only made the sweep noisy; at f/4 it rejected M = 5, 10 and 20 outright and
    # pushed the reported aspect ratio to 9.2 -- all of it truncation, not
    # geometry.  Merge a sub-tolerance sliver into the zone below it.
    if len(edges) > 1 and edges[-1] - edges[-2] < MIN_ZONE_WIDTH:
        edges.pop(-2)
    return edges


def widths(edges):
    return [b - a for a, b in zip([0.0] + edges[:-1], edges)]


# a zone that no ray lands in is invisible to the residual, so the
# radial sampling has to follow the zone count instead of staying at 96.  The
# fitting stage runs leaner than the evaluation because the LM Jacobian is
# [n_residuals, n_free] and both factors grow with the zone count.
def sampling(n_zones, per_zone, n_theta):
    return {"n_r": max(96, per_zone * n_zones), "n_theta": n_theta}


def fit_sampling(n_zones):
    return sampling(n_zones, per_zone=4, n_theta=32)


def eval_sampling(n_zones):
    return sampling(n_zones, per_zone=8, n_theta=64)


def near_flat_curvatures(seed=SEED):
    """AutoLens-style weak positive/negative curvatures, reproducibly seeded."""
    rng = random.Random(seed)
    return [rng.random() * CURVATURE_SCALE, -rng.random() * CURVATURE_SCALE]


def write_near_flat_start(order, path, seed=SEED):
    """Write a near-flat lens; M contributes boundaries but no pre-shaped relief."""
    edges = zone_radii(order)
    curvatures = near_flat_curvatures(seed)
    config = {
        "model": {
            "lens_parameterization": {
                "class_path": "eadld.optimization.parameterization.LensParameterization",
                "init_args": {
                    "lens_sequence": "sdRz-",
                    "s": [THICKNESS, EFL],
                    "c": curvatures,
                    "nd": [GLASS["nd"]],
                    "vd": [GLASS["vd"]],
                    "dpgf": [GLASS["dpgf"]],
                    "a": [],
                    # Front-face DOE coefficient; zero is a true no-phase start.
                    "d": [[0.0]],
                    "m": [],
                    # [delta_A1, A2, delta_Z, Rmax].  The complete physical
                    # relief starts at zero; only the M-defined topology exists.
                    "z": [
                        [[0.0, 0.0, 0.0, edge] for edge in edges]
                    ],
                    "nominal_wavelength": DESIGN,
                    "target_efl": EFL,
                    "solve_type": None,
                    "solve_idx": None,
                    "paraxial_image_solve": False,
                    "total_track_length_solve": None,
                    "qc_vars": False,
                    "scale_factor": RADIUS,
                    "bezier_aspherics": False,
                    "glass_file": "configs/recommended_ohara_glass.csv",
                    "misc_surface_model": None,
                },
            }
        }
    }
    banner = (
        f"# Reproducible AutoLens-style near-flat start: seed={seed}, M={order}, "
        f"glass={GLASS['name']}, {len(edges)} zones, "
        f"narrowest {min(widths(edges)) * 1e3:.1f} um.\n"
    )
    path.write_text(banner + yaml.safe_dump(config, sort_keys=False))
    return edges


def lens_from_parameters(path):
    """Rebuild a lens from a runtime ``lens_parameters`` snapshot."""
    p = yaml.safe_load(path.read_text())
    kw = {"dtype": torch.float64}
    z = torch.tensor(p["z"], **kw)
    return optics.Lens(
        sequence="sdRz-",
        s=torch.tensor(p["s"], **kw).reshape(-1, 1),
        c=torch.tensor(p["c"], **kw).reshape(-1, 1),
        nd=torch.tensor(p["nd"], **kw).reshape(-1, 1),
        vd=torch.tensor(p["vd"], **kw).reshape(-1, 1),
        dpgf=torch.tensor(p["dpgf"], **kw).reshape(-1, 1),
        a=torch.empty((0, 1, 0), **kw),
        d=torch.tensor(p["d"], **kw).reshape(1, 1, -1),
        m=torch.empty((0, 1, 0), **kw),
        z=z.reshape(1, 1, z.shape[-2], 4),
        w0=DESIGN,
    )


def spot_radii(lens, pupil_sampling_kwargs=None, mode="skew_uniform_zonal"):
    """Polychromatic RMS spot radius (um) per field, plus the valid-ray fraction."""
    rays = ri.RayInitialization(
        aperture=2 * RADIUS,
        aperture_type="epd",
        hfov=FIELD_DEG,
        n_fields=N_FIELDS,
        wavelengths=WAVELENGTHS,
        pupil_sampling_mode=mode,
        pupil_sampling_kwargs=pupil_sampling_kwargs or {"n_r": 96, "n_theta": 64},
        ray_aiming_steps=0,
    )
    r0, d0 = rays(lens)
    r, _, status, _ = list(lens.trace_rays(r0, d0, WAVELENGTHS, yield_on="end"))[-1]
    if mode == "skew_uniform_zonal":
        sampling = pupil_sampling_kwargs or {"n_r": 96, "n_theta": 64}
        pupil_weights = ri.zonal_pupil_weights(
            n_r=sampling["n_r"],
            n_theta=sampling["n_theta"],
            zone_edges=ri.zone_edges_from_lens(lens, 2 * RADIUS),
        )
    else:
        pupil_weights = torch.ones(r.shape[2], dtype=r.dtype, device=r.device)
        pupil_weights /= pupil_weights.sum()
    out = []
    for field in range(N_FIELDS):
        valid = status[field, ..., 0] == 0
        weights = pupil_weights[:, None] * valid
        weights = weights / weights.sum()
        xy = r[:2, field, ..., 0].where(valid[None], 0.0)
        centre = (xy * weights[None]).sum(dim=(1, 2), keepdim=True)
        rms = (weights * (xy - centre).square().sum(0)).sum().sqrt()
        out.append(float(rms) * 1e3)
    valid_fraction = (
        pupil_weights.view(1, -1, 1) * (status[..., 0] == 0)
    ).sum(dim=1).mean()
    return out, float(valid_fraction)


def run_lm(design, n_zones):
    """Run one LM stage and return (final snapshot, run directory).

    The run is identified by differencing the version directories, not by
    modification time: callbacks keep writing after a run finishes, so mtime
    ordering does not track creation order.
    """
    root = ROOT / "logs" / "singlet_achromat" / "sdRz-"
    before = {path.name for path in root.glob("version_*")} if root.exists() else set()
    # defaults.yml pins the f/8 aperture and a fixed 96x64 pupil, both
    # of which have to follow the F number and the zone count.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eadld.main",
            "fit",
            "-c",
            "configs/singlet_achromat/defaults.yml",
            "-c",
            str(design.relative_to(ROOT)).replace("\\", "/"),
            "-c",
            "configs/singlet_achromat/fit.yml",
            f"--model.ray_initialization.init_args.aperture={2 * RADIUS}",
            f"--model.ray_initialization.init_args.hfov={FIELD_DEG}",
            "--model.ray_initialization.init_args.pupil_sampling_kwargs="
            + json.dumps(fit_sampling(n_zones)),
            f"--trainer.max_steps={LM_STEPS}",
            f"--data.init_args.n_samples={LM_STEPS}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    created = [p for p in root.glob("version_*") if p.name not in before]
    if len(created) != 1:
        raise RuntimeError(f"expected one new run directory, got {created}")
    run = created[0]
    return sorted((run / "lens_parameters").glob("*.yml"))[-1], run


def manufacturing(lens):
    """Narrowest zone (um) and the worst step-height / zone-width ratio."""
    zones = lens.z[0, 0].detach()
    edges = zones[:, 3]
    zone_widths = torch.diff(torch.cat((edges.new_zeros(1), edges)))
    worst = 0.0
    for k in range(len(edges) - 1):
        radius = edges[k]
        inner = zones[k, 0] * radius**2 + zones[k, 1] * radius**4 + zones[k, 2]
        outer = (
            zones[k + 1, 0] * radius**2 + zones[k + 1, 1] * radius**4 + zones[k + 1, 2]
        )
        step = abs(float(outer - inner))
        span = min(float(zone_widths[k]), float(zone_widths[k + 1]))
        worst = max(worst, step / span)
    return float(zone_widths.min()) * 1e3, worst


def main():
    print(
        f"Specification: EFL {EFL} mm, f/{F_NUMBER:g}, {WAVELENGTHS} nm, "
        f"+/-{FIELD_DEG} deg.  Near-flat seed={SEED}, per-zone shape free, "
        f"{LM_STEPS} LM steps, metric = polychromatic RMS spot.\n"
    )
    print(
        f"{'M':>4} {'zones':>6} {'start width':>12} | {'spot um 0/0.5/1':>25}"
        f" | {'min width':>10} {'aspect':>7} {'valid':>7}"
    )
    best = None
    for order in ORDERS:
        edges = zone_radii(order)
        if min(widths(edges)) < MIN_ZONE_WIDTH or len(edges) > MAX_ZONES:
            continue
        # 生成处方属于运行产物，放在忽略目录中，示例执行后不会污染仓库。
        design = GENERATED_DESIGN
        write_near_flat_start(order, design)
        snapshot, run = run_lm(design, len(edges))
        lens = lens_from_parameters(snapshot)
        spots, valid = spot_radii(lens, eval_sampling(len(edges)))
        width, aspect = manufacturing(lens)
        print(
            f"{order:4d} {len(edges):6d} {min(widths(edges)) * 1e3:10.1f}um |"
            + "  ".join(f"{v:7.2f}" for v in spots)
            + f" | {width:8.1f}um {aspect:7.3f} {valid:7.4f}"
        )
        score = sum(spots) / len(spots)
        if best is None or score < best[0]:
            best = (score, order, run, spots, width, aspect)

    if best is None:
        raise SystemExit("no fold order satisfies the manufacturing limits")
    score, order, run, spots, width, aspect = best
    print(
        f"\nbest M={order}: mean RMS spot {score:.2f} um "
        f"(fields {'/'.join(f'{v:.2f}' for v in spots)}), "
        f"narrowest zone {width:.1f} um, aspect {aspect:.3f}"
    )
    print(f"run directory: {run.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
