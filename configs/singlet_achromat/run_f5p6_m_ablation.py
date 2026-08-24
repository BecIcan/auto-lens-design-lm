"""Profile harmonic order M for the f/5.6 annular singlet.

Every feasible order starts from the same deterministic weak near-flat lens and
uses the same constrained-LM merit. Invalid runs are removed; accepted runs are
kept with their complete native EISOPTX history and dense held-out evaluation.
"""

import argparse
import copy
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "paper" / "results" / "f5p6_m_ablation_exact"
F_NUMBER = 5.6
FIELD_DEG = 1.0
STEPS = 300
MIN_WIDTH_MM = 0.150
MAX_RELIEF_MM = 0.050
MAX_SLOPE = math.tan(math.radians(10.0))
MAX_STEP_WIDTH_RATIO = 0.25
MIN_VALID = 0.99
MAX_BRANCH_ERROR_MM = 1e-9
PARAMETER_KEYS = ("a", "c", "d", "dpgf", "m", "nd", "s", "vd", "z")


def load_demo():
    sys.argv = ["demo_singlet.py", str(F_NUMBER), str(FIELD_DEG)]
    spec = importlib.util.spec_from_file_location(
        "f5p6_demo_singlet", HERE / "demo_singlet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_overlay(order, n_zones, log_root, steps):
    return {
        "model": {
            "ray_initialization": {
                "init_args": {
                    "aperture": 100.0 / F_NUMBER,
                    "hfov": FIELD_DEG,
                    "pupil_sampling_mode": "skew_uniform_zonal",
                    "pupil_sampling_kwargs": {
                        "n_r": max(96, 4 * n_zones),
                        "n_theta": 32,
                    },
                }
            },
            "residuals": [
                {
                    "class_path": "eisoptx.optimization.residuals.TransverseRayAberrationResiduals",
                    "init_args": {"weight": 1.0},
                },
                {
                    "class_path": "eisoptx.optimization.residuals.FocalLengthResiduals",
                    "init_args": {"weight": 1e6},
                },
                {
                    "class_path": "eisoptx.optimization.residuals.DistortionResiduals",
                    "init_args": {"weight": 10.0, "threshold": 0.02},
                },
                {
                    "class_path": "eisoptx.optimization.residuals.HDOEPhaseResiduals",
                    "init_args": {
                        "weight": 1e6,
                        "diffraction_order": order,
                        "design_wavelength": 550.0,
                        "zonal_surface_index": 0,
                        "field_indices": [0],
                    },
                },
                {
                    "class_path": "eisoptx.optimization.residuals.HDOEBranchConstraintResiduals",
                    "init_args": {
                        "weight": 1.0,
                        "diffraction_order": order,
                        "design_wavelength": 550.0,
                        "n_zones": n_zones,
                        "zonal_surface_index": 0,
                        "field_indices": [0],
                    },
                },
            ],
        },
        "data": {"init_args": {"n_samples": steps}},
        "trainer": {
            "max_steps": steps,
            "logger": {"init_args": {"save_dir": log_root}},
        },
    }


def scalar_history(run, key):
    accumulator = EventAccumulator(str(run)).Reload()
    if key not in accumulator.Tags()["scalars"]:
        return []
    return [
        {"step": item.step, "value": item.value, "wall_time": item.wall_time}
        for item in accumulator.Scalars(key)
    ]


def save_final_design(initial, snapshot, output):
    parameters = yaml.safe_load(snapshot.read_text())
    final = copy.deepcopy(initial)
    args = final["model"]["lens_parameterization"]["init_args"]
    for key in PARAMETER_KEYS:
        args[key] = parameters[key]
    output.write_text(yaml.safe_dump(final, sort_keys=False))


def remove_run(run, run_root):
    resolved = run.resolve()
    if run_root.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing to remove unexpected run path: {resolved}")
    shutil.rmtree(resolved)
    for directory in (run_root, run_root.parent):
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()


def design_index(demo):
    value = demo.optics.hartmann_dispersion(
        demo.torch.tensor([demo.DESIGN]),
        demo.torch.tensor([demo.GLASS["nd"]]),
        demo.torch.tensor([demo.GLASS["vd"]]),
        demo.torch.tensor([demo.GLASS["dpgf"]]),
    )
    return float(value.squeeze())


def manufacturing_details(lens):
    zones = lens.z[0, 0].detach()
    base_curvature = lens.c[-1, 0].detach()
    lower = zones.new_zeros(1)
    widths = []
    steps = []
    max_slope = 0.0
    for index, zone in enumerate(zones):
        upper = zone[3]
        widths.append(float(upper - lower))
        radii = torch.linspace(float(lower), float(upper), 1025).to(zone)
        slope = (base_curvature + 2 * zone[0]) * radii + 4 * zone[1] * radii**3
        max_slope = max(max_slope, float(slope.abs().max()))
        if index + 1 < len(zones):
            inner = zone[0] * upper**2 + zone[1] * upper**4 + zone[2]
            next_zone = zones[index + 1]
            outer = (
                next_zone[0] * upper**2
                + next_zone[1] * upper**4
                + next_zone[2]
            )
            steps.append(abs(float(outer - inner)))
        lower = upper
    ratios = [
        step / min(widths[index], widths[index + 1])
        for index, step in enumerate(steps)
    ]
    return {
        "min_width_um": min(widths) * 1e3,
        "max_step_um": max(steps, default=0.0) * 1e3,
        "max_step_width_ratio": max(ratios, default=0.0),
        "max_slope": max_slope,
        "max_slope_deg": math.degrees(math.atan(max_slope)),
    }


def run_order(demo, order, steps, temporary):
    edges = demo.zone_radii(order)
    widths = demo.widths(edges)
    index_550 = design_index(demo)
    estimated_step_mm = order * demo.LAM0 / (index_550 - 1)
    screen = {
        "zones": len(edges),
        "min_width_um": min(widths) * 1e3,
        "estimated_step_um": estimated_step_mm * 1e3,
    }
    screen_reasons = []
    if len(edges) > demo.MAX_ZONES:
        screen_reasons.append(f"{len(edges)} zones > {demo.MAX_ZONES}")
    if min(widths) < MIN_WIDTH_MM:
        screen_reasons.append(
            f"minimum width {min(widths) * 1e3:.3f} um < 150 um"
        )
    if estimated_step_mm > MAX_RELIEF_MM:
        screen_reasons.append(
            f"estimated relief {estimated_step_mm * 1e3:.3f} um > 50 um"
        )
    if screen_reasons:
        return {
            "M": order,
            "status": "screened",
            "reason": "; ".join(screen_reasons),
            **screen,
        }

    design = temporary / f"M{order}_initial.yml"
    overlay = temporary / f"M{order}_fit.yml"
    demo.write_near_flat_start(order, design)
    initial = yaml.safe_load(design.read_text())
    log_root_relative = f"logs/f5p6_m_ablation_exact/M{order}"
    log_root = ROOT / log_root_relative / "sdRz-"
    overlay.write_text(
        yaml.safe_dump(
            fit_overlay(order, len(edges), log_root_relative, steps), sort_keys=False
        )
    )
    before = set(log_root.glob("version_*")) if log_root.exists() else set()
    print(f"START M={order}: {len(edges)} zones, min width {screen['min_width_um']:.1f} um", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eisoptx.main",
            "fit",
            "-c",
            "configs/singlet_achromat/defaults.yml",
            "-c",
            str(design),
            "-c",
            "configs/singlet_achromat/fit.yml",
            "-c",
            str(overlay),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    created = set(log_root.glob("version_*")) - before if log_root.exists() else set()
    if result.returncode or len(created) != 1:
        for run in created:
            remove_run(run, log_root)
        raise RuntimeError(result.stderr[-4000:] or f"unexpected runs: {created}")
    run = created.pop()
    snapshots = sorted((run / "lens_parameters").glob("*.yml"))
    if not snapshots:
        remove_run(run, log_root)
        raise RuntimeError(f"M={order} produced no parameter snapshots")

    histories = {
        name: scalar_history(run, key)
        for name, key in {
            "loss": "loss/scalar_loss",
            "geometric": "loss/transverse_ray_aberration",
            "phase": "loss/hdoe_phase",
            "branch": "loss/hdoe_branch_constraint",
            "valid": "ray_tracing/ray_valid",
            "constraint_rank": "optimizer/constraint_rank",
            "damping": "optimizer/lm_parameter",
        }.items()
    }
    lens = demo.lens_from_parameters(snapshots[-1])
    spots, dense_valid = demo.spot_radii(lens, demo.eval_sampling(len(edges)))
    manufacturing = manufacturing_details(lens)
    aspect = manufacturing["max_step_width_ratio"]
    ranks = [item["value"] for item in histories["constraint_rank"]]
    branch = histories["branch"][-1]["value"] if histories["branch"] else math.inf
    reasons = []
    if not ranks or min(ranks) < len(edges) - 1:
        reasons.append("constraint rank loss")
    if branch > MAX_BRANCH_ERROR_MM:
        reasons.append(f"branch error {branch:.3e} mm")
    if dense_valid < MIN_VALID:
        reasons.append(f"valid fraction {dense_valid:.6f}")
    if aspect > MAX_STEP_WIDTH_RATIO:
        reasons.append(f"step/width {aspect:.6f}")
    if manufacturing["max_step_um"] > MAX_RELIEF_MM * 1e3:
        reasons.append(f"step depth {manufacturing['max_step_um']:.3f} um")
    if manufacturing["max_slope"] > MAX_SLOPE:
        reasons.append(f"surface slope {manufacturing['max_slope_deg']:.3f} deg")
    if reasons:
        remove_run(run, log_root)
        record = {
            "M": order,
            "status": "rejected",
            "reason": "; ".join(reasons),
            **screen,
        }
        print(f"REJECT M={order}: {record['reason']}", flush=True)
        return record

    RESULTS.mkdir(parents=True, exist_ok=True)
    initial_output = RESULTS / f"M{order}_initial.yml"
    final_output = RESULTS / f"M{order}_final.yml"
    history_output = RESULTS / f"M{order}_history.csv"
    initial_output.write_text(yaml.safe_dump(initial, sort_keys=False))
    save_final_design(initial, snapshots[-1], final_output)
    names = list(histories)
    indexed = {
        name: {item["step"]: item["value"] for item in values}
        for name, values in histories.items()
    }
    with history_output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", *names])
        for step in sorted({step for values in indexed.values() for step in values}):
            writer.writerow([step, *(indexed[name].get(step) for name in names)])
    record = {
        "M": order,
        "status": "accepted",
        **screen,
        "trainable_parameters": int(3 * len(edges) + 2),
        "run": str(run.relative_to(ROOT)).replace("\\", "/"),
        "initial_design": str(initial_output.relative_to(ROOT)).replace("\\", "/"),
        "final_design": str(final_output.relative_to(ROOT)).replace("\\", "/"),
        "history_csv": str(history_output.relative_to(ROOT)).replace("\\", "/"),
        "loss_initial": histories["loss"][0]["value"],
        "loss_final": histories["loss"][-1]["value"],
        "elapsed_seconds": histories["loss"][-1]["wall_time"] - histories["loss"][0]["wall_time"],
        "spots_um": spots,
        "mean_spot_um": sum(spots) / len(spots),
        "dense_valid": dense_valid,
        "efl_mm": float(lens.efl.squeeze()),
        "final_branch_error_mm": branch,
        **manufacturing,
        "constraint_rank": min(ranks),
    }
    print(
        f"ACCEPT M={order}: RMS={record['mean_spot_um']:.3f} um, "
        f"branch={branch:.3e} mm, aspect={aspect:.3f}",
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", nargs="+", type=int)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("steps must be at least two")
    demo = load_demo()
    maximum_order = math.floor(
        MAX_RELIEF_MM * (design_index(demo) - 1) / demo.LAM0
    )
    orders = args.orders or list(range(1, maximum_order + 1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    protocol = {
        "topology": "exact spherical OPD",
        "f_number": F_NUMBER,
        "half_field_deg": FIELD_DEG,
        "wavelengths_nm": demo.WAVELENGTHS,
        "design_wavelength_nm": demo.DESIGN,
        "glass": demo.GLASS,
        "seed": demo.SEED,
        "steps": args.steps,
        "minimum_width_um": MIN_WIDTH_MM * 1e3,
        "maximum_relief_um": MAX_RELIEF_MM * 1e3,
        "maximum_slope_deg": math.degrees(math.atan(MAX_SLOPE)),
        "maximum_step_width_ratio": MAX_STEP_WIDTH_RATIO,
        "maximum_zones": demo.MAX_ZONES,
        "maximum_order": maximum_order,
        "orders": orders,
    }
    (RESULTS / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    with tempfile.TemporaryDirectory(prefix="eisoptx_f5p6_m_") as directory:
        temporary = Path(directory)
        for order in orders:
            records.append(run_order(demo, order, args.steps, temporary))
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
