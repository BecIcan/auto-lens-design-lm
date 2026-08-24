"""Run the same branch-constrained coherent LM solve for each candidate M.

The integer order and its zone table are never differentiated.  Each candidate
gets a fresh, equally perturbed design and an equal inner-LM budget; successful
runs are ranked by the converged normalized geometric-plus-coherent merit.

    python configs/raywave_zonal/profile_orders.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eisoptx.optimization.hdoe import (  # noqa: E402
    harmonic_zone_count,
    harmonic_zone_edges,
    normalized_phase_depth,
    scalar_blaze_efficiency,
)


HERE = Path(__file__).resolve().parent
CANDIDATES = {
    480: HERE / "designs" / "visible_f100_f2_m480.yml",
    627: HERE / "designs" / "visible_f100_f2_m627.yml",
}
N_STEPS = 15


def perturbed_design(source: Path):
    config = yaml.safe_load(source.read_text())
    args = config["model"]["lens_parameterization"]["init_args"]
    args["s"][-1] += 0.2
    for index, zone in enumerate(args["z"][0]):
        zone[1] *= 1.02 if index % 2 == 0 else 0.98
    return config


def inner_stage(order: int, n_zones: int):
    zone_freeze = [[[True, False, True, True] for _ in range(n_zones)]]
    return {
        "model": {
            "lens_parameterization": {
                "init_args": {
                    "freeze": {
                        "s": [True, False],
                        "c": True,
                        "g": True,
                        "a": True,
                        "d": True,
                        "m": True,
                        "z": zone_freeze,
                    }
                }
            },
            "ray_initialization": {
                "init_args": {
                    "hfov": 0.0,
                    "n_fields": 1,
                    "wavelengths": [486.1, 550.0, 656.3],
                    "wavelength_weights": [1.0, 1.0, 1.0],
                    "pupil_sampling_mode": "skew_uniform_zonal",
                    "pupil_sampling_kwargs": {"n_r": 48, "n_theta": 16},
                    "ray_aiming_steps": 0,
                }
            },
            "residuals+": [
                {
                    "class_path": (
                        "eisoptx.optimization.residuals."
                        "HDOEBranchConstraintResiduals"
                    ),
                    "init_args": {
                        "weight": 1.0,
                        "diffraction_order": order,
                        "design_wavelength": 550.0,
                        "n_zones": n_zones,
                        "zonal_surface_index": 0,
                        "field_indices": [0],
                    },
                },
                {
                    "class_path": (
                        "eisoptx.optimization.residuals.CoherentWavefrontResiduals"
                    ),
                    "init_args": {"weight": 1.0, "field_indices": [0]},
                },
            ],
            "lens_optimizer": {
                "class_path": "eisoptx.optimization.optimizers.LMOptimizer",
                "init_args": {
                    "lm_parameter": 1.0,
                    "damped_term_min": 1e-6,
                    "tolerance": 1.0,
                    "lam_increase_factor": 4.0,
                    "lam_decrease_factor": 2.0,
                    "lam_eps": 1e-10,
                    "beta": 0.95,
                },
            },
        },
        "data": {"init_args": {"n_samples": N_STEPS}},
        "trainer": {
            "max_steps": N_STEPS,
            "log_every_n_steps": 1,
            "precision": 64,
            "logger": {
                "init_args": {"save_dir": f"logs/order_profiled_sweep/M{order}"}
            },
            "callbacks": [
                {
                    "class_path": "eisoptx.utils.callbacks.ConfigFileCallback",
                    "init_args": {"every_n_steps": 5},
                },
                "eisoptx.main.CustomProgressBar",
            ],
        },
    }


def scalar_history(run: Path, key: str):
    events = EventAccumulator(str(run)).Reload().Scalars(key)
    return [{"step": event.step, "value": event.value} for event in events]


def run_candidate(order: int, source: Path, temporary: Path):
    design = perturbed_design(source)
    args = design["model"]["lens_parameterization"]["init_args"]
    n_zones = len(args["z"][0])
    summary_path = source.with_name(f"{source.stem}_summary.json")
    summary = json.loads(summary_path.read_text())
    design_summary = summary["design"]
    if design_summary["M"] != order:
        raise ValueError(f"{summary_path} does not describe M={order}.")
    ideal_n_zones = harmonic_zone_count(
        design_summary["diameter_mm"] / 2,
        design_summary["focal_length_mm"],
        order,
        design_summary["lambda0_nm"],
    )
    if n_zones != ideal_n_zones:
        raise ValueError(
            f"M={order} has {n_zones} generated zones; theory requires "
            f"{ideal_n_zones}."
        )

    material = summary["material_fit"]
    wavelengths_nm = material["wavelengths_nm"]
    delta_n = [index - 1.0 for index in material["raywave_n"]]
    nominal_index = wavelengths_nm.index(design_summary["lambda0_nm"])
    alpha = normalized_phase_depth(
        order,
        design_summary["lambda0_nm"],
        wavelengths_nm,
        delta_n,
        delta_n[nominal_index],
    )
    nearest_order, efficiency = scalar_blaze_efficiency(alpha)
    ideal_edges = harmonic_zone_edges(
        design_summary["diameter_mm"] / 2,
        design_summary["focal_length_mm"],
        order,
        design_summary["lambda0_nm"],
    )
    design_path = temporary / f"M{order}_design.yml"
    stage_path = temporary / f"M{order}_fit.yml"
    design_path.write_text(yaml.safe_dump(design, sort_keys=False))
    stage_path.write_text(yaml.safe_dump(inner_stage(order, n_zones), sort_keys=False))

    run_root = ROOT / "logs" / "order_profiled_sweep" / f"M{order}" / "sRz-"
    before = set(run_root.glob("version_*")) if run_root.exists() else set()
    command = [
        sys.executable,
        "-m",
        "eisoptx.main",
        "fit",
        "-c",
        "configs/raywave_zonal/defaults.yml",
        "-c",
        str(design_path),
        "-c",
        str(stage_path),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    created = set(run_root.glob("version_*")) - before if run_root.exists() else set()
    if result.returncode or len(created) != 1:
        for path in created:
            resolved = path.resolve()
            if run_root.resolve() not in resolved.parents:
                raise RuntimeError(
                    f"Refusing to remove unexpected run path: {resolved}"
                )
            shutil.rmtree(resolved)
        raise RuntimeError(
            result.stderr[-3000:] or f"Unexpected run directories: {created}"
        )

    run = created.pop()
    histories = {
        name: scalar_history(run, key)
        for name, key in {
            "merit": "loss/scalar_loss",
            "geometric": "loss/transverse_ray_aberration",
            "coherent": "loss/coherent_wavefront",
            "branch": "loss/hdoe_branch_constraint",
            "valid": "ray_tracing/ray_valid",
            "damping": "optimizer/lm_parameter",
        }.items()
    }
    return {
        "M": order,
        "n_zones": n_zones,
        "theory_screen": {
            "model": "ideal scalar continuous sawtooth; screening only",
            "wavelengths_nm": wavelengths_nm,
            "normalized_phase_depth": alpha.tolist(),
            "nearest_diffraction_order": nearest_order.tolist(),
            "scalar_blaze_efficiency": efficiency.tolist(),
            "ideal_zone_edges_mm": ideal_edges.tolist(),
        },
        "run": str(run.relative_to(ROOT)).replace("\\", "/"),
        "initial": {name: values[0]["value"] for name, values in histories.items()},
        "final": {name: values[-1]["value"] for name, values in histories.items()},
        "history": histories,
    }


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="eisoptx_order_profile_") as directory:
        temporary = Path(directory)
        for order, design in CANDIDATES.items():
            results.append(run_candidate(order, design, temporary))
    results.sort(key=lambda item: item["final"]["merit"])
    report = {
        "selection_metric": "0.5*(normalized geometric residual^2 + coherent residual^2)",
        "equal_inner_steps": N_STEPS,
        "winner_M": results[0]["M"],
        "candidates": results,
    }
    output = HERE / "order_profile_results.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
