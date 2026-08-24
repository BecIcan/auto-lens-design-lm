"""Repeat the exact-M f/5.6 experiment over deterministic near-flat seeds.

The shortlisted orders use the same seeds, residual vector, sampling and
hard-constrained LM settings. Plotting callbacks are disabled so elapsed time
measures optimization rather than report generation.
"""

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "paper" / "results" / "f5p6_m_multiseed_exact"
ORDERS = (29, 33, 38)
SEEDS = tuple(range(10))
STEPS = 300


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sources():
    sys.argv = ["demo_singlet.py", "5.6", "1.0"]
    demo = load_module("f5p6_multiseed_demo", HERE / "demo_singlet.py")
    ablation = load_module(
        "f5p6_multiseed_ablation", HERE / "run_f5p6_m_ablation.py"
    )
    return demo, ablation


def timing_overlay(ablation, order, n_zones, seed, steps):
    log_root = f"logs/f5p6_m_multiseed_exact/M{order}/seed_{seed:02d}"
    overlay = ablation.fit_overlay(order, n_zones, log_root, steps)
    overlay["trainer"]["callbacks"] = [
        {
            "class_path": "eisoptx.utils.callbacks.ConfigFileCallback",
            "init_args": {"every_n_steps": steps},
        }
    ]
    return overlay, log_root


def latest_snapshot(run):
    snapshots = sorted((run / "lens_parameters").glob("*.yml"))
    if not snapshots:
        raise RuntimeError(f"No parameter snapshot in {run}")
    return snapshots[-1]


def candidate_record(demo, ablation, order, seed, steps, temporary):
    edges = demo.zone_radii(order)
    design = temporary / f"M{order}_seed{seed:02d}_initial.yml"
    overlay_path = temporary / f"M{order}_seed{seed:02d}_fit.yml"
    demo.write_near_flat_start(order, design, seed=seed)
    overlay, log_root_relative = timing_overlay(
        ablation, order, len(edges), seed, steps
    )
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False))
    run_root = ROOT / log_root_relative / "sdRz-"
    before = set(run_root.glob("version_*")) if run_root.exists() else set()
    print(f"START M={order} seed={seed}", flush=True)
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
            str(overlay_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    created = set(run_root.glob("version_*")) - before if run_root.exists() else set()
    if result.returncode or len(created) != 1:
        for run in created:
            shutil.rmtree(run)
        raise RuntimeError(result.stderr[-4000:] or f"Unexpected runs: {created}")
    run = created.pop()
    histories = {
        name: ablation.scalar_history(run, key)
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
    lens = demo.lens_from_parameters(latest_snapshot(run))
    spots, dense_valid = demo.spot_radii(lens, demo.eval_sampling(len(edges)))
    manufacturing = ablation.manufacturing_details(lens)
    ranks = [item["value"] for item in histories["constraint_rank"]]
    branch = histories["branch"][-1]["value"] if histories["branch"] else float("inf")
    reasons = []
    if not ranks or min(ranks) < len(edges) - 1:
        reasons.append("constraint rank loss")
    if branch > ablation.MAX_BRANCH_ERROR_MM:
        reasons.append(f"branch error {branch:.3e} mm")
    if dense_valid < ablation.MIN_VALID:
        reasons.append(f"valid fraction {dense_valid:.6f}")
    if manufacturing["max_step_width_ratio"] > ablation.MAX_STEP_WIDTH_RATIO:
        reasons.append(f"step/width {manufacturing['max_step_width_ratio']:.6f}")
    if manufacturing["max_step_um"] > ablation.MAX_RELIEF_MM * 1e3:
        reasons.append(f"step depth {manufacturing['max_step_um']:.3f} um")
    if manufacturing["max_slope"] > ablation.MAX_SLOPE:
        reasons.append(f"surface slope {manufacturing['max_slope_deg']:.3f} deg")
    loss = histories["loss"]
    record = {
        "M": order,
        "seed": seed,
        "status": "accepted" if not reasons else "rejected",
        "reason": "; ".join(reasons),
        "zones": len(edges),
        "initial_curvatures_per_mm": demo.near_flat_curvatures(seed),
        "loss_initial": loss[0]["value"],
        "loss_final": loss[-1]["value"],
        "optimization_seconds": loss[-1]["wall_time"] - loss[0]["wall_time"],
        "spots_um": spots,
        "mean_spot_um": sum(spots) / len(spots),
        "dense_valid": dense_valid,
        "efl_mm": float(lens.efl.squeeze()),
        "final_branch_error_mm": branch,
        "constraint_rank": min(ranks) if ranks else 0,
        "final_damping": histories["damping"][-1]["value"],
        **manufacturing,
    }
    if reasons:
        shutil.rmtree(run)
    else:
        output = RESULTS / f"M{order}_seed{seed:02d}_history.csv"
        indexed = {
            name: {item["step"]: item["value"] for item in values}
            for name, values in histories.items()
        }
        with output.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", *histories])
            all_steps = sorted({step for values in indexed.values() for step in values})
            for step in all_steps:
                writer.writerow([step, *(indexed[name].get(step) for name in histories)])
        record["run"] = str(run.relative_to(ROOT)).replace("\\", "/")
        record["history_csv"] = str(output.relative_to(ROOT)).replace("\\", "/")
    print(
        f"{record['status'].upper()} M={order} seed={seed}: "
        f"RMS={record['mean_spot_um']:.4f} um, "
        f"branch={branch:.2e} mm, t={record['optimization_seconds']:.1f}s",
        flush=True,
    )
    return record


def summarize(records, orders):
    summary = []
    for order in orders:
        group = [item for item in records if item["M"] == order]
        accepted = [item for item in group if item["status"] == "accepted"]
        row = {
            "M": order,
            "runs": len(group),
            "accepted": len(accepted),
            "success_rate": len(accepted) / len(group),
        }
        for key in ("mean_spot_um", "optimization_seconds", "final_branch_error_mm"):
            if not accepted:
                row[key + "_median"] = None
                row[key + "_q1"] = None
                row[key + "_q3"] = None
                continue
            values = torch.tensor([item[key] for item in accepted], dtype=torch.float64)
            row[key + "_median"] = float(values.median())
            row[key + "_q1"] = float(torch.quantile(values, 0.25))
            row[key + "_q3"] = float(torch.quantile(values, 0.75))
        summary.append(row)
    return summary


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", nargs="+", type=int, default=list(ORDERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    if set(args.orders) - set(ORDERS):
        raise ValueError(f"Orders must be selected from {ORDERS}")
    if args.steps < 2:
        raise ValueError("steps must be at least two")
    if args.results_dir is not None:
        RESULTS = args.results_dir.resolve()
        if ROOT.resolve() not in RESULTS.parents:
            raise ValueError("results-dir must be inside the repository")
    demo, ablation = load_sources()
    RESULTS.mkdir(parents=True, exist_ok=True)
    protocol = {
        "purpose": "exact-M multi-seed stability",
        "orders": args.orders,
        "seeds": args.seeds,
        "steps": args.steps,
        "randomized_quantity": "two weak initial curvatures only",
        "curvature_range_per_mm": [-demo.CURVATURE_SCALE, demo.CURVATURE_SCALE],
        "plotting_disabled_during_timing": True,
        "shared_merit": "hard constrained LM from exact-M ablation",
    }
    (RESULTS / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    with tempfile.TemporaryDirectory(prefix="eisoptx_f5p6_multiseed_") as directory:
        temporary = Path(directory)
        for order in args.orders:
            for seed in args.seeds:
                records.append(candidate_record(demo, ablation, order, seed, args.steps, temporary))
                (RESULTS / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    summary = summarize(records, args.orders)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
