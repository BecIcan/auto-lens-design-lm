"""Consolidate and report the exact-M multi-seed experiment."""

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper" / "results" / "f5p6_m_multiseed_exact"
PARTS = (RESULTS, RESULTS / "M29_remaining", RESULTS / "M33", RESULTS / "M38")
ORDERS = (29, 33, 38)
SEEDS = tuple(range(10))


def read_records():
    records = []
    for part in PARTS:
        path = part / "results.json"
        if path.exists():
            records.extend(json.loads(path.read_text()))
    records.sort(key=lambda item: (item["M"], item["seed"]))
    keys = [(item["M"], item["seed"]) for item in records]
    expected = [(order, seed) for order in ORDERS for seed in SEEDS]
    if keys != expected:
        raise RuntimeError(f"Expected {expected}, found {keys}")
    return records


def copy_histories(records):
    for item in records:
        source = ROOT / item["history_csv"]
        target = RESULTS / f"M{item['M']}_seed{item['seed']:02d}_history.csv"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        item["history_csv"] = str(target.relative_to(ROOT)).replace("\\", "/")


def quartiles(values):
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def sign_test_two_sided(wins, trials):
    tail = sum(math.comb(trials, index) for index in range(wins + 1)) / 2**trials
    return min(1.0, 2 * tail)


def summarize(records):
    summary = []
    by_order = {}
    for order in ORDERS:
        group = [item for item in records if item["M"] == order]
        by_order[order] = {item["seed"]: item for item in group}
        summary.append(
            {
                "M": order,
                "zones": group[0]["zones"],
                "runs": len(group),
                "accepted": sum(item["status"] == "accepted" for item in group),
                "mean_spot_um": quartiles([item["mean_spot_um"] for item in group]),
                "branch_error_mm": quartiles(
                    [item["final_branch_error_mm"] for item in group]
                ),
                "dense_valid": quartiles([item["dense_valid"] for item in group]),
            }
        )
    paired = []
    for lower, upper in ((29, 33), (29, 38), (33, 38)):
        differences = [
            by_order[lower][seed]["mean_spot_um"]
            - by_order[upper][seed]["mean_spot_um"]
            for seed in SEEDS
        ]
        lower_wins = sum(value < 0 for value in differences)
        paired.append(
            {
                "comparison": f"M{lower}-M{upper}",
                "median_paired_difference_um": float(np.median(differences)),
                "lower_M_wins": lower_wins,
                "ties": sum(value == 0 for value in differences),
                "two_sided_exact_sign_p": sign_test_two_sided(
                    min(lower_wins, len(SEEDS) - lower_wins), len(SEEDS)
                ),
            }
        )
    return summary, paired


def history(item):
    with (ROOT / item["history_csv"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = np.asarray([float(row["loss"]) for row in rows], dtype=float)
    return values / values[0]


def plot_results(records):
    palette = {29: "#0072B2", 33: "#E69F00", 38: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for order in ORDERS:
        group = [item for item in records if item["M"] == order]
        curves = np.stack([history(item) for item in group])
        steps = np.arange(curves.shape[1])
        axes[0].plot(steps, np.median(curves, axis=0), color=palette[order], label=f"M={order}")
        axes[0].fill_between(
            steps,
            np.quantile(curves, 0.25, axis=0),
            np.quantile(curves, 0.75, axis=0),
            color=palette[order],
            alpha=0.18,
            linewidth=0,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("LM step")
    axes[0].set_ylabel("normalized merit")
    axes[0].legend(frameon=False)
    values = [
        [item["mean_spot_um"] for item in records if item["M"] == order]
        for order in ORDERS
    ]
    box = axes[1].boxplot(values, tick_labels=[str(order) for order in ORDERS], patch_artist=True)
    for patch, order in zip(box["boxes"], ORDERS):
        patch.set_facecolor(palette[order])
        patch.set_alpha(0.45)
    for seed in SEEDS:
        paired = [
            next(item["mean_spot_um"] for item in records if item["M"] == order and item["seed"] == seed)
            for order in ORDERS
        ]
        axes[1].plot(range(1, 4), paired, color="0.65", linewidth=0.5, alpha=0.7)
    axes[1].set_xlabel("harmonic order M")
    axes[1].set_ylabel("polychromatic RMS spot (um)")
    fig.savefig(RESULTS / "multiseed_stability.png", dpi=220)
    plt.close(fig)


def write_readme(summary, paired):
    lines = [
        "# Exact-M multi-seed stability (f/5.6)",
        "",
        "Thirty hard-constrained LM runs use the same ten weak near-flat seeds for each exact topology. Only the two initial base curvatures vary with seed; glass, merit, rays, zone boundaries and all solver settings are identical.",
        "",
        "| M | zones | success | median RMS [IQR] (um) | range (um) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        spot = row["mean_spot_um"]
        lines.append(
            f"| {row['M']} | {row['zones']} | {row['accepted']}/{row['runs']} | "
            f"{spot['median']:.4f} [{spot['q1']:.4f}, {spot['q3']:.4f}] | "
            f"{spot['minimum']:.4f}-{spot['maximum']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Paired exact sign tests use the same seed on both M values:",
            "",
            "| comparison | median paired difference (um) | lower-M wins | exact p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['comparison']} | {row['median_paired_difference_um']:.4f} | "
            f"{row['lower_M_wins']}/10 | {row['two_sided_exact_sign_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "All 30 runs pass the branch-rank, branch-error, ray-validity, relief, slope and step/width gates. The result supports a stable feasible M window and a performance/complexity trade-off; it does not establish a universal optimum M.",
            "",
            "Elapsed times are retained in `results.json` for provenance but are not used for optimizer-speed claims because the three M batches ran concurrently.",
            "",
            "![Multi-seed stability](multiseed_stability.png)",
            "",
        ]
    )
    (RESULTS / "README.md").write_text("\n".join(lines))


def remove_parts():
    for part in PARTS[1:]:
        if part.exists():
            if RESULTS.resolve() not in part.resolve().parents:
                raise RuntimeError(f"Refusing to remove unexpected path: {part}")
            shutil.rmtree(part)


def main():
    records = read_records()
    copy_histories(records)
    summary, paired = summarize(records)
    (RESULTS / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS / "paired_comparisons.json").write_text(json.dumps(paired, indent=2) + "\n")
    plot_results(records)
    write_readme(summary, paired)
    remove_parts()


if __name__ == "__main__":
    main()
