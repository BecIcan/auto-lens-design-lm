"""Build the paper-facing plots and summary for the f/5.6 M ablation."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper" / "results" / "f5p6_m_ablation_exact"


def main():
    records = json.loads((RESULTS / "results.json").read_text())
    accepted = [record for record in records if record["status"] == "accepted"]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for record in accepted:
        with (ROOT / record["history_csv"]).open() as stream:
            rows = list(csv.DictReader(stream))
        rows = [row for row in rows if row["loss"]]
        steps = [int(row["step"]) for row in rows]
        loss = [float(row["loss"]) for row in rows]
        normalized = [value / loss[0] for value in loss]
        ax.semilogy(
            steps,
            normalized,
            linewidth=1.2,
            label=f"M={record['M']} ({record['zones']} zones)",
        )
    ax.set(
        xlabel="LM step",
        ylabel="Normalized scalar loss",
        title="Identical constrained-LM merit",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "convergence_normalized.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    zones = [record["zones"] for record in accepted]
    rms = [record["mean_spot_um"] for record in accepted]
    ax.plot(zones, rms, "o-", linewidth=1.2, color="#1f77b4")
    for record, x, y in zip(accepted, zones, rms):
        x_offset = -28 if x == max(zones) else 4
        y_offset = -12 if y == max(rms) else 4
        ax.annotate(
            f"M={record['M']}",
            (x, y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        xlabel="Active annular zones",
        ylabel="Held-out mean RMS spot [um]",
        title="f/5.6 order-profiled performance",
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "m_tradeoff.png", dpi=180)
    plt.close(fig)

    protocol = json.loads((RESULTS / "protocol.json").read_text())
    optimized = [record for record in records if record["status"] != "screened"]
    diagnostics = []
    for record in accepted:
        accumulator = EventAccumulator(str(ROOT / record["run"])).Reload()
        loss = accumulator.Scalars("loss/scalar_loss")
        damping = accumulator.Scalars("optimizer/lm_parameter")
        ratio = accumulator.Scalars("optimizer/loss_ratio")
        tail = [item.value for item in loss[-50:]]
        diagnostics.append(
            {
                "M": record["M"],
                "terminal_loss": loss[-1].value,
                "tail_50_relative_span": (
                    (max(tail) - min(tail)) / max(abs(loss[-1].value), 1e-30)
                ),
                "final_lm_damping": damping[-1].value,
                "minimum_lm_damping": min(item.value for item in damping),
                "rejected_steps": sum(
                    not math.isfinite(item.value) or item.value > 1.0
                    for item in ratio
                ),
                "total_steps": len(loss),
            }
        )
    (RESULTS / "optimizer_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    lines = [
        "# Exact f/5.6 harmonic-order ablation",
        "",
        "Orders 1 through 47 were enumerated from the exact spherical-OPD "
        "topology. All optimized candidates use the same weak near-flat seed, "
        "real OHARA S-BSL7 glass, residual construction, constrained LM settings, "
        "300-step budget, and held-out pupil evaluation.",
        "",
        f"Numerical manufacturing window: minimum width "
        f"{protocol['minimum_width_um']:.0f} um, maximum relief "
        f"{protocol['maximum_relief_um']:.0f} um, maximum slope "
        f"{protocol['maximum_slope_deg']:.0f} deg, maximum step/width "
        f"{protocol['maximum_step_width_ratio']:.2f}, and maximum "
        f"{protocol['maximum_zones']} zones.",
        "",
        "| M | Status | Zones | Variables | Mean RMS [um] | Valid | Min width [um] | Step/width | Branch error [mm] | Time [s] |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in optimized:
        if record["status"] == "accepted":
            lines.append(
                f"| {record['M']} | accepted | {record['zones']} | "
                f"{record['trainable_parameters']} | {record['mean_spot_um']:.3f} | "
                f"{record['dense_valid']:.5f} | {record['min_width_um']:.1f} | "
                f"{record['max_step_width_ratio']:.3f} | "
                f"{record['final_branch_error_mm']:.3e} | "
                f"{record['elapsed_seconds']:.1f} |"
            )
        else:
            lines.append(
                f"| {record['M']} | {record['status']}: {record['reason']} | "
                f"{record['zones']} | -- | -- | -- | {record['min_width_um']:.1f} | "
                "-- | -- | -- |"
            )
    lines += [
        "",
        "![Normalized convergence](convergence_normalized.png)",
        "",
        "![Performance-complexity tradeoff](m_tradeoff.png)",
        "",
        "The geometric winner within the declared numerical feasible set is "
        "M=29. The "
        "accepted designs form a performance-complexity Pareto series; this "
        "experiment does not yet include wavelength-dependent diffraction "
        "efficiency, so it is not the final broadband order ranking.",
        "",
        "The final 50 steps are also audited in `optimizer_diagnostics.json`. "
        "A flat terminal loss with saturated damping is recorded as stable "
        "termination, not yet claimed as a projected-gradient convergence proof.",
    ]
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
