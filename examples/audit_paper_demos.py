"""Independent 11-field / 9-wavelength audit of the paper demonstrations.

This script only traces immutable prescriptions; it does not reuse the LM merit
function and never changes a design. Run it from the repository root:

    python examples/audit_paper_demos.py
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eadld.modeling import ray_initialization as ri
from eadld.optimization.parameterization import LensParameterization


WAVELENGTHS = np.linspace(450.0, 650.0, 9).tolist()
N_FIELDS = 11
PUPIL = {"n_r": 96, "n_theta": 32}
OUTPUT = ROOT / "outputs" / "paper_demos" / "audit.json"


@dataclass(frozen=True)
class Case:
    name: str
    reference: str
    final: str
    efl: float
    f_number: float
    half_field: float
    order: int


CASES = (
    Case("singlet", "configs/demo_cases/designs/singlet_initial.yml", "configs/demo_cases/designs/singlet_final.yml", 100.0, 8.0, 1.0, 30),
    Case("triplet", "configs/paper_demos/designs/triplet_refractive_reference.yml", "configs/paper_demos/designs/triplet_postrefine_100.yml", 100.0, 2.8, 5.0, 90),
    Case("four_element", "configs/paper_demos/designs/four_element_flat.yml", "configs/demo_cases/designs/four_element_final.yml", 28.0, 2.0, 15.88, 30),
)


def load_lens(relative_path: str):
    path = ROOT / relative_path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    args = config["model"]["lens_parameterization"]["init_args"]
    args.setdefault("dpgf", None)
    lens = LensParameterization(**args).double().lens
    return path, lens


@torch.no_grad()
def evaluate(case: Case, relative_path: str) -> dict:
    path, lens = load_lens(relative_path)
    has_zones = lens.z is not None and lens.z.numel() > 0
    mode = "skew_uniform_zonal" if has_zones else "skew_uniform"
    initializer = ri.RayInitialization(
        aperture=case.efl / case.f_number,
        aperture_type="epd",
        hfov=case.half_field,
        n_fields=N_FIELDS,
        wavelengths=WAVELENGTHS,
        wavelength_weights=[1.0] * len(WAVELENGTHS),
        pupil_sampling_mode=mode,
        pupil_sampling_kwargs=PUPIL,
        ray_aiming_steps=0,
    )
    r0, d0 = initializer(lens)
    r, _, status, _ = list(lens.trace_rays(r0, d0, WAVELENGTHS, yield_on="end"))[-1]
    xy, valid = r[:2, ..., 0], status[..., 0] == 0
    if has_zones:
        pupil_weights = ri.zonal_pupil_weights(
            **PUPIL,
            zone_edges=ri.zone_edges_from_lens(lens, case.efl / case.f_number),
        )
    else:
        pupil_weights = torch.full(
            (xy.shape[2],), 1.0 / xy.shape[2], dtype=xy.dtype, device=xy.device
        )

    # 加权 RMS 半径：sqrt(sum(w_i * ||r_i-r_bar||^2) / sum(w_i))。
    rms = np.full((N_FIELDS, len(WAVELENGTHS)), np.nan)
    throughput = np.zeros_like(rms)
    for field in range(N_FIELDS):
        for wave in range(len(WAVELENGTHS)):
            weights = pupil_weights * valid[field, :, wave]
            throughput[field, wave] = float(weights.sum())
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            points = xy[:, field, :, wave].where(valid[field, :, wave][None], 0.0)
            centroid = (points * weights[None]).sum(1, keepdim=True)
            rms[field, wave] = float(
                (weights * (points - centroid).square().sum(0)).sum().sqrt()
            ) * 1e3

    zones = 0 if not has_zones else int((lens.z[0, 0, :, 3] > 0).sum())
    efl = float(lens.efl)
    return {
        "source": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mean_rms_radius_um": float(np.nanmean(rms)),
        "worst_rms_radius_um": float(np.nanmax(rms)),
        "minimum_weighted_throughput": float(np.min(throughput)),
        "rms_radius_um": rms.tolist(),
        "efl_mm": efl if math.isfinite(efl) else None,
        "ttl_mm": float(lens.s.sum()),
        "active_zones": zones,
    }


def main() -> None:
    report = {
        "protocol": {
            "fields": N_FIELDS,
            "fields_are_uniform_from_zero_to_half_field": True,
            "wavelengths_nm": WAVELENGTHS,
            "uniform_wavelength_weights": True,
            "pupil": PUPIL,
            "metric": "weighted geometric RMS spot radius",
        },
        "cases": {},
    }
    for case in CASES:
        report["cases"][case.name] = {
            "specification": {
                "target_efl_mm": case.efl,
                "f_number": case.f_number,
                "half_field_deg": case.half_field,
                "annular_order_M": case.order,
            },
            "reference": evaluate(case, case.reference),
            "final": evaluate(case, case.final),
        }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
