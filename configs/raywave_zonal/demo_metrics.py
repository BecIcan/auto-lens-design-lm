"""Regenerate the acceptance metrics quoted in ``demo_report.md``.

add by cjy.  The numbers in that report predate the fix that stopped
``psf_mode: ray_wave`` from being silently replaced by the geometric spot-diagram
path, so they are recomputed here from a single auditable entry point.

Conditions match ``wave.yml``: on axis, 550 nm, 96 x 64 skew-uniform pupil,
65 x 65 PSF over a 10 um window.  Strehl is absolute (referenced to a
diffraction-limited sphere on the optical axis); the encircled-energy metrics
are relative to the energy inside that window, which holds the design order
only -- see the sizing rules in the README.

    python configs/raywave_zonal/demo_metrics.py
"""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eisoptx.modeling import ray_initialization as ri  # noqa: E402
from eisoptx.modeling.simulation import PSFSampler  # noqa: E402
from eisoptx.optimization.parameterization import LensParameterization  # noqa: E402

torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve().parent
WAVELENGTH, ORDER, EPD = 550.0, 480, 50.0
N_R, N_THETA, PSF_SHAPE, PSF_SIZE = 96, 64, 65, 0.01
DESIGNS = {
    "MATLAB reference": "visible_f100_f2_m480.yml",
    "Perturbed initial": "visible_f100_f2_m480_demo_start.yml",
    "Reloaded final": "visible_f100_f2_m480_demo_final.yml",
}


def load(name):
    args = yaml.safe_load((HERE / "designs" / name).read_text())
    args = args["model"]["lens_parameterization"]["init_args"]
    args.pop("class_path", None)
    args["freeze"] = {}
    return LensParameterization(**args).lens


RAYS = ri.RayInitialization(
    aperture=EPD,
    aperture_type="epd",
    hfov=0.0,
    n_fields=1,
    wavelengths=[WAVELENGTH],
    pupil_sampling_mode="skew_uniform",
    pupil_sampling_kwargs={"n_r": N_R, "n_theta": N_THETA},
    ray_aiming_steps=0,
)


def metrics(lens):
    r0, d0 = RAYS(lens)
    r, d, status, info = list(lens.trace_rays(r0, d0, [WAVELENGTH], yield_on="end"))[-1]
    valid = (status == 0).reshape(-1)

    # Same convention as TransverseRayAberrationResiduals: RMS about a centroid
    # whose x is fixed at zero by rotational symmetry.
    xy = r[:2].reshape(2, -1)[:, (status == 0).reshape(-1)]
    centre = torch.zeros(2, 1)
    centre[1] = xy[1].mean()
    spot = ((xy - centre) ** 2).sum(dim=0).mean().sqrt()

    # Branch-corrected wavefront: remove zone_index * M * lambda0 and the piston.
    distance = lens.s[-1].detach()
    back = distance / d[2]
    reference = (r[:2] - d[:2] * back).reshape(2, -1)[:, valid]
    opl = (info["opl"] - back).reshape(-1)[valid]
    common = opl + (reference.square().sum(dim=0) + distance**2).sqrt()
    edges = lens.z[0, 0, :, 3][(lens.z[0, 0] != 0).any(dim=-1)]
    zone = torch.searchsorted(edges.contiguous(), reference.norm(dim=0).contiguous())
    unwrapped = common - zone.to(common) * (ORDER * WAVELENGTH * 1e-6)
    wfe = (unwrapped - unwrapped.mean()).square().mean().sqrt()

    sampler = PSFSampler(
        psf_shape=(PSF_SHAPE, PSF_SHAPE),
        psf_abs_size=(PSF_SIZE, PSF_SIZE),
        wavelength_weights=[1.0],
        wavelengths=[WAVELENGTH],
        psf_normalization="strehl",
    )
    psf = sampler.forward_ray_wave(
        r, d, info["opl"], status, distance, torch.tensor([WAVELENGTH])
    )[0, 0, 0]

    pitch = PSF_SIZE / PSF_SHAPE
    axis = (torch.arange(PSF_SHAPE) - PSF_SHAPE // 2) * pitch
    radius = (axis[None, :] ** 2 + axis[:, None] ** 2).sqrt().reshape(-1)
    weight = psf.reshape(-1) / psf.sum()
    order = radius.argsort()
    encircled = weight[order].cumsum(0)
    first_zero = 1.22 * WAVELENGTH * 1e-6 * float(distance) / EPD
    index = round(first_zero / pitch)
    return {
        "Absolute Strehl": float(psf.max()),
        "Geometric RMS radius (um)": float(spot) * 1e3,
        "HDOE branch-corrected WFE RMS (nm)": float(wfe) * 1e6,
        "Scalar-wave RMS radius (um)": float((weight * radius**2).sum().sqrt()) * 1e3,
        "EE80 radius (um)": float(radius[order][(encircled >= 0.8).nonzero()[0, 0]])
        * 1e3,
        "Relative intensity at first zero": float(
            psf[PSF_SHAPE // 2, PSF_SHAPE // 2 + index] / psf.max()
        ),
        "Valid-ray fraction": float(valid.to(torch.get_default_dtype()).mean()),
    }


def main():
    results = {label: metrics(load(name)) for label, name in DESIGNS.items()}
    keys = list(next(iter(results.values())))
    width = max(len(k) for k in keys)
    print(f"{'Metric':<{width}} " + " ".join(f"{label:>20}" for label in results))
    for key in keys:
        print(
            f"{key:<{width}} "
            + " ".join(f"{results[label][key]:>20.6f}" for label in results)
        )


if __name__ == "__main__":
    main()
