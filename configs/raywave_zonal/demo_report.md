# Annular/HDOE recovery demo

> Material status: this demo uses a three-line `ZNS_broad` dispersion proxy,
> not a traced EISOPTX catalog material. It verifies software behavior only and
> is not publication-grade optical evidence.

This is a deliberately small, auditable EISOPTX demo. It proves the fixed-topology annular design and scalar
RayWave-compatible validation loop at 550 nm on axis. It is not a broadband-efficiency claim.

## Specification and lineage

- Target: EFL 100 mm, F/2, 550 nm, on axis.
- Annular topology: `sRz-`, `M=480`, 12 zones, 25 mm clear radius.
- Reference: MATLAB-generated `designs/visible_f100_f2_m480.yml`.
- Controlled starting point: image spacing +0.2 mm; consecutive A2 values alternately +2%/-2%.
- LM variables: image spacing and 12 A2 values. All Rmax, dA1, dZ, base curvature and material values stay fixed.
- Budget: 15 LM steps, 64x16 pupil samples; checkpoints and native plots every 5 steps.
- Independent gate: 96x32 pupil samples and a 65x65 scalar-wave PSF over a 10 um window.

## Independently reloaded result

| Metric | MATLAB reference | Perturbed initial | Reloaded final |
|---|---:|---:|---:|
| Absolute Strehl | 0.999999 | 0.023569 | 0.999999 |
| Geometric RMS radius (um) | 0.086043 | 40.629926 | 0.085659 |
| HDOE branch-corrected WFE RMS (nm) | 0.096034 | 1986.488481 | 0.098023 |
| Scalar-wave RMS radius (um) | 1.195813 | 3.461683 | 1.195813 |
| EE80 radius (um) | 0.935810 | 4.751829 | 0.935810 |
| Relative intensity at theoretical first zero | 0.000140 | 0.185490 | 0.000140 |
| Valid-ray fraction | 1.0 | 1.0 | 1.0 |

Regenerate with `python configs/raywave_zonal/demo_metrics.py`. Conditions match `wave.yml`: on axis, 550 nm,
96 x 64 skew-uniform pupil, 65 x 65 PSF over a 10 um window.

Two entries changed meaning since the first version of this report:

- **Absolute Strehl replaces the earlier "crop-normalized peak ratio"** (0.998248 / 0.300101 / 0.998248). The old
  quantity compared peaks after each PSF had been normalized to unit sum inside the analysis window, so it could not
  see energy leaving the window. The absolute Strehl references a diffraction-limited sphere converging on the
  optical axis, which is the `KirrchoffStrl_3D.m` convention. It is the stricter measure: the perturbed design scores
  0.024 rather than 0.300.
- The **HDOE WFE** is now computed by the script above rather than by `HDOEPhaseResiduals`, which normalizes and
  re-references differently; both agree that the reference and the recovered design sit near 0.1 nm, i.e. lambda/5000.

The four scalar-wave rows reproduce the original report exactly (EE80 to all six digits), which confirms they were
produced by calling the coherent kernel directly and were **not** affected by the defect that let
`psf_mode: ray_wave` fall back to the geometric spot-diagram path inside `try_build_simulation_model`. The geometric
RMS agrees with the `loss/transverse_ray_aberration` logged by the CLI to within 0.1%; the two use slightly different
centroid and normalization conventions.

The independently reloaded final design file reproduces the unperturbed MATLAB reference within the frozen gates.
All 12 Rmax values, dA1 values and dZ values have exactly zero change during the checkpointed optimization. The
runtime image spacing moves from 100.199997425 mm to 99.999982119 mm after the promoted YAML is reloaded.

The promoted checkpoint is `logs/annular_demo_fit/sRz-/version_2/checkpoints/last.ckpt`, SHA256
`6C97583CFCF452350B7D58F14FD87F4538241E955127F01AA8BD61D44F826C20`.

## Reproduce

Activate the tested environment:

```powershell
conda activate autoresearch
```

Run the 15-step demo:

```powershell
python -m eisoptx.main fit -c configs/raywave_zonal/defaults.yml -c configs/raywave_zonal/designs/visible_f100_f2_m480_demo_start.yml -c configs/raywave_zonal/demo_fit.yml
```

Validate the resulting checkpoint using its newly created `version_N` directory:

```powershell
python -m eisoptx.main test -c configs/raywave_zonal/defaults.yml -c configs/raywave_zonal/designs/visible_f100_f2_m480_demo_start.yml -c configs/raywave_zonal/demo_fit.yml -c configs/raywave_zonal/wave.yml --ckpt_path=logs/annular_demo_fit/sRz-/version_N/checkpoints/last.ckpt
```

The promoted final prescription can also be tested without a checkpoint:

```powershell
python -m eisoptx.main test -c configs/raywave_zonal/defaults.yml -c configs/raywave_zonal/designs/visible_f100_f2_m480_demo_final.yml -c configs/raywave_zonal/wave.yml
```

## Process record

- `demo_history.csv`: every LM step, including geometric RMS, HDOE WFE, defocus and damping.
- `demo_zone_changes.csv`: per-zone initial/final A2 changes and zero-change audit for dA1, dZ and Rmax.
- `demo_result.json`: machine-readable specification, lineage, independent metrics and limitations.
- `logs/annular_demo_fit/sRz-/version_2/`: merged config, checkpoints, TensorBoard events, runtime lens parameters,
  layouts, spot diagrams, scalar PSFs and zonal surface/change plots at steps 0/5/10/15.

Use `tensorboard --logdir logs/annular_demo_fit` to inspect the complete optimization trajectory.

## Physical boundary

The scalar propagator coherently evaluates the physical relief but normalizes each wavelength PSF inside the finite
image window. Therefore `crop-normalized peak ratio` is not an absolute diffraction efficiency. This demo does not
model multi-order energy splitting, Fresnel loss, polarization, sidewalls, rounding, roughness, thermal behavior or
tolerances, and it makes no claim about 486.1-656.3 nm broadband performance.
