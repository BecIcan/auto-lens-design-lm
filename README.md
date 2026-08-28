# EADLD

**English** | [中文](README.zh-CN.md)

**End-to-End Auto Diffractive Lens Design**

EADLD combines differentiable ray tracing, annular diffractive topology, constrained Levenberg–Marquardt optimization, and RayWave scalar-wave analysis.

## Demos

The singlet, Cooke triplet, and four-element C-mount cases are included. Final values use an independent 11-field × 9-wavelength audit with deterministic 96×32 pupil sampling.

| Case | Specification | Optimized annular topology | Mean RMS / µm | Worst RMS / µm | Minimum throughput |
|---|---|---|---:|---:|---:|
| [Singlet](configs/demo_cases/designs/singlet_final.yml) | EFL 100 mm · F/8 · ±1° | Rear surface, M=30, **12 zones** | 5.981 | 15.616 | 0.99580 |
| [Cooke triplet](configs/paper_demos/designs/triplet_postrefine_100.yml) | EFL 100 mm · F/2.8 · ±5° | Element 2 front, M=90, **41 zones** | 7.951 | 19.049 | 1.00000 |
| [Four-element C-mount](configs/demo_cases/designs/four_element_final.yml) | EFL 28 mm · F/2 · ±15.88° | Element 4 front, M=30, **19 zones** | 3.390 | 8.834 | 0.98655 |

```powershell
python examples/audit_paper_demos.py
```

The full 99-node audit is written to `outputs/paper_demos/audit.json`.

## Private Spec-to-Seed, Public Physics

EADLD now separates proprietary initial-structure intelligence from reproducible
physics verification. A private backend receives focal length, F-number, field,
spectrum, element count, and candidate count. It returns complete lens seeds through
the public [`InitialStructureBackend`](eadld/initialization/api.py) contract. EADLD
then applies a fixed-EPD mechanical gate, native real-ray tracing, candidate ranking,
and hash-bound layout/spot artifacts.

The inference path is deliberately **one shot**: no post-generation optimizer,
paraxial focal/image solve, or catalog-glass snapping modifies a candidate. The
backend architecture, training set, pair identities, prescriptions, and checkpoints
are not distributed in this repository.

![Private seed generator benchmark](docs/assets/initial_structure_benchmark.png)

| Spectrum | Elements | Retrieved source / µm | Generated / µm | Private reference / µm | Valid rays |
|---|---:|---:|---:|---:|---:|
| 435–656 nm | 8 | 39.56 | **12.55** | 11.28 | 97.47% |
| 435–656 nm | 9 | 15.60 | **9.96** | 8.52 | 100.00% |
| 435–656 nm | 10 | 12.79 | **12.53** | 10.33 | 99.63% |
| 435–850 nm | 8 | 23.46 | 67.41 | 27.45 | 95.53% |
| 435–850 nm | 9 | 22.66 | 23.30 | 15.53 | 100.00% |
| 435–850 nm | 10 | 22.10 | **19.41** | 16.86 | 99.77% |

These are five-field EADLD real-ray seed metrics from a small internal teacher set.
They are proof-of-concept results, not a held-out generalization, finished-lens, or
external-tool-equivalence claim. The wide-band 8-element regression is retained to
make the current limitation visible. Aggregate data are in
[`docs/initial_structure_benchmark.json`](docs/initial_structure_benchmark.json).

Representative 9-element visible result:

![Generated initial structure](docs/assets/initial_structure_layout.png)

![Generated initial-structure spot diagram](docs/assets/initial_structure_spots.png)

Private backend usage:

```powershell
python examples/generate_initial_structure.py --efl 74 --f-number 2.8 --half-field 6.17 --wavelengths 435 545.5 656 --elements 9 --candidate-count 3 --min-image-clearance 6.3 --max-package-length 55.5 --max-distortion 0.01 --target-cra 12 --backend private_seed.runtime:create_backend --backend-config D:\private\seed.toml --output-dir outputs\seed_demo
```

## Optical Model

Each annular zone uses a local sag profile:

$$
z_i(r)=\left(\frac{c}{2}+\Delta A_{1,i}\right)r^2+A_{2,i}r^4+\Delta Z_i,
\quad r\in(R_{i-1},R_i].
$$

Adjacent zones follow the full-system optical-path branch at design wavelength $\lambda_0$:

$$
\bar L_{i+1}-\bar L_i=M\lambda_0.
$$

RayWave evaluates the complex field by direct pupil summation:

$$
U(P)=\sum_j w_jK_j(P)e^{ikL_j(P)},\qquad \mathrm{PSF}(P)=|U(P)|^2.
$$

Constrained LM solves a feasible step in the null space $N$:

$$
\min_y\left\|r+J(\Delta x_p+Ny)\right\|_2^2+
\lambda\left\|D(\Delta x_p+Ny)\right\|_2^2,
\qquad A\Delta x_p=-c.
$$

## First- vs Second-Order Optimization

Five matched near-zero-power seeds, identical residuals and topology, 300 steps:

| Method | Order | 3×3 mean RMS / µm | 99-node mean RMS / µm | 48-node mean RMS / µm | 48-node worst RMS / µm |
|---|---|---:|---:|---:|---:|
| Adam | First-order | 5.195 ± 0.070 | 3.821 ± 0.075 | 3.301 | 5.544 |
| LM | Gauss–Newton | 5.563 ± 0.096 | 4.260 ± 0.042 | 3.926 | 5.956 |

## CODE V–RayWave PSF Validation

CODE V FFT PSF and EADLD RayWave use the same prescription, aperture, sampling, and image grid.

![CODE V and RayWave PSF validation](docs/assets/codev_raywave_psf.png)

| Wavelength / nm | PSF NRMSE | CODE V Strehl | RayWave Strehl | Strehl difference |
|---:|---:|---:|---:|---:|
| 486.1 | 0.0350 | 0.07694 | 0.07690 | −0.00004 |
| 550.0 | 0.0337 | 0.20753 | 0.20655 | −0.00097 |
| 656.3 | 0.0179 | 0.04831 | 0.04929 | +0.00098 |

## Multi-Element Optimization

The public example optimizes the annular front surface of the second element in a Cooke triplet.

![Cooke triplet automatic optimization](docs/assets/multi_element_optimization.gif)

![Optimized three-field spot diagram](docs/assets/multi_element_spot_diagram.png)

```powershell
python examples/run_multi_element.py
```

## Install

```powershell
conda env create -f environment.yml
conda activate eadld
pip install -e ".[dev]"
python -m pytest -q
```

RayWave check:

```powershell
python -m eadld.main test -c configs/raywave_zonal/defaults.yml -c configs/raywave_zonal/designs/visible_f100_f2_m480.yml -c configs/raywave_zonal/wave.yml
```

## Project

EADLD is based on [EISOPTX](https://light.princeton.edu/generalized-aberrations).
