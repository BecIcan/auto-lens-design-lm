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

## Initial Structure Generation

Starting structures are produced by a compact conditional network with structural
priors, then traced directly in EADLD. No optimization is applied after generation.
Network details and weights remain private.

74 mm · F/2.8 · ±6.17° · 435–656 nm · 9 elements

![Generated initial structure](docs/assets/initial_structure_layout.png)

![Generated initial-structure spot diagram](docs/assets/initial_structure_spots.png)

Mean/worst RMS: 9.97/12.09 µm. Valid rays: 100%.

```powershell
python examples/generate_initial_structure.py --efl 74 --f-number 2.8 --half-field 6.17 --wavelengths 435 545.5 656 --elements 9 --candidate-count 3 --min-image-clearance 6.3 --max-package-length 55.5 --max-distortion 0.01 --target-cra 12 --backend private_seed.runtime:create_backend --backend-config D:\private\seed.toml --output-dir outputs\seed_demo
```

[Web demo deployment](docs/web-demo.md)

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
