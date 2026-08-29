# EADLD

[English](README.md) | **中文**

**端到端自动衍射镜头设计**

EADLD 集成可微光线追迹、环带衍射拓扑、受约束 Levenberg–Marquardt 优化和 RayWave 标量波动分析。

## 演示案例

项目包含单片、Cooke 三片和四片 C-mount 案例。最终指标采用独立的 11 视场 × 9 波长审计和 96×32 确定性瞳面采样。

| 案例 | 系统规格 | 优化环带拓扑 | 均值 RMS / µm | 最差 RMS / µm | 最小通光率 |
|---|---|---|---:|---:|---:|
| [单片](configs/demo_cases/designs/singlet_final.yml) | EFL 100 mm · F/8 · ±1° | 后表面，M=30，**12 环** | 5.981 | 15.616 | 0.99580 |
| [Cooke 三片](configs/paper_demos/designs/triplet_postrefine_100.yml) | EFL 100 mm · F/2.8 · ±5° | 第二片前表面，M=90，**41 环** | 7.951 | 19.049 | 1.00000 |
| [四片 C-mount](configs/demo_cases/designs/four_element_final.yml) | EFL 28 mm · F/2 · ±15.88° | 第四片前表面，M=30，**19 环** | 3.390 | 8.834 | 0.98655 |

```powershell
python examples/audit_paper_demos.py
```

## 初始结构生成

初始结构由带结构先验的轻量条件网络生成，随后直接进入 EADLD 真实光线追迹；生成后不再优化。
网络细节和权重不公开。

[体验网站](https://spelling-popularity-honor-group.trycloudflare.com)

74 mm · F/2.8 · ±6.17° · 435–656 nm · 8 片

![网络生成初始结构](docs/assets/initial_structure_layout.png)

![网络生成初始结构点列图](docs/assets/initial_structure_spots.png)

平均/最差 RMS：9.97/12.09 µm；有效光线：100%。

## 光学模型

每个环带采用局部面型：

$$
z_i(r)=\left(\frac{c}{2}+\Delta A_{1,i}\right)r^2+A_{2,i}r^4+\Delta Z_i,
\quad r\in(R_{i-1},R_i].
$$

设计波长 $\lambda_0$ 下，相邻环带满足完整系统光程分支：

$$
\bar L_{i+1}-\bar L_i=M\lambda_0.
$$

RayWave 直接累加瞳面复振幅：

$$
U(P)=\sum_j w_jK_j(P)e^{ikL_j(P)},\qquad \mathrm{PSF}(P)=|U(P)|^2.
$$

受约束 LM 在零空间 $N$ 中求可行步：

$$
\min_y\left\|r+J(\Delta x_p+Ny)\right\|_2^2+
\lambda\left\|D(\Delta x_p+Ny)\right\|_2^2,
\qquad A\Delta x_p=-c.
$$

## 一阶与二阶优化

五个相同近零功率种子、相同残差与拓扑、300 步：

| 方法 | 阶次 | 3×3 均值 RMS / µm | 99 节点均值 RMS / µm | 48 节点均值 RMS / µm | 48 节点最差 RMS / µm |
|---|---|---:|---:|---:|---:|
| Adam | 一阶 | 5.195 ± 0.070 | 3.821 ± 0.075 | 3.301 | 5.544 |
| LM | Gauss–Newton | 5.563 ± 0.096 | 4.260 ± 0.042 | 3.926 | 5.956 |

## CODE V–RayWave PSF 验证

CODE V FFT PSF 与 EADLD RayWave 使用相同处方、孔径、采样和像面网格。

![CODE V 与 RayWave PSF 验证](docs/assets/codev_raywave_psf.png)

| 波长 / nm | PSF NRMSE | CODE V Strehl | RayWave Strehl | Strehl 差值 |
|---:|---:|---:|---:|---:|
| 486.1 | 0.0350 | 0.07694 | 0.07690 | −0.00004 |
| 550.0 | 0.0337 | 0.20753 | 0.20655 | −0.00097 |
| 656.3 | 0.0179 | 0.04831 | 0.04929 | +0.00098 |

## 多片自动优化

公开示例优化 Cooke 三片镜头第二片的环带前表面。

![Cooke 三片镜头自动优化](docs/assets/multi_element_optimization.gif)

![优化后三视场点列图](docs/assets/multi_element_spot_diagram.png)

```powershell
python examples/run_multi_element.py
```

## 安装

```powershell
conda env create -f environment.yml
conda activate eadld
pip install -e ".[dev]"
python -m pytest -q
```

RayWave 检查：

```powershell
python -m eadld.main test -c configs/raywave_zonal/defaults.yml -c configs/raywave_zonal/designs/visible_f100_f2_m480.yml -c configs/raywave_zonal/wave.yml
```

## 项目

EADLD 基于 [EISOPTX](https://light.princeton.edu/generalized-aberrations) 开发。
