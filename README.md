# EADLD

**End-to-End Auto Diffractive Lens Design｜端到端自动衍射镜头设计**

EADLD 将可微几何光线追迹、环带衍射拓扑、受约束 Levenberg–Marquardt（LM）优化和 RayWave 标量波动验证放在同一条设计链中。Python 包名与导入名统一为 `eadld`。

## 核心能力

- 从球面、非球面或近零功率种子自动生成固定环带拓扑。
- 联合优化曲率、厚度、玻璃和环带连续参数。
- 用完整光程差约束保持指定的整数衍射分支。
- 用 RayWave 兼容的直接 Kirchhoff 求和计算 PSF、Strehl 和 MTF。
- 支持单片与多片衍射镜头，核心计算保持 PyTorch 可微。

## 理论模型

第 i 个环带使用四参数局部面型：

$$
zᵢ(r) = (c/2 + ΔA₁,ᵢ)r² + A₂,ᵢr⁴ + ΔZᵢ,  r ∈ (Rᵢ₋₁, Rᵢ].
$$

设计波长 λ₀ 下，相邻环带满足完整系统光程分支：

$$
L̄ᵢ₊₁ − L̄ᵢ = Mλ₀.
$$

RayWave 直接累加各瞳面样本的复振幅：

$$
U(P) = Σⱼ wⱼKⱼ(P) exp(i k Lⱼ(P)),  PSF(P) = |U(P)|².
$$

受约束 LM 先求满足 AΔxₚ = −c 的特解，再在零空间 N 中求：

$$
minᵧ ‖r + J(Δxₚ + Ny)‖₂² + λ‖D(Δxₚ + Ny)‖₂².
$$

## 一阶与二阶优化对比

下表来自相同五个近零功率种子、相同 G+S+C 残差、相同环带拓扑和 300 步预算。数值为几何 RMS 半径；“±”表示种子间样本标准差。该实验中 Adam 的最终 RMS 更低，因此这里展示的是方法差异，不宣称 LM 在所有终点指标上必然占优。

| 方法 | 信息阶次 | 3×3 均值 RMS / µm | 99 节点均值 RMS / µm | 48 节点均值 RMS / µm | 48 节点最差 RMS / µm |
|---|---|---:|---:|---:|---:|
| Adam | 一阶梯度 | 5.195 ± 0.070 | 3.821 ± 0.075 | 3.301 | 5.544 |
| LM | Gauss–Newton 二阶近似 | 5.563 ± 0.096 | 4.260 ± 0.042 | 3.926 | 5.956 |

LM 的主要价值是把曲率信息与硬分支约束放进同一个可行步中；Adam 更轻量，但不直接利用 JᵀJ 或约束零空间。

## CODE V–RayWave PSF 精度

冻结同一环带处方、孔径、采样和像面网格后，CODE V FFT PSF 与 EADLD RayWave 的归一化 PSF NRMSE 分别为 3.50%、3.37% 和 1.79%。

![CODE V 与 RayWave PSF 精度验证](docs/assets/codev_raywave_psf.png)

| 波长 / nm | PSF NRMSE | CODE V Strehl | RayWave Strehl | Strehl 绝对差 |
|---:|---:|---:|---:|---:|
| 486.1 | 0.0350 | 0.07694 | 0.07690 | −0.00004 |
| 550.0 | 0.0337 | 0.20753 | 0.20655 | −0.00097 |
| 656.3 | 0.0179 | 0.04831 | 0.04929 | +0.00098 |

这是标量传播一致性验证，不包含偏振、台阶侧壁散射、菲涅耳损耗或绝对衍射效率。

## 多片自动优化示例

下面展示三片 Cooke 基线与第二片前表面环带设计的连续优化过程；项目只保留这一处多片公开示例。公开脚本从已记录的环带初始处方开始执行受约束 LM。

![三片环带镜头自动优化](docs/assets/multi_element_optimization.gif)

```powershell
python examples/run_multi_element.py
```

## 安装与验证

```powershell
conda env create -f environment.yml; conda activate eadld; pip install -e ".[dev]"
```

```powershell
python -m pytest -q
```

紧凑的 RayWave 检查：

```powershell
python -m eadld.main test -c configs/raywave_zonal/defaults.yml -c configs/raywave_zonal/designs/visible_f100_f2_m480.yml -c configs/raywave_zonal/wave.yml
```

## 目录

```text
eadld/        核心光学、波动传播和优化代码
configs/      三条最小可复现配置链
examples/     面向使用者的简洁入口
tests/        理论、梯度、约束和 RayWave 回归测试
docs/         展示图与验证指标
```

## 来源与许可

EADLD 基于 MIT 许可的 [EISOPTX / Generalized Aberrations](https://light.princeton.edu/generalized-aberrations) 开发。原许可证保留在 [LICENSE](LICENSE)，衍生工作说明见 [NOTICE.md](NOTICE.md)。论文使用时应同时引用上游工作，并明确说明 EADLD 的环带拓扑、完整光程分支约束、LM 优化和 RayWave 标量传播。
