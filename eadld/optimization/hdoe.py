"""谐衍射镜头的解析筛选公式。

论文中的理想球面光程差为

    W(r) = sqrt(f^2 + r^2) - f.

第 i 个环带边界满足 `W(r_i) = i M lambda_0`，因此
`r_i = sqrt(2 f i M lambda_0 + (i M lambda_0)^2)`。

理想连续闪耀模型使用 `eta_m = sinc^2(alpha - m)`。这些公式只负责
离散阶次和初始拓扑筛选；最终排序仍由完整系统光线追迹与 RayWave 完成。
"""

import numpy as np


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def ideal_opd(radius, focal_length: float):
    """返回精确球面光程差 `sqrt(f^2 + r^2) - f`，单位为 mm。"""
    radius = np.asarray(radius, dtype=float)
    if focal_length <= 0 or np.any(radius < 0):
        raise ValueError("Focal length must be positive and radius nonnegative.")
    return np.hypot(focal_length, radius) - focal_length


def harmonic_zone_count(
    aperture_radius: float,
    focal_length: float,
    diffraction_order: int,
    design_wavelength_nm: float,
) -> int:
    """Return the exact number of active right-closed zones."""
    diffraction_order = _positive_integer(diffraction_order, "Diffraction order")
    if design_wavelength_nm <= 0:
        raise ValueError("Design wavelength must be positive.")
    period = diffraction_order * design_wavelength_nm * 1e-6
    return max(1, int(np.ceil(ideal_opd(aperture_radius, focal_length) / period)))


def harmonic_zone_edges(
    aperture_radius: float,
    focal_length: float,
    diffraction_order: int,
    design_wavelength_nm: float,
):
    """返回满足 `W(r_i)=i M lambda_0` 的环带外边界。"""
    n_zones = harmonic_zone_count(
        aperture_radius, focal_length, diffraction_order, design_wavelength_nm
    )
    period = diffraction_order * design_wavelength_nm * 1e-6
    branch = np.arange(1, n_zones + 1, dtype=float) * period
    edges = np.sqrt(2 * focal_length * branch + branch**2)
    return np.minimum(edges, aperture_radius)


def normalized_phase_depth(
    diffraction_order: int,
    design_wavelength_nm: float,
    wavelengths_nm,
    delta_n,
    design_delta_n: float,
):
    """返回标量归一化浮雕深度 `alpha(lambda)`。"""
    diffraction_order = _positive_integer(diffraction_order, "Diffraction order")
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    delta_n = np.asarray(delta_n, dtype=float)
    if design_wavelength_nm <= 0 or np.any(wavelengths_nm <= 0):
        raise ValueError("Wavelengths must be positive.")
    if wavelengths_nm.shape != delta_n.shape:
        raise ValueError("Wavelengths and refractive-index contrast must match.")
    if design_delta_n == 0:
        raise ValueError("Design refractive-index contrast must be nonzero.")
    return (
        diffraction_order
        * design_wavelength_nm
        / wavelengths_nm
        * delta_n
        / design_delta_n
    )


def scalar_blaze_efficiency(alpha):
    """返回最近整数级次及理想效率 `sinc^2(alpha-m)`。"""
    alpha = np.asarray(alpha, dtype=float)
    nearest_order = np.rint(alpha).astype(int)
    efficiency = np.sinc(alpha - nearest_order) ** 2
    return nearest_order, efficiency
