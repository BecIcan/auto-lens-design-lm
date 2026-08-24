"""Analytical screening relations for harmonic diffractive lenses.

These functions describe the ideal scalar parent lens used to choose discrete
orders and validate generated topologies.  They do not replace the full-system
ray-wave model used to rank optimized designs.
"""

import numpy as np


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def ideal_opd(radius, focal_length: float):
    """Return the exact spherical optical-path departure in millimeters."""
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
    """Return ideal upper radii of all active zones, ending at the aperture."""
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
    """Return the scalar normalized relief depth alpha(lambda)."""
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
    """Return ideal continuous-sawtooth efficiency in the nearest order."""
    alpha = np.asarray(alpha, dtype=float)
    nearest_order = np.rint(alpha).astype(int)
    efficiency = np.sinc(alpha - nearest_order) ** 2
    return nearest_order, efficiency
