"""Minimal spherical CODE V SEQ export used by the public seed interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch

from eadld.modeling.optics import Lens


@dataclass(frozen=True)
class SphericalPrescription:
    title: str
    epd_mm: float
    wavelengths_nm: tuple[float, ...]
    reference_wavelength_index: int
    field_angles_deg: tuple[float, ...]
    radii_mm: tuple[float, ...]
    thicknesses_mm: tuple[float, ...]
    nd: tuple[float | None, ...]
    vd: tuple[float | None, ...]
    stop_after_surface: int


def _encode_fictitious_glass(nd: float, vd: float) -> str:
    n_code = round((min(max(nd, 1.000001), 2.199999) - 1.0) * 1_000_000)
    v_code = round(min(max(vd, 1.0), 99.9999) / 100.0 * 1_000_000)
    return f"{n_code:06d}.{v_code:06d}"


def lens_to_spherical_prescription(
    lens: Lens,
    *,
    title: str,
    epd_mm: float,
    wavelengths_nm: tuple[float, ...],
    field_angles_deg: tuple[float, ...],
) -> SphericalPrescription:
    """Convert one spherical EADLD lens to sequential surface rows."""
    sequence = lens.sequence.sequence
    if any(char not in "R-s" for char in sequence):
        raise ValueError("当前 SEQ 导出仅支持球面折射结构")
    surface_types = [char for char in sequence if char in "R-"]
    stop_offset = sequence.index("s")
    stop_after_surface = sum(char in "R-" for char in sequence[:stop_offset])
    if stop_after_surface >= len(surface_types):
        raise ValueError("光阑位置无法映射到 CODE V 表面")
    if len(surface_types) != lens.s.shape[0]:
        raise ValueError("厚度数量与表面拓扑不一致")

    def scalar(tensor: torch.Tensor, index: int) -> float:
        return float(tensor[index].reshape(-1)[0])

    radii: list[float] = []
    nd: list[float | None] = []
    vd: list[float | None] = []
    curvature_index = 0
    material_index = 0
    for index, surface_type in enumerate(surface_types):
        has_interface = surface_type == "R" or (
            surface_type == "-" and index > 0 and surface_types[index - 1] == "R"
        )
        if has_interface:
            curvature = scalar(lens.c, curvature_index)
            curvature_index += 1
            radii.append(0.0 if abs(curvature) < 1e-15 else 1.0 / curvature)
        else:
            radii.append(0.0)
        if surface_type == "R":
            nd.append(scalar(lens.nd, material_index))
            vd.append(scalar(lens.vd, material_index))
            material_index += 1
        else:
            nd.append(None)
            vd.append(None)
    if curvature_index != lens.c.shape[0] or material_index != lens.nd.shape[0]:
        raise ValueError("曲率或材料数量与表面拓扑不一致")

    ordered_wavelengths = tuple(sorted(wavelengths_nm, reverse=True))
    nominal_wavelength = float(lens.w0)
    reference = min(
        range(len(ordered_wavelengths)),
        key=lambda index: abs(ordered_wavelengths[index] - nominal_wavelength),
    )
    return SphericalPrescription(
        title=title,
        epd_mm=epd_mm,
        wavelengths_nm=ordered_wavelengths,
        reference_wavelength_index=reference + 1,
        field_angles_deg=field_angles_deg,
        radii_mm=tuple(radii),
        thicknesses_mm=tuple(scalar(lens.s, index) for index in range(lens.s.shape[0])),
        nd=tuple(nd),
        vd=tuple(vd),
        stop_after_surface=stop_after_surface,
    )


def write_codev_seq(prescription: SphericalPrescription, path: str | Path) -> Path:
    """Write a human-readable spherical SEQ for external validation."""
    safe_title = re.sub(r"[^A-Za-z0-9_.-]+", "_", prescription.title)[:64] or "EADLD_SEED"
    lines = [
        "RDM;LEN",
        f"TITLE '{safe_title}'",
        f"EPD   {prescription.epd_mm:.12g}",
        "DIM   M",
        "WL    " + " ".join(f"{value:.12g}" for value in prescription.wavelengths_nm),
        f"REF   {prescription.reference_wavelength_index}",
        "WTW   " + " ".join("1" for _ in prescription.wavelengths_nm),
        "XAN   " + " ".join("0" for _ in prescription.field_angles_deg),
        "YAN   " + " ".join(f"{value:.12g}" for value in prescription.field_angles_deg),
        "WTF   " + " ".join("1" for _ in prescription.field_angles_deg),
    ]
    for index, (radius, thickness, nd, vd) in enumerate(
        zip(
            prescription.radii_mm,
            prescription.thicknesses_mm,
            prescription.nd,
            prescription.vd,
        )
    ):
        suffix = "" if nd is None or vd is None else f" {_encode_fictitious_glass(nd, vd)}"
        lines.append(f"S     {radius:.12g} {thickness:.12g}{suffix}")
        if index == prescription.stop_after_surface:
            lines.append("  STO")
    lines.extend(("SI    0.0 0.0", "GO"))
    output = Path(path)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
