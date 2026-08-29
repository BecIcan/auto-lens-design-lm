"""Opaque seed-generator interface with public EADLD real-ray auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from eadld.modeling.optics import Lens
from eadld.modeling.ray_initialization import RayInitialization
from eadld.utils.visualization import generate_layout_plot, generate_spot_plot
from eadld.initialization.codev_seq import lens_to_spherical_prescription, write_codev_seq


STOP_SURFACE_CLEARANCE_MM = 0.2
STOP_REPAIR_MARGIN_MM = 0.02
LAYOUT_DIAMETER_SCALER = 17 / 16


def _is_surface_mounted_stop(sequence: str) -> bool:
    """光阑与前/后镜面共面，是合法的镜面安装方式。"""
    return "-sR" in sequence or "Rs-" in sequence


@dataclass(frozen=True)
class DesignSpec:
    """Public design contract passed to a private seed backend."""

    effective_focal_length_mm: float
    f_number: float
    max_field_angle_deg: float
    wavelengths_nm: tuple[float, ...]
    elements: int
    candidate_count: int = 3
    min_image_clearance_mm: float | None = None
    max_package_length_mm: float | None = None
    max_distortion_fraction: float | None = None
    target_chief_ray_angle_deg: float | None = None
    min_relative_illumination_fraction: float | None = None
    max_efl_error_fraction: float = 0.05

    @property
    def entrance_pupil_diameter_mm(self) -> float:
        return self.effective_focal_length_mm / self.f_number

    def validate(self) -> None:
        if self.effective_focal_length_mm <= 0 or self.f_number <= 0:
            raise ValueError("焦距和 F/# 必须为正")
        if not 0 <= self.max_field_angle_deg < 60:
            raise ValueError("半视场必须在 [0, 60) 度")
        if not 3 <= self.elements <= 10:
            raise ValueError("当前接口支持 3..10 片")
        if len(self.wavelengths_nm) < 1 or min(self.wavelengths_nm) <= 0:
            raise ValueError("至少需要一个正波长")
        if self.candidate_count <= 0:
            raise ValueError("候选数必须为正")
        if not 0 < self.max_efl_error_fraction < 1:
            raise ValueError("焦距相对误差门槛必须在 (0, 1)")


@dataclass(frozen=True)
class LensSeed:
    """Backend-neutral lens seed accepted by the native EADLD tracer."""

    candidate_id: str
    lens_sequence: str
    spacings_mm: tuple[float, ...]
    curvatures_per_mm: tuple[float, ...]
    refractive_indices_d: tuple[float, ...]
    abbe_numbers: tuple[float, ...]
    partial_dispersion_deviation: tuple[float, ...] = ()
    asphere_coefficients: tuple[tuple[float, ...], ...] = ()
    nominal_wavelength_nm: float = 550.0

    def to_lens(self) -> Lens:
        """Convert without a focal solve, image solve, material snap, or optimizer."""
        dtype = torch.float64
        dpgf = self.partial_dispersion_deviation or (0.0,) * len(self.refractive_indices_d)
        a_width = max((len(row) for row in self.asphere_coefficients), default=1)
        a = torch.zeros((len(self.asphere_coefficients), 1, a_width), dtype=dtype)
        for index, row in enumerate(self.asphere_coefficients):
            a[index, 0, : len(row)] = torch.tensor(row, dtype=dtype)
        empty = torch.empty((0, 1, 1), dtype=dtype)
        return Lens(
            sequence=self.lens_sequence,
            s=torch.tensor(self.spacings_mm, dtype=dtype),
            c=torch.tensor(self.curvatures_per_mm, dtype=dtype),
            nd=torch.tensor(self.refractive_indices_d, dtype=dtype),
            vd=torch.tensor(self.abbe_numbers, dtype=dtype),
            dpgf=torch.tensor(dpgf, dtype=dtype),
            a=a,
            d=empty,
            m=empty.clone(),
            z=None,
            w0=self.nominal_wavelength_nm,
        )


@runtime_checkable
class InitialStructureBackend(Protocol):
    """Private packages implement this small interface."""

    def generate(self, spec: DesignSpec) -> list[LensSeed]: ...

    def public_metadata(self) -> dict[str, str | int | float | bool | None]: ...


def load_backend(factory_path: str, config_path: str | Path | None = None) -> InitialStructureBackend:
    """Load ``package.module:factory`` without exposing backend internals here."""
    module_name, separator, factory_name = factory_path.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("backend 必须写成 package.module:factory")
    factory = getattr(importlib.import_module(module_name), factory_name)
    backend = factory(None if config_path is None else Path(config_path))
    if not isinstance(backend, InitialStructureBackend):
        raise TypeError("私有 backend 未实现 InitialStructureBackend 接口")
    return backend


def _initializer(spec: DesignSpec, n_fields: int = 5) -> RayInitialization:
    return RayInitialization(
        aperture=spec.entrance_pupil_diameter_mm,
        aperture_type="epd",
        hfov=spec.max_field_angle_deg,
        n_fields=n_fields,
        wavelengths=list(spec.wavelengths_nm),
        wavelength_weights=[1.0] * len(spec.wavelengths_nm),
        pupil_sampling_mode="skew_uniform",
        pupil_sampling_kwargs={"n_r": 10, "n_theta": 20},
        ray_aiming_steps=0,
    )


@torch.no_grad()
def _trace(lens: Lens, initializer: RayInitialization) -> tuple[dict, torch.Tensor, torch.Tensor]:
    r0, d0 = initializer(lens)
    r, _, status, _ = list(
        lens.trace_rays(r0, d0, initializer.wavelengths, yield_on="end")
    )[-1]
    xy = r[:2, ..., 0]
    valid = status[..., 0] == 0
    rms = []
    for field in range(xy.shape[1]):
        points = [
            xy[:, field, valid[field, :, wave], wave]
            for wave in range(xy.shape[3])
            if valid[field, :, wave].any()
        ]
        if not points:
            rms.append(float("inf"))
            continue
        merged = torch.cat(points, dim=1)
        centered = merged - merged.mean(dim=1, keepdim=True)
        rms.append(float(centered.square().sum(dim=0).mean().sqrt()) * 1e3)
    return (
        {
            "efl_mm": float(lens.efl),
            "bfl_mm": float(lens.bfl),
            "ttl_mm": float(lens.s.sum()),
            "valid_ray_fraction": float(valid.float().mean()),
            "rms_radius_um_by_field": rms,
            "mean_rms_radius_um": float(np.mean(rms)),
            "worst_rms_radius_um": float(np.max(rms)),
        },
        xy,
        valid,
    )


@torch.no_grad()
def _mechanical_gate(lens: Lens, initializer: RayInitialization) -> dict:
    r0, d0 = initializer(lens, n_fields=3, pupil_sampling_mode="skew_uniform", n_r=8, n_theta=12)
    diameters = []
    for r, _, status, _ in lens.trace_rays(
        r0, d0, initializer.wavelengths, yield_on="position"
    ):
        radius = r[:2].norm(dim=0).where(status == 0, 0.0)
        diameters.append(2 * float(radius.max()))
    geometry = list(lens.return_geometry())
    refractive_event_indices = [
        index
        for index, event in enumerate(lens.sequence.events)
        if event["type"] == "r"
    ]
    optical = [
        (geometry_index, event_index, item, diameters[geometry_index])
        for event_index, (geometry_index, item) in zip(
            refractive_event_indices,
            (
                (geometry_index, item)
                for geometry_index, item in enumerate(geometry)
                if item[0] == "r"
            ),
            strict=True,
        )
    ]
    clearances = []
    for index in range(1, len(optical)):
        _, left_event, (_, left_z, left_sag, _), left_diameter = optical[index - 1]
        _, right_event, (_, right_z, right_sag, closes_glass), right_diameter = optical[index]
        semi = 0.525 * min(left_diameter, right_diameter)
        y = torch.linspace(0.0, max(semi, 1e-6), 96, dtype=lens.c.dtype)
        left = left_z + (left_sag(y) if left_sag else 0.0)
        right = right_z + (right_sag(y) if right_sag else 0.0)
        minimum = float((right - left).min())
        required = 0.3 if closes_glass else 0.05
        spacing_indices = tuple(
            dict.fromkeys(
                event["s"]
                for event in lens.sequence.events[left_event + 1 : right_event]
                if event["type"] == "p" and isinstance(event.get("s"), int)
            )
        )
        clearances.append(
            {
                "minimum_mm": minimum,
                "required_mm": required,
                "passed": minimum >= required,
                "spacing_indices": spacing_indices,
            }
        )

    stop_geometry_index = next(
        index for index, item in enumerate(geometry) if item[0] == "s"
    )
    left = next(
        (item for item in reversed(optical) if item[0] < stop_geometry_index),
        None,
    )
    right = next(
        (item for item in optical if item[0] > stop_geometry_index),
        None,
    )

    def surface_bounds(item, diameter: float) -> tuple[float, float]:
        _, z, sag_fn, _ = item
        y = torch.linspace(
            0.0,
            max(0.5 * diameter * LAYOUT_DIAMETER_SCALER, 1e-6),
            128,
            dtype=lens.c.dtype,
        )
        surface_z = z + (sag_fn(y) if sag_fn else 0.0)
        return float(surface_z.min()), float(surface_z.max())

    stop_z = float(geometry[stop_geometry_index][1])
    left_edge = float("-inf")
    right_edge = float("inf")
    if left is not None:
        _, _, left_item, left_diameter = left
        _, left_edge = surface_bounds(left_item, left_diameter)
    if right is not None:
        _, _, right_item, right_diameter = right
        right_edge, _ = surface_bounds(right_item, right_diameter)
    stop_clearance = {
        "front_mm": stop_z - left_edge,
        "rear_mm": right_edge - stop_z,
        "required_mm": STOP_SURFACE_CLEARANCE_MM,
        "surface_mounted": _is_surface_mounted_stop(lens.sequence.sequence),
    }
    stop_clearance["passed"] = (
        stop_clearance["surface_mounted"]
        or (
            stop_clearance["front_mm"] >= STOP_SURFACE_CLEARANCE_MM
            and stop_clearance["rear_mm"] >= STOP_SURFACE_CLEARANCE_MM
        )
    )
    return {
        "passed": bool(clearances)
        and all(row["passed"] for row in clearances)
        and stop_clearance["passed"],
        "clearances": clearances,
        "stop_clearance": stop_clearance,
    }


def _relocate_stop_in_air_gap(
    seed: LensSeed,
    initializer: RayInitialization,
) -> LensSeed | None:
    """在不改变相邻镜片位置的前提下，把光阑移到空气隙内。"""
    sequence = seed.lens_sequence
    if _is_surface_mounted_stop(sequence):
        return seed
    stop_offset = sequence.index("s")
    if (
        stop_offset == 0
        or stop_offset == len(sequence) - 1
        or sequence[stop_offset - 1 : stop_offset + 2] != "-s-"
    ):
        return None

    spacing_before_stop = sum(char in "R-" for char in sequence[:stop_offset]) - 1
    spacing_after_stop = spacing_before_stop + 1
    repaired = seed
    for _ in range(6):
        lens = repaired.to_lens()
        stop = _mechanical_gate(lens, initializer)["stop_clearance"]
        if stop["passed"]:
            return repaired

        geometry = list(lens.return_geometry())
        stop_z = float(next(item[1] for item in geometry if item[0] == "s"))
        repair_clearance = STOP_SURFACE_CLEARANCE_MM + STOP_REPAIR_MARGIN_MM
        lower_z = stop_z - stop["front_mm"] + repair_clearance
        upper_z = stop_z + stop["rear_mm"] - repair_clearance
        if lower_z > upper_z:
            return None
        target_z = min(max(stop_z, lower_z), upper_z)

        spacings = list(repaired.spacings_mm)
        total_gap = spacings[spacing_before_stop] + spacings[spacing_after_stop]
        shifted_before = spacings[spacing_before_stop] + target_z - stop_z
        shifted_after = total_gap - shifted_before
        if min(shifted_before, shifted_after) < 0.05:
            return None
        spacings[spacing_before_stop] = shifted_before
        spacings[spacing_after_stop] = shifted_after
        repaired = replace(repaired, spacings_mm=tuple(spacings))
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_generation_audit(
    spec: DesignSpec,
    backend: InitialStructureBackend,
    output_dir: str | Path,
) -> dict:
    """Generate once, audit with native rays, and rank without changing any seed."""
    spec.validate()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录: {output}")
    output.mkdir(parents=True)
    seeds = backend.generate(spec)
    if not 1 <= len(seeds) <= spec.candidate_count:
        raise ValueError("backend 返回的候选数必须在 1..candidate_count")
    lenses = []
    for seed in seeds:
        lens = seed.to_lens()
        actual_elements = lens.sequence.n_refractive
        if actual_elements != spec.elements:
            raise ValueError(
                f"候选 {seed.candidate_id} 的实际片数为 {actual_elements}，"
                f"与请求的 {spec.elements} 片不一致"
            )
        lenses.append(lens)
    initializer = _initializer(spec)
    records = []
    runtime = []
    for seed, lens in zip(seeds, lenses):
        try:
            metrics, xy, valid = _trace(lens, initializer)
            mechanics = _mechanical_gate(lens, initializer)
            first_order = {
                "efl_relative_error": abs(
                    metrics["efl_mm"] - spec.effective_focal_length_mm
                )
                / spec.effective_focal_length_mm,
                "efl_passed": abs(metrics["efl_mm"] - spec.effective_focal_length_mm)
                / spec.effective_focal_length_mm
                <= spec.max_efl_error_fraction,
                "image_clearance_passed": spec.min_image_clearance_mm is None
                or metrics["bfl_mm"] >= spec.min_image_clearance_mm,
                "package_length_passed": spec.max_package_length_mm is None
                or metrics["ttl_mm"] <= spec.max_package_length_mm,
            }
            first_order["passed"] = all(
                first_order[key]
                for key in (
                    "efl_passed",
                    "image_clearance_passed",
                    "package_length_passed",
                )
            )
            passed = (
                mechanics["passed"]
                and first_order["passed"]
                and metrics["valid_ray_fraction"] >= 0.5
                and math.isfinite(metrics["mean_rms_radius_um"])
            )
            records.append(
                {
                    "candidate_id": seed.candidate_id,
                    "elements": lens.sequence.n_refractive,
                    "passed": passed,
                    "metrics": metrics,
                    "mechanics": mechanics,
                    "first_order": first_order,
                }
            )
            runtime.append((lens, xy, valid) if passed else None)
        except (AssertionError, RuntimeError, ValueError) as error:
            records.append(
                {"candidate_id": seed.candidate_id, "passed": False, "error": f"{type(error).__name__}: {error}"}
            )
            runtime.append(None)
    viable = sorted(
        (row["metrics"]["mean_rms_radius_um"], index)
        for index, row in enumerate(records)
        if row["passed"]
    )
    if not viable:
        raise RuntimeError("没有候选通过机械与真实光线门槛")
    selected_index = viable[0][1]
    selected_artifacts = None
    for rank, (_, candidate_index) in enumerate(viable, start=1):
        lens, xy, valid = runtime[candidate_index]
        candidate_output = output if rank == 1 else output / f"candidate-{rank:02d}"
        candidate_output.mkdir(exist_ok=rank == 1)
        layout = candidate_output / "layout.png"
        spots = candidate_output / "spots.png"
        generate_layout_plot(
            lens,
            initializer,
            list(spec.wavelengths_nm),
            n_rays=5,
            n_fields=3,
        ).savefig(layout, dpi=180, bbox_inches="tight")
        generate_spot_plot(
            xy,
            valid,
            spec.wavelengths_nm,
            spec.max_field_angle_deg,
        ).savefig(spots, dpi=180, bbox_inches="tight")
        candidate_artifacts = {
            "layout": {"path": str(layout.resolve()), "sha256": _sha256(layout)},
            "spots": {"path": str(spots.resolve()), "sha256": _sha256(spots)},
        }
        if all(char in "R-s" for char in lens.sequence.sequence):
            fields = (
                (0.0,)
                if spec.max_field_angle_deg == 0
                else (
                    0.0,
                    spec.max_field_angle_deg / math.sqrt(2.0),
                    spec.max_field_angle_deg,
                )
            )
            prescription = lens_to_spherical_prescription(
                lens,
                title=f"EADLD_{records[candidate_index]['candidate_id']}",
                epd_mm=spec.entrance_pupil_diameter_mm,
                wavelengths_nm=spec.wavelengths_nm,
                field_angles_deg=fields,
            )
            seq = write_codev_seq(prescription, candidate_output / "initial_structure.seq")
            candidate_artifacts["seq"] = {
                "path": str(seq.resolve()),
                "sha256": _sha256(seq),
            }
        records[candidate_index]["rank"] = rank
        records[candidate_index]["artifacts"] = candidate_artifacts
        if rank == 1:
            selected_artifacts = candidate_artifacts
    manifest = {
        "status": "private_generation_public_physics_audit",
        "spec": {**asdict(spec), "wavelengths_nm": list(spec.wavelengths_nm)},
        "backend": backend.public_metadata(),
        "runtime_contract": {
            "optimizer_invocations": 0,
            "paraxial_solves": 0,
            "catalog_glass_snapping": False,
            "candidate_ranking_changes_prescription": False,
            "aperture_audit": "fixed target EPD",
        },
        "selected_candidate_id": records[selected_index]["candidate_id"],
        "candidates": records,
        "artifacts": selected_artifacts,
        "evidence_boundary": "These are EADLD native real-ray seed metrics, not a finished-lens or external-tool equivalence claim.",
    }
    manifest = _json_safe(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest
