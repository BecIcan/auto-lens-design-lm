"""桌面控制台与 EADLD 优化器之间的轻量运行层。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml

from eadld.utils.visualization import (
    LensLayout,
    SpotDiagrams,
    VisualizationCallback,
    WaveMTF,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "outputs" / "desktop" / "runs"

CASE_LABELS = {
    "singlet": "单片环带",
    "triplet": "三片 Cooke 环带",
    "four_element": "四片 C-mount 环带",
}

CASE_PRESETS = {
    "singlet": {
        "demo_case": "singlet",
        "target_efl": 100.0,
        "f_number": 8.0,
        "half_field": 1.0,
        "wavelengths": [486.1, 550.0, 656.3],
        "wavelength_weights": [1.0, 1.0, 1.0],
        "primary_wavelength": 1,
        "n_fields": 3,
        "n_r": 96,
        "n_theta": 64,
        "steps": 200,
        "visual_every": 1,
        "lm_parameter": 1.0,
        "distortion_percent": 1.0,
        "zone_count": 12,
        "seed": 0,
    },
    "triplet": {
        "demo_case": "triplet",
        "target_efl": 100.0,
        "f_number": 2.8,
        "half_field": 5.0,
        "wavelengths": [486.1, 550.0, 656.3],
        "wavelength_weights": [1.0, 1.0, 1.0],
        "primary_wavelength": 1,
        "n_fields": 5,
        "n_r": 164,
        "n_theta": 32,
        "steps": 200,
        "visual_every": 1,
        "lm_parameter": 1.0,
        "distortion_percent": 1.0,
        "zone_count": 41,
        "seed": 42,
    },
    "four_element": {
        "demo_case": "four_element",
        "target_efl": 28.0,
        "f_number": 2.0,
        "half_field": 15.88,
        "wavelengths": [486.1, 587.6, 656.3],
        "wavelength_weights": [1.0, 1.0, 1.0],
        "primary_wavelength": 1,
        "n_fields": 11,
        "n_r": 96,
        "n_theta": 32,
        "steps": 750,
        "visual_every": 1,
        "lm_parameter": 1e6,
        "distortion_percent": 2.0,
        "zone_count": 19,
        "seed": 0,
    },
}

CASE_MODELS = {
    "singlet": {
        "design": "configs/paper_demos/designs/singlet_recorded_seed.yml",
        "final_design": "configs/demo_cases/designs/singlet_final.yml",
        "psf_shape": [129, 129],
        "psf_abs_size": 0.08,
    },
    "triplet": {
        "design": "configs/paper_demos/designs/triplet_prerefine.yml",
        "final_design": "configs/paper_demos/designs/triplet_postrefine_100.yml",
        "psf_shape": [129, 129],
        "psf_abs_size": 0.04,
    },
    "four_element": {
        "design": "configs/paper_demos/designs/four_element_stage4_seed.yml",
        "final_design": "configs/demo_cases/designs/four_element_final.yml",
        "psf_shape": [65, 65],
        "psf_abs_size": 0.02,
    },
}

DEFAULTS = CASE_PRESETS["triplet"]
DEMO_DEFAULTS = CASE_PRESETS["four_element"]

LIMITS = {
    "target_efl": (20.0, 500.0, float),
    "f_number": (1.4, 22.0, float),
    "half_field": (0.0, 30.0, float),
    "n_fields": (1, 11, int),
    "n_r": (8, 512, int),
    "n_theta": (8, 1024, int),
    "steps": (1, 5000, int),
    "visual_every": (1, 500, int),
    "lm_parameter": (1e-8, 1e6, float),
    "distortion_percent": (0.0, 25.0, float),
    "seed": (0, 2**31 - 1, int),
}

METRICS = {
    "loss": "loss/scalar_loss_post_step",
    "damping": "optimizer/lm_parameter",
    "efl": "lens/efl",
    "ttl": "lens/ttl",
    "valid": "ray_tracing/ray_valid",
}


class DesktopVisualizationCallback(VisualizationCallback):
    """每个 LM 步刷新轻量光路图。"""

    def __init__(
        self,
        every_n_steps=1,
        wavelength_0=486.1,
        wavelength_1=550.0,
        wavelength_2=656.3,
        n_fields=3,
    ):
        wavelengths = (wavelength_0, wavelength_1, wavelength_2)
        super().__init__(
            visualizations=[
                LensLayout(
                    wavelengths=wavelengths,
                    n_fields=min(3, n_fields),
                    n_rays=5,
                    label_materials=True,
                ),
            ],
            save_formats=("png",),
            every_n_steps=every_n_steps,
            rc_params={
                "figure.figsize": (8, 3.6),
                "figure.constrained_layout.use": True,
                "figure.dpi": 140,
                "font.size": 8,
            },
        )


class DesktopAnalysisCallback(VisualizationCallback):
    """低频刷新高成本点列图和 RayWave MTF，并始终保存最终状态。"""

    def __init__(
        self,
        every_n_steps=50,
        n_fields=3,
    ):
        field_indices = (
            list(range(n_fields))
            if n_fields <= 3
            else [0, n_fields // 2, n_fields - 1]
        )
        super().__init__(
            visualizations=[
                SpotDiagrams(field_indices=field_indices),
                WaveMTF(field_indices=field_indices),
            ],
            save_formats=("png",),
            every_n_steps=every_n_steps,
            rc_params={
                "figure.figsize": (8, 3.6),
                "figure.constrained_layout.use": True,
                "figure.dpi": 140,
                "font.size": 8,
            },
        )


def validate_parameters(payload: dict) -> dict:
    """在启动高成本求解前校验所有界面输入。"""
    demo_case = payload.get("demo_case", DEFAULTS["demo_case"])
    if demo_case not in CASE_PRESETS:
        raise ValueError("demo_case 必须是 singlet、triplet 或 four_element")
    params = {}
    for name, (lower, upper, converter) in LIMITS.items():
        try:
            value = converter(payload.get(name, DEFAULTS[name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 不是有效数值") from exc
        if not lower <= value <= upper:
            raise ValueError(f"{name} 必须位于 {lower}–{upper}")
        params[name] = value

    wavelengths = payload.get("wavelengths", DEFAULTS["wavelengths"])
    if not isinstance(wavelengths, list) or len(wavelengths) != 3:
        raise ValueError("必须输入三个波长")
    try:
        params["wavelengths"] = [float(value) for value in wavelengths]
    except (TypeError, ValueError) as exc:
        raise ValueError("波长必须是数值") from exc
    if any(not 380.0 <= value <= 1700.0 for value in params["wavelengths"]):
        raise ValueError("波长必须位于 380–1700 nm")

    weights = payload.get("wavelength_weights", DEFAULTS["wavelength_weights"])
    if not isinstance(weights, list) or len(weights) != 3:
        raise ValueError("必须输入三个波长权重")
    try:
        params["wavelength_weights"] = [float(value) for value in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError("波长权重必须是数值") from exc
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1e6
        for value in params["wavelength_weights"]
    ):
        raise ValueError("波长权重必须位于 0–1000000")
    if sum(params["wavelength_weights"]) <= 0:
        raise ValueError("至少一个波长权重必须大于 0")

    try:
        primary = int(payload.get("primary_wavelength", DEFAULTS["primary_wavelength"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("主波长序号无效") from exc
    if primary not in range(3):
        raise ValueError("主波长必须是第 1–3 行")
    params["primary_wavelength"] = primary
    params["demo_case"] = demo_case
    params["visual_every"] = min(params["visual_every"], params["steps"])
    return params


def build_overlay(params: dict, save_dir: Path) -> dict:
    """把用户规格映射到单片、Cooke 三片或 C-mount 四片配置链。"""
    constraint_weight = 10.0
    demo_case = params["demo_case"]
    if demo_case == "singlet":
        freeze = {
            "s": [True, False],
            "c": True,
            "g": True,
            "a": True,
            "m": True,
            "d": False,
        }
        residuals = [
            {
                "class_path": "eadld.optimization.residuals.TransverseRayAberrationResiduals",
                "init_args": {"weight": 1.0},
            },
            {
                "class_path": "eadld.optimization.residuals.HDOEPhaseResiduals",
                "init_args": {
                    "weight": 1000.0,
                    "diffraction_order": 30,
                    "design_wavelength": params["wavelengths"][
                        params["primary_wavelength"]
                    ],
                    "zonal_surface_index": 0,
                    "field_indices": [0],
                },
            },
        ]
    elif demo_case == "triplet":
        freeze = {
            "s": [False, False, False, False, False, False, True],
            "c": [False, False, True, False, False, False],
            "g": True,
            "m": True,
            "d": True,
            "a": [[True, False, False, False, False]],
            "z": [[[False, False, True, True]]],
        }
        residuals = [
            {
                "class_path": "eadld.optimization.residuals.TransverseRayAberrationResiduals",
                "init_args": {"weight": 1.0},
            },
            {
                "class_path": "eadld.optimization.residuals.GlassVariableResiduals",
                "init_args": {"weight": 0.001},
            },
            {
                "class_path": "eadld.optimization.residuals.FocalLengthResiduals",
                "init_args": {"weight": 1e6},
            },
            {
                "class_path": "eadld.optimization.residuals.RayPathResiduals",
                "init_args": {
                    "weight": constraint_weight,
                    "min_cutoff": 0.5,
                    "max_cutoff": 40.0,
                    "other_max_cutoffs": [[-1, float("inf")]],
                    "min_cutoff_refractive": 1.5,
                    "max_cutoff_refractive": 15.0,
                    "min_cutoff_refractive_relative": 0.0,
                    "max_cutoff_refractive_relative": 100.0,
                },
            },
            {
                "class_path": "eadld.optimization.residuals.DistortionResiduals",
                "init_args": {
                    "weight": constraint_weight,
                    "threshold": params["distortion_percent"] / 100.0,
                },
            },
            {
                "class_path": "eadld.optimization.residuals.SurfaceNormalResiduals",
                "init_args": {"weight": constraint_weight, "max_angle": 30.0},
            },
            {
                "class_path": "eadld.optimization.residuals.RayAngleResiduals",
                "init_args": {"weight": constraint_weight, "max_angle": 60.0},
            },
        ]
    else:
        freeze = {
            "s": [False] * 8 + [True],
            "c": [False] * 6 + [True, False],
            "g": True,
            "m": True,
            "d": True,
            "a": [[True, False, False, False, False]],
            "z": [[[False, False, True, True]]],
        }
        residuals = [
            {
                "class_path": "eadld.optimization.residuals.TransverseRayAberrationResiduals",
                "init_args": {"weight": 1.0},
            },
            {
                "class_path": "eadld.optimization.residuals.RayPathResiduals",
                "init_args": {
                    "weight": 20.0,
                    "min_cutoff": 0.25,
                    "max_cutoff": float("inf"),
                    "min_cutoff_refractive": 1.0,
                    "max_cutoff_refractive": 5.0,
                    "min_cutoff_refractive_relative": 0.3333,
                    "max_cutoff_refractive_relative": 3.0,
                    "other_min_cutoffs": [[-1, 20.0]],
                    "other_max_cutoffs": [],
                },
            },
            {
                "class_path": "eadld.optimization.residuals.RayAngleResiduals",
                "init_args": {"weight": 10.0, "max_angle": 60.0},
            },
            {
                "class_path": "eadld.optimization.residuals.DistortionResiduals",
                "init_args": {
                    "weight": 10.0,
                    "threshold": params["distortion_percent"] / 100.0,
                },
            },
            {
                "class_path": "eadld.optimization.residuals.GlassVariableResiduals",
                "init_args": {"weight": None},
            },
            {
                "class_path": "eadld.optimization.residuals.GlassMeshDistanceResiduals",
                "init_args": {
                    "weight": 0.001,
                    "threshold": 0.0,
                    "max_simplex_size": 1.25,
                },
            },
            {
                "class_path": "eadld.optimization.residuals.ImageHeightResiduals",
                "init_args": {
                    "weight": 10.0,
                    "target": params["target_efl"]
                    * math.tan(math.radians(params["half_field"])),
                },
            },
            {
                "class_path": "eadld.optimization.residuals.TotalTrackLengthResiduals",
                "init_args": {"weight": 10.0, "target": 39.3},
            },
        ]
    model = CASE_MODELS[demo_case]
    optimizer = (
        {
            "lm_parameter": params["lm_parameter"],
            "damped_term_min": 1e-6,
            "tolerance": 1.25,
            "lam_increase_factor": 2.0,
            "lam_decrease_factor": 3.0,
            "lam_eps": 1e-8,
            "beta": 0.9,
        }
        if demo_case == "four_element"
        else {
            "lm_parameter": params["lm_parameter"],
            "damped_term_min": 1e-6,
            "tolerance": 1.0,
            "lam_increase_factor": 4.0,
            "lam_decrease_factor": 2.0,
            "lam_eps": 1e-10,
            "beta": 0.95,
        }
    )
    analysis_every = min(50, params["steps"])
    callbacks = [
        {
            "class_path": "eadld.utils.callbacks.ConfigFileCallback",
            "init_args": {"every_n_steps": params["visual_every"]},
        },
        "eadld.main.CustomProgressBar",
    ]
    if demo_case == "four_element":
        callbacks.extend(
            [
                {
                    "class_path": "eadld.utils.callbacks.IncreaseGlassVariableResidualsWeightCallback",
                    "init_args": {
                        "initial_step": 0.0,
                        "final_step": 0.5,
                        "n_increments": 25,
                        "initial_weight": 0.00001525878,
                        "final_weight": 512,
                    },
                },
                {
                    "class_path": "eadld.utils.callbacks.ToggleGlassOptimizationCallback",
                    "init_args": {
                        "initial_step": 0.0,
                        "final_step": 0.5,
                        "n_cycles": 25,
                    },
                },
                {
                    "class_path": "eadld.utils.callbacks.BindMaterialsCallback",
                    "init_args": {"step": 0.5},
                },
            ]
        )
    callbacks.extend(
        [
            {
                "class_path": "eadld.desktop.backend.DesktopAnalysisCallback",
                "init_args": {
                    "every_n_steps": analysis_every,
                    "n_fields": params["n_fields"],
                },
            },
            {
                "class_path": "eadld.desktop.backend.DesktopVisualizationCallback",
                "init_args": {
                    "every_n_steps": params["visual_every"],
                    "wavelength_0": params["wavelengths"][0],
                    "wavelength_1": params["wavelengths"][1],
                    "wavelength_2": params["wavelengths"][2],
                    "n_fields": params["n_fields"],
                },
            },
        ]
    )
    return {
        "seed_everything": params["seed"],
        "model": {
            # 单片与四片历史早于面积求积；兼容重放后仍用独立面积审计验收。
            "optimization_pupil_quadrature": demo_case == "triplet",
            "ray_initialization": {
                "init_args": {
                    "aperture": params["target_efl"] / params["f_number"],
                    "hfov": params["half_field"],
                    "n_fields": params["n_fields"],
                    "wavelengths": params["wavelengths"],
                    "wavelength_weights": params["wavelength_weights"],
                    "pupil_sampling_mode": "skew_uniform_zonal",
                    "pupil_sampling_kwargs": {
                        "n_r": params["n_r"],
                        "n_theta": params["n_theta"],
                    },
                }
            },
            "lens_parameterization": {
                "init_args": {
                    "target_efl": params["target_efl"],
                    "nominal_wavelength": params["wavelengths"][
                        params["primary_wavelength"]
                    ],
                    "freeze": freeze,
                }
            },
            "residuals": residuals,
            "optics_simulator": {
                "init_args": {
                    "shape": model["psf_shape"],
                    "psf_abs_size": model["psf_abs_size"],
                    "psf_grid_shape": [1, params["n_fields"]],
                }
            },
            "lens_optimizer": {
                "class_path": "eadld.optimization.optimizers.LMOptimizer",
                "init_args": optimizer,
            },
        },
        "data": {"init_args": {"n_samples": params["steps"]}},
        "trainer": {
            "max_steps": params["steps"],
            "log_every_n_steps": 1,
            "enable_checkpointing": False,
            "enable_model_summary": False,
            "logger": {
                "class_path": "eadld.main.CustomLogger",
                "init_args": {"save_dir": save_dir.as_posix()},
            },
            "callbacks": callbacks,
        },
    }


def build_command(config_path: Path, design_path: Path | None = None) -> list[str]:
    if design_path is None:
        design_path = ROOT / CASE_MODELS[DEFAULTS["demo_case"]]["design"]
    return [
        sys.executable,
        "-m",
        "eadld.main",
        "fit",
        "-c",
        "configs/multi_element/defaults.yml",
        "-c",
        design_path.relative_to(ROOT).as_posix(),
        "-c",
        config_path.relative_to(ROOT).as_posix(),
    ]


@dataclass
class OptimizationRun:
    run_id: str
    params: dict
    root: Path
    state: str = "starting"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    return_code: int | None = None
    process: subprocess.Popen | None = None
    log_lines: deque = field(default_factory=lambda: deque(maxlen=160))

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        overlay_path = self.root / "desktop.yml"
        overlay_path.write_text(
            yaml.safe_dump(
                build_overlay(self.params, self.root),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        design_path = ROOT / CASE_MODELS[self.params["demo_case"]]["design"]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            build_command(overlay_path, design_path),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
            env=os.environ | {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        self.state = "running"
        threading.Thread(target=self._collect_output, daemon=True).start()

    def _collect_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        with (self.root / "console.log").open("w", encoding="utf-8") as log_file:
            for line in self.process.stdout:
                self.log_lines.append(line.rstrip())
                log_file.write(line)
                log_file.flush()
        self.return_code = self.process.wait()
        self.finished_at = time.time()
        if self.state != "stopped":
            self.state = "completed" if self.return_code == 0 else "failed"

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.state = "stopped"
            self.process.terminate()

    def read_metrics(self) -> dict[str, list[dict]]:
        event_files = list(self.root.rglob("events.out.tfevents.*"))
        if not event_files:
            return {}
        latest = max(event_files, key=lambda path: path.stat().st_mtime_ns)
        try:
            accumulator = EventAccumulator(
                str(latest), size_guidance={"scalars": 5000}
            )
            accumulator.Reload()
            available = set(accumulator.Tags().get("scalars", []))
            return {
                key: [
                    {"step": event.step, "value": event.value}
                    for event in accumulator.Scalars(tag)
                ]
                for key, tag in METRICS.items()
                if tag in available
            }
        except (OSError, RuntimeError, KeyError):
            return {}

    def latest_artifact(self, kind: str) -> Path | None:
        files = list(self.root.rglob(f"{kind}/*.png"))
        return max(files, key=lambda path: path.stat().st_mtime_ns) if files else None

    def snapshot(self) -> dict:
        metrics = self.read_metrics()
        last_zero_based_step = max(
            (point["step"] for series in metrics.values() for point in series),
            default=-1,
        )
        step = last_zero_based_step + 1
        end_time = self.finished_at or time.time()
        return {
            "run_id": self.run_id,
            "state": self.state,
            "step": step,
            "max_steps": self.params["steps"],
            "elapsed_seconds": round(end_time - self.started_at, 1),
            "return_code": self.return_code,
            "metrics": metrics,
            "artifacts": {
                kind: self.latest_artifact(kind)
                for kind in ("layout", "spot_diagrams", "mtf")
            },
            "logs": list(self.log_lines)[-80:],
        }


class RunManager:
    def __init__(self) -> None:
        self.current: OptimizationRun | None = None
        self.lock = threading.Lock()

    def start(self, payload: dict) -> dict:
        params = validate_parameters(payload)
        with self.lock:
            if self.current is not None and self.current.state in {"starting", "running"}:
                raise RuntimeError("已有优化任务正在运行")
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self.current = OptimizationRun(run_id, params, RUNS_ROOT / run_id)
            self.current.start()
        return self.snapshot()

    def stop(self) -> dict:
        with self.lock:
            if self.current is not None:
                self.current.stop()
        return self.snapshot()

    def snapshot(self) -> dict:
        if self.current is None:
            return {
                "state": "idle",
                "step": 0,
                "max_steps": 0,
                "elapsed_seconds": 0,
                "return_code": None,
                "metrics": {},
                "artifacts": {},
                "logs": [],
            }
        return self.current.snapshot()
