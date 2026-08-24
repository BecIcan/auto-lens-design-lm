import os

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import callbacks

from eadld.imaging_system import ImagingSystemModule
from eadld.modeling import optics, ray_analysis as ra, misc_surfaces as ms
from eadld.utils.utils import retrieve_log_dir, apply_recursively


class ToggleIRMOptimizerCallback(callbacks.Callback):
    """Callback that disables or re-enables the optimizer of the IRM module at a specific training step."""

    def __init__(
        self, disable_step: int | float = None, enable_step: int | float | None = None
    ):
        """Constructor.

        Args:
            disable_step: If an integer, disable the optimizer after this training step;
                if a float, disable the optimizer after this fraction of the total number of training steps.
            enable_step: If an integer, re-enable the optimizer after this training step;
                if a float, re-enable the optimizer after this fraction of the total number of training steps.
        """
        super().__init__()
        self.disable_step = disable_step
        self.enable_step = enable_step
        self.initialized = False

    def initialize(self, trainer: pl.Trainer):
        """Initialize the callback.

        Args:
            trainer: The PyTorch Lightning trainer.
        """
        self.disable_step = (
            int(self.disable_step * trainer.max_steps)
            if isinstance(self.disable_step, float)
            else self.disable_step
        )
        self.enable_step = (
            int(self.enable_step * trainer.max_steps)
            if isinstance(self.enable_step, float)
            else self.enable_step
        )
        self.initialized = True

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: ImagingSystemModule, *_
    ):
        if not self.initialized:
            self.initialize(trainer)
        if self.disable_step is not None and pl_module.global_step == self.disable_step:
            pl_module.irm_optimization_disabled = True
        if self.enable_step is not None and pl_module.global_step == self.enable_step:
            pl_module.irm_optimization_disabled = False


class ToggleGlassOptimizationCallback(callbacks.Callback):
    """Callback that disables and re-enables glass variable optimization cyclically."""

    def __init__(
        self, initial_step: int | float, final_step: int | float, n_cycles: int = 1
    ):
        """Constructor.

        Each cycle is split into two approximately equal parts: the first part enables the optimization,
        the second part disables it.

        Args:
            initial_step: If an integer, start the process after this training step;
                if a float, start the process after this fraction of the total number of training steps.
            final_step: If an integer, stop the process after this training step;
                if a float, stop the process after this fraction of the total number of training steps.
            n_cycles: The number of cycles to perform between start and stop steps.
        """
        super().__init__()
        self.initial_step = initial_step
        self.final_step = final_step
        self.enable_steps = None
        self.disable_steps = None
        self.glass_indices = None
        self.n_cycles = n_cycles
        self.initialized = False

    def initialize(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        """Initialize the callback.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The ImagingSystemModule instance.
        """
        self.initial_step = (
            int(self.initial_step * trainer.max_steps)
            if isinstance(self.initial_step, float)
            else self.initial_step
        )
        self.final_step = (
            int(self.final_step * trainer.max_steps)
            if isinstance(self.final_step, float)
            else self.final_step
        )
        steps = np.linspace(
            self.initial_step, self.final_step, self.n_cycles * 2 + 1
        ).astype(int)

        if pl_module.get_optimizers()[0] is None:  # No lens optimizer
            self.enable_steps = ()
            self.disable_steps = ()
            return

        self.enable_steps = steps[:-1:2]
        self.disable_steps = steps[1::2]
        g_optimization_mask = ~pl_module.parameterization.freeze_dict["g"]
        self.glass_indices = torch.where(g_optimization_mask.flatten(1).any(dim=-1))[
            0
        ].tolist()
        self.initialized = True

    @torch.no_grad()
    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: ImagingSystemModule, *_
    ):
        if not self.initialized:
            self.initialize(trainer, pl_module)
        if pl_module.global_step in self.enable_steps:
            pl_module.parameterization.freeze_glass_variables(
                self.glass_indices, unfreeze=True
            )
            self.reset_lens_optimizer(pl_module)
        if pl_module.global_step in self.disable_steps:
            pl_module.parameterization.freeze_glass_variables(self.glass_indices)
            self.reset_lens_optimizer(pl_module)

    @torch.no_grad()
    def reset_lens_optimizer(self, imaging_system: ImagingSystemModule):
        """Reset the lens optimizer to the default values.

        Args:
            imaging_system: The ImagingSystemModule instance.
        """
        lens_optimizer, _ = imaging_system.get_optimizers()
        if lens_optimizer is not None:
            for p in lens_optimizer.param_groups:
                p.update(lens_optimizer.defaults)


class BindMaterialsCallback(callbacks.Callback):
    """Callback that binds the optimized materials to catalog materials at a specific training step.

    The lens optimizer state is reset to the default values.
    """

    def __init__(self, step: int | float, end_step: int | float | None = None):
        """Constructor.

        Args:
            step: If an integer, bind materials after this training step;
                if a float, bind materials after this fraction of the total number of training steps.
            end_step: If provided, progressively bind materials one by one from "step" to "end_step".
        """
        super().__init__()
        self.step = step
        self.end_step = end_step
        self.step_to_glass_indices = None

    def initialize(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        """Initialize the callback.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The ImagingSystemModule instance.
        """
        # Determine start and end steps
        start_step = (
            int(self.step * trainer.max_steps)
            if isinstance(self.step, float)
            else self.step
        )
        if self.end_step is not None:
            end_step = (
                int(self.end_step * trainer.max_steps)
                if isinstance(self.end_step, float)
                else self.end_step
            )
        else:
            end_step = start_step

        self.step_to_glass_indices = {}

        if pl_module.get_optimizers()[0] is None:  # No lens optimizer
            return

        # Determine optimizable material indices
        g_optimization_mask = ~pl_module.parameterization.freeze_dict["g"]
        optimizable_glass_indices = torch.where(
            g_optimization_mask.flatten(1).any(dim=-1)
        )[0].tolist()
        n_optimizable = len(optimizable_glass_indices)

        # Map steps to corresponding material indices
        steps = np.linspace(start_step, end_step, n_optimizable).astype(int).tolist()
        for s, i in zip(steps, optimizable_glass_indices):
            if s in self.step_to_glass_indices:
                self.step_to_glass_indices[s] += (i,)
            else:
                self.step_to_glass_indices[s] = (i,)

    def on_train_batch_start(
        self, trainer: pl.trainer, pl_module: ImagingSystemModule, *_
    ):
        if self.step_to_glass_indices is None:
            self.initialize(trainer, pl_module)
        if pl_module.global_step in self.step_to_glass_indices:
            glass_indices = self.step_to_glass_indices[pl_module.global_step]
            self.bind_materials(pl_module, glass_indices)
            self.reset_lens_optimizer(pl_module)

    @torch.no_grad()
    def bind_materials(
        self, imaging_system: ImagingSystemModule, glass_indices: tuple[int]
    ):
        """Trigger the binding of the optimized materials to the catalog materials.

        Args:
            imaging_system: The ImagingSystemModule instance.
            glass_indices: The indices of the optimized materials.
        """
        parameterization = imaging_system.parameterization
        parameterization.freeze_glass_variables(glass_indices, bind_to_catalog=True)

    @torch.no_grad()
    def reset_lens_optimizer(self, imaging_system: ImagingSystemModule):
        """Reset the lens optimizer to the default values.

        Args:
            imaging_system: The ImagingSystemModule instance.
        """
        lens_optimizer, _ = imaging_system.get_optimizers()
        if lens_optimizer is not None:
            for p in lens_optimizer.param_groups:
                p.update(lens_optimizer.defaults)


class IncreaseGlassVariableResidualsWeightCallback(callbacks.Callback):
    """Callback that exponentially increases the weight of glass variable residuals at discrete steps."""

    def __init__(
        self,
        initial_step: int | float,
        final_step: int | float,
        initial_weight: float,
        final_weight: float,
        n_increments: int,
    ):
        """Constructor.

        Args:
            initial_step: If an integer, the step at which the weight starts to increase;
                if a float, the fraction of the total number of training steps.
            final_step: If an integer, the step at which the weight reaches its final value;
                if a float, the fraction of the total number of training steps.
            initial_weight: The initial weight.
            final_weight: The final weight.
            n_increments: The number of increments between the initial and final steps.
        """
        super().__init__()
        self.initial_step = initial_step
        self.final_step = final_step
        # Find all increments on log scale
        self.increments = np.logspace(
            np.log10(initial_weight), np.log10(final_weight), n_increments + 1
        )
        self.step_to_weight = None

    def initialize(self, trainer: pl.Trainer):
        """Initialize the callback.

        Args:
            trainer: The PyTorch Lightning trainer.
        """
        max_steps = trainer.max_steps
        start_step = (
            int(self.initial_step * max_steps)
            if isinstance(self.initial_step, float)
            else self.initial_step
        )
        end_step = (
            int(self.final_step * max_steps)
            if isinstance(self.final_step, float)
            else self.final_step
        )
        steps = (
            np.linspace(start_step, end_step, len(self.increments)).astype(int).tolist()
        )
        self.step_to_weight = dict(zip(steps, self.increments))

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: ImagingSystemModule, *_
    ):
        if self.step_to_weight is None:
            self.initialize(trainer)
            assert "glass_variable" in [
                residuals.name for residuals in pl_module.residuals
            ], 'The "glass_variable" residuals are not present in the residuals list.'
        if pl_module.global_step in self.step_to_weight:
            weight = self.step_to_weight[pl_module.global_step]
            pl_module.weight_dict["glass_variable"] = float(weight)


class ExtendedLoggingCallback(callbacks.Callback):
    """Callback that logs additional metrics, usually computationally expensive ones, every n steps."""

    def __init__(self, every_n_steps: int = 10):
        """Constructor.

        Args:
            every_n_steps: Log additional metrics every n steps.
        """
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_train_start(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        self.log_extended_metrics(pl_module)

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: ImagingSystemModule, *_
    ):
        if pl_module.global_step % self.every_n_steps == 0:
            self.log_extended_metrics(pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        self.log_extended_metrics(pl_module)

    def log_extended_metrics(self, imaging_system: ImagingSystemModule):
        """Compute and log the extended metrics.

        Args:
            imaging_system: The ImagingSystemModule instance.
        """
        lens = imaging_system.lens

        logs = {
            "ray_tracing/distortion": self.compute_max_distortion(
                imaging_system, lens
            ).item(),
            "ray_tracing/max_ray_aiming_error": self.compute_max_ray_aiming_error(
                imaging_system, lens
            ).item(),
            **self.compute_phase_profile_range(imaging_system, lens),
        }

        tb_logger = imaging_system.try_find_tensorboard_logger()
        for k, v in logs.items():
            tb_logger.add_scalar(k, v, imaging_system.global_step)

    @torch.inference_mode()
    def compute_max_distortion(
        self,
        imaging_system: ImagingSystemModule,
        lens: optics.Lens,
        xy: torch.Tensor | None = None,
    ):
        """Return the maximum distortion from the spot diagrams.

        Args:
            imaging_system: The ImagingSystemModule instance.
            lens: The lens instance.
            xy: The spot diagram coordinates (x, y) at the image plane.
        """
        # Compute the spot diagram centroid at maximum field
        # Average over all wavelengths and pupil intersections
        if xy is None and imaging_system.xy_updated:
            xy = imaging_system.last_xy
        else:
            xy = imaging_system.compute_spot_diagrams()
        ray_valid = xy.isfinite().all(dim=0)
        _, y = xy
        y_centroid = ra.evaluate_mean_ray_height(
            y, ray_valid, (1, 2), imaging_system.wavelength_weights
        )

        # Compute paraxial heights
        field_angles = (
            torch.linspace(0, imaging_system.ray_initialization.hfov, y.shape[0])
            .to(xy)
            .deg2rad()
        )
        heights = lens.evaluate_paraxial_heights_at_image_plane(field_angles).squeeze(
            dim=0
        )

        # Compute
        distortion = (y_centroid.squeeze() - heights) / heights[-1]

        return distortion[distortion.abs().argmax()]

    @torch.inference_mode()
    def compute_max_ray_aiming_error(
        self, imaging_system: ImagingSystemModule, lens: optics.Lens
    ):
        """Return the maximum ray aiming error from the spot diagrams.

        Args:
            imaging_system: The ImagingSystemModule instance.
            lens: The lens instance.
        """
        # Initialize rays from specifications
        n_fields = 1
        # On-axis (only at reference wavelength)
        ray_initialization = imaging_system.ray_initialization
        ref_r, ref_d = ray_initialization(lens, None, 0.0, 1, [lens.w0], "marginal", 0)
        (_, ref_y, _), _, ray_status_ref, _ = next(
            lens.trace_rays(ref_r, ref_d, (lens.w0,), yield_on="stop")
        )
        # Off-axis
        r, d = ray_initialization(lens, n_fields=n_fields, pupil_sampling_mode="tee")
        (x, y, _), _, ray_status, _ = next(
            lens.trace_rays(r, d, ray_initialization.wavelengths, yield_on="stop")
        )
        bottom_y, top_y, _ = y.unbind(dim=1)
        _, _, right_x = x.unbind(dim=1)
        # Compute error
        relative_coordinates = (
            torch.stack((bottom_y, top_y, right_x), dim=1).abs() / ref_y
        )
        max_error = (1 - relative_coordinates).abs().max()
        return max_error

    @torch.inference_mode()
    def compute_phase_profile_range(
        self, imaging_system: ImagingSystemModule, lens: optics.Lens
    ):
        """Return the range of the phase profile of the dispersion-engineered metasurface.

        Args:
            imaging_system: The ImagingSystemModule instance.
            lens: The lens instance.
        """
        misc_surface_model = lens.misc_surface_model

        if not isinstance(misc_surface_model, ms.DispersionEngineeredMetasurface):
            return {}
        if lens.sequence.n_misc_surfaces < 1:
            return {}

        ray_initialization = imaging_system.ray_initialization

        # Estimate diameters
        wavelengths = torch.tensor(ray_initialization.wavelengths).to(lens.c)
        r, d = ray_initialization(
            lens,
            n_fields=2,
            wavelengths=wavelengths,
            pupil_sampling_mode="skew_outer_edge_uniform",
            n_rays=32,
        )
        diameters = lens.estimate_diameters(r, d, wavelengths)

        # Get metasurface indices
        m_idx = []
        for i, (event_type, *_) in enumerate(lens.return_geometry()):
            if event_type == "m":
                m_idx.append(i)
        assert len(m_idx) == lens.sequence.n_misc_surfaces, (
            "The number of miscellaneous surfaces does not match the sequence."
        )

        diameters = diameters[m_idx].squeeze(dim=1)

        # Get rho
        n_points = 1000
        rho = torch.linspace(0, 1, n_points).to(diameters)
        rho = rho[:, None] * diameters / 2

        # Compute phase components
        m = lens.m.squeeze(dim=1)
        phase = misc_surface_model.phase_at_design_wavelength(rho, m)
        group_delay = misc_surface_model.group_delay(rho, m) * 1e15  # fs
        phase_range = phase.max(dim=0)[0] - phase.min(dim=0)[0]
        group_delay_range = group_delay.max(dim=0)[0] - group_delay.min(dim=0)[0]

        # Add output to dictionary
        out_dict = {}
        for i, (pr, gdr) in enumerate(zip(phase_range, group_delay_range)):
            out_dict[f"metasurface/phase_range{i + 1}"] = pr.item()
            out_dict[f"metasurface/group_delay_range{i + 1}"] = gdr.item()

        return out_dict


class ConfigFileCallback(callbacks.Callback):
    """Callback that saves the lens parameters in .yml format every n steps."""

    def __init__(self, every_n_steps: int = 10):
        """Constructor.

        Args:
            every_n_steps: Save the lens parameters every n steps.
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self._last_saved_step = None

    def on_test_start(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        self.save_config_file(trainer, pl_module)

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: ImagingSystemModule, *_
    ):
        if pl_module.global_step % self.every_n_steps == 0:
            self.save_config_file(trainer, pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        """Save the final updated lens even when max_steps is not a logging step."""
        if self._last_saved_step != pl_module.global_step:
            self.save_config_file(trainer, pl_module)

    def save_config_file(
        self, trainer: pl.Trainer, imaging_system: ImagingSystemModule
    ):
        """Save the lens parameters in .yml format.

        Args:
            trainer: The PyTorch Lightning trainer.
            imaging_system: The ImagingSystemModule instance.
        """
        log_dir = retrieve_log_dir(trainer)
        lens = imaging_system.lens
        lens_dict = {
            k: getattr(lens, k).squeeze(1).tolist()
            for k in imaging_system.parameterization.parameter_keys
        }
        dir_path = f"{log_dir}/lens_parameters"
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.normpath(
            os.path.abspath(f"{dir_path}/{imaging_system.global_step:05d}.yml")
        ).replace("\\", "/")
        with open(file_path, "w") as f:
            def float_precision(x):
                return float(f"{x:.12g}") if isinstance(x, float) else x

            yaml.dump(
                apply_recursively(lens_dict, float_precision),
                f,
                default_flow_style=None,
            )
        self._last_saved_step = imaging_system.global_step


class CodeVSeqFileCallback(callbacks.Callback):
    """Callback that saves the Code V sequence file in the log directory."""

    def __init__(self, use_private_catalog: bool = True):
        """Constructor.

        Args:
            use_private_catalog: If True, create private catalog glass for each glass;
                otherwise, use Code V dispersion.
        """
        super().__init__()
        self.use_private_catalog = use_private_catalog

    def on_test_start(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        self.save_codev_file(trainer, pl_module)

    def save_codev_file(self, trainer: pl.Trainer, imaging_system: ImagingSystemModule):
        """Save the Code V sequence file in the log directory.

        Args:
            trainer: The PyTorch Lightning trainer.
            imaging_system: The ImagingSystemModule instance.
        """
        log_dir = retrieve_log_dir(trainer)
        codev_file = make_codev_file(imaging_system, self.use_private_catalog)
        file_path = os.path.normpath(os.path.abspath(f"{log_dir}/lens.seq")).replace(
            "\\", "/"
        )
        with open(file_path, "w") as f:
            f.write(codev_file)
        # Print clickable link to file in console
        print(f"Code V sequence file: file:///{file_path}")


def make_codev_file(imaging_system: ImagingSystemModule, use_private_catalog: bool):
    """Generate a Code V sequence file from the imaging system module.

    Args:
        imaging_system: Imaging system module
        use_private_catalog: If True, create private catalog glass for each glass; otherwise, use Code V dispersion

    """
    lens = imaging_system.lens
    if getattr(lens, "z", None) is not None and lens.z.numel() > 0:
        raise NotImplementedError(
            "Code V export is unsupported for lenses with zonal surfaces."
        )
    ray_initialization = imaging_system.ray_initialization.convert_to_absolute(lens)
    epd = float(ray_initialization.epd)

    commands = [
        "RDM N",  # Curvature mode
        "DDM M",  # Use millimeters
        "LEN",  # Fresh lens
        f"EPD {epd:.8g}",  # EPD
    ]

    # Fields and vignetting
    r0, d0 = ray_initialization(lens, pupil_sampling_mode="tee")
    x0, y0, _ = r0.numpy(force=True)
    vig_y_l = 1 + y0[:, 0, len(ray_initialization.wavelengths) // 2] / (epd / 2)
    vig_y_u = 1 - y0[:, 1, len(ray_initialization.wavelengths) // 2] / (epd / 2)
    vig_x = 1 - x0[:, 2, len(ray_initialization.wavelengths) // 2] / (epd / 2)
    commands.extend(
        [
            "YAN "
            + " ".join(
                f"{f:.8g}"
                for f in np.linspace(
                    0, ray_initialization.hfov, ray_initialization.n_fields
                )
            ),  # Fields
            "VUY "
            + " ".join(
                f"{f:.8g}"
                for f in np.clip(vig_y_u.flat, a_min=-float("inf"), a_max=1.0)
            ),  # Upper y-vignetting
            "VLY "
            + " ".join(
                f"{f:.8g}"
                for f in np.clip(vig_y_l.flat, a_min=-float("inf"), a_max=1.0)
            ),  # Lower y-vignetting
            "VUX "
            + " ".join(
                f"{f:.8g}" for f in np.clip(vig_x.flat, a_min=-float("inf"), a_max=1.0)
            ),  # X-vignetting
            "VLX "
            + " ".join(
                f"{f:.8g}" for f in np.clip(vig_x.flat, a_min=-float("inf"), a_max=1.0)
            ),  # X-vignetting
        ]
    )

    # Wavelengths and weights
    wavelength_weights = ray_initialization.wavelength_weights
    if wavelength_weights is None:
        wavelength_weights = np.ones_like(ray_initialization.wavelengths).astype(int)
    else:
        wavelength_weights = (wavelength_weights * 1000).astype(int)
    commands.extend(
        [
            "WL "
            + " ".join(
                f"{wav:.8g}" for wav in ray_initialization.wavelengths[::-1]
            ),  # Wavelengths
            "WTW " + " ".join(f"{w:.8g}" for w in wavelength_weights[::-1]),  # Weights
            f"REF {(len(ray_initialization.wavelengths) + 1) // 2}",  # Set middle wavelength as reference
        ]
    )

    # Put the lens variables in tabular format
    tabular_data = lens.as_tabular()

    # Glass catalog
    if use_private_catalog:
        n = lens.get_n(
            torch.tensor(ray_initialization.wavelengths).to(lens.nd)
        ).squeeze(dim=1)
        for i, nn in enumerate(n):
            glass_name = "glass" + str(i + 1)
            commands.extend(
                [
                    "PRV",
                    "PWL "
                    + " ".join(f"{wav:.8g}" for wav in ray_initialization.wavelengths),
                    f'"{glass_name}" ' + " ".join(f"{nn:.8g}" for nn in nn),
                    "END",
                ]
            )

    glass_count = 0
    for i, surface in enumerate(tabular_data):
        new_surface = (
            f"S {surface['c'] if 'c' in surface else 0:.8g} {surface['s']:.8g}"
        )
        if "nd" in surface:
            glass_count += 1
            if use_private_catalog:
                new_surface += f' "glass{glass_count}"'
            else:
                new_surface += f" {surface['nd']:.8g}:{surface['vd']:.8g}"
        commands.append(new_surface)
        if "a" in surface:
            commands.append(f"ASP S{i + 1}")
            commands.extend(
                [f"{k} S{i + 1} {v:.8g}" for k, v in zip("KABCDEFGHJ", surface["a"])]
            )
        if "stop" in surface:
            commands.append(f"STO S{i + 1}")
        if "d" in surface:
            commands.append(f"DIF S{i + 1} DOE")
            commands.append(f"HCT S{i + 1} R")
            commands.append(f"HOR S{i + 1} 1")
            commands.append(f"HWL S{i + 1} {lens.w0}")
            commands.extend(
                [
                    f"HCO S{i + 1} C{k + 1} {v:.8g}"
                    for k, v in zip(range(10), surface["d"])
                ]
            )
        if "m" in surface:
            commands.append(f"DIF S{i + 1} DOE")
            commands.append(f"HCT S{i + 1} R")
            commands.append(f"HOR S{i + 1} 1")
            commands.append(f"HWL S{i + 1} {lens.w0}")
            commands.extend(
                [
                    f"HCO S{i + 1} C{k + 1} {v:.8g}"
                    for k, v in zip(range(10), surface["m"])
                ]
            )

    # Field stop
    field_stop_position = ray_initialization.field_stop_position
    if field_stop_position is not None:
        commands.extend(
            [
                "INS S1..2",  # Add 2 additional surfaces
                f"THI S1 {field_stop_position:.8g}",
                f"THI S2 {-field_stop_position:.8g}",
                f"CIR S2 {epd / 2:.8g}",  # Field stop semi-aperture
            ]
        )

    # Re-compute vignetting factors
    commands.append("set vig")
    return "\n".join(commands)
