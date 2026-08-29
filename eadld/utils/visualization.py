import os

import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import numpy as np
import torch
from pytorch_lightning import callbacks
import pytorch_lightning as pl
import torchvision.utils

from eadld.imaging_system import ImagingSystemModule, _combine_ray_weights
from eadld.modeling import (
    simulation as sim,
    misc_surfaces as ms,
    ray_initialization as ri,
    ray_analysis as ra,
    optics,
)
from eadld.utils import utils


class Visualization:
    """Base class for visualizations."""

    name = ""

    def __init__(self, rc_params: dict | None = None):
        """Constructor.

        Args:
            rc_params: Matplotlib rcParams that are applied when producing the figure.
        """
        if rc_params is None:
            rc_params = {}
        self.rc_params = rc_params

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        """Generate the figure and return either the figure or an RGB image.

        Args:
            trainer: PyTorch Lightning trainer.
            imaging_system: ImagingSystemModule instance.
            fig: Matplotlib figure to plot on.
        """
        return NotImplementedError


class VisualizationCallback(callbacks.Callback):
    """A callback to generate visualizations and log them to Tensorboard and/or save them as files."""

    def __init__(
        self,
        visualizations: list[Visualization] | None,
        save_formats: list[str] = (),
        rc_params: dict[str, str] | None = None,
        every_n_steps: int = 100,
    ):
        """Constructor.

        Args:
            visualizations: List of figures with their corresponding args.
            save_formats: List of formats to save the images as;
                supported formats are 'jpg', 'png', 'svg', 'pdf', etc.
            rc_params: Matplotlib rcParams.
            every_n_steps: Log images every n steps.
        """
        super().__init__()
        self.visualizations = {}
        duplicates = {}
        for vis in visualizations:
            name = vis.name
            if name in self.visualizations:
                duplicates[name] = duplicates.get(name, 0) + 1
                name += f"_{duplicates[name]}"
            self.visualizations[name] = vis
        self.save_formats = save_formats
        self.rc_params = rc_params
        if rc_params is None:
            self.rc_params = {}
        self.every_n_steps = every_n_steps
        self._last_logged_step = None

    def on_test_start(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        self.log_plots(trainer, pl_module)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        self.log_plots(trainer, pl_module)

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: ImagingSystemModule, *_
    ):
        if pl_module.lens_optimizer is None:
            # If lens is not being optimized, only generate plots once at the beginning
            if trainer.global_step == 0:
                self.log_plots(trainer, pl_module)
        else:
            if trainer.global_step % self.every_n_steps == 0:
                self.log_plots(trainer, pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: ImagingSystemModule):
        """Log the final updated lens even when max_steps is not a logging step."""
        if self._last_logged_step != pl_module.global_step:
            self.log_plots(trainer, pl_module)

    @torch.no_grad()
    def log_plots(self, trainer: pl.Trainer, imaging_system: ImagingSystemModule):
        """Generate and log the visualizations.

        Args:
            trainer: PyTorch Lightning trainer.
            imaging_system: ImagingSystemModule instance.
        """
        tb_logger = imaging_system.try_find_tensorboard_logger()
        log_dir = None if tb_logger is None else tb_logger.log_dir
        for k, v in self.visualizations.items():
            with mpl.rc_context({**self.rc_params, **v.rc_params}):
                fig = v.generate_figure(trainer, imaging_system)
                if fig is None:
                    continue
                img = plot_to_image(fig) if isinstance(fig, mpl.figure.Figure) else fig
                if tb_logger is not None:
                    tb_logger.add_image(
                        k, img, imaging_system.global_step, dataformats="CHW"
                    )
                if log_dir is not None:
                    if isinstance(fig, mpl.figure.Figure):
                        for fmt in self.save_formats:
                            os.makedirs(f"{log_dir}/{k}", exist_ok=True)
                            fig.savefig(
                                f"{log_dir}/{k}/{imaging_system.global_step:05d}.{fmt}"
                            )
                    elif isinstance(fig, np.ndarray):
                        for fmt in self.save_formats:
                            os.makedirs(f"{log_dir}/{k}", exist_ok=True)
                            img = PIL.Image.fromarray(np.moveaxis(fig, 0, 2))
                            img.save(
                                f"{log_dir}/{k}/{imaging_system.global_step:05d}.{fmt}"
                            )
        self._last_logged_step = imaging_system.global_step


class LensLayout(Visualization):
    """2D layout of the lens system.

    In addition to the lens layout, the rays are plotted for the given fields and wavelengths.
    """

    name = "layout"

    def __init__(
        self,
        wavelengths: list[float] | tuple[float, ...] = (450.0, 650.0, 550.0),
        n_rays: int = 3,
        n_fields: int = 2,
        label_materials: bool = False,
        highlight_materials_for_n_steps: int = 0,
        diameter_scaler: float = 17 / 16,
        scale_bar: bool = True,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            wavelengths: Wavelengths to plot the rays for (in nm).
            n_rays: Number of rays to plot per field and wavelength.
            n_fields: Number of equidistant fields to plot.
            label_materials: Whether to label the materials.
            highlight_materials_for_n_steps: Highlight the glass labels for this many steps after a change;
                set to 0 to disable highlighting.
            diameter_scaler: Set to >1 to increase the diameter of the lens elements;
                if set to 1, the diameters are set precisely so that all rays pass through.
            scale_bar: Whether to add a scale bar below the lens layout.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.wavelengths = wavelengths
        self.n_rays = n_rays
        self.n_fields = n_fields
        self.label_materials = label_materials
        self.highlight_glass_for_n_steps = highlight_materials_for_n_steps
        self.last_labels = []
        self.highlight_steps = []
        self.diameter_scaler = diameter_scaler
        self.scale_bar = scale_bar

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        glass_labels = None
        parameterization = imaging_system.parameterization
        glass_model = parameterization.glass_model
        if self.label_materials and glass_model.catalog_glass_names is not None:
            closest_catalog_indices = (
                parameterization.glass_model.get_catalog_glass_indices(
                    parameterization.lens_variables["g"]
                )
                .squeeze(dim=1)
                .numpy(force=True)
            )
            glass_labels = glass_model.catalog_glass_names[closest_catalog_indices]
        fig = generate_layout_plot(
            imaging_system.lens,
            imaging_system.ray_initialization,
            self.wavelengths,
            self.n_rays,
            self.n_fields,
            glass_labels,
            self.diameter_scaler,
            self.scale_bar,
            fig=fig,
        )
        # Highlight glass labels if the glasses have changed
        if glass_labels is not None and self.highlight_glass_for_n_steps > 0:
            for i, text in enumerate(fig.axes[0].texts):
                label = text._text
                if label in glass_labels:
                    if len(self.last_labels) <= i:
                        self.last_labels.append(label)
                        self.highlight_steps.append(0)
                    elif self.last_labels[i] != label:
                        self.last_labels[i] = label
                        self.highlight_steps[i] = (
                            trainer.global_step + self.highlight_glass_for_n_steps
                        )
                    if self.highlight_steps[i] > trainer.global_step:
                        text.set_color("red")

        return fig


class ZonalSurfaceProfile(Visualization):
    """Plot the physical sag of a zonal refractive surface."""

    name = "zonal_surface_profile"

    def __init__(
        self,
        idx: int = 0,
        samples_per_zone: int = 64,
        sag_limits: tuple[float, float] | list[float] | None = None,
        show_change: bool = False,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            idx: Index of the zonal surface to plot.
            samples_per_zone: Number of radial samples drawn inside each zone.
            sag_limits: Optional fixed y-axis limits in mm.
            show_change: Plot the change from the first recorded profile.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.idx = idx
        self.samples_per_zone = samples_per_zone
        self.sag_limits = sag_limits
        self.show_change = show_change
        self._initial_profiles = None

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        del trainer
        lens = imaging_system.lens
        if len(lens) != 1:
            raise ValueError("ZonalSurfaceProfile only supports a single lens.")
        if lens.z is None or self.idx >= lens.sequence.n_zonal:
            raise ValueError(f"Zonal surface index {self.idx} does not exist.")

        event = next(
            event
            for event in lens.sequence.events
            if event["type"] == "r" and event.get("z") == self.idx
        )
        c = lens.c[event["c"], 0].detach().cpu()
        zones = lens.z[self.idx, 0].detach().cpu()
        zones = zones[zones[:, 3] > 0]

        if fig is None:
            if self.show_change:
                fig, (ax, change_ax) = plt.subplots(
                    2, 1, sharex=True, height_ratios=(2, 1)
                )
            else:
                fig, ax = plt.subplots()
                change_ax = None
        else:
            if self.show_change:
                ax, change_ax = fig.subplots(
                    2, 1, sharex=True, height_ratios=(2, 1)
                )
            else:
                ax = fig.subplots()
                change_ax = None
        profiles = []
        previous_radius = zones.new_zeros(())
        previous_sag = None
        for zone_idx, (delta_a1, a2, delta_z, rmax) in enumerate(zones):
            radius = torch.linspace(
                previous_radius,
                rmax,
                self.samples_per_zone,
                dtype=zones.dtype,
            )
            rho = radius.square()
            sag = (0.5 * c + delta_a1) * rho + a2 * rho.square() + delta_z
            profiles.append((radius, sag))
            ax.plot(
                radius.numpy(),
                sag.numpy(),
                color="C0",
                label="Current" if zone_idx == 0 and self.show_change else None,
            )
            if previous_sag is not None:
                # The dotted connector records the profile discontinuity only;
                # it is not a traced sidewall in the optical model.
                ax.plot(
                    [previous_radius.item(), previous_radius.item()],
                    [previous_sag.item(), sag[0].item()],
                    color="0.55",
                    linestyle=":",
                    linewidth=0.8,
                )
            previous_radius = rmax
            previous_sag = sag[-1]

        if self.show_change:
            if self._initial_profiles is None:
                self._initial_profiles = [
                    (radius.clone(), sag.clone()) for radius, sag in profiles
                ]
            for zone_idx, ((radius, sag), (initial_radius, initial_sag)) in enumerate(
                zip(profiles, self._initial_profiles)
            ):
                if not torch.equal(radius, initial_radius):
                    raise ValueError(
                        "ZonalSurfaceProfile change plot requires fixed zone boundaries."
                    )
                ax.plot(
                    initial_radius.numpy(),
                    initial_sag.numpy(),
                    color="C1",
                    linestyle="--",
                    label="Initial" if zone_idx == 0 else None,
                )
                change_ax.plot(
                    radius.numpy(),
                    ((sag - initial_sag) * 1e3).numpy(),
                    color="C2",
                )
            ax.legend(loc="best")
            change_ax.set(xlabel="Radius (mm)", ylabel="Change (um)")
            change_ax.grid(alpha=0.2)

        ax.set(
            ylabel="Sag (mm)",
            title=f"Zonal surface {self.idx + 1}: {len(zones)} zones",
        )
        if not self.show_change:
            ax.set_xlabel("Radius (mm)")
        if self.sag_limits is not None:
            ax.set_ylim(*self.sag_limits)
        ax.grid(alpha=0.2)
        return fig


class DispersionEngineeredPhase(Visualization):
    """Plot the phase and group delay of an idealized phase profile as a function of radius."""

    name = "dispersion_engineered_phase"

    def __init__(
        self,
        max_radius: float | None = None,
        idx: int | None = None,
        which: str = "both",
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            max_radius: Maximum radius to plot; if None, the radius is estimated by exact ray tracing.
            idx: Index of the misc surface to plot, if there are multiple.
            which: Whether to plot 'both', 'phase', or 'group_delay'.
        """
        super().__init__(rc_params)
        self.max_radius = max_radius
        self.idx = idx
        self.which = which

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        if fig is None:
            fig = mpl.figure.Figure()
        lens = imaging_system.lens
        model = lens.misc_surface_model
        if not isinstance(model, ms.DispersionEngineeredMetasurface):
            return None
        if self.idx is None:
            idx_list = range(lens.sequence.n_misc_surfaces)
        else:
            if self.idx >= lens.sequence.n_misc_surfaces:
                return None
            idx_list = [self.idx]
        if len(idx_list) == 0:
            return None

        if self.which == "both":
            axes = fig.subplots(2, 1, sharex="all", squeeze=False).flat
            ax_phase, ax_group_delay = axes
        elif self.which == "phase":
            axes = fig.subplots(squeeze=False).flat
            ax_phase = axes[0]
            ax_group_delay = None
        elif self.which == "group_delay":
            axes = fig.subplots(squeeze=False).flat
            ax_phase = None
            ax_group_delay = axes[0]
        else:
            raise ValueError(
                f'which should be "both", "phase", or "group_delay", got {self.which}'
            )

        # Compute phase components
        for idx in idx_list:
            if self.max_radius is not None:
                max_radius = [self.max_radius]
            else:
                max_radius = estimate_misc_surface_radius(
                    lens, idx, imaging_system.ray_initialization
                )
            rho = torch.linspace(0, max_radius, 1000)
            m = lens.m[idx, :1].to(rho.device)
            phase_at_design_wavelength = model.phase_at_design_wavelength(rho, m)
            group_delay = model.group_delay(rho, m) * 1e15  # fs
            if ax_phase is not None:
                ax_phase.plot(
                    rho,
                    phase_at_design_wavelength,
                    label=f"Surface {idx + 1}",
                    zorder=1 - idx,
                )
            if ax_group_delay is not None:
                ax_group_delay.plot(
                    rho, group_delay, label=f"Surface {idx + 1}", zorder=1 - idx
                )

        # Plot phase and group delay
        if ax_phase is not None:
            ax_phase.set_ylabel(r"$\phi(r, \lambda_\mathrm{ref})$")
        if ax_group_delay is not None:
            ax_group_delay.set_ylabel(
                r"$\tau(r, \lambda_\mathrm{ref})$ $[\mathrm{fs}]$"
            )
        fig.align_ylabels(axes)
        axes[-1].set_xlabel(r"$r$ [mm]")
        if len(idx_list) > 1:
            axes[0].legend(loc="best")

        return fig


class GlassPlot(Visualization):
    """Plot the refractive index and Abbe number of the refractive materials, as well as catalog glasses."""

    name = "glass"

    def __init__(
        self,
        plot_selected_glass: bool = True,
        plot_partial_dispersion: bool = False,
        single_column: bool = True,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            plot_selected_glass: Plot the selected glass in the optimization process.
            plot_partial_dispersion: Plot the partial dispersion of the glasses.
            single_column: Whether to use a single column layout if partial dispersion is plotted.
        """
        super().__init__(rc_params)
        self.plot_selected_glass = plot_selected_glass
        self.plot_partial_dispersion = plot_partial_dispersion
        self.single_column = single_column

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        parameterization = imaging_system.parameterization
        glass_model = parameterization.glass_model

        if fig is None:
            fig = mpl.figure.Figure()
        if self.plot_partial_dispersion and self.single_column:
            ax_nd, ax_dpgf = fig.subplots(2, sharex="all")
        elif self.plot_partial_dispersion:
            ax_nd, ax_dpgf = fig.subplots(1, 2)
        else:
            ax_nd = fig.subplots(1)

        # Plot catalog glasses
        if glass_model.catalog_g is not None:
            cat_nd, cat_vd, cat_dpgf = glass_model.nd_vd_dpgf_from_g(
                glass_model.catalog_g
            )
            ax_nd.scatter(
                cat_vd.cpu(),
                cat_nd.cpu(),
                marker="+",
                s=16,
                label="Catalog Glasses",
                c="k",
            )
            if self.plot_partial_dispersion:
                ax_dpgf.scatter(cat_vd.cpu(), cat_dpgf.cpu(), marker="+", s=16, c="k")

        # Plot lens glasses
        nd, vd, dpgf = parameterization.get_nd_vd_dpgf(
            parameterization.lens_variables["g"], False
        )
        color_cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color_cycle)
        colors = np.array(color_cycle)[np.arange(len(nd)) % n_colors].tolist()

        # Plot selected glasses
        if self.plot_selected_glass and glass_model.catalog_g is not None:
            fix_nd, fix_vd, fix_dpgf = parameterization.get_nd_vd_dpgf(
                parameterization.lens_variables["g"], True
            )
            ax_nd.scatter(
                fix_vd.view(-1).cpu(),
                fix_nd.view(-1).cpu(),
                marker="s",
                s=16,
                c=colors,
                zorder=0,
                label="Closest Catalog Glass",
                alpha=0.5,
            )
            if self.plot_partial_dispersion:
                ax_dpgf.scatter(
                    fix_vd.view(-1).cpu(),
                    fix_dpgf.view(-1).cpu(),
                    marker="s",
                    s=16,
                    c=colors,
                    zorder=0,
                    alpha=0.5,
                )
            # Add lines between the selected glass and the optimized glasses
            for i in range(len(nd)):
                ax_nd.plot(
                    [fix_vd[i].cpu(), vd[i].cpu()],
                    [fix_nd[i].cpu(), nd[i].cpu()],
                    c=colors[i],
                    lw=0.5,
                    zorder=0,
                    alpha=0.5,
                )
                if self.plot_partial_dispersion:
                    ax_dpgf.plot(
                        [fix_vd[i].cpu(), vd[i].cpu()],
                        [fix_dpgf[i].cpu(), dpgf[i].cpu()],
                        c=colors[i],
                        lw=0.5,
                        zorder=0,
                        alpha=0.5,
                    )

        # Plot optimized glasses
        points_x = 4
        points_y = 3
        for i in range(len(nd)):
            ax_nd.scatter(
                vd[i].cpu(),
                nd[i].cpu(),
                marker="x",
                s=16,
                c=colors[i],
                label=f"Glass {i + 1}",
            )
            # Annotate the index of each optimized glass
            kwargs = dict(
                xytext=(
                    points_x * (-1 if i % 2 == 0 else 1),
                    points_y * (-1 if (i // 2) % 2 == 1 else 1),
                ),
                textcoords="offset points",
                ha="center",
                va="center_baseline",
                fontsize="small",
                color=colors[i],
            )
            ax_nd.annotate(str(i + 1), (vd[i].cpu(), nd[i].cpu()), **kwargs)
            if self.plot_partial_dispersion:
                ax_dpgf.scatter(
                    vd[i].cpu(), dpgf[i].cpu(), marker="x", s=16, c=colors[i]
                )
                ax_dpgf.annotate(str(i + 1), (vd[i].cpu(), dpgf[i].cpu()), **kwargs)

        # Labels and format
        ax_nd.invert_xaxis()
        ax_nd.set_ylabel("Refractive Index (587.6 nm)")
        handles, labels = ax_nd.get_legend_handles_labels()
        legend1_indices = [
            i for i, label in enumerate(labels) if label.startswith("Glass")
        ]
        legend1 = ax_nd.legend(
            [handles[i] for i in legend1_indices],
            [labels[i] for i in legend1_indices],
            loc="upper left",
            ncol=2,
        )
        ax_nd.add_artist(legend1)
        ax_nd.legend(
            [handles[i] for i in range(len(labels)) if i not in legend1_indices],
            [labels[i] for i in range(len(labels)) if i not in legend1_indices],
            loc="lower right",
            ncol=1,
        )

        if self.plot_partial_dispersion:
            ax_dpgf.invert_xaxis()
            ax_dpgf.set_xlabel("Abbe Number")
            ax_dpgf.set_ylabel("Partial Dispersion Deviation")
        if not (self.plot_partial_dispersion and self.single_column):
            ax_nd.set_xlabel("Abbe Number")
        fig.align_ylabels()

        return fig


class RayFanPlot(Visualization):
    """Plot the tangential and sagittal ray fan diagrams."""

    name = "ray_fan"

    def __init__(
        self,
        wavelengths: list[float] | tuple[float, ...] = (
            550.0,
            450.0,
            500.0,
            600.0,
            650.0,
        ),
        n_fields: int = 4,
        ref_w: int = 0,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            wavelengths: Wavelengths to plot the rays for (in nm).
            n_fields: Number of equidistant fields to plot.
            ref_w: Reference wavelength to use for the tangential ray fan.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.wavelengths = wavelengths
        self.n_fields = n_fields
        self.ref_w = ref_w

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        if fig is None:
            fig = mpl.figure.Figure()
        lens = imaging_system.lens
        hfov = imaging_system.ray_initialization.hfov
        n_wavelengths = len(self.wavelengths)
        rel_fields = np.linspace(0, 1, self.n_fields)
        n_rays = 129
        specs = dict(
            n_fields=self.n_fields, wavelengths=self.wavelengths, n_rays=n_rays
        )

        # Tangential
        r0, d0 = imaging_system.ray_initialization(
            lens, pupil_sampling_mode="meridional_uniform", **specs
        )
        r, *_ = next(lens.trace_rays(r0, d0, self.wavelengths))
        _, y, *_ = r.cpu()

        # Sagittal
        r0, d0 = imaging_system.ray_initialization(
            lens, pupil_sampling_mode="sagittal_uniform", **specs
        )
        r, *_ = next(lens.trace_rays(r0, d0, self.wavelengths))
        x, *_ = r.cpu()

        ray_fan_subfig, legend_subfig = fig.subfigures(
            2, 1, height_ratios=(4 * self.n_fields, 1)
        )
        axes = ray_fan_subfig.subplots(
            self.n_fields, 2, sharex="col", sharey="all", width_ratios=(2, 1)
        )
        colors = wavelengths2color(self.wavelengths)
        for i, ((ax_left, ax_right), rel_field) in enumerate(
            zip(axes[::-1, :], rel_fields)
        ):
            ax_left.text(
                0.05,
                0.95,
                f"{rel_field * hfov:0.1f}°",
                va="top",
                ha="left",
                transform=ax_left.transAxes,
            )
            y_ref = y[i, n_rays // 2, self.ref_w]
            for j in range(n_wavelengths):
                ax_left.plot(
                    np.linspace(-1, 1, n_rays),
                    (y[i, :, j] - y_ref).numpy(),
                    color=colors[j],
                )
                ax_right.plot(
                    np.linspace(0, 1, n_rays), x[i, :, j].numpy(), color=colors[j]
                )
            ax_left.set_xlim(-1, 1)
            ax_left.set_xticks([-1, 1])
            ax_right.set_xlim(0, 1)
            ax_right.set_xticks([1])
            for ax in (ax_left, ax_right):
                ax.spines["left"].set_position("zero")
                ax.spines["right"].set_color("none")
                ax.spines["bottom"].set_position("zero")
                ax.spines["top"].set_color("none")
                ax.set_xticklabels([])
        axes[0, 0].set_title("Tangential")
        axes[0, 1].set_title("Sagittal")
        lim = max([abs(x) for ax in axes.flat for x in ax.get_ylim()])
        exponent = np.floor(np.log10(lim))
        possible = np.array([1, 1.2, 1.5, 2, 3, 4, 6, 8, 10])
        lim = possible[np.argmax((possible - lim / 10**exponent) > 0)] * 10**exponent
        for ax in axes.flat:
            ax.set_ylim(-lim, lim)
            ax.set_yticks([-lim, lim])

        # Split the plot in two with the bottom part being only for the legend
        legend_ax = legend_subfig.subplots(1, 1)
        legend_ax.axis("off")
        handles = [
            mpl.lines.Line2D([0], [0], color=colors[i], label=f"{int(wav)} nm")
            for i, wav in enumerate(self.wavelengths)
        ]
        handles = sorted(handles, key=lambda h: int(h.get_label().split()[0]))
        legend_ax.legend(
            handles=handles, loc="center", ncol=len(self.wavelengths), frameon=False
        )

        return fig


def _format_length_um(value: float) -> str:
    """显示小尺寸时保留有效数字，避免真实非零值被写成 0.0。"""
    value = abs(value)
    if value >= 100:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.3g}"


class SpotDiagrams(Visualization):
    """Plot the spot diagrams for the given fields and wavelengths."""

    name = "spot_diagrams"

    def __init__(
        self,
        field_indices: list[int] | None = None,
        n_rows: int | None = None,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            field_indices: Indices of the fields to plot.
            n_rows: Number of rows to plot; if None, compute automatically.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.field_indices = (
            field_indices if field_indices is not None else slice(None, None)
        )
        self.n_rows = n_rows

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        if fig is None:
            fig = mpl.figure.Figure()

        # Spot diagrams
        if imaging_system.last_xy is None:
            imaging_system.compute_residuals_dict_and_logs()
        xy = imaging_system.last_xy
        xy = xy[:, self.field_indices]
        spectral_weights = imaging_system.wavelength_weights
        x, y = xy
        ray_valid = torch.isfinite(xy).all(dim=0)
        ray_initialization = imaging_system.ray_initialization.convert_to_absolute(
            imaging_system.lens
        )
        pupil_weights = None
        if ray_initialization.pupil_sampling_mode == "skew_uniform_zonal":
            epd = ray_initialization.epd
            if callable(epd):
                epd = epd(imaging_system.lens.efl)
            pupil_weights = ri.zonal_pupil_weights(
                zone_edges=ri.zone_edges_from_lens(imaging_system.lens, epd),
                **ray_initialization.pupil_sampling_kwargs,
            ).to(xy).view(1, -1, 1, 1)
        ray_weights = _combine_ray_weights(
            spectral_weights, pupil_weights
        )
        rms = ra.compute_rms_spot_size(
            x, y, ray_valid, (1, 2), weights=ray_weights
        )
        y_centroid = ra.evaluate_mean_ray_height(
            y, ray_valid, (1, 2), ray_weights
        )
        y = y - y_centroid.expand_as(y).where(ray_valid, 0.0)
        x = x.cpu()
        y = y.cpu()
        ray_valid = ray_valid.cpu()

        # Specs
        n_fields = x.shape[0]
        if n_fields == 1:
            fig.set_size_inches(4.8, 4.0)
        ray_initialization = imaging_system.ray_initialization
        hfov = ray_initialization.hfov
        fields = np.linspace(0, hfov, ray_initialization.n_fields)[self.field_indices]
        wavelengths = np.asarray(imaging_system.ray_initialization.wavelengths)
        wavelengths = wavelengths.tolist()
        wavelength_colors = wavelengths2color(np.asarray(wavelengths))
        primary_index = (
            int(spectral_weights.flatten().argmax().item())
            if spectral_weights is not None
            else len(wavelengths) // 2
        )
        epd = float(ray_initialization.convert_to_absolute(imaging_system.lens).epd)
        target_efl = float(
            torch.as_tensor(imaging_system.parameterization.target_efl)
            .abs()
            .mean()
            .cpu()
        )
        airy_radius_um = (
            1.22 * wavelengths[primary_index] * 1e-3 * target_efl / epd
        )

        # Plot
        n_rows = self.n_rows or int(np.floor(np.sqrt(n_fields)))
        n_cols = int(np.ceil(n_fields / n_rows))
        axes = fig.subplots(n_rows, n_cols, sharex="all", sharey="all", squeeze=False)
        for i, (ax, xx, yy) in enumerate(zip(axes.flat, x.unbind(), y.unbind())):
            ax.set_facecolor("#F8FAFC")
            ax.axvline(0, c="#AAB7C7", alpha=0.8, lw=0.6)
            ax.axhline(0, c="#AAB7C7", alpha=0.8, lw=0.6)
            markers = (".", "x", "+")
            for wavelength_index, (wavelength, color) in enumerate(
                zip(wavelengths, wavelength_colors)
            ):
                valid = ray_valid[i, :, wavelength_index, :]
                wave_x = xx[:, wavelength_index, :][valid]
                wave_y = yy[:, wavelength_index, :][valid]
                ax.scatter(
                    torch.cat((-wave_x, wave_x)) * 1000,
                    torch.cat((wave_y, wave_y)) * 1000,
                    color=color,
                    marker=markers[wavelength_index % len(markers)],
                    s=1.4,
                    linewidths=0.25,
                    alpha=0.72,
                    rasterized=True,
                    label=f"{wavelength:.1f} nm" if i == 0 else None,
                )
            ax.add_patch(
                mpl.patches.Circle(
                    (0, 0),
                    airy_radius_um,
                    fill=False,
                    color="#0D1C2E",
                    linewidth=0.9,
                    linestyle=":",
                    label="Airy" if i == 0 else None,
                )
            )
            ax.set_aspect("equal")
            ax.tick_params(colors="#718096", labelsize=7, length=2)
            for spine in ax.spines.values():
                spine.set_color("#CBD5E1")
            ax.text(
                1 / 32,
                31 / 32,
                f"{fields[i]:.2f}°",
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
            ax.text(
                1 / 32,
                1 / 32,
                f"RMS  {_format_length_um(rms[i].item() * 1e3)} μm",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                fontsize="small",
                color="#0D1C2E",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#DCE4ED",
                    "alpha": 0.92,
                },
            )
        # Remove empty axes
        for i in range(n_fields, n_cols * n_rows):
            fig.delaxes(axes.flat[i])
        fig.supxlabel("x [\u03bcm]")
        fig.supylabel("y [\u03bcm]")
        axes.flat[0].legend(frameon=False, fontsize=6.5, loc="upper right")
        return fig


class PupilSampling(Visualization):
    """Plot the rays after being initialized at the entrance pupil, or pupil sampling scheme."""

    name = "pupil_sampling"

    def __init__(
        self,
        wavelength_indices: list[int] | None = None,
        field_indices: list[int] | None = None,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            wavelength_indices: Indices of the wavelengths to plot.
            field_indices: Indices of the fields to plot.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.wavelength_indices = (
            wavelength_indices if wavelength_indices is not None else slice(None, None)
        )
        self.field_indices = (
            field_indices if field_indices is not None else slice(None, None)
        )

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        if fig is None:
            fig = mpl.figure.Figure()
        ray_initialization = imaging_system.ray_initialization
        lens = imaging_system.lens
        r0, d0 = ray_initialization(lens)

        # Retrieve specs
        hfov = ray_initialization.hfov
        wavelengths = ray_initialization.wavelengths
        epd = ray_initialization.convert_to_absolute(lens).epd
        fields = np.linspace(0, hfov, ray_initialization.n_fields)[self.field_indices]
        wavelengths = np.array(wavelengths)[self.wavelength_indices]
        n_fields = len(fields)
        n_wavelengths = len(wavelengths)

        xy0 = r0[:2][:, self.field_indices][
            :, :, :, self.wavelength_indices
        ].cpu()  # [xy, field, ray, wavelength, 1]

        # Plot x and y for each field and wavelength
        axes = fig.subplots(
            n_wavelengths, n_fields, sharex=True, sharey=True, squeeze=False
        )
        for i, wavelength in enumerate(wavelengths):
            axes[i, 0].set_ylabel(f"{int(wavelength)} nm")
            for j, field in enumerate(fields):
                ax = axes[i, j]
                x, y = xy0[:, j, :, i]
                x = torch.cat((-x, x), dim=0)
                y = torch.cat((y, y), dim=0)
                ax.scatter(x, y, s=0.5, c="k")
                ax.set_aspect("equal")
                ax.tick_params(
                    bottom=False, labelbottom=False, left=False, labelleft=False
                )
                if i == 0:
                    axes[-1, j].set_xlabel(f"{field:0.1f}°")
                ax.set_aspect("equal")
                circle = mpl.patches.Circle(
                    (0, 0), epd / 2, fill=False, color="C1", lw=1
                )
                ax.add_patch(circle)

        return fig


class PSFs(Visualization):
    """Plot the point spread functions for the given fields and wavelengths."""

    name = "psfs"

    def __init__(
        self,
        wavelength_indices: list[int] | None = None,
        field_indices: list[int] | None = None,
        psf_sampler: sim.PSFSampler | None = None,
        sqrt_scale: bool = False,
        normalize_by_max: bool = True,
        cmap: str = "inferno",
        no_labels: bool = False,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            wavelength_indices: Indices of the wavelengths to plot.
            field_indices: Indices of the fields to plot.
            psf_sampler: PSF sampler to use, or None to use the imaging system's PSF sampler.
            sqrt_scale: Whether to visualize the square root of the PSFs.
            normalize_by_max: Whether to normalize the PSFs by their maximum.
            cmap: Colormap to use.
            no_labels: Whether to remove labels.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.wavelength_indices = (
            wavelength_indices if wavelength_indices is not None else slice(None, None)
        )
        self.field_indices = (
            field_indices if field_indices is not None else slice(None, None)
        )
        self.psf_sampler = psf_sampler
        self.sqrt_scale = sqrt_scale
        self.normalize_by_max = normalize_by_max
        self.cmap = cmap
        self.no_labels = no_labels

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
        eps: float = 1e-6,
    ):
        if fig is None:
            fig = mpl.figure.Figure()

        psf_sampler = self.psf_sampler
        specs = imaging_system.ray_initialization
        psfs = None

        # Without PSF sampler, get PSFs from lens module
        if psf_sampler is None:
            assert imaging_system.optics_simulator is not None, (
                "No OpticsSimulator available."
            )
            psf_sampler = imaging_system.optics_simulator.psf_sampler
            imaging_system.try_build_simulation_model()
            psfs = imaging_system.last_psfs
        else:
            psf_sampler = psf_sampler.to(imaging_system.device)
        if psfs is None:
            xy = imaging_system.compute_spot_diagrams()
            psfs = psf_sampler(xy)

        # Retrieve specs
        fields = np.linspace(0, specs.hfov, specs.n_fields)[self.field_indices]
        wavelengths = np.array(specs.wavelengths)[self.wavelength_indices]
        size = set(psf_sampler.psf_abs_size)
        assert len(size) == 1, "Only square PSFs are supported"
        size = next(iter(size))

        # Select PSFs
        psfs = psfs[self.field_indices][:, 0, self.wavelength_indices]

        # Compute RMS size of PSFs
        rms_size = compute_rms_size(psfs, size, size)

        # Normalize
        if self.normalize_by_max:
            psfs = psfs / psfs.view(*psfs.shape[:2], 1, -1).max(dim=-1, keepdim=True)[0]
        if self.sqrt_scale:
            psfs = psfs.clip(min=0).sqrt()

        # Create figure
        axes = fig.subplots(
            len(wavelengths), len(fields), sharex=True, sharey=True, squeeze=False
        )

        # Plot
        psfs = psfs.cpu()
        for i, wavelength in enumerate(wavelengths):
            if not self.no_labels:
                axes[i, 0].set_ylabel(f"{int(wavelength)} nm")
                labels = [
                    f"{row[i].item() * 1000:.1f}" + " \u03bcm" for row in rms_size
                ]
                if i == 0:
                    labels[0] = "RMS: " + labels[0]
            else:
                labels = None
            plot_psf_row(axes[i], psfs[:, i], labels, cmap=self.cmap)
        for j, field in enumerate(fields):
            if not self.no_labels:
                axes[-1, j].set_xlabel(f"{field:0.1f}°")

        # Scale bar
        if not self.no_labels:
            add_psf_scale_bar(axes[-1, 0], size)
        return fig


def compute_mtf_slices(psf, sample_pitch_mm):
    """从中心采样 PSF 返回正频率方向的弧矢与子午 MTF。"""
    psf = np.asarray(psf, dtype=float)
    if psf.ndim != 2 or min(psf.shape) < 3:
        raise ValueError("PSF must be a two-dimensional image.")
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    mtf = np.abs(otf)
    cy, cx = np.array(mtf.shape) // 2
    mtf /= max(mtf[cy, cx], np.finfo(float).eps)
    fx = np.fft.fftshift(np.fft.fftfreq(psf.shape[1], sample_pitch_mm))
    fy = np.fft.fftshift(np.fft.fftfreq(psf.shape[0], sample_pitch_mm))
    return fx[cx:], mtf[cy, cx:], fy[cy:], mtf[cy:, cx]


def diffraction_limited_mtf(frequency, wavelength_nm, f_number):
    """圆形非相干孔径的理论衍射极限 MTF。"""
    cutoff = 1.0 / (float(wavelength_nm) * 1e-6 * float(f_number))
    normalized = np.clip(np.asarray(frequency, dtype=float) / cutoff, 0.0, 1.0)
    mtf = 2.0 / np.pi * (
        np.arccos(normalized) - normalized * np.sqrt(1.0 - normalized**2)
    )
    return np.where(np.asarray(frequency) <= cutoff, mtf, 0.0)


def mtf_frequency_limit(wavelengths, f_number, sample_pitch):
    """Return the lower of the optical cutoff and sampled PSF Nyquist limit."""
    optical_cutoff = max(
        1.0 / (float(wavelength) * 1e-6 * f_number)
        for wavelength in wavelengths
    )
    return min(optical_cutoff, 0.5 / sample_pitch)


class WaveMTF(Visualization):
    """用 RayWave PSF 绘制弧矢、子午 MTF 与圆孔衍射极限。"""

    name = "mtf"

    def __init__(
        self,
        field_indices: list[int] | None = None,
        rc_params: dict | None = None,
    ):
        super().__init__(rc_params)
        self.field_indices = (
            field_indices if field_indices is not None else slice(None, None)
        )

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        del trainer
        simulator = imaging_system.optics_simulator
        if simulator is None or simulator.psf_mode != "ray_wave":
            raise ValueError("WaveMTF requires a RayWave optics simulator.")
        if imaging_system.last_psfs is None or not imaging_system.psfs_updated:
            imaging_system.try_build_simulation_model()

        psfs = imaging_system.last_psfs[self.field_indices, 0]
        mono = psfs / psfs.sum(dim=(-2, -1), keepdim=True).clip(min=1e-30)
        strehl = psfs.amax(dim=(-2, -1)).detach().cpu().numpy()

        specs = imaging_system.ray_initialization.convert_to_absolute(
            imaging_system.lens
        )
        epd = specs.epd
        if callable(epd):
            epd = epd(imaging_system.lens.efl)
        target_efl = imaging_system.parameterization.target_efl
        target_efl = float(torch.as_tensor(target_efl).abs().mean().cpu())
        f_number = target_efl / float(epd)
        wavelengths = np.asarray(specs.wavelengths, dtype=float)
        wavelength_colors = wavelengths2color(wavelengths)
        fields = np.linspace(0, specs.hfov, specs.n_fields)[self.field_indices]
        pitch = float(simulator.psf_abs_size[0] / simulator.psf_shape[0])

        if fig is None:
            fig = mpl.figure.Figure()
        fig.set_size_inches(6.0, max(3.8, 1.9 * len(fields)))
        axes = fig.subplots(
            len(fields), 1, sharex=True, sharey=True, squeeze=False
        )[:, 0]
        # PSF 网格无法表示超过奈奎斯特频率的信息，高频曲线必须在此截断。
        max_frequency = mtf_frequency_limit(wavelengths, f_number, pitch)
        mono = mono.detach().cpu().numpy()
        for field_index, (ax, field_psfs, field) in enumerate(
            zip(axes, mono, fields)
        ):
            for wavelength_index, (psf, wavelength, color) in enumerate(
                zip(field_psfs, wavelengths, wavelength_colors)
            ):
                fx, sagittal, fy, tangential = compute_mtf_slices(psf, pitch)
                frequency = fx[fx <= max_frequency * 1.05]
                ax.plot(
                    frequency,
                    sagittal[: len(frequency)],
                    color=color,
                    lw=1.55,
                    label=f"{wavelength:.1f} nm" if field_index == 0 else None,
                )
                ax.plot(
                    frequency,
                    tangential[: len(frequency)],
                    color=color,
                    lw=1.35,
                    linestyle="--",
                )
                # 每个波长使用各自的衍射截止频率，不能以主波长代替。
                ax.plot(
                    frequency,
                    diffraction_limited_mtf(frequency, wavelength, f_number),
                    color=color,
                    lw=1.0,
                    linestyle=":",
                    alpha=0.9,
                )
            ax.set(
                xlim=(0, max_frequency * 1.05),
                ylim=(0, 1.03),
                ylabel="MTF",
            )
            ax.set_title(
                f"F{field_index + 1}  ·  {field:.2f}°",
                loc="left",
                fontsize=8,
                color="#0D1C2E",
            )
            ax.text(
                0.99,
                0.92,
                "SR  " + " / ".join(f"{value:.3f}" for value in strehl[field_index]),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                color="#526176",
            )
            ax.set_facecolor("#F8FAFC")
            ax.grid(True, color="#DCE4ED", linewidth=0.7, alpha=0.8)
            ax.tick_params(colors="#718096", labelsize=7, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[-1].set_xlabel("Spatial frequency [lp/mm]")
        wavelength_handles = [
            mpl.lines.Line2D([], [], color=color, lw=1.6, label=f"{wavelength:.1f} nm")
            for wavelength, color in zip(wavelengths, wavelength_colors)
        ]
        style_handles = [
            mpl.lines.Line2D([], [], color="#526176", lw=1.5, label="S"),
            mpl.lines.Line2D(
                [], [], color="#526176", lw=1.5, linestyle="--", label="T"
            ),
            mpl.lines.Line2D(
                [], [], color="#526176", lw=1.2, linestyle=":", label="Diff."
            ),
        ]
        axes[0].legend(
            handles=wavelength_handles + style_handles,
            frameon=False,
            fontsize=6,
            ncol=3,
            loc="lower left",
        )
        return fig


class RGBPSFs(Visualization):
    """Plot the RGB point spread functions for the given fields."""

    name = "rgb_psfs"

    def __init__(
        self,
        field_indices: list[int] | None = None,
        sqrt_scale: bool = False,
        normalize_by_max: bool = True,
        cmap: str = "inferno",
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            field_indices: Indices of the fields to plot.
            sqrt_scale: Whether to visualize the square root of the PSFs.
            normalize_by_max: Whether to normalize the PSFs by their maximum.
            cmap: Colormap to use.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.field_indices = (
            field_indices if field_indices is not None else slice(None, None)
        )
        self.sqrt_scale = sqrt_scale
        self.normalize_by_max = normalize_by_max
        self.cmap = cmap

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
        eps: float = 1e-6,
    ):
        if fig is None:
            fig = mpl.figure.Figure()

        optics_simulator = imaging_system.optics_simulator
        assert optics_simulator is not None, "No OpticsSimulator available."
        if not imaging_system.psfs_updated:
            imaging_system.try_build_simulation_model()
        rgb_psfs = imaging_system.last_rgb_psfs

        # Retrieve specs
        hfov = imaging_system.ray_initialization.hfov
        fields = np.linspace(0, hfov, imaging_system.ray_initialization.n_fields)[
            self.field_indices
        ]
        size = set(optics_simulator.psf_sampler.psf_abs_size)
        assert len(size) == 1, "Only square PSFs are supported"
        size = next(iter(size))

        # Select PSFs
        rgb_psfs = rgb_psfs[self.field_indices][:, 0, :]

        # Compute RMS size of PSFs
        rms_size = compute_rms_size(rgb_psfs, size, size)

        # Normalize
        if self.normalize_by_max:
            rgb_psfs = (
                rgb_psfs
                / rgb_psfs.view(*rgb_psfs.shape[:2], 1, -1).max(dim=-1, keepdim=True)[0]
            )
        if self.sqrt_scale:
            rgb_psfs = rgb_psfs.clip(min=0).sqrt()

        # Create figure
        axes = fig.subplots(3, len(fields), sharex=True, sharey=True, squeeze=False)

        # Plot
        rgb_psfs = rgb_psfs.cpu()
        for i, channel in enumerate("RGB"):
            axes[i, 0].set_ylabel(channel, rotation=0, ha="right", va="center")
            labels = [f"{row[i].item() * 1000:.1f}" + " \u03bcm" for row in rms_size]
            if i == 0:
                labels[0] = "RMS: " + labels[0]
            plot_psf_row(axes[i], rgb_psfs[:, i], labels, cmap=self.cmap)
        for j, field in enumerate(fields):
            axes[-1, j].set_xlabel(f"{field:0.1f}°")

        # Scale bar
        add_psf_scale_bar(axes[-1, 0], size)
        return fig


class CombinedRGBPSFs(Visualization):
    """Plot the combined RGB PSFs (single plot for all color channels) for the given fields."""

    name = "combined_rgb_psfs"

    def __init__(
        self, field_indices: list[int] | None = None, rc_params: dict | None = None
    ):
        """Constructor.

        Args:
            field_indices: Indices of the fields to plot.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.field_indices = (
            field_indices if field_indices is not None else slice(None, None)
        )

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
        eps: float = 1e-6,
    ):
        if fig is None:
            fig = mpl.figure.Figure()

        optics_simulator = imaging_system.optics_simulator
        assert optics_simulator is not None, "No OpticsSimulator available."
        if not imaging_system.psfs_updated:
            imaging_system.try_build_simulation_model()
        rgb_psfs = imaging_system.last_rgb_psfs

        # Retrieve specs
        hfov = imaging_system.ray_initialization.hfov
        fields = np.linspace(0, hfov, imaging_system.ray_initialization.n_fields)[
            self.field_indices
        ]
        size = set(optics_simulator.psf_sampler.psf_abs_size)
        assert len(size) == 1, "Only square PSFs are supported"
        size = next(iter(size))

        # Select PSFs
        rgb_psfs = rgb_psfs[self.field_indices][:, 0, :]

        # Normalize by max
        rgb_psfs = rgb_psfs / rgb_psfs.flatten(-3).max(dim=-1).values.view(-1, 1, 1, 1)

        # Compute RMS size of PSFs
        rms_size = compute_rms_size(
            rgb_psfs, size, size, additional_reduce_dimension=-3
        )

        # Create figure
        axes = fig.subplots(1, len(fields), sharex="all", sharey="all", squeeze=False)
        axes = axes.flatten()

        # Plot
        rgb_psfs = rgb_psfs.cpu()
        labels = []
        for j, field in enumerate(fields):
            ax = axes[j]
            ax.set_xlabel(f"{field:0.1f}°")
            labels.append(f"{rms_size[j].item() * 1000:.1f}" + " \u03bcm")
            if j == 0:
                labels[-1] = "RMS: " + labels[-1]
        plot_psf_row(axes, rgb_psfs.permute(0, 2, 3, 1), labels, cmap=None)

        # Scale bar
        add_psf_scale_bar(axes[-1], size)
        return fig


class PSFGrid(Visualization):
    """Plot the PSF grid for the given sensor region."""

    name = "psf_grid"

    def __init__(
        self,
        optics_simulator: sim.OpticsSimulator | None = None,
        ray_initialization: ri.RayInitialization | None = None,
        tile_idx_rows_cols: tuple[int, int, int] | None = None,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            optics_simulator: Optics simulator to use, or None to use the imaging system's.
            ray_initialization: Ray initialization to use, or None to use the imaging system's.
            tile_idx_rows_cols: Tile index of the sensor region and number of rows and columns of the tile layout;
                if None, the PSF grid corresponds to the entire sensor.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.optics_simulator = optics_simulator
        self.ray_initialization = ray_initialization
        self.tile_idx_rows_cols = tile_idx_rows_cols

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
        eps: float = 1e-6,
    ):
        optics_simulator = self.optics_simulator
        ray_initialization = self.ray_initialization
        rgb_psfs = None

        # Without optics simulator or ray initialization, get PSFs from lens module
        if optics_simulator is None and ray_initialization is None:
            optics_simulator = imaging_system.optics_simulator
            assert optics_simulator is not None, "No OpticsSimulator available."
            if not imaging_system.psfs_updated:
                imaging_system.try_build_simulation_model()
            rgb_psfs = imaging_system.last_rgb_psfs

        # Otherwise, recompute them
        if rgb_psfs is None:
            if ray_initialization is None:
                ray_initialization = imaging_system.ray_initialization
            if optics_simulator is None:
                optics_simulator = imaging_system.optics_simulator
                assert optics_simulator is not None, "No OpticsSimulator available."
            # Build model
            lens = imaging_system.lens
            _, rgb_psfs = optics_simulator.build_optics_model(lens, ray_initialization)

        rgb_psfs = rgb_psfs.squeeze(dim=1)  # Assume only 1 lens

        # Retrieve PSF grid
        if self.tile_idx_rows_cols is None:
            # Use default image size of optics simulator; compute PSF grid for entire image instead of image patch
            field_lims = None
            im_size = optics_simulator.default_image_size
        else:
            field_lims = (
                utils.compute_field_lims(*self.tile_idx_rows_cols)
                .to(rgb_psfs)
                .view(1, -1)
            )
            rows_cols = np.array(self.tile_idx_rows_cols[1:3])
            im_size = np.round(optics_simulator.default_image_size / rows_cols).astype(
                int
            )
        psf_grid = optics_simulator.compute_psf_grid(rgb_psfs, *im_size, field_lims)

        # Create mask to select only a subset of PSFs
        grid_h, grid_w = optics_simulator.psf_grid_shape
        select_mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        if self.tile_idx_rows_cols is None:
            # Due to x-y symmetry, we only show the PSFs corresponding to the upper right quadrant
            n_rows = (grid_h + 1) // 2
            n_cols = (grid_w + 1) // 2
        else:
            # Without x-y symmetry, we display all PSFs
            n_rows = grid_h
            n_cols = grid_w
        select_mask[:n_rows, -n_cols:] = True
        psf_grid = psf_grid.view(grid_h, grid_w, *psf_grid.shape[2:])[select_mask]

        # Make grid
        img = torchvision.utils.make_grid(
            psf_grid, nrow=n_cols, normalize=True, scale_each=True, pad_value=1.0
        )
        return (img.cpu().numpy() * 255.0).astype(np.uint8)


class OpticsSimulation(Visualization):
    """Simulate aberrations on a given image and plot the result."""

    name = "simulation"

    def __init__(
        self,
        image_path: str,
        apply_image_restoration: bool = True,
        field_lims: tuple[float, float, float, float] | None = None,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            image_path: Path to the image to simulate aberrations on.
            apply_image_restoration: Whether to apply the image restoration model.
            field_lims: Field limits to simulate aberrations for.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.image_path = image_path
        self.apply_image_restoration = apply_image_restoration
        self.field_lims = field_lims

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        image = np.moveaxis(np.asarray(PIL.Image.open(self.image_path)), 2, 0)[:3]
        image = torch.tensor(np.array(image) / 255.0, device=imaging_system.device)[
            None, ...
        ]

        # Simulate aberrations
        field_lims = self.field_lims
        if field_lims is not None:
            field_lims = torch.tensor([self.field_lims]).to(
                dtype=imaging_system.dtype, device=imaging_system.device
            )
        output, psf_grid = imaging_system.simulate_aberrations(
            image, field_lims=field_lims
        )

        # Noise
        output = imaging_system.apply_noise(output)

        # Apply image restoration
        if (
            self.apply_image_restoration
            and imaging_system.image_restoration_model is not None
        ):
            output = imaging_system.image_restoration_model(output, psf_grid)

        output = output.clip(0, 1)
        output_image = (output[0].cpu().numpy() * 255.0).astype(np.uint8)
        return output_image


class MetricPlot(Visualization):
    """Plot a given metric."""

    name = "metric"

    def __init__(
        self,
        metric_name: str,
        ylabel: str,
        ylim: tuple[float, float],
        ylog: bool = False,
        rc_params: dict | None = None,
    ):
        """Constructor.

        Args:
            metric_name: Name of the metric to plot.
            ylabel: Label of the y-axis.
            ylim: Limits of the y-axis.
            ylog: Whether to use a logarithmic scale for the y-axis.
            rc_params: Matplotlib rcParams.
        """
        super().__init__(rc_params)
        self.metric_name = metric_name
        self.x_history = []
        self.y_history = []
        self.ylabel = ylabel
        self.ylim = ylim
        self.ylog = ylog

    def generate_figure(
        self,
        trainer: pl.Trainer,
        imaging_system: ImagingSystemModule,
        fig: mpl.figure.Figure = None,
    ):
        if fig is None:
            fig = mpl.figure.Figure()
        max_steps = trainer.max_steps
        step = trainer.global_step
        metric = (
            imaging_system.metrics[self.metric_name].item()
            if self.metric_name in imaging_system.metrics
            else None
        )
        ax = fig.add_subplot()
        if metric is not None:
            self.x_history.append(step)
            self.y_history.append(metric)
            ax.plot(self.x_history, self.y_history)
        ax.set_xlabel("Step")
        ax.set_ylabel(self.ylabel)
        ax.grid(axis="y", which="both")
        if self.ylog:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(-max_steps / 32, max_steps * 33 / 32)
        ax.set_ylim(*self.ylim)
        return fig


def plot_to_image(fig: mpl.figure.Figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.

    The supplied figure is closed and inaccessible after this call.

    Args:
        fig: Matplotlib figure to convert to an image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = np.moveaxis(np.asarray(PIL.Image.open(buf)), 2, 0)
    return img


def estimate_misc_surface_radius(
    lens: optics.Lens, idx: int, ray_initialization: ri.RayInitialization
):
    """Estimate the radius of a misc surface in the system.

    Args:
        lens: Lens system to estimate the misc surface radius for.
        idx: Index of the misc surface to estimate the radius.
        ray_initialization: Ray initialization to use for the estimation.
    """
    # Find the index of the misc surface in the system
    m_idx = i = 0
    for event_type, *_ in lens.return_geometry():
        if event_type == "m":
            if m_idx == idx:
                break
            m_idx += 1
        i += 1
    # Find the corresponding diameter by ray tracing
    wavelengths = torch.tensor(ray_initialization.wavelengths).to(lens.c)
    r, d = ray_initialization(
        lens,
        n_fields=2,
        wavelengths=wavelengths,
        pupil_sampling_mode="skew_outer_edge_uniform",
        n_rays=32,
    )
    diameter = lens.estimate_diameters(r, d, wavelengths)[i].item()
    max_radius = diameter / 2
    return max_radius


def compute_rms_size(
    psfs: torch.Tensor,
    size_x: int | None = None,
    size_y: int | None = None,
    additional_reduce_dimension: int | None = None,
    eps: float = 1e-6,
):
    """Calculate the RMS size of given PSFs, with optional reduction over a specified dimension.

    Args:
        psfs: PSFs to calculate the RMS size of (shape: [*, H, W]).
        size_x: Size of the PSFs in the x direction.
        size_y: Size of the PSFs in the y direction.
        additional_reduce_dimension: Additional dimension to reduce the PSFs along.
        eps: Small value to avoid division by zero.
    """
    if size_x is None:
        size_x = psfs.shape[-1]
    if size_y is None:
        size_y = psfs.shape[-2]

    reduce_dims = (-2, -1)
    if additional_reduce_dimension is not None:
        reduce_dims = reduce_dims + (additional_reduce_dimension,)

    # Calculate the sum along the spatial dimensions to normalize
    psf_sum = psfs.sum(dim=reduce_dims, keepdim=True)

    # Normalize PSF
    psfs = psfs / psf_sum.clip(min=eps)

    # Get the grid for calculating the centroid
    h, w = psfs.shape[-2:]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h).to(psfs), torch.arange(w).to(psfs), indexing="ij"
    )

    # Calculate the centroid
    x_centroid = (psfs * x_grid).sum(dim=reduce_dims, keepdim=True)
    y_centroid = (psfs * y_grid).sum(dim=reduce_dims, keepdim=True)

    # Calculate the squared distances from the centroid
    x_dist = (x_grid - x_centroid) * size_x / w
    y_dist = (y_grid - y_centroid) * size_y / h
    sq_distances = x_dist**2 + y_dist**2

    # Calculate the mean squared distance weighted by the PSF
    mean_sq_distance = (psfs * sq_distances).sum(dim=reduce_dims)

    # RMS size is the square root of the mean squared distance
    rms_size = mean_sq_distance.sqrt()

    return rms_size


def plot_psf_row(
    ax_list: list[mpl.axes.Axes],
    psf_list: list[torch.Tensor],
    labels: list[str] | None = None,
    cmap: str = "inferno",
):
    """Plot a row of PSFs.

    Args:
        ax_list: List of axes to plot the PSFs on.
        psf_list: List of PSFs to plot.
        labels: List of labels for the PSFs.
        cmap: Colormap to use.
    """
    if labels is None:
        labels = [None] * len(ax_list)
    elif isinstance(labels, str):
        labels = [labels] * len(ax_list)
    for ax, psf, label in zip(ax_list, psf_list, labels):
        ax.imshow(psf, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_aspect("equal")
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        if label is not None:
            kwargs = dict(
                ha="center", va="top", transform=ax.transAxes, color="w", size="x-small"
            )
            ax.text(0.5, 31 / 32, label, **kwargs)


def add_psf_scale_bar(ax: mpl.axes.Axes, size: float):
    """Add a scale bar to a PSF plot.

    Args:
        ax: Axes to add the scale bar to.
        size: Size of the PSF.
    """
    should_round = ((size * 1000) + 0.025) % 1 < 0.05
    size_label = ("{:.0f}" if should_round else "{:.1f}").format(
        size * 1000
    ) + " \u03bcm"
    ax.text(
        0.5,
        1 / 32,
        size_label,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        color="w",
        size="x-small",
    )
    ax.annotate(
        "",
        xy=(0, 1 / 32),
        xytext=(1, 1 / 32),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<|-|>", color="w", lw=3 / 8, shrinkA=0, shrinkB=0),
    )


def generate_layout_plot(
    lens: optics.Lens,
    ray_initialization: ri.RayInitialization,
    wavelengths: list[float],
    n_rays: int = 3,
    n_fields: int = 2,
    glass_labels: list[str] | None = None,
    diameter_scaler: float = 17 / 16,
    scale_bar: bool = True,
    fig: mpl.figure.Figure = None,
):
    """Generate a layout plot of the given lens system.

    Args:
        lens: Lens system to plot.
        ray_initialization: Ray initialization to use for the plot.
        wavelengths: Wavelengths to use for the plot.
        n_rays: Number of rays to use for the plot.
        n_fields: Number of fields to use for the plot.
        glass_labels: Labels for the glass elements in the lens system.
        diameter_scaler: Scaler for the diameter of the refractive elements.
        scale_bar: Whether to include a scale bar in the plot.
        fig: Matplotlib figure to plot on.
    """
    if fig is None:
        fig = mpl.figure.Figure()
    ax = fig.add_subplot()

    r, d = ray_initialization(
        lens,
        n_fields=n_fields,
        wavelengths=wavelengths,
        pupil_sampling_mode="meridional_uniform",
        n_rays=n_rays,
    )
    plot_rays(ax, lens, r, d, wavelengths)
    r, d = ray_initialization(
        lens,
        n_fields=2,
        wavelengths=wavelengths,
        pupil_sampling_mode="skew_outer_edge_uniform",
        n_rays=32,
    )
    diameters = lens.estimate_diameters(r, d, wavelengths)
    plot_layout(
        ax,
        lens,
        diameters,
        scale_bar=scale_bar,
        glass_labels=glass_labels,
        diameter_scaler=diameter_scaler,
    )
    return fig


def generate_spot_plot(
    xy: torch.Tensor,
    ray_valid: torch.Tensor,
    wavelengths: list[float] | tuple[float, ...],
    hfov: float,
    fig: mpl.figure.Figure = None,
):
    """Plot image-plane intercepts produced by :meth:`Lens.trace_rays`.

    This lightweight entry point uses the same centering convention and units as
    ``SpotDiagrams`` without requiring a Lightning trainer or imaging module.
    """
    n_fields = xy.shape[1]
    fields = np.linspace(0, hfov, n_fields)
    colors = wavelengths2color(np.asarray(wavelengths))
    if fig is None:
        fig = mpl.figure.Figure(figsize=(2.8 * n_fields, 2.8))
    axes = fig.subplots(1, n_fields, squeeze=False)[0]
    max_extent = 0.0
    for field, ax in enumerate(axes):
        all_points = []
        for wave, (wavelength, color) in enumerate(zip(wavelengths, colors)):
            mask = ray_valid[field, :, wave]
            points = xy[:, field, mask, wave]
            if points.numel() == 0:
                continue
            all_points.append(points)
            center = points.mean(dim=1, keepdim=True)
            centered = (points - center).detach().cpu() * 1e3
            ax.scatter(
                centered[0],
                centered[1],
                s=4,
                color=color,
                alpha=0.72,
            )
            max_extent = max(max_extent, float(centered.abs().max()))
        if all_points:
            points = torch.cat(all_points, dim=1)
            center = points.mean(dim=1, keepdim=True)
            rms = float((points - center).square().sum(0).mean().sqrt()) * 1e3
            max_extent = max(
                max_extent,
                float(((points - center) * 1e3).abs().max()),
            )
            label = f"{fields[field]:.2f}° · RMS {rms:.2f} μm"
        else:
            label = f"{fields[field]:.2f}° · RMS —"
        ax.text(
            0.5,
            0.02,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="black",
            fontsize=8,
        )
        ax.set_axis_off()
    limit = max(max_extent * 1.08, 1e-3)
    for ax in axes:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.04)
    return fig


def plot_layout(
    ax: mpl.axes.Axes,
    lens: optics.Lens,
    diameters: torch.Tensor,
    stop_color: str = "k",
    scale_bar: bool = False,
    glass_labels: list[str] | None = None,
    diameter_scaler: float = 1.0,
    **kwargs,
):
    """Plot the layout of the given lens system.

    Args:
        ax: Axes to plot on.
        lens: Lens system to plot.
        diameters: Diameters of the lens elements.
        stop_color: Color of the aperture stop.
        scale_bar: Whether to include a scale bar in the plot.
        glass_labels: Labels for the glass elements in the lens system.
        diameter_scaler: Scaler for the diameter of the refractive elements.
        kwargs: Additional keyword arguments to pass to the plot
    """
    assert len(lens) == 1
    if glass_labels is not None:
        assert len(glass_labels) == lens.sequence.n_refractive

    default_kwargs = dict(solid_capstyle="round", c="k")
    default_kwargs.update(kwargs)
    doe_params = {"path.sketch": (0.25, 3, 1)}

    min_z = max_z = min_y = max_y = 0

    last_y = last_edge_z = last_vertex_z = 0
    refractive_element_idx = 0

    geometry = list(lens.return_geometry())
    surface_mounted_stop = "-sR" in lens.sequence.sequence or "Rs-" in lens.sequence.sequence
    for k, ((surface_type, z, sag_fn, is_refractive), diameter) in enumerate(
        zip(geometry, diameters)
    ):
        diameter = diameter.item()
        diameter = diameter * diameter_scaler
        z = z.item()
        y = diameter / 2

        # Draw the surface
        if surface_type == "s":
            color = stop_color
            ratio = 7 / 6
            if surface_mounted_stop:
                neighbor = k + 1 if "-sR" in lens.sequence.sequence else k - 1
                if 0 <= neighbor < len(diameters):
                    y = max(y, float(diameters[neighbor]) * diameter_scaler / 2)
            left_z = right_z = current_edge_z = z
            ax.vlines([z, z], [-y, y], [-y * ratio, y * ratio], colors=color, zorder=-1)
            y = y * ratio
        else:
            yy = (
                torch.linspace(-y, y, 128).to(y)
                if sag_fn
                else torch.linspace(-y, y, 2).to(y)
            )
            sag = sag_fn(yy) if sag_fn else torch.zeros_like(yy)
            left_z = z + sag.min().item()
            right_z = z + sag.max().item()
            zz = z + sag.cpu()
            current_edge_z = zz[-1]
            color = "steelblue" if surface_type in ("d", "m") else "k"
            with mpl.rc_context(doe_params if surface_type in ("d", "m") else {}):
                ax.plot(zz, yy, **(default_kwargs | {"c": color}))

        # Close the glass elements
        if is_refractive:
            highest_y = max(last_y, y)
            # Vertical lines
            if last_y != highest_y:
                ax.plot(
                    [last_edge_z, last_edge_z], [last_y, highest_y], **default_kwargs
                )
                ax.plot(
                    [last_edge_z, last_edge_z], [-last_y, -highest_y], **default_kwargs
                )
            if y != highest_y:
                ax.plot(
                    [current_edge_z, current_edge_z], [y, highest_y], **default_kwargs
                )
                ax.plot(
                    [current_edge_z, current_edge_z], [-y, -highest_y], **default_kwargs
                )
            # Horizontal line
            ax.plot(
                [last_edge_z, current_edge_z], [highest_y, highest_y], **default_kwargs
            )
            ax.plot(
                [last_edge_z, current_edge_z],
                [-highest_y, -highest_y],
                **default_kwargs,
            )
            if glass_labels is not None:
                glass_label = glass_labels[refractive_element_idx]
                z_glass_pos = (last_vertex_z + z) / 2
                ax.annotate(
                    glass_label,
                    (z_glass_pos, 0),
                    ha="center",
                    va="center",
                    rotation=90,
                    textcoords="offset points",
                    xytext=(0, 0),
                    bbox=dict(
                        boxstyle="round,pad=.125",
                        edgecolor="black",
                        facecolor="white",
                        alpha=0.5,
                        lw=0.5,
                    ),
                )
            refractive_element_idx += 1

        # 光阑不是镜片表面，不能覆盖下一条玻璃闭合边的起点。
        if surface_type != "s":
            last_y = y
            last_edge_z = current_edge_z
            last_vertex_z = z
        min_z = min(min_z, left_z)
        max_z = max(max_z, right_z)
        min_y = min(min_y, -y)
        max_y = max(max_y, y)

    # Format ax and set plot limits
    # ax.axis('off')
    width = max_z - min_z
    z_pad = width / 128
    y_pad = max_y / 64
    ax.set_xlim(min_z - z_pad, max_z + z_pad)
    ax.set_ylim(min_y - y_pad, max_y + y_pad)
    ax.set_xticks(())
    ax.set_yticks(())
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    ax.tick_params(
        left=False,
        labelbottom=False,
        labelleft=False,
        colors="silver",
        direction="inout",
        which="both",
    )

    if scale_bar:
        ax.spines["bottom"].set_color("silver")
        ax.spines["bottom"].set_visible(True)
        ax.set_ylim(ax.get_ylim()[0] - 5 * y_pad, ax.get_ylim()[1])
        ticks = np.arange(max_z, min_z, -1)[::-1]
        ax.set_xticks(ticks[::-5], minor=False)
        ax.set_xticks(ticks, minor=True)
        label_right = True
        if label_right:
            ax.text(
                max_z - 3 * z_pad,
                -max_y - 2 * y_pad,
                "[mm]",
                va="bottom",
                ha="right",
                color="silver",
            )
        else:
            ax.text(
                min_z + 3 * z_pad,
                -max_y - 2 * y_pad,
                "[mm]",
                va="bottom",
                ha="left",
                color="silver",
            )


def wavelengths2color(
    wavelengths: list[float] | np.ndarray,
    cmap: mpl.colors.Colormap = mpl.colormaps["nipy_spectral"],
):
    """Convert wavelengths to colors using a colormap.

    By default, the colormap is 'nipy_spectral'.
    This is a good choice for visualizing wavelengths in the visible spectrum.

    Args:
        wavelengths: Wavelengths to convert to colors.
        cmap: Colormap to use.
    """
    return [cmap((w - 400) / 300) for w in wavelengths]


def plot_rays(
    ax: mpl.axes.Axes,
    lens: optics.Lens,
    r: torch.Tensor,
    d: torch.Tensor,
    wavelengths: list[float],
):
    """Plot rays on a given lens system.

    Args:
        ax: Axes to plot on.
        lens: Lens system used to trace the rays.
        r: Initial ray position vectors.
        d: Initial ray direction vectors.
        wavelengths: Wavelengths of the rays.
    """
    # Trace rays
    r_list = []
    status_list = []
    for r, d, status, *_ in lens.trace_rays(r, d, wavelengths, yield_on="position"):
        r_list.append(r)
        status_list.append(status)

    # Replace invalid ray coordinates with NaNs
    r_stack = torch.stack(r_list, dim=1)
    validity_mask = torch.stack(status_list, dim=0) < 2
    validity_mask = validity_mask[None, ...].expand_as(r_stack)
    r_stack[~validity_mask] = np.nan

    # Unpack ray coordinates
    x_stack, y_stack, z_stack = r_stack.cpu()
    n_surfaces = len(r_list)
    n_rays = z_stack[0].numel()

    # Plot lines
    ax.plot(
        z_stack.view(n_surfaces, -1).numpy(),
        y_stack.view(n_surfaces, -1).numpy(),
        alpha=1 / 2,
        linewidth=0.55,
    )

    # Edit colors according to wavelength
    colors = wavelengths2color(np.array(wavelengths * (n_rays // len(wavelengths))))
    [line.set_color(color) for line, color in zip(ax.lines[-n_rays:], colors)]
