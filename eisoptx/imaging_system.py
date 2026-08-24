import torch
import torch.utils.data
import torch.func
import torch.optim
import torchvision.utils
import torchmetrics.image
import pytorch_lightning as pl
from pytorch_lightning import cli, loggers
import numpy as np

from eisoptx.modeling import (
    ray_initialization as ri,
    ray_analysis as ra,
    simulation as sim,
    optics,
)
from eisoptx.optimization import parameterization as param, residuals as res


class ImagingSystemModule(pl.LightningModule):
    """Class to optimize and test imaging systems composed of lenses and image restoration models."""

    def __init__(
        self,
        # Lens optimization
        lens_parameterization: param.LensParameterization | None = None,
        ray_initialization: ri.RayInitialization | None = None,
        lens_optimizer: cli.OptimizerCallable | None = None,
        lens_lr_scheduler: cli.LRSchedulerCallable | None = None,
        residuals: list[res.Residuals] = (),
        # Image-driven lens optimization
        optics_simulator: sim.OpticsSimulator | None = None,
        noise_sigma: float | None = None,
        image_restoration_model: torch.nn.Module | None = None,
        e2e_vector_mode: bool | None = None,
        e2e_loss_weight: float | None = None,
        e2e_loss_fn: str | tuple[str, float] = "mae",
        # Joint optimization of image restoration model
        irm_e2e_loss_fn: str | tuple[str, float] = None,
        irm_optimizer: cli.OptimizerCallable | None = None,
        irm_lr_scheduler: cli.LRSchedulerCallable | None = None,
    ):
        """Constructor.

        Args:
            lens_parameterization: LensParameterization instance to represent the lens.
            ray_initialization: RayInitialization instance to initialize rays based on system specifications.
            lens_optimizer: Optimizer for the lens parameters; if not provided, the lens is not optimized.
            lens_lr_scheduler: Learning rate scheduler for the lens optimizer (does not work with LMOptimizer).
            residuals: List of Residuals objects to build the least-squares optimization objective;
                if there are duplicates, only the last occurrence is kept.
            optics_simulator: OpticsSimulator instance to simulate aberrations on provided images.
            noise_sigma: Standard deviation of the Gaussian noise to be added to the simulated images.
            image_restoration_model: Image restoration model (IRM) to be used in the end-to-end pipeline.
            e2e_vector_mode: Whether to use generalized transverse ray aberrations (True) or the scalar loss (False);
                if None, end-to-end optimization is disabled.
            e2e_loss_weight: Weight of the end-to-end loss; if None, it is only monitored.
            e2e_loss_fn: Image discrepancy loss function for end-to-end optimization;
                either a string ('mae', 'mse') or a tuple ('truncated_mae', truncation_value).
            irm_e2e_loss_fn: Image discrepancy loss function for the IRM; if None, the same loss as e2e_loss_fn is used.
            irm_optimizer: Optimizer for the image restoration model; if not provided, the IRM is not optimized.
            irm_lr_scheduler: Learning rate scheduler for the image restoration model optimizer.
        """
        super().__init__()
        self.automatic_optimization = False

        # Initialize lens
        self.parameterization = lens_parameterization
        if lens_parameterization is None or ray_initialization is None:
            assert lens_optimizer is None, (
                "Lens optimizer must be None if lens parameterization is not provided."
            )
            assert lens_lr_scheduler is None, (
                "Lens LR scheduler must be None if lens parameterization is not provided."
            )
            assert len(residuals) == 0, (
                "Residuals must be empty if lens parameterization is not provided."
            )
            assert optics_simulator is not None, (
                "OpticsSimulator must be provided if lens parameterization is not provided."
            )

        # Residuals (duplicated entries are overwritten by the last occurrence)
        self.residuals = list(
            {residual.name: residual for residual in residuals}.values()
        )
        self.constraint_dict = {
            residual.name: residual.constraint for residual in self.residuals
        }
        self.weight_dict = {
            residual.name: residual.weight for residual in self.residuals
        }
        # Find what intermediate ray-tracing data needs to be collected to compute residuals
        self.data_collection_keys = set(
            k for residual in self.residuals for k in residual.data_keys
        )

        # Image-driven loss function
        self.lens_e2e_loss_fn = get_image_discrepancy_loss_fn(e2e_loss_fn)
        if irm_e2e_loss_fn is None:
            self.irm_e2e_loss_fn = self.lens_e2e_loss_fn
        else:
            self.irm_e2e_loss_fn = get_image_discrepancy_loss_fn(irm_e2e_loss_fn)
        self.e2e_vector_mode = e2e_vector_mode
        self.e2e_loss_weight = e2e_loss_weight
        if self.e2e_vector_mode is not None:
            assert optics_simulator is not None, (
                "OpticsSimulator must be provided if end-to-end losses are used."
            )
            self.e2e_enabled = True
            name = (
                "generalized_transverse_ray_aberration"
                if self.e2e_vector_mode
                else "e2e_scalar_loss"
            )
            self.constraint_dict[name] = False
            self.weight_dict[name] = e2e_loss_weight
        else:
            self.e2e_enabled = False
            assert e2e_loss_weight is None, (
                "Value end_to_end_loss_weight must be None if end-to-end mode is disabled."
            )

        # Lens simulation
        self.ray_initialization = ray_initialization
        if (
            ray_initialization is not None
            and ray_initialization.wavelength_weights is not None
        ):
            wavelength_weights = torch.tensor(
                ray_initialization.wavelength_weights
            ).view(-1, 1)
            self.register_buffer("wavelength_weights", wavelength_weights)
        else:
            self.wavelength_weights = None
        self.optics_simulator = optics_simulator

        # Image restoration model
        self.image_restoration_model = image_restoration_model
        self.noise_sigma = noise_sigma

        # Optimizers
        self.lens_optimizer = lens_optimizer
        self.irm_optimizer = irm_optimizer
        self.lens_lr_scheduler = lens_lr_scheduler
        self.irm_lr_scheduler = irm_lr_scheduler
        self.lens_optimization_disabled = (
            False  # Flag to externally disable lens optimization
        )
        self.irm_optimization_disabled = (
            False  # Flag to externally disable IRM optimization
        )

        # Logging
        self.metrics = {}
        self.last_xy = self.last_psfs = self.last_rgb_psfs = None
        self.xy_updated = self.psfs_updated = False

    @property
    def lens(self):
        """Lens instance."""
        return self.parameterization.lens

    def compute_residuals_dict_and_logs(
        self, batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
    ):
        """Compute residuals and logs for the current lens configuration.

        Args:
            batch: Tuple of images and field limits; if None, only the end-to-end loss is computed.
        """
        lens = self.lens

        # Initialize rays from specifications
        ray_initialization = self.ray_initialization.convert_to_absolute(lens)
        r0, d0 = ray_initialization(lens)
        wavelengths = ray_initialization.wavelengths

        # Trace rays and gather intermediate information for the penalty terms
        keys = self.data_collection_keys
        rt_info = {k: [] for k in keys}
        for r, d, ray_status, event_info in lens.trace_rays(
            r0, d0, wavelengths, yield_on="all"
        ):
            for k, v in event_info.items():
                if k in self.data_collection_keys:
                    rt_info[k].append(v)

        # Stack lists
        rt_info = {
            k: torch.stack(v, dim=-1)
            if len(v) > 0
            else torch.tensor([], device=r0.device)
            for k, v in rt_info.items()
        }
        xy = r[:2]
        pupil_weights = None
        if ray_initialization.pupil_sampling_mode == "skew_uniform_zonal":
            epd = ray_initialization.epd
            if callable(epd):
                epd = epd(lens.efl)
            pupil_weights = ri.zonal_pupil_weights(
                zone_edges=ri.zone_edges_from_lens(lens, epd),
                **ray_initialization.pupil_sampling_kwargs,
            ).to(xy).view(1, -1, 1, 1)
        ray_weights = self.wavelength_weights
        if pupil_weights is not None:
            ray_weights = ray_weights * pupil_weights
        y_centroid = ra.evaluate_mean_ray_height(
            xy[1], ray_status == 0, (1, 2), ray_weights
        )
        xy_centroid = torch.stack((torch.zeros_like(y_centroid), y_centroid), dim=0)
        xy = xy.where(
            ray_status == 0, float("inf")
        )  # Invalid rays are set to an invalid value

        # For logging
        self.last_xy = xy.detach()
        self.xy_updated = True

        residuals_dict = {}
        logs = {}

        if self.e2e_enabled and batch is not None:
            monitor_only = self.e2e_loss_weight is None
            if monitor_only:  # Only log the end-to-end loss
                e2e_loss, _ = self.scalar_e2e_loss(
                    xy.detach(), batch, self.lens_e2e_loss_fn
                )
            elif self.e2e_vector_mode:
                y_boundary, x_boundary = self.optics_simulator.psf_abs_size / 2
                residual_vector, e2e_loss = (
                    self.evaluate_generalized_transverse_ray_aberrations(
                        xy, batch, (x_boundary, y_boundary), xy_centroid
                    )
                )
                logs["loss/generalized_transverse_ray_aberration"] = (
                    (residual_vector**2).sum().sqrt().item()
                    if residual_vector.numel() > 0
                    else 0.0
                )
                residuals_dict["generalized_transverse_ray_aberration"] = (
                    residual_vector
                )
            else:
                e2e_loss, _ = self.evaluate_e2e_loss(xy, batch, self.lens_e2e_loss_fn)
                # Take square root to get the scalar loss since it's squared in the sum-of-squares objective
                residuals_dict["e2e_scalar_loss"] = (2 * e2e_loss).sqrt()
            logs["loss/e2e_loss"] = e2e_loss.item()

        design_residuals_dict, design_residuals_logs = (
            self.compute_lens_design_residuals(
                lens,
                ray_initialization,
                xy,
                xy_centroid,
                r0,
                d,
                rt_info,
                pupil_weights,
                ray_weights,
            )
        )

        residuals_dict = {**residuals_dict, **design_residuals_dict}

        # Log metrics
        logs = {
            **logs,
            **design_residuals_logs,
            **self.log_ray_tracing_status(ray_status),
        }

        return residuals_dict, self.weight_dict, self.constraint_dict, logs

    def compute_lens_design_residuals(
        self,
        lens: optics.Lens,
        ray_initialization: ri.RayInitialization,
        xy: torch.Tensor,
        xy_centroid: torch.Tensor,
        r0: torch.Tensor,
        d: torch.Tensor,
        rt_info: dict[str, torch.Tensor],
        pupil_weights: torch.Tensor | None = None,
        ray_weights: torch.Tensor | None = None,
    ):
        """Compute design residuals and return as a dict with corresponding logs.

        All residuals that do not involve the end-to-end pipeline are computed here.

        Args:
            lens: Lens instance.
            ray_initialization: RayInitialization instance.
            xy: Spot diagrams in image space (shape: [2, n_fields, n_rays, n_wavelengths, n_lens (1)]).
            xy_centroid: Centroids of the spot diagrams (shape: [2, 1, 1, 1, 1]).
            r0: Initial ray position vectors (shape: [3, n_fields, n_rays, n_wavelengths, n_lens (1)]).
            d: Ray direction vectors (shape: [3, n_fields, n_rays, n_wavelengths, n_lens (1)]).
            rt_info: Intermediate ray-tracing information.
            pupil_weights: Pupil quadrature weights, broadcastable to a ray bundle.
            ray_weights: Combined pupil and spectral weights.
        """
        weight_dict = self.weight_dict
        logs, residuals_dict = {}, {}

        for component in self.residuals:
            # Optical performance targets
            if isinstance(
                component, res.SpotSizeResiduals
            ):  # TransverseRayAberrationResiduals is preferred
                residual_vector = component(xy, ray_weights)
            elif isinstance(
                component,
                (res.TransverseRayAberrationResiduals, res.RayBoundaryResiduals),
            ):
                residual_vector = component(xy, xy_centroid, ray_weights)
            elif isinstance(component, res.DistortionResiduals):
                residual_vector = component(
                    lens, xy_centroid[1], ray_initialization.hfov
                )
            elif isinstance(component, res.RimmerRelativeIlluminationResiduals):
                residual_vector = component(lens, ray_initialization)
            elif isinstance(component, res.RelativeIlluminationResiduals):
                residual_vector = component(d, self.wavelength_weights)

            # Ray behavior constraints
            elif isinstance(component, res.RayPathResiduals):
                residual_vector = component(lens.sequence, rt_info["delta_z"], lens.s)
            elif isinstance(component, res.MarginalRayPathResiduals):
                # Find marginal ray indices
                marginal_ray_idx = r0[:2, 0:1].norm(dim=0).argmax()
                indices = np.unravel_index(marginal_ray_idx.item(), r0[0, 0:1].shape)
                residual_vector = component(rt_info["delta_z"], indices)
            elif isinstance(component, res.RayAngleResiduals):
                residual_vector = component(
                    torch.cat((rt_info["cos2_theta"], rt_info["cos2_prime"]), dim=-1)
                )
            elif isinstance(component, res.SurfaceNormalResiduals):
                residual_vector = component(rt_info["cos_n"])
            elif isinstance(component, res.HDOEPhaseResiduals):
                component_pupil_weights = (
                    pupil_weights.reshape(-1) if pupil_weights is not None else None
                )
                if isinstance(component, res.HDOEBranchConstraintResiduals):
                    residual_vector = component(
                        rt_info["opl"],
                        rt_info["zonal_index"],
                        xy,
                        d,
                        lens.s[-1],
                        ray_initialization.wavelengths,
                        component_pupil_weights,
                    )
                else:
                    residual_vector = component(
                        rt_info["opl"],
                        rt_info["zonal_index"],
                        xy,
                        d,
                        lens.s[-1],
                        ray_initialization.wavelengths,
                        component_pupil_weights,
                    )
            elif isinstance(component, res.CoherentWavefrontResiduals):
                component_pupil_weights = (
                    pupil_weights.reshape(-1) if pupil_weights is not None else None
                )
                residual_vector = component(
                    rt_info["opl"],
                    xy,
                    d,
                    lens.s[-1],
                    ray_initialization.wavelengths,
                    self.wavelength_weights,
                    component_pupil_weights,
                )

            # First-order constraints
            elif isinstance(component, res.FocalLengthResiduals):
                residual_vector = component(
                    -lens.get_abcd(reduce=True)[0, 1, 0],
                    1 / self.parameterization.target_efl,
                )
            elif isinstance(component, res.ImageHeightResiduals):
                # add by cjy: the residual is documented to take the *maximum* y
                # centroid, but squeeze() left the field axis in place -- under
                # this ray initialization xy_centroid is [2, n_fields, 1, 1,
                # n_lens], so [..., -1] picks the last lens, not the last field.
                # Every field was being driven to the one target: at +/-1 deg the
                # 0.5 and 1.0 degree centroids were both pulled to 1.7455 mm, and
                # the on-axis field contributed a constant, gradient-free
                # -target that no step could reduce.  Take the outer field.
                residual_vector = component(
                    xy_centroid[1, ..., -1].reshape(-1).abs().max()
                )
            elif isinstance(component, res.TotalTrackLengthResiduals):
                residual_vector = component(lens.s)

            # Design variable constraints
            elif isinstance(
                component, (res.GlassVariableResiduals, res.GlassMeshDistanceResiduals)
            ):
                residual_vector = component(
                    lens.nd,
                    lens.vd,
                    lens.dpgf,
                    self.parameterization.glass_model,
                    self.parameterization.freeze_dict["g"],
                )
            elif isinstance(component, res.GroupDelayResiduals):
                residual_vector = component(rt_info["group_delay"])

            else:
                raise NotImplementedError(f"Loss {component} is not implemented.")
            name = component.name
            logs[f"loss/{name}"] = (
                (residual_vector**2).sum().sqrt().item()
                if residual_vector.numel() > 0
                else 0.0
            )
            if weight_dict[name] is not None:
                residuals_dict[name] = residual_vector
        return residuals_dict, logs

    def scalar_e2e_loss(
        self,
        xy: torch.Tensor | None,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        loss_fn: callable,
        apply_noise: bool = True,
        rgb_psfs: torch.Tensor | None = None,
    ):
        """Return the scalar end-to-end loss as well as the aberrated and restored images.

        Spot diagrams can be replaced by RGB PSFs if differentiability is not required.

        Args:
            xy: Spot diagrams in image space (shape: [2, n_fields, n_rays, n_wavelengths, n_lens (1)]);
                if None, RGB PSFs must be provided.
            batch: Tuple of images and field limits.
            loss_fn: Image discrepancy loss function; expected to have the following signature:
                loss_fn(predicted images, target images) -> loss
            apply_noise: Whether to apply noise to the simulated images.
            rgb_psfs: RGB PSFs to simulate aberrations; only used if xy is None.
        """
        imgs, field_lims = process_batch(batch)

        with torch.autocast(device_type=self.device.type):
            # Image simulation
            if xy is not None:
                aberrated_imgs, psf_grid = self.simulate_aberrations_from_xy(
                    imgs, xy, field_lims
                )
            else:
                assert rgb_psfs is not None, (
                    "Either spot diagrams xy or RGB PSFs must be provided."
                )
                aberrated_imgs, psf_grid = self.simulate_aberrations_from_rgb_psfs(
                    imgs, rgb_psfs, field_lims
                )

            # Noise
            if apply_noise:
                aberrated_imgs = self.apply_noise(aberrated_imgs)

            # Image restoration
            if self.image_restoration_model is not None:
                restored_imgs = self.image_restoration_model(aberrated_imgs, psf_grid)
            else:
                restored_imgs = aberrated_imgs

            # Image discrepancy loss
            e2e_loss = loss_fn(restored_imgs, imgs)
        return e2e_loss, (aberrated_imgs, restored_imgs)

    def evaluate_e2e_loss(
        self,
        xy: torch.Tensor,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        loss_fn: callable,
    ):
        """Return the scalar end-to-end loss and the auxiliary data.

        Under forward-mode AD:
        For typical e2e losses, forward-mode AD is either too costly or not supported by PyTorch.
        With a scalar end-to-end loss, a backward pass is much more efficient.
        The loss is evaluated at the spot diagram "xy0": L(xy0) === L0.
        The gradient of the loss is computed w.r.t. the spot diagram "xy0": grad(L)(xy0) === grad0.
        We reformulate the loss as a simple linear mapping using detach tricks:
            L(xy) = L0 + (xy - xy0) @ grad0
        For first-order optimization, this is equivalent to computing the loss directly.

        Outside forward-mode AD:
        We just compute the loss directly.

        Args:
            xy: Spot diagrams in image space (shape: [2, n_fields, n_rays, n_wavelengths, n_lens (1)]).
            batch: Tuple of images and field limits.
            loss_fn: Image discrepancy loss function; expected to have the following signature:
                loss_fn(predicted images, target images) -> loss
        """
        # TODO: find better way to detect whether forward-mode AD is employed
        if torch._C._functorch.is_gradtrackingtensor(xy):
            with torch.inference_mode(False):
                grad, (value, aux) = torch.func.grad_and_value(
                    self.scalar_e2e_loss, 0, has_aux=True
                )(xy.detach(), batch, loss_fn)
            e2e_loss = value.detach() + (xy - xy.detach()).view(
                1, -1
            ) @ grad.detach().view(-1, 1)
        else:
            e2e_loss, aux = self.scalar_e2e_loss(xy, batch, loss_fn)

        return e2e_loss, aux

    def evaluate_generalized_transverse_ray_aberrations(
        self,
        xy: torch.Tensor,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        boundaries: tuple[float, float] | None = None,
        xy_centroid: torch.Tensor | None = None,
    ):
        """Compute the scalar end-to-end loss and convert it into a least-squares objective.

        The loss is evaluated at the spot diagram "xy0": L(xy0) === L0.
        The gradient of the loss is computed w.r.t. the spot diagram "xy0": grad(L)(xy0) === grad0.
        Then, the scalar loss is locally approximated as a function L(xy) = L(xy0) + (xy - xy0) @ grad(L)(xy0).
        An approximation gives L(xy) = 1/2 || l(xy) ||^2
            where l(xy) = w * (xy - xy')
            with xy' = xy0 - grad0 * 2L0 / || grad0 ||^2
            and w = || grad0 || / sqrt(2L0)

        If boundaries and ray centroids are provided:
        The spot diagrams are clipped to prevent the optimization process from displacing rays outside the PSF grid.
        A compensation factor is computed to account for the clipping.
        This way, the sum-of-squares loss has the same value as if there were no clipping.

        Args:
            xy: Spot diagrams in image space (shape: [2, n_fields, n_rays, n_wavelengths, n_lens (1)]).
            batch: Tuple of images and field limits.
            boundaries: Boundaries to clip xy; if None, no clipping is performed.
            xy_centroid: Centroids of the spot diagrams (shape: [2, 1, 1, 1, 1]);
                if None, the centroid is not used.
        """
        # Compute L0 and grad0 (note that they are returned as constants)
        with torch.inference_mode(False):  # Disable inference mode to compute gradients
            with torch.no_grad():  # Set requires_grad to False to ignore potential optimizable parameters in IRM
                grad, (scalar_loss, _) = torch.func.grad_and_value(
                    self.scalar_e2e_loss, 0, has_aux=True
                )(xy.detach(), batch, self.lens_e2e_loss_fn)

        # Compute || grad0 ||^2 by excluding ray failures
        ray_valid = xy.isfinite().all(dim=0)
        grad_valid = grad.where(ray_valid, 0.0).view(-1)
        grad_norm_squared = grad_valid @ grad_valid

        # Compute xy'
        xy_control = xy.detach() - grad * 2 * scalar_loss / grad_norm_squared

        # Clip xy' to prevent the optimization process from displacing rays outside the PSF grid
        if boundaries is not None and xy_centroid is not None:
            boundaries = (
                torch.tensor(boundaries)
                .to(xy)
                .view(-1, *(1 for _ in range(xy.dim() - 1)))
            )
            xy_min = xy_centroid - boundaries
            xy_max = xy_centroid + boundaries
            clipped_xy_control = xy_control.clip(min=xy_min, max=xy_max)
            # Compute compensation factor to account for the clipping
            # We want the sum-of-squares loss to have the same value as if there were no clipping
            # Compensation factor corresponds of the ratio of the squared L2 norms
            compensation_factor = torch.dot(
                *((xy.detach() - xy_control).view(-1),) * 2
            ) / torch.dot(*((xy.detach() - clipped_xy_control).view(-1),) * 2)
            xy_control = clipped_xy_control
        else:
            compensation_factor = 1.0

        # Compute the weight of the residuals
        weight = (compensation_factor * grad_norm_squared / (2 * scalar_loss)).sqrt()

        # Compute the residual vector l(xy)
        residual_vector = weight * (xy - xy_control)
        residual_vector = residual_vector[ray_valid.broadcast_to(xy.shape)]

        return residual_vector, scalar_loss

    def compute_spot_diagrams(
        self,
        lens: optics.Lens | None = None,
        ray_initialization: ri.RayInitialization | None = None,
    ):
        """Compute spot diagrams for the current lens state.

        If lens and ray_initialization are not provided, the current lens and ray_initialization are used.
        In this case, the spot diagrams are stored in self.last_xy and self.xy_updated is set to True.

        Args:
            lens: Lens instance; if None, the current lens is used.
            ray_initialization: RayInitialization instance; if None, the current ray initialization is used.
        """
        if lens is None:
            lens = self.lens
        if ray_initialization is None:
            ray_initialization = self.ray_initialization
            # If self.ray_initialization is used, xy can be updated for logging purposes
            update_xy = True
        else:
            update_xy = False
        # Initialize rays from specifications
        r0, d0 = ray_initialization(lens)
        wavelengths = ray_initialization.wavelengths
        r, d, ray_status, event_info = next(
            lens.trace_rays(r0, d0, wavelengths, yield_on="end")
        )
        xy = r[:2]
        xy = xy.where(
            ray_status == 0, float("inf")
        )  # Invalid rays are set to an invalid value
        if update_xy:
            self.last_xy = xy.detach()
            self.xy_updated = True
        return xy

    def training_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        batch_idx: int,
    ):
        """Training step for the imaging system optimization.

        The training step is divided into two parts: lens optimization and image restoration model optimization.
        Both steps are optional and can be disabled by setting the corresponding optimizers to None.
        The same input batch is used for both steps.

        Args:
            batch: Tuple of images and field limits.
            batch_idx: Index of the current batch.
        """
        lens_optimizer, irm_optimizer = self.get_optimizers()
        if lens_optimizer is None and irm_optimizer is None:
            raise RuntimeError("At least one optimizer must be provided when fitting.")

        logs = {}

        # Lens optimization
        if lens_optimizer is not None:
            optimizer_logs = self.lens_optimization_step(batch, lens_optimizer)
            logs = {**logs, **optimizer_logs}

        # Image restoration model optimization
        if irm_optimizer is not None:
            irm_logs = self.irm_optimization_step(batch, irm_optimizer)
            logs = {**logs, **irm_logs}

        # Update schedulers if any
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            self.lr_scheduler_step(schedulers, None)

        # Retrieve logs from closure and log to logger
        for k, v in logs.items():
            self.log(k, v)

    def lens_optimization_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        lens_optimizer: pl.core.optimizer.LightningOptimizer,
    ):
        """Perform a single optimization step on the lens.

        Args:
            batch: Tuple of images and field limits.
            lens_optimizer: Lens optimizer.
        """
        # For logging purposes, initialize lens before it is updated
        lens = self.lens

        # Optimization step
        with lens_optimizer.toggle_model():
            # Define closure for optimizer
            # This implementation is required for compatibility with LM optimizer
            closure = CustomClosure(self, lens_optimizer, batch)
            lens_optimizer.zero_grad()
            scalar_loss = lens_optimizer.step(closure=closure)
            # Optimizer closures evaluate the objective before the parameter
            # update. Re-evaluate at the accepted state so the reported final
            # objective and the final parameter snapshot describe one design.
            with torch.inference_mode():
                post_step_scalar_loss = closure.evaluate_least_squares_loss()
        optimizer_logs = (
            lens_optimizer.optimizer.logs
            if hasattr(lens_optimizer.optimizer, "logs")
            else {}
        )

        logs = {
            **closure.logs,
            **{f"optimizer/{k}": v for k, v in optimizer_logs.items()},
            **self.parameterization.log_lens_parameterization(lens),
            **self.log_lens_first_order_data(lens),
            "loss/scalar_loss": scalar_loss,
            "loss/scalar_loss_post_step": post_step_scalar_loss,
        }

        self.xy_updated = self.psfs_updated = False

        return logs

    def irm_optimization_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        irm_optimizer: pl.core.optimizer.LightningOptimizer,
    ):
        """Perform a single optimization step on the image restoration model.

        Args:
            batch: Tuple of images and field limits.
            irm_optimizer: Image restoration model optimizer.
        """
        logs = {"optimizer/irm_lr": irm_optimizer.param_groups[0]["lr"]}
        with irm_optimizer.toggle_model():
            with torch.inference_mode(True):
                rgb_psfs = self.try_build_simulation_model()
            e2e_loss, _ = self.scalar_e2e_loss(
                None, batch, self.irm_e2e_loss_fn, rgb_psfs=rgb_psfs
            )
            irm_optimizer.zero_grad()
            self.manual_backward(e2e_loss)
            irm_optimizer.step()
        scalar_loss = e2e_loss
        logs["loss/e2e_loss_irm"] = scalar_loss
        return logs

    def validation_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        batch_idx: int,
    ):
        """Validation step for the imaging system.

        The validation step includes the test step, but only for the first batch.
        Image quality metrics (PSNR and SSIM) are computed and logged.
        A subset of validation images is displayed on Tensorboard.

        Args:
            batch: Tuple of images and field limits.
            batch_idx: Index of the current batch.
        """
        if batch_idx == 0:
            # On first batch, log and compute everything that does not relate to image processing
            self.test_step(None, 0)

        if batch is not None:
            rgb_psfs = self.try_build_simulation_model()
            e2e_loss, (aberrated_imgs, restored_imgs) = self.scalar_e2e_loss(
                None, batch, self.irm_e2e_loss_fn, rgb_psfs=rgb_psfs
            )
            batch_size = batch[0].shape[0]
            log_kwargs = {"on_step": False, "on_epoch": True, "batch_size": batch_size}
            self.log("val/e2e_loss", e2e_loss, **log_kwargs, prog_bar=True)
            imgs = batch[0]

            # Compute and log image quality metrics
            metrics = {
                "psnr": torchmetrics.image.PeakSignalNoiseRatio(
                    data_range=1.0, reduction=None, dim=(-3, -2, -1)
                ),
                "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure(
                    data_range=1.0, reduction=None
                ),
            }
            for k, metric in metrics.items():
                self.log(
                    f"val/{k}_aberrated",
                    metric(aberrated_imgs, imgs).mean(),
                    **log_kwargs,
                )
                self.log(
                    f"val/{k}_aberrated_squared",
                    (metric(aberrated_imgs, imgs) ** 2).mean(),
                    **log_kwargs,
                )
                if self.image_restoration_model is not None:
                    self.log(
                        f"val/{k}_restored",
                        metric(restored_imgs, imgs).mean(),
                        **log_kwargs,
                    )
                    self.log(
                        f"val/{k}_restored_squared",
                        (metric(restored_imgs, imgs) ** 2).mean(),
                        **log_kwargs,
                    )

            # Display subset of validation images on Tensorboard (only for first batch)
            tb_logger = self.try_find_tensorboard_logger()
            if batch_idx == 0 and tb_logger is not None:
                n_imgs = min(8, batch_size)
                log_imgs = torch.cat(
                    (imgs[:n_imgs], aberrated_imgs[:n_imgs], restored_imgs[:n_imgs]),
                    dim=0,
                )
                log_img = (
                    torchvision.utils.make_grid(log_imgs, nrow=n_imgs).cpu().numpy()
                )
                tb_logger.add_image(
                    "val/img", log_img, self.global_step, dataformats="CHW"
                )

    def test_step(self, *_):
        """Test step for the imaging system.

        The test step is used to log various metrics and data for the current lens state.
        """
        if self.parameterization is None:
            return
        e2e_enabled = self.e2e_enabled
        # Locally disable end-to-end optimization
        self.e2e_enabled = False
        residuals_dict, weight_dict, constraint_dict, logs = (
            self.compute_residuals_dict_and_logs(None)
        )
        # Re-enable
        self.e2e_enabled = e2e_enabled
        lens = self.lens
        logs = {
            **logs,
            **self.parameterization.log_lens_parameterization(lens),
            **self.log_lens_first_order_data(lens),
        }
        for k, v in logs.items():
            self.log(k, v)

    def log(
        self,
        name: str,
        value: float | int | torch.Tensor,
        batch_size: int = 1,
        **kwargs,
    ):
        """Log a metric to the logger and store it in the metrics dictionary.

        Args:
            name: Name of the metric.
            value: Value of the metric.
            batch_size: Batch size used to compute the metric.
            **kwargs: Additional arguments for the logger.
        """
        super().log(name, value, **kwargs, batch_size=batch_size)
        self.metrics[name] = value

    @torch.inference_mode()
    def log_ray_tracing_status(self, ray_status: torch.Tensor):
        """Return dict of ray-tracing status statistics.

        Args:
            ray_status: Tensor corresponding to status of each ray.
        """
        logs = {
            "ray_tracing/ray_valid": (ray_status == 0).float().mean().item(),
            "ray_tracing/ray_backtrack": (ray_status == 1).float().mean().item(),
            "ray_tracing/ray_tir": (ray_status == 2).float().mean().item(),
            "ray_tracing/ray_backward": (ray_status == 3).float().mean().item(),
            "ray_tracing/ray_miss": (ray_status == 4).float().mean().item(),
        }
        return logs

    @torch.inference_mode()
    def log_lens_first_order_data(self, lens: optics.Lens):
        """Return dict of first-order data for the lens.

        Args:
            lens: Lens instance.
        """
        hfov_rad = torch.tensor(self.ray_initialization.hfov).deg2rad().to(lens.c)
        efl = lens.efl.item()
        bfl = lens.bfl.item()
        logs = {
            "lens/efl": efl,
            "lens/bfl": bfl,
            "lens/ttl": lens.s.flip(0).cumsum(dim=0).max().item(),
            "lens/paraxial_chief_ray_height": lens.evaluate_paraxial_heights_at_image_plane(
                hfov_rad
            ).item(),
            "lens/defocus": (lens.s[-1:] - lens.bfl).item(),
        }
        return logs

    def simulate_aberrations(
        self, imgs: torch.Tensor, field_lims: torch.Tensor | None = None
    ):
        """Simulate aberrations on the provided images.

        Args:
            imgs: Virtual scenes to be simulated (shape: [n_images, n_channels, height, width]).
            field_lims: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [B, 4]).
        """
        return self.simulate_aberrations_from_xy(imgs, None, field_lims)

    def simulate_aberrations_from_xy(
        self,
        imgs: torch.Tensor,
        xy: torch.Tensor | None | None,
        field_lims: torch.Tensor | None = None,
    ):
        """Simulate aberrations on the provided images using spot diagrams.

        Args:
            imgs: Virtual scenes to be simulated (shape: [n_images, n_channels, height, width]).
            xy: Spot diagrams in image space (shape: [2, n_fields, n_rays, n_wavelengths, n_lens (1)]);
                if None, the spot diagrams are computed from the current lens state.
            field_lims: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [B, 4]).
        """
        rgb_psfs = self.try_build_simulation_model(xy=xy)
        return self.simulate_aberrations_from_rgb_psfs(imgs, rgb_psfs, field_lims)

    def simulate_aberrations_from_rgb_psfs(
        self,
        imgs: torch.Tensor,
        rgb_psfs: torch.Tensor,
        field_lims: torch.Tensor | None = None,
    ):
        """Simulate aberrations on the provided images using RGB PSFs.

        Args:
            imgs: Virtual scenes to be simulated (shape: [n_images, n_channels, height, width]).
            rgb_psfs: RGB PSFs to simulate aberrations (shape: [n_fields, n_channels, height, width]).
            field_lims: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [B, 4]).
        """
        psf_grid = self.optics_simulator.compute_psf_grid(
            rgb_psfs, *imgs.shape[-2:], field_lims
        )
        aberrated_imgs = self.optics_simulator.apply_optics_model(imgs, psf_grid)
        return aberrated_imgs, psf_grid

    def apply_noise(self, imgs: torch.Tensor):
        """Apply noise from a Gaussian distribution to the images and return the noisy images.

        Args:
            imgs: Images to which noise is applied (shape: [n_images, n_channels, height, width]).
        """
        if self.noise_sigma is not None:
            noise = torch.randn_like(imgs) * self.noise_sigma
            imgs = imgs + noise
        return imgs

    def try_build_simulation_model(self, xy: torch.Tensor | None = None):
        """Build the simulation model and return the RGB PSFs.

        If the spot diagrams are provided, the PSFs are built from the spot diagrams (required for differentiability).

        The PSFs are stored for future retrieval (e.g., for visualization or in inference).
        The PSFs and RGB PSFs are stored in self.last_psfs and self.last_rgb_psfs, respectively.
        The flag self.psfs_updated is set to True.

        Args:
            xy: Spot diagrams in image space (shape: [2, n_fields, n_rays, n_wavelengths, n_lens (1)]).
        """
        # add by cjy: the coherent path needs the full ray state (directions and
        # optical path), not just the image-plane intercepts, so the spot-diagram
        # shortcut has to be bypassed for "ray_wave"; otherwise the mode is
        # silently ignored and the geometric PSF is returned instead.
        ray_wave = getattr(self.optics_simulator, "psf_mode", None) == "ray_wave"
        if xy is not None and not ray_wave:
            psfs, rgb_psfs = self.optics_simulator.build_optics_model_from_xy(xy)
        elif self.psfs_updated:
            psfs = self.last_psfs
            rgb_psfs = self.last_rgb_psfs
        else:
            if self.xy_updated and not ray_wave:
                xy = self.last_xy
                psfs, rgb_psfs = self.optics_simulator.build_optics_model_from_xy(xy)
            elif hasattr(self.optics_simulator, "psfs"):
                psfs, rgb_psfs = self.optics_simulator.build_optics_model()
            else:
                lens = self.lens
                psfs, rgb_psfs = self.optics_simulator.build_optics_model(
                    lens, self.ray_initialization
                )

        # For logging
        self.last_psfs = psfs.detach()
        self.last_rgb_psfs = rgb_psfs.detach()
        self.psfs_updated = True

        return rgb_psfs.squeeze(dim=1)  # Assume only 1 lens

    def configure_optimizers(self):
        """Configure the optimizers for the imaging system."""
        opt_list = []

        def configure_optimizer(optimizer, lr_scheduler, parameters):
            if optimizer is None:
                for p in parameters:
                    p.requires_grad_(False)
                return None
            opt = optimizer(parameters)
            opt_dict = {"optimizer": opt}
            if lr_scheduler is not None:
                opt_dict["lr_scheduler"] = lr_scheduler(opt)
            return opt_dict

        if self.parameterization is not None:
            opt_dict = configure_optimizer(
                self.lens_optimizer,
                self.lens_lr_scheduler,
                self.parameterization.parameters(),
            )
            if opt_dict is not None:
                opt_list.append(opt_dict)

        if self.image_restoration_model is not None:
            opt_dict = configure_optimizer(
                self.irm_optimizer,
                self.irm_lr_scheduler,
                self.image_restoration_model.parameters(),
            )
            if opt_dict is not None:
                opt_list.append(opt_dict)

        return opt_list

    def get_optimizers(self):
        """Return the lens or image restoration model optimizers."""
        if self.lens_optimizer is not None and self.irm_optimizer is not None:
            lens_optimizer, irm_optimizer = self.optimizers()
        elif self.lens_optimizer is not None:
            lens_optimizer, irm_optimizer = self.optimizers(), None
        elif self.irm_optimizer is not None:
            lens_optimizer, irm_optimizer = None, self.optimizers()
        else:
            return None, None
        lens_optimization_on = (
            lens_optimizer is not None and self.lens_optimization_disabled is False
        )
        irm_optimization_on = (
            irm_optimizer is not None and self.irm_optimization_disabled is False
        )
        if lens_optimization_on and irm_optimization_on:
            # Fix issue with Lightning to prevent optimizer index from being incremented twice per actual step
            lens_optimizer._on_before_step = lambda: self.trainer.profiler.start(
                "optimizer_step"
            )
            lens_optimizer._on_after_step = lambda: self.trainer.profiler.stop(
                "optimizer_step"
            )
            return lens_optimizer, irm_optimizer
        elif lens_optimization_on:
            return lens_optimizer, None
        elif irm_optimization_on:
            return None, irm_optimizer

    def try_find_tensorboard_logger(self):
        """Find the Tensorboard logger in the trainer if there is one."""
        for logger in self.trainer.loggers:
            if isinstance(logger, loggers.TensorBoardLogger):
                return logger.experiment
        else:
            return None


class LossWrapper(torch.nn.Module):
    """Wrapper around the imaging system module to return the vector of residuals.

    This wrapper is required to compute the Jacobian using torch.func.functional_call.
    """

    def __init__(self, imaging_system: ImagingSystemModule):
        """Constructor.

        Args:
            imaging_system: ImagingSystemModule instance.
        """
        super().__init__()
        self.imaging_system = imaging_system
        self.constraint_mask = None
        self.residual_vector = None
        self.residual_slices = None
        self.logs = None

    def forward(self, batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None):
        """Forward pass of the loss wrapper.

        Computes the residual vector and stores intermediate results as attributes for later retrieval.

        Args:
            batch: Tuple of images and field limits.
        """
        residuals_dict, weight_dict, constraint_dict, logs = (
            self.imaging_system.compute_residuals_dict_and_logs(batch)
        )
        keys = residuals_dict.keys()
        weight_dict = {k: weight_dict[k] for k in keys}
        # Apply sqrt on the weights since the loss is the sum of squares
        weighted_residuals_dict = {
            k: np.sqrt(weight_dict[k]) * residuals_dict[k] for k in keys
        }
        flattened = [weighted_residuals_dict[k].view(-1) for k in keys]
        self.residual_vector = torch.cat(flattened)
        offset = 0
        self.residual_slices = {}
        for key, values in zip(keys, flattened):
            self.residual_slices[key] = slice(offset, offset + values.numel())
            offset += values.numel()

        # Store bool tensor to specify which residual is a constraint
        self.constraint_mask = torch.tensor(
            [constraint_dict[k] for k in keys for _ in range(residuals_dict[k].numel())]
        ).to(self.residual_vector.device)

        # Store logs
        self.logs = logs

        return self.residual_vector


class CustomClosure:
    """Custom closure for the optimizer that computes the scalar loss and updates the gradients.

    This closure is required for compatibility with the LM optimizer.
    In particular, the function get_least_squares_quantities is used to compute the Jacobian.
    """

    def __init__(
        self,
        imaging_system: ImagingSystemModule,
        optimizer: pl.core.optimizer.LightningOptimizer,
        batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ):
        """Constructor.

        Args:
            imaging_system: ImagingSystemModule instance.
            optimizer: LightningOptimizer instance.
            batch: Tuple of images and field limits.
        """
        self.optimizer = optimizer
        self.batch = batch
        self.loss_wrapper = LossWrapper(imaging_system)
        self.logs = None

    def __call__(self):
        """Default function for the closure, when used by regular PyTorch optimizers.

        Computes and returns the least-squares loss, and updates the gradients with autograd.
        """
        least_squares_loss = self.evaluate_least_squares_loss()
        self.optimizer.zero_grad()
        self.loss_wrapper.imaging_system.manual_backward(least_squares_loss)
        self.logs = self.loss_wrapper.logs
        return least_squares_loss

    def evaluate_least_squares_loss(self):
        """Compute the least-squares loss and return it."""
        residual_vector = self.loss_wrapper(self.batch)
        constraint_mask = self.loss_wrapper.constraint_mask
        least_squares_loss = 0.5 * (residual_vector[~constraint_mask] ** 2).sum()
        return least_squares_loss

    def evaluate_constraint_norms(self):
        """Return L2 and infinity norms from the most recently evaluated state."""
        residual_vector = self.loss_wrapper.residual_vector
        constraint_mask = self.loss_wrapper.constraint_mask
        constraints = residual_vector[constraint_mask]
        if constraints.numel() == 0:
            return 0.0, 0.0
        return float(constraints.norm()), float(constraints.abs().max())

    def get_least_squares_quantities(self):
        """Return the residual vector, the least-squares loss, the Jacobian, and the constraint mask.

        This function is called by the LMOptimizer to compute quantities for solving the least-squares problem.
        """

        def compute_residual_vector(p):
            return torch.func.functional_call(
                self.loss_wrapper, dict(zip(parameter_dict.keys(), p)), self.batch
            )

        parameter_dict = {
            k: v.detach()
            for k, v in self.loss_wrapper.named_parameters()
            if v.requires_grad
        }
        assert len(parameter_dict) == 1, (
            "The module should have only one trainable parameter tensor."
        )
        parameters = tuple(parameter_dict.values())
        # Forward-mode differentiation
        jacobian = torch.func.jacfwd(compute_residual_vector, randomness="same")(
            parameters
        )[0]

        constraint_mask = self.loss_wrapper.constraint_mask
        residual_vector = self.loss_wrapper.residual_vector
        least_squares_loss = 0.5 * (residual_vector[~constraint_mask] ** 2).sum()
        self.logs = self.loss_wrapper.logs
        return residual_vector, least_squares_loss, jacobian, constraint_mask


def get_image_discrepancy_loss_fn(parameters: str | tuple[str, float]):
    """Return the image discrepancy loss function.

    Args:
        parameters: Image discrepancy loss name or tuple of loss name and additional arguments.
    """
    if isinstance(parameters, str):
        loss_name = parameters
        args = []
    else:
        loss_name, *args = parameters

    if loss_name == "mse":
        return torch.nn.functional.mse_loss
    elif loss_name == "mae":
        return torch.nn.functional.l1_loss
    elif loss_name == "truncated_mae":
        return lambda predictions, targets: (
            ((predictions - targets).abs() - args[0]).clip(min=0).mean()
        )
    else:
        raise ValueError(f'Image discrepancy loss "{loss_name}" is not supported.')


def process_batch(batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
    """Process the batch and return the images and field limits.

    Args:
        batch: Tuple of images and field limits or images;
            batch[0]: Images (shape: [n_images, n_channels, height, width]);
            batch[1]: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [n_images, 4]).
    """
    if isinstance(batch, torch.Tensor):
        imgs = batch
        field_lims = None
    else:
        imgs, field_lims = batch
        assert all(
            (
                isinstance(field_lims, torch.Tensor),
                field_lims.dim() == 2,
                imgs.shape[0] == field_lims.shape[0],
                field_lims.shape[1] == 4,
            )
        ), "Image dataset is not properly set up for end-to-end optimization."
    assert all(
        (
            isinstance(imgs, torch.Tensor),
            imgs.dim() == 4,
            imgs.shape[1] == 3,
        )
    ), "Image dataset is not properly set up for end-to-end optimization."
    return imgs, field_lims
