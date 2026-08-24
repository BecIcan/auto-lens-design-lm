import math
import warnings

import torch
import numpy as np

from eisoptx.modeling import ray_analysis as ra, ray_initialization as ri, optics


# add by cjy: a Fresnel-type surface ties the zone count to the finest period,
# and both the pupil and the image grid have to be sized from it.  The rules are
# derived in the README section "Sizing the engine to the device"; undersampling
# is silent and can fabricate a focus, so it is reported rather than assumed.
def check_ray_wave_sampling(lens, ray_initialization, psf_sampler, wavelengths):
    """Return one message per unmet coherent-path sampling requirement.

    Args:
        lens: Lens object being simulated.
        ray_initialization: Ray initialization providing the pupil sampling.
        psf_sampler: PSF sampler providing the image grid.
        wavelengths: Traced wavelengths (nm).
    """
    kwargs = ray_initialization.pupil_sampling_kwargs
    if "n_r" not in kwargs or "n_theta" not in kwargs or lens.z.numel() == 0:
        return []
    n_r, n_theta = kwargs["n_r"], kwargs["n_theta"]

    epd = ray_initialization.epd
    radius = float(epd(lens.efl) if callable(epd) else epd) / 2
    distance = float(lens.s[-1])
    wavelength = float(min(wavelengths)) * 1e-6
    edges = lens.z[0, 0, :, 3][(lens.z[0, 0] != 0).any(dim=-1)]
    half_width = float(max(psf_sampler.psf_abs_size)) / 2
    pitch = float(min(psf_sampler.psf_abs_size / psf_sampler.psf_shape))
    narrowest = float(torch.diff(torch.cat((edges.new_zeros(1), edges))).min())

    zonal = ray_initialization.pupil_sampling_mode == "skew_uniform_zonal"
    shells = edges.numel() * (1 if zonal else 8)
    azimuth = math.ceil(8 * math.pi * radius * half_width / (wavelength * distance))
    finest = wavelength * distance / (4 * radius)
    separation = wavelength * distance / narrowest

    messages = []
    if n_r < shells:
        messages.append(
            f"n_r={n_r} below {shells} needed for {edges.numel()} zones"
            f"{'' if zonal else '; pupil_sampling_mode=skew_uniform_zonal needs 8x fewer'}"
        )
    if n_theta < azimuth:
        messages.append(f"n_theta={n_theta} below {azimuth} needed by the PSF window")
    if pitch > finest:
        messages.append(f"PSF pitch {pitch * 1e3:.3f} um above {finest * 1e3:.3f} um")
    if half_width < separation:
        messages.append(
            f"PSF half-width {half_width * 1e3:.1f} um below the {separation * 1e3:.1f} um "
            "order separation: only the design order is captured"
        )
    return messages


class OpticsSimulator(torch.nn.Module):
    """An optics simulator that generates realistic optical aberrations on virtual scenes."""

    def __init__(
        self,
        shape: int | tuple[int, int],
        psf_abs_size: float | tuple[float, float],
        sensor_diagonal: float | tuple[float, float],
        psf_grid_shape: tuple[int, int],
        wavelength_weights: tuple[list[float], list[float], list[float]],
        patch_overlap: float,
        shape_type: str = "image",
        kernel_type: str = "linear",
        sigma_rel: float = 1.0,
        diffraction_f_number: float | None = None,
        wavelengths: list[float] | None = None,
        psf_mode: str = "geometric",
        psf_normalization: str = "energy",
        wave_chunk_size: int = 512,
    ):
        """Constructor.

        Args:
            shape: Height and width of either PSFs or image (see shape_type).
            psf_abs_size: Height and width of the PSF in absolute units (mm).
            sensor_diagonal: Sensor diagonal (mm); a tuple of the EFL (mm) and HFOV (deg) can also be provided.
            psf_grid_shape: Height and width of the PSF grid.
            wavelength_weights: Non normalized wavelength weights for R, G, and B channels.
            patch_overlap: Fraction of overlap between patches.
            shape_type: What the shape corresponds to ('image' or 'psf').
            kernel_type: Type of kernel for PSF sampling ('cosine' or 'linear').
            sigma_rel: Kernel size relative to the size of a bin.
            diffraction_f_number: F-number of the lens for simulating pseudo-diffraction-limited PSFs;
                if None, no diffraction is simulated.
            wavelengths: Wavelengths (in nm) for simulating diffraction-limited PSFs;
                required if ``diffraction_f_number`` is not None or
                ``psf_mode`` is ``ray_wave``.
            psf_mode: PSF model (``geometric`` or ``ray_wave``). ``ray_wave``
                coherently propagates the traced wavefront using the RayWave
                Kirchhoff kernel.
            psf_normalization: ``energy`` scales each PSF to unit sum inside the
                analysis window (required by the imaging convolution). ``strehl``
                divides by the diffraction-limited reference instead, so the PSF
                is an absolute Strehl map; use it for analysis and merit
                functions, not for the imaging path.
            wave_chunk_size: Number of image-plane samples evaluated per coherent
                propagation chunk.
        """
        super().__init__()
        if psf_mode not in ("geometric", "ray_wave"):
            raise ValueError(f"Unknown PSF mode: {psf_mode}")
        # add by cjy: absolute normalization is only defined for the coherent path.
        if psf_normalization not in ("energy", "strehl"):
            raise ValueError(f"Unknown PSF normalization: {psf_normalization}")
        if psf_normalization == "strehl" and psf_mode != "ray_wave":
            raise ValueError("Strehl normalization requires ray-wave PSFs.")
        self.psf_normalization = psf_normalization
        if psf_mode == "ray_wave" and diffraction_f_number is not None:
            raise ValueError(
                "Analytic Airy convolution cannot be combined with ray-wave PSFs."
            )
        if psf_mode == "ray_wave" and wavelengths is None:
            raise ValueError("Wavelengths are required for ray-wave PSFs.")
        self.psf_mode = psf_mode

        # Scaling
        if isinstance(sensor_diagonal, float):
            self.sensor_diagonal = sensor_diagonal
        else:
            efl, hfov = sensor_diagonal
            self.sensor_diagonal = 2 * efl * np.tan(np.deg2rad(hfov))
        if isinstance(shape, int):
            shape = (shape,) * 2
        if isinstance(psf_abs_size, float):
            psf_abs_size = (psf_abs_size,) * 2
        if shape_type == "image":
            self.default_image_size = np.array(shape)
            image_diag = torch.tensor(shape).float().norm().item()
            pixel_abs_size = self.sensor_diagonal / image_diag
            psf_shape = (
                ((np.array(psf_abs_size) / pixel_abs_size) // 2 * 2 + 1)
                .astype(int)
                .tolist()
            )
        elif shape_type == "psf":
            psf_shape = shape
            # Assume square aspect ratio for default image shape
            self.default_image_size = np.round(
                np.array(psf_shape)
                * (self.sensor_diagonal / np.array(psf_abs_size))
                / np.sqrt(2)
            ).astype(int)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        # PSF and convolution parameters
        wavelength_weights = torch.tensor(wavelength_weights)
        wavelength_weights = wavelength_weights / wavelength_weights.sum(
            dim=1, keepdim=True
        )
        self.register_buffer(
            "wavelength_weights", wavelength_weights.T, persistent=False
        )
        self.psf_grid_shape = np.array(psf_grid_shape)

        # Misc
        wavelength_weights_mono = wavelength_weights.mean(dim=0).tolist()
        self.psf_sampler = PSFSampler(
            psf_shape,
            psf_abs_size,
            wavelength_weights_mono,
            kernel_type,
            sigma_rel,
            diffraction_f_number,
            wavelengths,
            wave_chunk_size=wave_chunk_size,
            psf_normalization=psf_normalization,  # add by cjy
        )
        self.convolution = SVOLAConvolution(psf_grid_shape, patch_overlap)

    @property
    def psf_shape(self):
        """Return the shape of the PSFs (height, width)."""
        return self.psf_sampler.psf_shape

    @property
    def psf_abs_size(self):
        """Return the size of the PSFs in mm (height, width)."""
        return self.psf_sampler.psf_abs_size

    def build_optics_model(
        self, lens: optics.Lens, ray_initialization: ri.RayInitialization
    ):
        """Build optics model by sampling the PSFs from the lens via ray tracing.

        Args:
            lens: Lens object to simulate.
            ray_initialization: Ray initialization object for system specifications.
        """
        wavelengths = ray_initialization.wavelengths
        assert self.wavelength_weights.shape[0] == len(wavelengths)
        wavelengths = torch.tensor(wavelengths).to(lens.c)

        # Compute spot diagrams or a coherent wavefront via ray tracing.
        r, d = ray_initialization(lens)
        r, d, ray_status, event_info = next(
            lens.trace_rays(r, d, wavelengths, yield_on="end")
        )

        if self.psf_mode == "ray_wave":
            # add by cjy: "skew_uniform_zonal" is also deterministic and
            # full-pupil; it trades equal areas for zone alignment plus weights.
            if ray_initialization.pupil_sampling_mode not in (
                "skew_uniform",
                "skew_uniform_zonal",
            ):
                raise ValueError(
                    "Ray-wave PSFs require deterministic, full-pupil sampling "
                    "(pupil_sampling_mode='skew_uniform' or 'skew_uniform_zonal')."
                )
            if ray_initialization.ray_aiming_steps != 0:
                raise ValueError("Ray-wave PSFs do not yet support ray aiming.")
            final_event = lens.sequence.events[-1]
            if final_event["type"] != "p" or final_event["s"] is None:
                raise ValueError(
                    "Ray-wave PSFs require a trailing image-plane propagation."
                )
            for message in check_ray_wave_sampling(  # add by cjy
                lens, ray_initialization, self.psf_sampler, wavelengths
            ):
                warnings.warn(f"Ray-wave sampling: {message}", stacklevel=2)
            reference_distance = lens.s[final_event["s"]]
            # add by cjy: the weights are a pure function of the same arguments
            # the sampler used, so they are recomputed here instead of being
            # carried as state on the ray initialization.
            pupil_weights = None
            if ray_initialization.pupil_sampling_mode == "skew_uniform_zonal":
                epd = ray_initialization.epd
                if callable(epd):
                    epd = epd(lens.efl)
                pupil_weights = ri.zonal_pupil_weights(
                    zone_edges=ri.zone_edges_from_lens(lens, epd),
                    **ray_initialization.pupil_sampling_kwargs,
                )
            psfs = self.psf_sampler.forward_ray_wave(
                r,
                d,
                event_info["opl"],
                ray_status,
                reference_distance,
                wavelengths,
                pupil_weights=pupil_weights,
            )
            rgb_psfs = torch.einsum("fbwij,wc->fbcij", psfs, self.wavelength_weights)
            return psfs, rgb_psfs

        xy = r[:2]
        # Only status 0 is physically valid. Backtracked rays (status 1) must not
        # contribute to image metrics or PSFs.
        xy = xy.where(ray_status == 0, float("inf"))

        return self.build_optics_model_from_xy(xy)

    def build_optics_model_from_xy(self, xy: torch.Tensor):
        """Build optics model from spot diagrams.

        Args:
            xy: Spot diagrams (shape: [2, n_fields, n_rays, n_wavelengths, n_lens]).
        """
        assert self.wavelength_weights.shape[0] == xy.shape[-2], (
            "Number of wavelengths must match."
        )
        psfs = self.psf_sampler(xy)

        # Compute RGB PSFs
        # Matrix multiplication with the wavelength weight matrix to convert wavelengths into RGB channels
        rgb_psfs = torch.einsum("fbwij,wc->fbcij", psfs, self.wavelength_weights)
        return psfs, rgb_psfs

    def compute_psf_grid(
        self,
        psfs: torch.Tensor,
        im_h: int,
        im_w: int,
        field_lims: torch.Tensor | None = None,
    ):
        """Return PSF grid for spatially-varying convolution, computed from radially sampled RGB PSFs.

        Args:
            psfs: RGB PSFs (shape: [n_fields, n_channels (3), psf_h, psf_w]).
            im_h: Image height.
            im_w: Image width.
            field_lims: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [B, 4]).
        """
        # Compute weights for PSF interpolation
        if field_lims is None:
            # Assume that the image occupies the full field of view
            field_lims = (
                torch.tensor(define_default_field_lims(im_h, im_w)).view(1, -1).to(psfs)
            )

        # Interpolate PSFs from the sampled PSFs
        interpolated_psfs = self.interpolate_psfs(psfs, field_lims)

        # Merge batch and patches
        shape = interpolated_psfs.shape
        interpolated_psfs = interpolated_psfs.view(-1, *shape[2:])

        # Rotate
        rotated_psfs = self.rotate_psfs(interpolated_psfs, field_lims)

        # Resize
        resized_psfs = self.resize_psfs(rotated_psfs, im_h, im_w, field_lims)

        # Separate batch from patches
        resized_psfs = resized_psfs.view((*shape[:-2], *resized_psfs.shape[-2:]))

        return resized_psfs

    def interpolate_psfs(self, psfs: torch.Tensor, field_lims: torch.Tensor):
        """Return interpolated PSFs to match the field of view.

        Args:
            psfs: PSFs (shape: [n_fields, n_channels, psf_h, psf_w]).
            field_lims: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [B, 4]).
        """
        # Weights are prenormalized
        psf_weights = get_psf_weights(*self.psf_grid_shape, field_lims, psfs.shape[0])
        interpolated_psfs = torch.einsum("fcij,bpf->bpcij", psfs, psf_weights)
        return interpolated_psfs

    def rotate_psfs(self, psfs: torch.Tensor, field_lims: torch.Tensor):
        """Return rotated PSFs to match the field of view.

        Args:
            psfs: PSFs (shape: [n_batch x n_patches, n_channels, psf_h, psf_w]).
            field_lims: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [B, 4]).
        """
        x0, x1, y0, y1 = field_lims.T[..., None]

        # Compute angles for rotation
        # Find the center of each patch in relative coordinates
        grid_h, grid_w = self.psf_grid_shape
        x_center = (torch.arange(grid_w).to(field_lims) + 0.5) / grid_w * (x1 - x0) + x0
        y_center = (torch.arange(grid_h).to(field_lims) + 0.5) / grid_h * (y1 - y0) + y0
        angles = torch.arctan2(-x_center[:, None, :], y_center[:, :, None]).reshape(
            field_lims.shape[0], -1
        )
        angles = angles.view(-1)

        # Note that this rotation implementation preserves the area
        rotated_psfs = rotate2d(psfs, angles, "bilinear", "constant", 0.0)

        # We assume that a PSF area < 1 means that some of the energy was lost during rotation (in corners)
        # We redistribute this energy uniformly over the PSF area
        # This has no physical meaning; it's only to penalize energy loss as if it were scattered
        n_bins = torch.prod(torch.tensor(rotated_psfs.shape[-2:]))
        psf_area = rotated_psfs.sum(dim=(-1, -2), keepdim=True)
        rotated_psfs = rotated_psfs + (1 - psf_area) / n_bins

        return rotated_psfs

    def resize_psfs(
        self, psfs: torch.Tensor, im_h: int, im_w: int, field_lims: torch.Tensor
    ):
        """Return resized PSFs to match the image size.

        Args:
            psfs: PSFs (shape: [n_batch x n_patches, n_channels, psf_h, psf_w]).
            im_h: Image height.
            im_w: Image width.
            field_lims: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [B, 4]).
        """
        x0, x1, y0, y1 = field_lims.T[..., None]

        # Compute the required size for the PSFs in the PSF grid
        # Since the sensor and image aspect ratio don't necessarily match, we assume that the diagonal is the same
        # Compute width and height in pixels on the bounding box of the full-resolution image
        # Should be the same for all input images
        im_extent_x = (im_w / ((x1 - x0) / 2)).mean().item()
        im_extent_y = (im_h / ((y0 - y1) / 2)).mean().item()
        im_extent = np.array((im_extent_y, im_extent_x))
        # Round to the nearest odd number
        resized_psf_shape = (
            self.psf_abs_size / (self.sensor_diagonal / im_extent) // 2 * 2 + 1
        ).astype(int)

        # Resize if size has changed
        if tuple(psfs.shape[-2:]) != tuple(resized_psf_shape):
            psfs = torch.nn.functional.interpolate(
                psfs,
                resized_psf_shape.tolist(),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # Re-normalize after rescaling
            psfs = psfs / psfs.sum(dim=(-1, -2), keepdim=True)

        return psfs

    def apply_optics_model(self, im: torch.Tensor, psf_grid: torch.Tensor):
        """Return aberrated image after applying the optics model.

        Args:
            im: Image(s) to be processed (shape: [B, C, H, W]).
            psf_grid: PSF grid (shape: [N, C, KH, KW]).
        """
        return self.convolution(im, psf_grid)


class FixedPSFsOpticsSimulator(OpticsSimulator):
    """An optics simulator that uses fixed PSFs for the simulation."""

    def __init__(
        self,
        psf_file: str,
        psf_abs_size: float | tuple[float, float],
        sensor_diagonal: float | tuple[float, float],
        psf_grid_shape: tuple[int, int],
        wavelength_weights: tuple[list[float], list[float], list[float]],
        patch_overlap: float,
    ):
        """Constructor.

        Args:
            psf_file: Path to the .npy file containing the PSFs.
            psf_abs_size: Height and width of the PSF in absolute units (mm).
            sensor_diagonal: Sensor diagonal (mm); a tuple of the EFL (mm) and HFOV (deg) can also be provided.
            psf_grid_shape: Height and width of the PSF grid.
            wavelength_weights: Non normalized wavelength weights for R, G, and B channels.
            patch_overlap: Fraction of overlap between patches.
        """
        # Read PSFs
        psfs = np.load(psf_file)
        assert psfs.ndim == 4, "PSFs must have 4 dimensions."
        psfs = torch.tensor(psfs)[:, None]  # Add dummy dimension for batch
        shape = psfs.shape[-2:]

        super().__init__(
            shape=shape,
            psf_abs_size=psf_abs_size,
            sensor_diagonal=sensor_diagonal,
            psf_grid_shape=psf_grid_shape,
            wavelength_weights=wavelength_weights,
            patch_overlap=patch_overlap,
            shape_type="psf",
        )

        self.register_buffer("psfs", psfs, persistent=False)

    def build_optics_model(self, *_):
        """Build optics model using the fixed PSFs."""
        return self.build_optics_model_from_xy()

    def build_optics_model_from_xy(self, *_):
        """Build optics model using the fixed PSFs."""
        psfs = self.psfs

        # Compute RGB PSFs
        # Matrix multiplication with the wavelength weight matrix to convert wavelengths into RGB channels
        rgb_psfs = torch.einsum("fbwij,wc->fbcij", psfs, self.wavelength_weights)
        return psfs, rgb_psfs


class PSFSampler(torch.nn.Module):
    """A differentiable PSF sampler that generates diffraction-compensated PSFs from spot diagrams."""

    def __init__(
        self,
        psf_shape: int | tuple[int, int],
        psf_abs_size: float | tuple[float, float],
        wavelength_weights: list[float] | None,
        kernel_type: str = "linear",
        sigma_rel: float = 1.0,
        diffraction_f_number: float | None = None,
        wavelengths: list[float] | None = None,
        distribute_unaccounted_energy: bool = True,
        wave_chunk_size: int = 512,
        psf_normalization: str = "energy",
    ):
        """Constructor.

        Args:
            psf_shape: Grid height and width for PSFs.
            psf_abs_size: Height and width of the PSF in absolute units (mm).
            wavelength_weights: Non normalized wavelength weights for computing centroid.
            kernel_type: Type of kernel for PSF sampling ('cosine' or 'linear').
            sigma_rel: Kernel size relative to the size of a bin.
            diffraction_f_number: F-number of the lens for simulating pseudo-diffraction-limited PSFs;
                if None, no diffraction is simulated.
            wavelengths: Wavelengths (in nm) for simulating diffraction-limited PSFs;
                only required if diffraction_f_number is not None.
            distribute_unaccounted_energy: Whether to distribute the unaccounted energy uniformly over the PSF area.
            wave_chunk_size: Number of image-plane samples evaluated per coherent
                propagation chunk.
            psf_normalization: ``energy`` (unit sum in the window) or ``strehl``
                (absolute, referenced to the diffraction limit).
        """
        super().__init__()
        self.psf_normalization = psf_normalization  # add by cjy
        self.psf_shape = np.array(
            (psf_shape,) * 2 if isinstance(psf_shape, int) else psf_shape
        )
        assert tuple(self.psf_shape % 2) == (1, 1)
        self.psf_abs_size = np.array(
            (psf_abs_size,) * 2 if isinstance(psf_abs_size, float) else psf_abs_size
        )
        self.kernel_type = kernel_type
        self.sigma_rel = sigma_rel
        self.diffraction_f_number = diffraction_f_number
        self.diffraction_mode = (
            "airy_field"  # Flag for manually changing diffraction mode
        )
        self.distribute_unaccounted_energy = distribute_unaccounted_energy
        self.wave_chunk_size = wave_chunk_size
        if self.diffraction_f_number is not None:
            assert wavelengths is not None, (
                "Wavelengths must be provided for simulating diffraction-limited PSFs."
            )
        self.wavelengths = wavelengths

        if wavelength_weights is None:
            wavelength_weights = [1.0]
        wavelength_weights = torch.tensor(wavelength_weights)
        wavelength_weights = wavelength_weights / wavelength_weights.sum()
        self.register_buffer(
            "wavelength_weights", wavelength_weights.view(-1, 1), persistent=False
        )

    def forward(self, xy: torch.Tensor, eps: float = 1e-6):
        """Return the PSFs that are generated from spot diagrams.

        It is assumed that invalid rays are given the value inf.

        Args:
            xy: Spot diagrams (shape: [2, n_fields, n_rays, n_wavelengths, n_lens]).
            eps: Small value to avoid division by zero.

        Returns:
            PSFs (shape: [n_fields, n_lens, n_wavelengths, psf_h, psf_w]).
        """
        # Compute grid
        xy_increment = self.psf_abs_size[::-1] / self.psf_shape[::-1]
        x_increment, y_increment = [torch.tensor(item).to(xy) for item in xy_increment]
        x_grid, y_grid = ra.define_psf_grid(*self.psf_shape, x_increment, y_increment)

        # Center rays
        x, y = xy
        ray_valid = torch.isfinite(xy).all(dim=0)
        y_centroid = ra.evaluate_mean_ray_height(
            y, ray_valid, (1, 2), self.wavelength_weights
        )
        y = y - y_centroid.expand_as(y).where(ray_valid, 0.0)

        # Compute PSFs individually for each wavelength
        x = x.permute(0, 3, 2, 1)  # Batch, field, wavelength, ray
        y = y.permute(0, 3, 2, 1)
        psfs = ra.compute_psfs(
            x, y, x_grid, y_grid, kernel_type=self.kernel_type, sigma_rel=self.sigma_rel
        )

        if self.distribute_unaccounted_energy:
            # We assume that a PSF area < 1 means that some of the energy was lost because of rays not hitting the grid
            # We redistribute it uniformly over the PSF area
            # This has no physical meaning; it's only to penalize energy loss as if it were scattered
            psf_area = psfs.sum(dim=(-1, -2), keepdim=True)
            psfs = psfs + (1 - psf_area) / math.prod(psfs.shape[-2:])

        if self.diffraction_f_number is not None:
            # Simulate diffraction-limited PSFs
            wavelengths = torch.tensor(self.wavelengths).to(psfs)
            xy_increment = torch.tensor(xy_increment).to(psfs)
            diffraction_kernels = generate_diffraction_kernels(
                self.diffraction_f_number,
                wavelengths,
                xy_increment,
                mode=self.diffraction_mode,
            )

            # Convolve the PSFs with the diffraction kernels
            shape = psfs.shape
            if self.diffraction_mode == "airy_field":
                # Approximate optical field (square root of intensity)
                psfs = psfs.abs().clip(min=eps**2).sqrt()
            kernel_size = diffraction_kernels.shape[-1]
            psfs = torch.nn.functional.conv2d(
                psfs.view(-1, *psfs.shape[-3:]),
                diffraction_kernels[:, None, ...],
                padding=kernel_size // 2,
                groups=len(wavelengths),
            )
            psfs = psfs.view(shape)
            if self.diffraction_mode == "airy_field":
                # Retrieve intensity by squaring the optical field
                psfs = psfs**2
                psfs = psfs / psfs.sum(dim=(-1, -2), keepdim=True)

        if self.distribute_unaccounted_energy:
            psf_area = psfs.sum(dim=(-1, -2), keepdim=True)
            psfs = psfs + (1 - psf_area) / math.prod(psfs.shape[-2:])

        return psfs

    def forward_ray_wave(
        self,
        r_image: torch.Tensor,
        d_image: torch.Tensor,
        opl_image: torch.Tensor,
        ray_status: torch.Tensor,
        reference_distance: torch.Tensor,
        wavelengths: torch.Tensor,
        eps: float = 1e-12,
        pupil_weights: torch.Tensor | None = None,
    ):
        """Return scalar PSFs from traced optical path lengths.

        Rays are projected from the image plane back to the nominal vertex
        plane of the final optical surface.  Their complex amplitudes are then
        propagated to a common physical image grid with the same obliquity
        kernel used by ``RayWaveModel_Thermal/Kirrchoff_3D_test.m``.

        Args:
            r_image: Ray coordinates at the image plane
                (shape: [3, n_fields, n_rays, n_wavelengths, n_lens]).
            d_image: Ray directions at the image plane, with the same shape.
            opl_image: Accumulated optical path lengths in mm
                (shape: [n_fields, n_rays, n_wavelengths, n_lens]).
            ray_status: Ray tracing status, with the same shape as ``opl_image``.
            reference_distance: Nominal last-surface-to-image distance in mm
                (shape: [n_lens]).
            wavelengths: Wavelengths in nm (shape: [n_wavelengths]).
            eps: Small value used for normalization.

        Returns:
            PSFs (shape: [n_fields, n_lens, n_wavelengths, psf_h, psf_w]).
        """
        if r_image.dtype != torch.float64:
            raise ValueError("Ray-wave PSFs require float64 ray tracing.")

        n_fields, n_rays, n_wavelengths, n_lens = opl_image.shape
        if len(wavelengths) != n_wavelengths:
            raise ValueError("Number of wavelengths must match the traced wavefront.")

        ray_valid = ray_status == 0
        wavelength_weights = self.wavelength_weights
        if len(wavelength_weights) != n_wavelengths:
            raise ValueError("Number of wavelength weights must match the wavefront.")

        # Use a common, spectrum-weighted physical grid for every wavelength so
        # that lateral color is retained in the spectral and RGB PSFs.
        x_centroid = ra.evaluate_mean_ray_height(
            r_image[0], ray_valid, (1, 2), wavelength_weights
        )
        y_centroid = ra.evaluate_mean_ray_height(
            r_image[1], ray_valid, (1, 2), wavelength_weights
        )

        # RayWave defines the complex pupil on the nominal vertex plane of the
        # final surface.  Back-project both position and eikonal from the image
        # plane; the output medium is air in the EISOPTX sequential convention.
        reference_distance_broadcast = reference_distance.view(1, 1, 1, n_lens)
        distance_to_reference = reference_distance_broadcast / d_image[2]
        r_reference = r_image - d_image * distance_to_reference
        opl_reference = torch.nan_to_num(opl_image) - distance_to_reference
        r_reference = r_reference.where(ray_valid, 0.0)
        opl_reference = opl_reference.where(ray_valid, 0.0)

        # Flatten field/wavelength/lens bundles while keeping pupil samples in
        # the final dimension.
        def flatten_bundles(tensor):
            return tensor.permute(0, 2, 3, 1).reshape(-1, n_rays)

        x_reference = flatten_bundles(r_reference[0])
        y_reference = flatten_bundles(r_reference[1])
        opl_reference = flatten_bundles(opl_reference)
        ray_valid = flatten_bundles(ray_valid)

        center_x = (
            x_centroid[:, 0, 0, :][:, None, :]
            .expand(n_fields, n_wavelengths, n_lens)
            .reshape(-1, 1)
        )
        center_y = (
            y_centroid[:, 0, 0, :][:, None, :]
            .expand(n_fields, n_wavelengths, n_lens)
            .reshape(-1, 1)
        )
        propagation_distance = (
            reference_distance[None, None, :]
            .expand(n_fields, n_wavelengths, n_lens)
            .reshape(-1, 1)
        )
        wavelength_mm = (
            wavelengths[None, :, None]
            .expand(n_fields, n_wavelengths, n_lens)
            .reshape(-1, 1)
            * 1e-6
        )

        # Stable reference-sphere phase: remove a bundle-wise piston at the
        # grid center and add only the small pixel-dependent distance change.
        center_distance = (
            (x_reference - center_x) ** 2
            + (y_reference - center_y) ** 2
            + propagation_distance**2
        ).sqrt()
        total_center_path = opl_reference + center_distance
        valid_float = ray_valid.to(total_center_path)
        # add by cjy: zone-aligned sampling has unequal shell areas, so the
        # quadrature weights enter every pupil sum -- field, reference and the
        # piston alike.  Uniform sampling passes None and keeps unit weights.
        if pupil_weights is not None:
            valid_float = valid_float * pupil_weights.to(valid_float).reshape(1, -1)
        piston = (total_center_path * valid_float).sum(dim=1, keepdim=True) / (
            valid_float.sum(dim=1, keepdim=True).clip(min=1)
        )
        base_path = (total_center_path - piston).where(ray_valid, 0.0)
        wavenumber = 2 * torch.pi / wavelength_mm

        # add by cjy: RayWave absolute reference (``KirrchoffStrl_3D.m``).  The
        # reference is the SAME Kirchhoff sum evaluated for a perfect converging
        # sphere over the SAME pupil samples, so quadrature error cancels
        # against the numerator.  The reference sphere converges on the optical
        # AXIS, not on the chief ray: every phase then aligns and the sum
        # collapses to the obliquity sum.  Using the chief ray instead biases
        # off-axis Strehl by ~3e-4 at 1.5 deg and grows with field.
        axis_distance = (
            x_reference.square() + y_reference.square() + propagation_distance.square()
        ).sqrt()
        ideal_amplitude = (
            valid_float * (axis_distance + propagation_distance) / (2 * axis_distance)
        ).sum(dim=1)

        psf_h, psf_w = self.psf_shape.tolist()
        xy_increment = self.psf_abs_size[::-1] / self.psf_shape[::-1]
        x_increment, y_increment = [
            torch.tensor(item).to(r_image) for item in xy_increment
        ]
        x_grid, y_grid = ra.define_psf_grid(psf_w, psf_h, x_increment, y_increment)
        pixel_x = x_grid[None, :].expand(psf_h, psf_w).reshape(-1)
        pixel_y = y_grid[:, None].expand(psf_h, psf_w).reshape(-1)

        intensities = []
        for start in range(0, pixel_x.numel(), self.wave_chunk_size):
            stop = min(start + self.wave_chunk_size, pixel_x.numel())
            image_x = center_x + pixel_x[start:stop]
            image_y = center_y + pixel_y[start:stop]
            distance = (
                (x_reference[..., None] - image_x[:, None, :]) ** 2
                + (y_reference[..., None] - image_y[:, None, :]) ** 2
                + propagation_distance[..., None] ** 2
            ).sqrt()
            phase = wavenumber[..., None] * (
                base_path[..., None] + distance - center_distance[..., None]
            )
            obliquity = (distance + propagation_distance[..., None]) / (2 * distance)
            field = (valid_float[..., None] * obliquity * torch.exp(1j * phase)).sum(
                dim=1
            )
            intensities.append(field.abs().square())

        intensity = torch.cat(intensities, dim=-1).view(-1, psf_h, psf_w)
        if self.psf_normalization == "strehl":  # add by cjy
            psfs = intensity / ideal_amplitude.square()[..., None, None].clip(min=eps)
        else:
            psfs = intensity / intensity.sum(dim=(-1, -2), keepdim=True).clip(min=eps)
        psfs = psfs.view(n_fields, n_wavelengths, n_lens, psf_h, psf_w).permute(
            0, 2, 1, 3, 4
        )
        return psfs


class SVOLAConvolution(torch.nn.Module):
    """A spatially-varying convolution with overlap-add method (SVOLA)."""

    def __init__(self, psf_grid_shape: tuple[int, int], patch_overlap: float):
        """Constructor.

        Args:
            psf_grid_shape: Height and width of the PSF grid.
            patch_overlap: Fraction of overlap between patches.
        """
        super().__init__()
        self.psf_grid_shape = psf_grid_shape
        self.patch_overlap = patch_overlap

        # Buffer initialization
        self.im_h = self.im_w = None
        self.patch_weights = None
        self.patch_corners = None
        self.overlap_size = None

    def forward(self, im: torch.Tensor, psf_grid: torch.Tensor):
        """Return the image after applying spatially-varying convolution.

        Args:
            im: Image(s) to be processed (shape: [B, C, H, W]).
            psf_grid: PSF grid (shape: [B, N, C, KH, KW]).
        """
        self.try_recompute_operands(*im.shape[-2:])

        # Pad
        overlap_size = self.overlap_size
        im = torch.nn.functional.pad(
            im, [overlap_size[i] for i in (1, 1, 0, 0)], mode="replicate"
        )

        # Patch-wise convolution
        patch_corners = torch.tensor(self.patch_corners, device=im.device)
        patches = patch_convolve(im, psf_grid, patch_corners)

        # Patch stitching
        patch_weights = torch.tensor(self.patch_weights).to(patches)
        im_out = reconstruct_patches(patches, patch_weights, patch_corners)

        # Remove padding
        im_out = im_out[
            ..., overlap_size[0] : -overlap_size[0], overlap_size[1] : -overlap_size[1]
        ]

        return im_out

    def try_recompute_operands(self, im_h: int, im_w: int, eps: float = 1e-6):
        """Recompute intermediate operands that depend on the image size for the convolution.

        Args:
            im_h: Image height.
            im_w: Image width.
            eps: Small value to avoid division by zero.
        """
        if self.im_h != im_h or self.im_w != im_w:
            self.im_h = im_h
            self.im_w = im_w

            # SVOLA convolution padding
            # Compute the beginning and end coordinates of all image patches
            # Padding due to overlap is considered, but not the one due to the kernel size
            # If the image shape is not a multiple of the grid shape, we stretch those coordinates
            # so that outside patches start or end at the (padded) border
            im_size_orig = np.array((im_h, im_w))
            self.overlap_size = (
                self.patch_overlap * im_size_orig / self.psf_grid_shape
            ).astype(int)
            patch_size = im_size_orig // self.psf_grid_shape + self.overlap_size * 2
            im_size = im_size_orig + 2 * self.overlap_size
            rows_0 = np.round(
                np.linspace(0, 1, self.psf_grid_shape[0]) * (im_size[0] - patch_size[0])
            ).astype(int)
            cols_0 = np.round(
                np.linspace(0, 1, self.psf_grid_shape[1]) * (im_size[1] - patch_size[1])
            ).astype(int)
            rows_1 = rows_0 + patch_size[0]
            cols_1 = cols_0 + patch_size[1]
            rows_0, cols_0 = np.meshgrid(rows_0, cols_0, indexing="ij")
            rows_1, cols_1 = np.meshgrid(rows_1, cols_1, indexing="ij")
            patch_corners = np.stack(
                (rows_0.ravel(), rows_1.ravel(), cols_0.ravel(), cols_1.ravel()), axis=1
            )
            self.patch_corners = patch_corners

            # Compute the normalized weights (contribution for each pixel of a patch to the final image)
            window_type = "hann"
            window_fn = {
                "boxcar": lambda x: np.ones_like(x),
                "hann": lambda x: np.sin(math.pi * x) ** 2,
            }
            row_window = window_fn[window_type](
                np.linspace(0, 1, patch_size[0] + 2)[1:-1]
            )
            col_window = window_fn[window_type](
                np.linspace(0, 1, patch_size[1] + 2)[1:-1]
            )
            window = (row_window[:, None] * col_window[None, :]).clip(min=eps)
            im_patch_weights = np.zeros(
                (patch_corners.shape[0], *im_size), dtype=np.float16
            )
            for i, (r0, r1, c0, c1) in enumerate(patch_corners):
                im_patch_weights[i, r0:r1, c0:c1] = window
            im_patch_weights /= im_patch_weights.sum(axis=0)
            patch_weights = np.stack(
                [
                    patch[r0:r1, c0:c1]
                    for patch, (r0, r1, c0, c1) in zip(im_patch_weights, patch_corners)
                ]
            )
            patch_weights = patch_weights.astype(window.dtype)
            self.patch_weights = patch_weights


def define_default_field_lims(im_h: int, im_w: int):
    """Return relative coordinates in object space of the image corners.

    The coordinates are normalized such that x**2 + y**2 = 1
        corresponds to the outer edge of the circular full field of view.

    Args:
        im_h: Image height.
        im_w: Image width.
    """
    # Sample the ROI
    # Retrieve the image limits in object-space coordinates
    diag = np.sqrt(im_h**2 + im_w**2)
    y0 = im_h / diag
    y1 = -im_h / diag
    x0 = -im_w / diag
    x1 = im_w / diag
    return x0, x1, y0, y1


def generate_diffraction_kernels(
    f_number: float,
    wavelengths: torch.Tensor,
    xy_increment: torch.Tensor,
    mode: str = "airy",
    sampling_factor: int = 3,
    eps: float = 1e-6,
):
    """Return diffraction kernels that can partially account for diffraction effects.

    Args:
        f_number: F-number of the lens.
        wavelengths: Wavelengths in nm (shape: [n_wavelengths]).
        xy_increment: Increment in x and y in mm (shape: [2]).
        mode: Type of diffraction pattern ('gaussian', 'airy', or 'airy_field').
        sampling_factor: Factor for oversampling the diffraction pattern.
        eps: Small value to avoid division by zero.
    """
    na = 1 / (2 * f_number)
    scale_per_mm = na / (wavelengths * 1e-6)
    scale_pix = scale_per_mm[None, :] * xy_increment[:, None]

    def get_pixel_location(kernel_size: int, sampling_factor: int):
        pixel_location = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).to(
            scale_pix
        )
        sampling_offset = (
            torch.arange(sampling_factor).to(scale_pix) - sampling_factor // 2
        ) / sampling_factor
        soft_pixel_location = pixel_location[:, None] + sampling_offset[None, :]
        return soft_pixel_location

    # Compute Gaussian kernel for each wavelength and normalize
    if mode == "gaussian":
        # Approximate the Airy diffraction pattern with a Gaussian kernel
        # (sigma = 0.21 λ / NA, see https://doi.org/10.1364/AO.46.001819)
        sigma_pix = 0.21 / scale_pix
        # At least 3 times sigma on each side
        kernel_size = int(torch.ceil(sigma_pix.max() * 3) * 2 + 1)
        soft_pixel_location = get_pixel_location(kernel_size, sampling_factor)
        kernel_x = (
            -0.5 * (soft_pixel_location / sigma_pix[0, ..., None, None]) ** 2
        ).exp()
        kernel_y = (
            -0.5 * (soft_pixel_location / sigma_pix[1, ..., None, None]) ** 2
        ).exp()
        diffraction_kernels = (
            kernel_x[..., None, :, None, :] * kernel_y[..., :, None, :, None]
        )
    elif "airy" in mode:
        airy_disk_radius_pix = 0.61 / scale_pix
        # At least 3 times Airy radius on each side
        kernel_size = 2 * int((3 * airy_disk_radius_pix).max().ceil().item()) + 1
        soft_pixel_location = get_pixel_location(kernel_size, sampling_factor)
        airy_input = (
            (soft_pixel_location * scale_pix[..., None, None] * 2 * math.pi)
            .abs()
            .clip(min=eps)
        )
        airy_input_r = (
            airy_input[0, ..., None, :, None, :] ** 2
            + airy_input[1, ..., :, None, :, None] ** 2
        ).sqrt()
        diffraction_kernels = 2 * torch.special.bessel_j1(airy_input_r) / airy_input_r
        if mode == "airy":
            diffraction_kernels = diffraction_kernels**2
        else:
            assert mode == "airy_field"
    else:
        raise ValueError(f"Unknown mode: {mode}.")

    # Normalize the kernels
    diffraction_kernels = diffraction_kernels.mean(dim=(-1, -2))
    diffraction_kernels = diffraction_kernels / diffraction_kernels.sum(
        dim=(-1, -2), keepdim=True
    )
    return diffraction_kernels


def get_psf_weights(
    grid_h: int, grid_w: int, field_lims: torch.Tensor, n_fields: int, n_patch: int = 32
):
    """Return PSF interpolation weights based on weighted sum of the sampled PSFs.

    For a PSF corresponding to a given patch, the weights are proportional
        to the number of pixels that are closest to each field in that given patch.

    Args:
        grid_h: Height of PSF grid.
        grid_w: Width of PSF grid.
        field_lims: Field limits (x0, x1, y0, y1) in relative coordinates (shape: [n_batch, 4]).
        n_fields: Number of sampled fields.
        n_patch: Size of dummy patch used for interpolation.

    Returns:
        PSF interpolation weights (shape: [n_batch, grid_h * grid_w, n_fields]).
    """
    # Create a field map
    n = field_lims.shape[0]
    x0, x1, y0, y1 = field_lims.T[..., None]
    x_coords = torch.linspace(0, 1, n_patch * grid_w).to(field_lims) * (x1 - x0) + x0
    y_coords = torch.linspace(0, 1, n_patch * grid_h).to(field_lims) * (y1 - y0) + y0
    field_map = (x_coords[:, None, :] ** 2 + y_coords[:, :, None] ** 2).sqrt()

    # Discretize it and compare fields
    field = field_map * (n_fields - 1)
    field_l = (field // 1).to(int)
    field_u = ((field_l + 1).clip(max=n_fields - 1)).to(int)
    weight_u = field % 1
    weight_l = 1 - weight_u
    fields = torch.arange(n_fields, device=field_lims.device)
    match_l = field_l[..., None] == fields
    match_u = field_u[..., None] == fields
    effective_match = match_l * weight_l[..., None] + match_u * weight_u[..., None]

    # Tile and reduce
    weights = (
        effective_match.view(n, grid_h, n_patch, grid_w, n_patch, n_fields)
        .mean(dim=(2, 4))
        .view(n, -1, n_fields)
    )
    return weights


def rotate2d(
    tensor: torch.Tensor,
    angles: torch.Tensor,
    mode: str,
    padding_mode: str,
    value: float = 0.0,
):
    """Return a tensor that is rotateed by a given angle using the three-shear method.

    Adapted from https://github.com/teboli/pytorch_rotation.

    Args:
        tensor: Tensor to be rotated (shape: [B, C, H, W]).
        angles: Rotation angles (in radians) (shape: [B]).
        mode: Interpolation mode ('bilinear' or 'bicubic').
        padding_mode: Padding mode (e.g., 'constant').
        value: Padding value if padding_mode == 'constant'.
    """
    assert mode in ("bilinear", "bicubic"), (
        "Only bilinear and bicubic interpolation are supported."
    )

    im_h, im_w = tensor.shape[-2:]

    pad_h = im_h // 2
    pad_w = im_w // 2
    tensor = torch.nn.functional.pad(
        tensor, [pad_w, pad_w, pad_h, pad_h], mode=padding_mode, value=value
    )
    im_h, im_w = tensor.shape[-2:]

    # Rotate the image by 180 first if the angle is outside the range [-pi / 2, pi / 2]
    angles_offset = angles + math.pi / 2
    rotate_180 = angles_offset % (math.pi * 2) > math.pi
    tensor = tensor.where(
        ~rotate_180.view(-1, *((1,) * (tensor.dim() - 1))),
        tensor.rot90(k=2, dims=(-2, -1)),
    )
    angles = angles_offset % math.pi - math.pi / 2

    # Shear grids
    shear_h = torch.zeros(tensor.shape[0], 2, 3).to(tensor)
    shear_h[:, 0, 0] = shear_h[:, 1, 1] = 1.0
    shear_v = shear_h.clone()
    shear_h[:, 0, 1] = -torch.tan(angles / 2) * im_h / im_w
    shear_v[:, 1, 0] = torch.sin(angles) * im_w / im_h
    shear_h_grid = torch.nn.functional.affine_grid(
        shear_h, tensor.shape, align_corners=False
    )
    shear_v_grid = torch.nn.functional.affine_grid(
        shear_v, tensor.shape, align_corners=False
    )

    # Apply the three shearing operations
    tensor = torch.nn.functional.grid_sample(
        tensor, shear_h_grid, mode=mode, align_corners=False
    )
    tensor = torch.nn.functional.grid_sample(
        tensor, shear_v_grid, mode=mode, align_corners=False
    )
    tensor = torch.nn.functional.grid_sample(
        tensor, shear_h_grid, mode=mode, align_corners=False
    )

    return tensor[..., pad_h:-pad_h, pad_w:-pad_w]


def patch_convolve(
    im: torch.Tensor, psf_grid: torch.Tensor, patch_corners: torch.Tensor
):
    """Return image patches after extraction from the image and convolution with the PSF grid.

    Args:
        im: Image(s) to be processed (shape: [B, C, H, W]).
        psf_grid: PSF grid (shape: [B, N, C, KH, KW]).
        patch_corners: Corners of the patches (row0, row1, col0, col1) (shape: [N, 4]).
    """
    pad = np.array(psf_grid.shape[-2:]) // 2

    im = torch.nn.functional.pad(im, [pad[i] for i in (1, 1, 0, 0)], mode="replicate")

    patches = torch.stack(
        [
            im[..., r0 : r1 + 2 * pad[0], c0 : c1 + 2 * pad[1]]
            for r0, r1, c0, c1 in patch_corners
        ],
        dim=1,
    )

    # Fourier space convolution
    patches_f = torch.fft.rfft2(patches)
    psf_grid_f = torch.fft.rfft2(psf_grid, patches.shape[-2:])
    patches_f = patches_f * psf_grid_f
    patches_out = torch.fft.irfft2(patches_f, patches.shape[-2:])

    # Trim
    patches_out = patches_out[..., 2 * pad[0] :, 2 * pad[1] :]

    return patches_out


def reconstruct_patches(patches, patch_weights, patch_corners):
    """
        Reconstruct the image from the patches.

    Args:
        patches: Patches to be reconstructed (shape: [N, B, C, H, W])
        patch_weights: Normalized weights of the patches (shape: [N, H, W])
        patch_corners: Corners of the patches (row0, row1, col0, col1) (shape: [N, 4])
    """
    im_h = patch_corners[:, :2].max()
    im_w = patch_corners[:, 2:].max()
    im_out = torch.zeros((patches.shape[0], patches.shape[2], im_h, im_w)).to(patches)
    weighted_patches = patch_weights[None, :, None, ...] * patches
    # Accumulate the results from every patch to limit memory use
    for patch, (r0, r1, c0, c1) in zip(weighted_patches.unbind(dim=1), patch_corners):
        padding = (c0, im_w - c1, r0, im_h - r1)
        padded_weighted_patch = torch.nn.functional.pad(patch, padding, "constant")
        im_out = im_out + padded_weighted_patch

    return im_out
