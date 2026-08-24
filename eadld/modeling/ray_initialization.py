import numpy as np
import torch

from eadld.modeling import optics


class RayInitialization:
    """Initialization of rays at the entrance pupil of a lens based on system specifications."""

    def __init__(
        self,
        aperture: float,
        hfov: float,
        n_fields: int,
        wavelengths: list[float],
        pupil_sampling_mode: str,
        pupil_sampling_kwargs: dict[str, int] | None = None,
        aperture_type: str = "f_number",
        ray_aiming_steps: int = 0,
        ray_aiming_locations: str = "tbs",
        field_stop_position: float | None = None,
        wavelength_weights: list[float] | None = None,
    ):
        """Constructor.

        Args:
            aperture: Aperture of the system; used to initialize rays; see aperture_type.
            hfov: Half field of view of the system; used to initialize rays.
            n_fields: Number of fields between [0, hfov], spread uniformly.
            wavelengths: Wavelengths [nm]; used to compute the refractive indices and diffraction angle.
            pupil_sampling_mode: Name of the pattern used to sample the entrance pupil.
            pupil_sampling_kwargs: Other parameters used in the pupil sampling (related to number of rays).
            aperture_type: Type of aperture specification ('f_number' or 'epd').
            ray_aiming_steps: Number of ray-aiming correction steps.
            ray_aiming_locations: Locations where ray-aiming correction is applied (e.g., 'tbs' or 'ts');
                'tbs' applies to the top, bottom, and side edges of the pupil.
            field_stop_position: Position of field stop in object space w.r.t. first surface (size is set to EPD).
            wavelength_weights: Weights for each wavelength; not used for ray initialization.
        """
        super().__init__()
        if aperture_type == "epd":
            self.epd = aperture
        elif aperture_type == "f_number":
            self.epd = lambda efl: efl / aperture
        else:
            raise ValueError(f"Unknown aperture type: {aperture_type}")
        self.hfov = hfov
        self.pupil_sampling_mode = pupil_sampling_mode
        self.pupil_sampling_kwargs = pupil_sampling_kwargs
        self.ray_aiming_steps = ray_aiming_steps
        self.ray_aiming_locations = ray_aiming_locations
        self.field_stop_position = field_stop_position
        self.wavelengths = wavelengths
        self.n_fields = n_fields
        if wavelength_weights is not None:
            assert len(wavelength_weights) == len(self.wavelengths)
            self.wavelength_weights = np.array(wavelength_weights) / np.sum(
                wavelength_weights
            )
        else:
            self.wavelength_weights = None

    def __call__(
        self,
        lens: optics.Lens,
        epd: float | None = None,
        hfov: float | None = None,
        n_fields: int | None = None,
        wavelengths: list[float] | None = None,
        pupil_sampling_mode: str | None = None,
        ray_aiming_steps: int | None = None,
        ray_aiming_locations: str | None = None,
        **pupil_sampling_kwargs,
    ):
        """Return ray positions and directions at the entrance pupil of the lens provided as input.

        Optional arguments can be provided to override the initialization parameters.

        Args:
            lens: Lens object.
            epd: Entrance pupil diameter.
            hfov: Half field of view.
            n_fields: Number of fields.
            wavelengths: Wavelengths.
            pupil_sampling_mode: Name of the pattern used to sample the entrance pupil.
            ray_aiming_steps: Number of ray-aiming correction steps.
            ray_aiming_locations: Locations where ray-aiming correction is applied.
            pupil_sampling_kwargs: Other parameters used in the pupil sampling (related to number of rays).
        """
        if epd is None:
            epd = self.epd
            if callable(epd):  # If f-number
                epd = epd(lens.efl)
        if hfov is None:
            hfov = self.hfov
        if n_fields is None:
            n_fields = self.n_fields
        if wavelengths is None:
            wavelengths = self.wavelengths
        if len(pupil_sampling_kwargs) == 0:
            if pupil_sampling_mode is None:
                pupil_sampling_kwargs = self.pupil_sampling_kwargs
            else:
                pupil_sampling_kwargs = {}
        if pupil_sampling_mode is None:
            pupil_sampling_mode = self.pupil_sampling_mode
        if ray_aiming_steps is None:
            ray_aiming_steps = self.ray_aiming_steps
        if ray_aiming_locations is None:
            ray_aiming_locations = self.ray_aiming_locations
        # the zone table lives on the lens, so the edges are injected
        # here and the pupil samplers stay pure functions of their kwargs.
        if pupil_sampling_mode == "skew_uniform_zonal":
            pupil_sampling_kwargs = {
                "zone_edges": zone_edges_from_lens(lens, epd),
                **pupil_sampling_kwargs,
            }

        r, d = initialize_rays(
            lens,
            hfov,
            n_fields,
            wavelengths,
            epd,
            pupil_sampling_mode,
            ray_aiming_steps,
            ray_aiming_locations,
            field_stop_position=self.field_stop_position,
            **pupil_sampling_kwargs,
        )
        return r, d

    def convert_to_absolute(self, lens: optics.Lens):
        """Convert specifications that depend on the lens (e.g., f-number) to their absolute counterparts.

        Args:
            lens: Lens object.
        """
        if callable(self.epd):
            epd = self.epd(lens.efl)
        else:
            epd = self.epd
        return type(self)(
            epd,
            self.hfov,
            self.n_fields,
            self.wavelengths,
            self.pupil_sampling_mode,
            self.pupil_sampling_kwargs,
            "epd",
            self.ray_aiming_steps,
            self.ray_aiming_locations,
            self.field_stop_position,
            self.wavelength_weights,
        )


def initialize_rays(
    lens: optics.Lens,
    hfov: float | torch.Tensor,
    n_fields: int,
    wavelengths: list[float] | torch.Tensor,
    epd: float | torch.Tensor,
    pupil_sampling_mode: str,
    ray_aiming_steps: int = 0,
    ray_aiming_locations: str = "tbs",
    field_stop_position: float | None = None,
    **pupil_sampling_kwargs,
):
    """Return ray positions and directions at the entrance pupil.

    Args:
        lens: Lens object.
        hfov: Half field of view.
        n_fields: Number of fields.
        wavelengths: Wavelengths.
        epd: Entrance pupil diameter.
        pupil_sampling_mode: Name of the pattern used to sample the entrance pupil.
        ray_aiming_steps: Number of ray-aiming correction steps.
        ray_aiming_locations: Locations where ray-aiming correction is applied.
        field_stop_position: Position of field stop in object space w.r.t. first surface (size is set to EPD).
        pupil_sampling_kwargs: Other parameters used in the pupil sampling (related to number of rays).
    """
    z = lens.pupil_position
    if not isinstance(epd, torch.Tensor):
        epd = torch.tensor(epd).to(z)
    if not isinstance(hfov, torch.Tensor):
        hfov = torch.tensor(hfov).to(z)
    if not isinstance(wavelengths, torch.Tensor):
        wavelengths = torch.tensor(wavelengths).to(z)

    # Ray directions
    d = initialize_ray_directions(n_fields, hfov)

    # Correction factors for the entrance pupil where the rays are initialized
    scale = offset = None

    # Model the impact of a field stop located in object space
    if field_stop_position is not None:
        scale, offset = field_stop_correction(d, epd, z, field_stop_position)

    # Apply ray-aiming correction to ray positions
    if ray_aiming_steps > 0:
        scale, offset = ray_aiming(
            lens,
            epd,
            z,
            n_fields,
            hfov,
            wavelengths,
            ray_aiming_steps,
            ray_aiming_locations,
            scale,
            offset,
        )

    # Ray positions at entrance pupil
    r = initialize_ray_positions(epd, z, pupil_sampling_mode, **pupil_sampling_kwargs)

    if scale is not None and offset is not None:
        # Rescale entrance pupil coordinates as a function of field and wavelength
        r = r * scale + offset

    # Broadcast
    r, d, _ = torch.broadcast_tensors(r, d, r.new_ones(len(wavelengths), 1))

    return r, d


def field_stop_correction(
    d: torch.Tensor,
    epd: torch.Tensor,
    entrance_pupil_location: torch.Tensor,
    field_stop_location: float,
):
    """Compute correction factors to apply to the entrance pupil coordinates to account for a field stop.

    The field stop is assumed to be located in object space.
    The field stop is assumed to have the same diameter as the entrance pupil.

    Args:
        d: Ray direction vectors (shape: [3, *, n_lens]).
        epd: Entrance pupil diameter (shape: [n_lens]).
        entrance_pupil_location: Location of the entrance pupil in object space (shape: [n_lens]).
        field_stop_location: Location of the field stop in object space.
    """
    # For each field, find the displacement in "y" of the entrance pupil at the field stop
    dist = (entrance_pupil_location - field_stop_location) / d[2].clip(min=1e-6)
    delta_y = dist * d[1]

    # Rescale the y coordinates
    # Assume that the top of the entrance pupil is not impacted; only the bottom is clipped
    scale_y = 1 - delta_y / epd
    offset_y = delta_y / 2

    # Scale coordinates in "x" as well to make sure the shrunk entrance pupil is inside the original pupil
    scale_x = ((1 - delta_y / epd / 2) ** 2).sqrt()

    # Compute scale and offset factors
    scale = torch.stack((scale_x, scale_y, torch.ones_like(scale_x)))
    offset = torch.stack(
        (torch.zeros_like(offset_y), offset_y, torch.zeros_like(offset_y))
    )
    return scale, offset


def ray_aiming(
    lens: optics.Lens,
    epd: torch.Tensor,
    entrance_pupil_location: torch.Tensor,
    n_fields: int,
    hfov: torch.Tensor,
    wavelengths: torch.Tensor,
    ray_aiming_steps: int,
    ray_aiming_locations: str,
    scale: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
):
    """Return scale and offset factors relative to entrance pupil size to apply to ray bundles.

    The scale and offset factors are computed to correct the ray positions at the entrance pupil.
    The correction factors are refined iteratively for a predefined number of steps.

    Args:
        lens: Lens object.
        epd: Entrance pupil diameter.
        entrance_pupil_location: Location of the entrance pupil in object space.
        n_fields: Number of fields.
        hfov: Half field of view.
        wavelengths: Wavelengths.
        ray_aiming_steps: Number of ray-aiming correction steps.
        ray_aiming_locations: Locations where ray-aiming correction is applied.
        scale: Initial correction factors for the entrance pupil coordinates.
        offset: Initial correction factors for the entrance pupil coordinates.
    """
    # Estimate stop radius
    r = initialize_ray_positions(epd, entrance_pupil_location, "marginal")
    d = initialize_ray_directions(1, entrance_pupil_location.new_zeros(1))
    r, d = torch.broadcast_tensors(r, d)
    pupil_radius = epd / 2
    (_, stop_radius, _), *_ = next(lens.trace_rays(r, d, (lens.w0,), yield_on="stop"))

    # Estimate target coordinates for the "tee" rays at the stop
    zeros = torch.zeros_like(stop_radius)
    targets_y_l = torch.stack((zeros, -stop_radius, zeros))
    targets_y_h = torch.stack((zeros, stop_radius, zeros))
    targets_x = torch.stack((stop_radius, zeros, zeros))
    targets = torch.stack((targets_y_l, targets_y_h, targets_x), dim=-1).view(
        3, 1, 3, 1, 1
    )

    # Scale targets if initial correction factors are provided
    if scale is not None and offset is not None:
        mag = stop_radius / (epd / 2)
        targets = targets * scale + mag * offset
        delta_xp = (targets[0, :, 2:3] - stop_radius) / mag
        delta_yp_l = (targets[1, :, 0:1] + stop_radius) / mag
        delta_yp_h = (targets[1, :, 1:2] - stop_radius) / mag
    else:
        scale = 1.0
        offset = 0.0
        delta_xp = delta_yp_l = delta_yp_h = epd.new_zeros(1)

    # Initialize lower and upper meridional rays, and sagittal ray
    r_tee = initialize_ray_positions(epd, entrance_pupil_location, "tee")
    d_tee = initialize_ray_directions(n_fields, hfov)
    r_tee, d_tee, _ = torch.broadcast_tensors(
        r_tee, d_tee, r.new_ones(len(wavelengths), 1)
    )
    delta_xp, delta_yp_l, delta_yp_h, _ = torch.broadcast_tensors(
        delta_xp, delta_yp_l, delta_yp_h, r_tee[0, :, 0:1]
    )

    for _ in range(ray_aiming_steps):
        # Scale rays with current correction factors
        r_tee = r_tee * scale + offset

        # Update the correction factors
        yp_l_step, yp_h_step, xp_step = ray_aiming_step(
            r_tee, d_tee, targets, lens, wavelengths
        )
        # Only update the correction factors if the corresponding location is enabled
        if "t" in ray_aiming_locations:
            delta_yp_h = delta_yp_h + yp_h_step
        if "b" in ray_aiming_locations:
            delta_yp_l = delta_yp_l + yp_l_step
        if "s" in ray_aiming_locations:
            delta_xp = delta_xp + xp_step

        # Compute scale and offset
        x_scale = 1 + delta_xp / pupil_radius
        y_scale = 1 + (delta_yp_h - delta_yp_l) / pupil_radius / 2
        y_offset = (delta_yp_h + delta_yp_l) / 2
        scale = torch.stack((x_scale, y_scale, torch.ones_like(x_scale)))
        offset = torch.stack(
            (torch.zeros_like(y_offset), y_offset, torch.zeros_like(y_offset))
        )

    return scale, offset


def rs_from_rp(
    r_p: torch.Tensor, d_tee: torch.Tensor, wavelengths: torch.Tensor, lens: optics.Lens
):
    """Return ray positions at the aperture stop based on ray positions at the entrance pupil.

    Args:
        r_p: Ray position vectors at the entrance pupil (shape: [3, n_fields, 3, n_wavelengths, n_lens]).
        d_tee: Ray direction vectors (shape: [3, n_fields, 3, n_wavelengths, n_lens]).
        wavelengths: Wavelengths to be traced (shape: [n_wavelengths]).
        lens: Lens object.
    """
    # Trace rays from entrance pupil to stop
    r_s, *_ = next(lens.trace_rays(r_p, d_tee, wavelengths, yield_on="stop"))

    return r_s


def ray_aiming_step(r_tee, d_tee, targets, lens, wavelengths):
    """Return updated pupil corrections after single ray-aiming correction step.

    Args:
        r_tee: Initial ray position vectors (lower meridional ray, upper meridional ray, right sagittal ray)
         (shape: [3, n_fields, 3, n_wavelengths, n_lens]).
        d_tee: Ray direction vectors (shape: [3, n_fields, 3, n_wavelengths, n_lens]).
        targets: Target ray positions at the stop (shape: [3, n_fields, 3, n_wavelengths, n_lens]).
        lens: Lens object.
        wavelengths: Wavelengths to be traced (shape: [n_wavelengths]).
    """
    # Evaluate derivatives and ray positions at the stop
    # We are only interested in the "y" derivatives for rays 0 and 1, and the "x" derivative for ray 2
    mask = torch.zeros_like(r_tee, dtype=torch.bool)
    mask[torch.tensor((1, 1, 0)), :, torch.tensor((0, 1, 2))] = True
    # Remove inference mode so we can use autograd
    with torch.inference_mode(mode=False):

        def grad_fn(*inputs):
            rs = rs_from_rp(*inputs)
            return rs[mask.clone()].sum(), rs

        func = torch.func.grad(grad_fn, 0, has_aux=True)
        rs_grad, rs = func(r_tee.clone(), d_tee.clone(), wavelengths, lens.clone())

    # Retrieve derivatives
    ys_l_grad = rs_grad[1, :, 0:1]
    ys_h_grad = rs_grad[1, :, 1:2]
    xs_grad = rs_grad[0, :, 2:3]

    # Retrieve errors
    delta = rs - targets
    delta_ys_l = delta[1, :, 0:1]
    delta_ys_h = delta[1, :, 1:2]
    delta_xs = delta[0, :, 2:3]

    # Update correction factors at the pupil
    delta_yp_l = -delta_ys_l / ys_l_grad
    delta_yp_h = -delta_ys_h / ys_h_grad
    delta_xp = -delta_xs / xs_grad

    # Numerical stability
    delta_yp_l = delta_yp_l.where(torch.isfinite(delta_yp_l), 0.0)
    delta_yp_h = delta_yp_h.where(torch.isfinite(delta_yp_h), 0.0)
    delta_xp = delta_xp.where(torch.isfinite(delta_xp), 0.0)

    return delta_yp_l, delta_yp_h, delta_xp


def initialize_ray_directions(n_fields: int, hfov: torch.Tensor):
    """Return ray directions at the entrance pupil.

    Args:
        n_fields: Number of fields.
        hfov: Half field of view.
    """
    rel_fields = np.linspace(1, 0, n_fields)[
        ::-1
    ].tolist()  # Upside down so the field is maximal if only one
    u = torch.tensor(rel_fields).to(hfov) * hfov.deg2rad()
    cy = u.sin().view(-1, 1, 1, 1)
    cx = torch.zeros_like(cy)
    cz = (1 - cx**2 - cy**2).sqrt()
    d = torch.stack(torch.broadcast_tensors(cx, cy, cz))
    return d


def initialize_ray_positions(
    epd: torch.tensor,
    z: torch.tensor,
    pupil_sampling_mode: str,
    **pupil_sampling_kwargs,
):
    """Return ray positions at the entrance pupil.

    Args:
        epd: Entrance pupil diameter.
        z: Location of the entrance pupil w.r.t. the first lens surface.
        pupil_sampling_mode: Name of the pattern used to sample the entrance pupil.
        pupil_sampling_kwargs: Other parameters used in the pupil sampling (related to number of rays).
    """
    x_rel, y_rel = mode_to_pupil_span(pupil_sampling_mode)(**pupil_sampling_kwargs)
    x_rel = x_rel.view(1, -1, 1, 1).to(z)
    y_rel = y_rel.view(1, -1, 1, 1).to(z)
    x, y = scale_to_epd(epd, x_rel, y_rel)
    r = torch.stack(torch.broadcast_tensors(x, y, z))
    return r


# zone-aware pupil quadrature.  An annular surface makes the pupil
# integrand piecewise smooth with a jump at every zone boundary; a global
# equal-area grid puts shells across those jumps and the coherent sum then
# converges at O(1/N).  Partitioning the pupil on the boundaries keeps every
# shell inside one zone, at the cost of unequal shell areas -- hence the
# matching explicit quadrature weights.
def _zonal_partition(n_r: int, n_theta: int, zone_edges):
    """Return pupil samples and quadrature weights aligned on the zone edges.

    Args:
        n_r: Total number of radial shells, distributed by zone area.  At least
            one shell is kept per zone, so a design with more zones than ``n_r``
            realises more rays than ``n_r * n_theta``.
        n_theta: Number of rays per shell.
        zone_edges: Ascending zone outer radii, normalized to the pupil radius,
            with the last entry equal to 1.
    """
    # The pupil samplers all return host tensors; initialize_ray_positions casts
    # them to the lens device, so keep the partition on the host as well.
    edges = torch.as_tensor(zone_edges, dtype=torch.get_default_dtype()).cpu()
    rho = torch.cat((edges.new_zeros(1), edges.square()))
    widths = rho[1:] - rho[:-1]
    counts = (n_r * widths / widths.sum()).round().clamp(min=1).to(torch.int64)

    # Index of each shell inside its own zone, without a per-zone Python loop.
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    within = torch.arange(int(counts.sum())) - starts.repeat_interleave(counts)
    shell_area = (widths / counts).repeat_interleave(counts)
    r = (rho[:-1].repeat_interleave(counts) + (within + 0.5) * shell_area).sqrt()

    stagger = (torch.arange(r.numel()) + 1) % 2 / n_theta / 2
    theta = (
        (stagger[:, None] + torch.tensor(np.linspace(0, 1, n_theta, endpoint=False)))
        * 2
        * np.pi
    )
    x = (r[:, None] * theta.cos()).reshape(-1)
    y = (r[:, None] * theta.sin()).reshape(-1)
    weights = (shell_area[:, None] / n_theta).expand(-1, n_theta).reshape(-1)
    return x, y, weights / weights.sum()


def skew_uniform_zonal(n_r: int, n_theta: int, zone_edges):
    """Return zone-aligned pupil samples (see ``_zonal_partition``)."""
    x, y, _ = _zonal_partition(n_r, n_theta, zone_edges)
    return x, y


def zonal_pupil_weights(n_r: int, n_theta: int, zone_edges):
    """Return the quadrature weights matching ``skew_uniform_zonal``."""
    return _zonal_partition(n_r, n_theta, zone_edges)[2]


def zone_edges_from_lens(lens, epd, zonal_surface_index: int = 0):
    """Return one zonal surface's outer radii normalized to the pupil radius.

    A single entrance-pupil quadrature cannot in general align simultaneously
    with independent boundaries on several zonal surfaces. Callers working with
    more than one such surface must select and document the reference surface.
    """
    # Rmax is frozen topology, so the edges must not carry a tangent: the LM
    # Jacobian is built with forward-mode AD, which torch.unique cannot follow.
    if not 0 <= zonal_surface_index < lens.z.shape[0]:
        raise IndexError("zonal_surface_index is outside the lens zonal surfaces.")
    zones = lens.z[zonal_surface_index, 0].detach()
    valid = (zones != 0).any(dim=-1)
    edges = (zones[valid, 3] / (epd / 2)).clamp(max=1.0)
    edges = torch.unique(edges, sorted=True)
    if float(edges[-1]) < 1.0:
        edges = torch.cat((edges, edges.new_ones(1)))
    return edges


def mode_to_pupil_span(mode: str):
    """Return the function that generates the pupil span based on the mode.

    Args:
        mode: Name of the pattern used to sample the entrance pupil.
    """
    return {
        "tee": tee,
        "chief": chief,
        "marginal": marginal,
        "meridional_uniform": meridional_uniform,
        "sagittal_uniform": sagittal_uniform,
        "skew_outer_edge_uniform": skew_outer_edge_uniform,
        "skew_uniform": skew_uniform,
        "skew_uniform_zonal": skew_uniform_zonal,
        "skew_uniform_pseudo_random": skew_uniform_pseudorandom,
        "skew_uniform_jittered": skew_uniform_jittered,
    }[mode]


def tee():
    """Return bottom meridional ray, top meridional ray, and positive sagittal ray."""
    y = torch.tensor([-1.0, 1.0, 0.0])
    x = torch.tensor([0.0, 0.0, 1.0])

    return x, y


def chief():
    """Return the relative position of the chief ray."""
    x = torch.zeros(1)
    y = torch.zeros(1)

    return x, y


def marginal():
    """Return the relative position of the marginal ray (top of aperture)."""
    x = torch.zeros(1)
    y = torch.ones(1)

    return x, y


def meridional_uniform(n_rays: int):
    """Compute 'n_rays' x and y relative meridional pupil intersections to span the pupil uniformly.

    Args:
        n_rays: Number of rays.
    """
    y = torch.linspace(-1, 1, n_rays)
    x = torch.zeros_like(y)

    return x, y


def sagittal_uniform(n_rays):
    """Return 'n_rays' x and y relative positive sagittal pupil intersections to span the right-side pupil uniformly.

    Args:
        n_rays: Number of rays.
    """
    x = torch.linspace(0, 1, n_rays)
    y = torch.zeros_like(x)

    return x, y


def skew_outer_edge_uniform(n_rays: int):
    """Return 'n_rays' x and y relative pupil intersections on the outer edge of the right half of the pupil.

    Args:
        n_rays: Number of rays.
    """
    theta = torch.tensor(np.linspace(-np.pi / 2, np.pi / 2, n_rays))
    x = theta.cos()
    y = theta.sin()

    return x, y


def skew_uniform(n_r: int, n_theta: int):
    """Return 'n_r' * 'n_theta' x and y relative pupil intersections to span the pupil uniformly.

    Args:
        n_r: Number of concentric shells.
        n_theta: Number of rays per shell.
    """
    r_squared = torch.tensor(np.linspace(0, 1, n_r * 2, endpoint=False))[1::2]
    r = r_squared.sqrt()

    delta_theta = (torch.tensor(np.arange(n_r)) + 1) % 2 / n_theta / 2
    theta_increments = torch.tensor(np.linspace(0, 1, n_theta, endpoint=False))
    theta = (delta_theta + theta_increments.view(-1, 1)) * 2 * np.pi

    x = r * theta.cos()
    y = r * theta.sin()

    return x.view(-1), y.view(-1)


def skew_uniform_pseudorandom(n_r: int, n_theta: int):
    """Return 'n_r' * 'n_theta' x and y relative pupil intersections to span the pupil uniformly and pseudo-randomly.

    Args:
        n_r: Number of concentric shells.
        n_theta: Number of rays per shell.
    """
    delta_r_squared = torch.rand(n_r, n_theta) / n_r
    r_squared_increments = torch.tensor(np.linspace(0, 1, n_r, endpoint=False)).view(
        -1, 1
    )
    r_squared = delta_r_squared + r_squared_increments
    r = r_squared.sqrt()

    delta_theta = torch.rand(n_r, n_theta) / n_theta
    theta_increments = torch.tensor(np.linspace(0, 1, n_theta, endpoint=False)).view(
        1, -1
    )
    theta = (delta_theta + theta_increments) * 2 * np.pi

    x = r * theta.cos()
    y = r * theta.sin()

    return x.view(-1), y.view(-1)


def skew_uniform_jittered(n_r: int, n_i: int, eps: float = 1e-6):
    """Compute (n_r ** 2)(n_i) x and y relative pupil intersections to span the right half of the pupil uniformly.

    The sampling pattern is slightly biased but allows the sampling of the outer edge of the pupil.

    Args:
        n_r: Number of concentric shells.
        n_i: Multiplier for the number of rays per shell.
        eps: Small number to avoid division by zero.
    """
    assert n_r % 2 == 0, "The number of shells n_r must be even."
    rays_per_shell = np.arange(1, 2 * n_r, 2) * n_i
    shell_idx = np.repeat(np.arange(n_r), rays_per_shell)
    even_shell = shell_idx % 2 == 0
    half_rays = rays_per_shell[shell_idx] // 2
    ray_idx = np.array([j for i in range(n_r) for j in range(rays_per_shell[i])])

    # Initial estimate for r
    shell_limits = np.linspace(0, 1, n_r + 1)
    inner_r = shell_limits[:-1]
    outer_r = shell_limits[1:]
    shell_mean_r = np.sqrt((outer_r**2 + inner_r**2) / 2)
    r = shell_mean_r[shell_idx]

    # Initial estimate for theta
    theta = np.array(
        [(i / n - 0.5) * np.pi for n in rays_per_shell for i in (np.arange(n) + 0.5)]
    )

    # Apply jittering to r (heuristic to sample outer edge of pupil and to move rays apart from each other)
    # Only odd-numbered shells are affected (including the outer shell)
    # To minimize bias, rays are jittered by pairs: one pushed outside and one pushed inside
    delta_r_inner = (shell_mean_r - inner_r)[shell_idx]
    delta_r_outer = (outer_r - shell_mean_r)[shell_idx]
    jitter_up = ray_idx % 2
    jitter_down = 1 - jitter_up
    jittering_r = jitter_up * delta_r_outer - jitter_down * delta_r_inner
    rays_to_jitter_r = ~even_shell * (ray_idx % half_rays.clip(min=eps) != 0)
    r = r + rays_to_jitter_r * jittering_r

    # Apply jittering to theta (heuristic to move rays apart from each other)
    # Odd/even shells are respectively stretched/compressed towards x- and y-axis by the same factor
    shell_delta_theta = np.array([np.pi / n for n in rays_per_shell])[shell_idx]
    invert = (ray_idx > half_rays).astype(float) - (ray_idx < half_rays).astype(float)
    theta_mul = 2 * (
        (np.abs(ray_idx - half_rays) - 1).clip(min=eps) / (half_rays - 1).clip(min=eps)
        - 0.5
    )
    # Odd shells are stretched by up to a quarter of delta theta
    theta = theta + ~even_shell * theta_mul * invert * shell_delta_theta / 4
    # Even shells are compressed by up to a quarter of delta theta
    theta = theta + even_shell * theta_mul * invert * shell_delta_theta / -4
    # Rays in even shells are also displaced two-by-two by up to a sixth of delta theta
    # The further they are from the x- and y-axis, the bigger the displacement
    delta_theta = (
        invert
        * (jitter_up - jitter_down)
        * (1 - np.abs(theta_mul))
        * shell_delta_theta
        / 6
    )
    theta = theta + even_shell * (ray_idx != half_rays) * delta_theta

    x = torch.tensor(r * np.cos(theta))
    y = torch.tensor(r * np.sin(theta))

    return x.view(-1), y.view(-1)


def scale_to_epd(epd: torch.Tensor, *args):
    """Scale transverse ray coordinates in relative units to absolute units.

    Args:
        epd: Entrance pupil diameter (shape: [n_lens]).
        args: Transverse ray coordinates (shape: [*, n_lens]).
    """
    return [x * epd / 2 for x in args]
