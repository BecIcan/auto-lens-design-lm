import torch
import numpy as np
import scipy.spatial

from eadld.modeling import ray_initialization as ri, ray_analysis as ra, optics
from eadld.optimization import parameterization as param


class Residuals(torch.nn.Module):
    """A generic handler for defining categories of residuals for optimization."""

    def __init__(
        self,
        weight: float | None,
        reduce_to_scalar: bool = False,
        name: str | None = None,
        **kwargs,
    ):
        """Constructor.

        Args:
            weight: Weight on residuals.
            reduce_to_scalar: Whether to reduce the residuals to a scalar.
            name: Overrides the class-level name.  ImagingSystemModule keys
                residuals by name and keeps only the last of each, so that a
                later config file can override an earlier one.  Give distinct
                names when several instances of one class are wanted at once
               .
        """
        self.weight = weight
        self.reduce_to_scalar = reduce_to_scalar
        if name is not None:
            self.name = name
        super().__init__()

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if self.reduce_to_scalar:
            return (loss**2).sum().sqrt()
        else:
            return loss

    def compute_loss(self, *args, **kwargs):
        """Compute residuals.

        Args:
            *args: Arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.
        """
        raise NotImplementedError


class SpotSizeResiduals(Residuals):
    """Residuals for field-wise RMS spot size."""

    name = "spot_size"
    constraint = False
    data_keys = ()

    def compute_loss(
        self, xy: torch.Tensor, weights: torch.Tensor | None = None
    ):
        """Forward pass.

        Args:
            xy: Ray coordinates at the image plane (shape: [2, *]).
            weights: Relative ray weights, broadcastable to ``xy[0]``.
        """
        ray_valid = xy.isfinite().all(dim=0)
        x, y = xy
        residuals = ra.compute_rms_spot_size(
            x, y, ray_valid, reduce_dims=(1, 2), weights=weights
        ) / np.sqrt(xy.shape[1])
        return residuals


class TransverseRayAberrationResiduals(Residuals):
    """Residuals for transverse ray aberrations at the image plane."""

    name = "transverse_ray_aberration"
    constraint = False
    data_keys = ()

    def compute_loss(
        self,
        xy: torch.Tensor,
        xy_centroid: torch.Tensor,
        weights: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            xy: Ray coordinates at the image plane (shape: [2, n_fields, *, n_wavelength, 1]).
            xy_centroid: Ray centroid coordinates at the image plane (shape: [2, n_fields, *, 1]).
            weights: Relative weights for rays (shape: [n_wavelengths, 1]).
        """
        bundle_valid = xy.isfinite().all(dim=0)
        ray_valid = bundle_valid.broadcast_to(xy.shape)
        delta_xy = xy - xy_centroid

        # Normalize by weights
        # Rays should have a weight of 1 on average
        if weights is not None:
            weights = weights / weights.mean()
            # Extract square root since the weights are squared in the sum of squares
            weights = weights.sqrt()
        else:
            weights = 1
        delta_xy_weighted = delta_xy * weights
        ray_valid_weighted = ray_valid * weights

        # Normalize by the square root of the number of rays (which will be squared in the sum of squares)
        weighted_valid_rays = (ray_valid_weighted**2).sum() / 2
        normalization = weighted_valid_rays.clamp_min(
            torch.finfo(xy.dtype).eps
        ).sqrt()
        residuals = delta_xy_weighted.where(ray_valid, 0.0).reshape(-1) / normalization
        # 固定维度的丢光惩罚防止优化器通过让全部光线失效来获得空残差。
        invalid_penalty = (~bundle_valid).to(xy).mean().sqrt().reshape(1)
        return torch.cat((residuals, invalid_penalty))


class RayBoundaryResiduals(Residuals):
    """Residuals for rays that exceed a transverse boundary box at the image plane."""

    name = "ray_boundary"
    constraint = False
    data_keys = ()

    def __init__(self, weight: float | None, x_boundary: float, y_boundary: float):
        """Constructor.

        Args:
            weight: Weight on residuals.
            x_boundary: Boundary box size in the x direction in mm.
            y_boundary: Boundary box size in the y direction in mm.
        """
        super().__init__(weight)
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary

    def compute_loss(
        self,
        xy: torch.Tensor,
        xy_centroid: torch.Tensor,
        weights: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            xy: Ray coordinates at the image plane (shape: [2, n_fields, *, n_wavelength, 1]).
            xy_centroid: Ray centroid coordinates at the image plane (shape: [2, n_fields, *, 1]).
            weights: Relative weights for rays (shape: [n_wavelengths, 1]).
        """
        ray_valid = xy.isfinite().all(dim=0)
        delta_xy = xy - xy_centroid
        boundary = (
            torch.tensor([self.x_boundary, self.y_boundary])
            .to(xy)
            .view(2, *([1] * (xy.dim() - 1)))
        )
        mask_xy = (delta_xy.abs() > boundary) & ray_valid
        delta_xy_clipped = delta_xy - boundary * delta_xy.sign()

        # Normalize by weights
        # Rays should have a weight of 1 on average
        if weights is not None:
            weights = weights / weights.mean()
            # Extract square root since the weights are squared in the sum of squares
            weights = weights.sqrt()
        else:
            weights = 1
        delta_xy_weighted = delta_xy_clipped * weights
        ray_valid_weighted = ray_valid * weights
        weighted_valid_rays = (ray_valid_weighted**2).sum() / 2

        residuals = delta_xy_weighted[mask_xy] / weighted_valid_rays.sqrt()
        return residuals


class RayPathResiduals(Residuals):
    """Residuals for bounding horizontal path of rays across different spacings."""

    name = "ray_path"
    constraint = False
    data_keys = ("delta_z",)

    def __init__(
        self,
        weight,
        min_cutoff: float = -float("inf"),
        max_cutoff: float = float("inf"),
        min_cutoff_refractive: float = -float("inf"),
        max_cutoff_refractive: float = float("inf"),
        min_cutoff_refractive_relative: float = -float("inf"),
        max_cutoff_refractive_relative: float = float("inf"),
        other_min_cutoffs: list[tuple[int, float]] = (),
        other_max_cutoffs: list[tuple[int, float]] = (),
    ):
        """Constructor.

        Args:
            weight: Weight on residuals.
            min_cutoff: Minimum general cutoff in airspaces for horizontal ray path in mm.
            max_cutoff: Maximum general cutoff in airspaces for horizontal ray path in mm.
            min_cutoff_refractive: Minimum cutoff in refractive surfaces in mm.
            max_cutoff_refractive: Maximum cutoff in refractive surfaces in mm.
            min_cutoff_refractive_relative: Minimum cutoff in refractive surfaces relative to central thickness.
            max_cutoff_refractive_relative: Maximum cutoff in refractive surfaces relative to central thickness.
            other_min_cutoffs: List of tuples with indices and minimum cutoffs for specific surfaces.
            other_max_cutoffs: List of tuples with indices and maximum cutoffs for specific surfaces.
        """
        super().__init__(weight)
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff
        self.min_cutoff_refractive = min_cutoff_refractive
        self.max_cutoff_refractive = max_cutoff_refractive
        self.min_cutoff_refractive_relative = min_cutoff_refractive_relative
        self.max_cutoff_refractive_relative = max_cutoff_refractive_relative
        self.other_min_cutoffs = other_min_cutoffs
        self.other_max_cutoffs = other_max_cutoffs

    def compute_loss(
        self,
        lens_sequence: optics.LensSequence,
        delta_z: torch.Tensor,
        spacings: torch.Tensor,
    ):
        """Forward pass.

        Args:
            lens_sequence: Lens sequence.
            delta_z: Horizontal ray path lengths (shape: [n_spacings, n_fields, n_rays, n_wavelengths, 1]).
            spacings: Spacings between surfaces (shape: [n_spacings, 2]).
        """
        min_cutoff, max_cutoff = self.get_cutoffs(lens_sequence, spacings)
        min_cutoff = min_cutoff.to(delta_z)
        max_cutoff = max_cutoff.to(delta_z)
        n_rays = np.prod(delta_z.shape[:-1])
        residuals = torch.maximum(
            (min_cutoff - delta_z).clip(min=0), (delta_z - max_cutoff).clip(min=0)
        )
        # Return only positive values; zero values do not have a gradient due to the ramp function
        residuals = residuals[(residuals > 0) & residuals.isfinite()] / np.sqrt(n_rays)
        return residuals

    def get_cutoffs(self, lens_sequence: optics.LensSequence, spacings: torch.Tensor):
        """Return cutoffs for horizontal ray path residuals.

        Args:
            lens_sequence: Lens sequence.
            spacings: Spacings between surfaces (shape: [n_surfaces, 2]).
        """
        propagation_events = [
            event
            for event in lens_sequence.events
            if event["type"] == "p" and event["s"] is not None
        ]
        is_refractive = ["refractive" in event for event in propagation_events]
        min_cutoff = torch.ones(len(propagation_events)).to(spacings) * self.min_cutoff
        max_cutoff = torch.ones(len(propagation_events)).to(spacings) * self.max_cutoff
        min_cutoff_relative = (
            spacings[is_refractive, 0] * self.min_cutoff_refractive_relative
        )
        max_cutoff_relative = (
            spacings[is_refractive, 0] * self.max_cutoff_refractive_relative
        )
        min_cutoff[is_refractive] = min_cutoff_relative.clip(
            min=self.min_cutoff_refractive
        )
        max_cutoff[is_refractive] = max_cutoff_relative.clip(
            max=self.max_cutoff_refractive
        )
        for i, cutoff in self.other_min_cutoffs:
            min_cutoff[i] = cutoff
        for i, cutoff in self.other_max_cutoffs:
            max_cutoff[i] = cutoff
        return min_cutoff, max_cutoff


class MarginalRayPathResiduals(Residuals):
    """Residuals for bounding horizontal path of marginal rays across different spacings."""

    name = "marginal_ray_path"
    constraint = False
    data_keys = ("delta_z",)

    def compute_loss(self, delta_z: torch.Tensor, marginal_ray_indices: torch.Tensor):
        """Forward pass.

        Args:
            delta_z: Horizontal ray path lengths (shape: [n_spacings, n_fields, n_rays, n_wavelengths, 1]).
            marginal_ray_indices: Indices of marginal rays (shape: [n_spacings, n_fields, n_rays, n_wavelengths, 1]).
        """
        # Follow the marginal ray and penalize backtracking, i.e., horizontal path < 0
        # This can be used to prevent the aperture stop from overlapping with other surfaces
        residuals = (-delta_z[marginal_ray_indices]).clip(min=0)
        return residuals[residuals > 0]


class RayAngleResiduals(Residuals):
    """Residuals for bounding angles of incidence/refraction at all interfaces."""

    name = "ray_angle"
    constraint = False
    data_keys = ("cos2_theta", "cos2_prime")

    def __init__(self, weight: float | None, max_angle: float):
        """Constructor.

        Args:
            weight: Weight on residuals.
            max_angle: Maximum angle in degrees.
        """
        super().__init__(weight)
        self.threshold = np.cos(np.deg2rad(max_angle)) ** 2

    def compute_loss(self, cos2: torch.Tensor):
        """Forward pass.

        Args:
            cos2: Squared cosine of angles (shape: [n_interfaces, n_fields, n_rays, n_wavelengths, 1]).
        """
        n_rays = np.prod(cos2.shape[:-1])
        residuals = (self.threshold - cos2).clip(min=0)
        residuals = residuals[residuals > 0] / np.sqrt(n_rays)
        return residuals


class SurfaceNormalResiduals(Residuals):
    """Residuals for bounding surface normals at all refractive interfaces."""

    name = "surface_normal"
    constraint = False
    data_keys = ("cos_n",)

    def __init__(self, weight: float | None, max_angle: float):
        """Constructor.

        Args:
            weight: Weight on residuals.
            max_angle: Maximum angle in degrees.
        """

        super().__init__(weight)
        self.threshold = np.cos(np.deg2rad(max_angle)) ** 2

    def compute_loss(self, cos_n: torch.Tensor):
        """Forward pass.

        Args:
            cos_n: Cosine of surface normals (shape: [n_interfaces, n_fields, n_rays, n_wavelengths, 1]).
        """
        if cos_n.numel() == 0:
            return torch.tensor([], device=cos_n.device)
        n_rays = np.prod(cos_n.shape[:-1])
        residuals = (self.threshold - cos_n**2).clip(min=0)
        residuals = residuals[residuals > 0] / np.sqrt(n_rays)
        return residuals


class HDOEPhaseResiduals(Residuals):
    """保持环带 HDOE 的生成光程分支。

    设计波长下相邻环带满足 ``L_(i+1)-L_i=M*lambda_0``。减去已知整数分支后，
    每个视场和镜头只允许保留一个公共 piston。
    """

    name = "hdoe_phase"
    constraint = False
    data_keys = ("opl", "zonal_index")

    def __init__(
        self,
        weight: float | None,
        diffraction_order: int,
        design_wavelength: float,
        zonal_surface_index: int = 0,
        field_indices: tuple[int, ...] | list[int] | None = (0,),
        reduce_to_scalar: bool = False,
        name: str | None = None,
    ):
        """Constructor.

        Args:
            weight: Residual weight used by EADLD.
            diffraction_order: Positive integer harmonic diffraction order ``M``.
            design_wavelength: HDOE design wavelength in nm.
            zonal_surface_index: Zonal surface whose integer branch is used.
            field_indices: Fields on which to enforce the phase-wrap relation;
                ``None`` selects every field.
            reduce_to_scalar: Whether to reduce the residual vector to a scalar.
            name: Distinct name for this instance; required when several are
                used at once, one per wavelength.
        """
        super().__init__(weight, reduce_to_scalar, name=name)
        if diffraction_order <= 0 or int(diffraction_order) != diffraction_order:
            raise ValueError("HDOE diffraction order must be a positive integer.")
        self.diffraction_order = diffraction_order
        self.design_wavelength = design_wavelength
        self.zonal_surface_index = zonal_surface_index
        self.field_indices = field_indices

    def compute_loss(
        self,
        opl: torch.Tensor,
        zonal_index: torch.Tensor,
        xy: torch.Tensor,
        direction: torch.Tensor,
        image_distance: torch.Tensor,
        wavelengths: list[float] | tuple[float, ...] | torch.Tensor,
        pupil_weights: torch.Tensor | None = None,
    ):
        """Return phase-wrap residuals in mm of optical path.

        Args:
            opl: OPL collected at every ray-tracing event
                (shape: [field, ray, wavelength, lens, event]).
            zonal_index: Selected zone at each zonal surface
                (shape: [field, ray, wavelength, lens, zonal_surface]).
            xy: Final image-plane ray coordinates
                (shape: [2, field, ray, wavelength, lens]).
            direction: Final ray directions
                (shape: [3, field, ray, wavelength, lens]).
            image_distance: Nominal final propagation distance in mm
                (shape: [lens]).
            wavelengths: Traced wavelengths in nm.
            pupil_weights: Pupil quadrature weights matching the ray dimension.
        """
        wavelengths = torch.as_tensor(wavelengths, device=opl.device, dtype=opl.dtype)
        wavelength_index = (wavelengths - self.design_wavelength).abs().argmin()
        wavelength_error = (
            wavelengths[wavelength_index] - self.design_wavelength
        ).abs()
        if wavelength_error > 1e-6:
            raise ValueError(
                f"HDOE design wavelength {self.design_wavelength} nm is not traced."
            )

        opl = opl[:, :, wavelength_index, :, -1]
        zonal_index = zonal_index[:, :, wavelength_index, :, self.zonal_surface_index]
        xy = xy[:, :, :, wavelength_index, :]
        direction = direction[:, :, :, wavelength_index, :]

        valid = opl.isfinite() & xy.isfinite().all(dim=0)
        n_rays = opl.shape[1]
        if pupil_weights is None:
            pupil_weights = opl.new_ones(n_rays)
        else:
            pupil_weights = torch.as_tensor(pupil_weights).to(opl)
        if pupil_weights.numel() != n_rays:
            raise ValueError("Pupil weights must match the traced ray count.")
        xy = xy.where(valid[None], 0.0)
        target_weights = pupil_weights.view(1, -1, 1) * valid
        target_xy = (xy * target_weights[None]).sum(
            dim=2, keepdim=True
        ) / target_weights.sum(dim=1, keepdim=True)[None].clip(min=1e-30)

        # Move from each ray's own image intercept back to the nominal final
        # surface plane, then propagate to one common image point.  This makes
        # the residual a wavefront error rather than a geometric spot error.
        image_distance = image_distance.view(1, 1, -1)
        back_distance = image_distance / direction[2]
        reference_xy = xy - direction[:2] * back_distance
        opl_reference = torch.nan_to_num(opl) - back_distance
        common_distance = (
            (reference_xy - target_xy).square().sum(dim=0) + image_distance.square()
        ).sqrt()
        common_opl = opl_reference + common_distance

        if self.field_indices is not None:
            field_indices = torch.as_tensor(
                self.field_indices, device=opl.device, dtype=torch.long
            )
            common_opl = common_opl.index_select(0, field_indices)
            zonal_index = zonal_index.index_select(0, field_indices)
            valid = valid.index_select(0, field_indices)
            target_weights = target_weights.index_select(0, field_indices)

        period = common_opl.new_tensor(
            self.diffraction_order * self.design_wavelength * 1e-6
        )
        unwrapped_opl = common_opl - zonal_index.to(common_opl) * period
        normalized_weights = target_weights / target_weights.sum(
            dim=1, keepdim=True
        ).clip(min=1e-30)
        piston = (unwrapped_opl.where(valid, 0.0) * normalized_weights).sum(
            dim=1, keepdim=True
        )
        residuals = (unwrapped_opl - piston) * normalized_weights.sqrt()
        return residuals[valid]


class HDOEBranchConstraintResiduals(HDOEPhaseResiduals):
    """约束相邻环带的系统平均 OPL 相差一个谐波周期。

    约束量为 ``c_i = mean(L_(i+1)) - mean(L_i) - M*lambda_0``。这里使用
    到公共像点的完整追迹光程，而不是薄界面高度近似；固定边界后该约束
    对连续镜头变量可微，可直接进入 LM 零空间步骤。
    """

    name = "hdoe_branch_constraint"
    constraint = True

    def __init__(
        self,
        weight: float | None,
        diffraction_order: int,
        design_wavelength: float,
        n_zones: int,
        zonal_surface_index: int = 0,
        field_indices: tuple[int, ...] | list[int] | None = (0,),
        reduce_to_scalar: bool = False,
        name: str | None = None,
    ):
        super().__init__(
            weight=weight,
            diffraction_order=diffraction_order,
            design_wavelength=design_wavelength,
            zonal_surface_index=zonal_surface_index,
            field_indices=field_indices,
            reduce_to_scalar=reduce_to_scalar,
            name=name,
        )
        if n_zones < 2:
            raise ValueError("HDOE branch constraints require at least two zones.")
        self.n_zones = n_zones

    def compute_loss(
        self,
        opl: torch.Tensor,
        zonal_index: torch.Tensor,
        xy: torch.Tensor,
        direction: torch.Tensor,
        image_distance: torch.Tensor,
        wavelengths: list[float] | tuple[float, ...] | torch.Tensor,
        pupil_weights: torch.Tensor | None = None,
    ):
        wavelengths = torch.as_tensor(wavelengths, device=opl.device, dtype=opl.dtype)
        wavelength_index = (wavelengths - self.design_wavelength).abs().argmin()
        if (wavelengths[wavelength_index] - self.design_wavelength).abs() > 1e-6:
            raise ValueError(
                f"HDOE design wavelength {self.design_wavelength} nm is not traced."
            )

        opl = opl[:, :, wavelength_index, :, -1]
        zonal_index = zonal_index[:, :, wavelength_index, :, self.zonal_surface_index]
        xy = xy[:, :, :, wavelength_index, :]
        direction = direction[:, :, :, wavelength_index, :]
        valid = opl.isfinite() & xy.isfinite().all(dim=0) & direction.isfinite().all(dim=0)
        n_rays = opl.shape[1]
        if pupil_weights is None:
            pupil_weights = opl.new_ones(n_rays)
        else:
            pupil_weights = torch.as_tensor(pupil_weights).to(opl)
        if pupil_weights.numel() != n_rays:
            raise ValueError("Pupil weights must match the traced ray count.")

        safe_xy = xy.where(valid[None], 0.0)
        target_weights = pupil_weights.view(1, -1, 1) * valid
        target_xy = (
            safe_xy * target_weights[None]
        ).sum(dim=2, keepdim=True) / target_weights.sum(
            dim=1, keepdim=True
        )[None].clip(min=1e-30)
        image_distance = image_distance.view(1, 1, -1)
        safe_direction = direction.where(valid[None], 0.0)
        back_distance = image_distance / safe_direction[2].where(valid, 1.0)
        reference_xy = safe_xy - safe_direction[:2] * back_distance[None]
        opl_reference = torch.nan_to_num(opl) - back_distance
        common_distance = (
            (reference_xy - target_xy).square().sum(dim=0) + image_distance.square()
        ).sqrt()
        common_opl = opl_reference + common_distance

        if self.field_indices is not None:
            field_indices = torch.as_tensor(
                self.field_indices, device=opl.device, dtype=torch.long
            )
            common_opl = common_opl.index_select(0, field_indices)
            zonal_index = zonal_index.index_select(0, field_indices)
            valid = valid.index_select(0, field_indices)

        membership = torch.nn.functional.one_hot(
            zonal_index.clip(0, self.n_zones - 1), self.n_zones
        ).to(common_opl)
        weights = pupil_weights.view(1, -1, 1, 1) * membership
        weights = weights * valid[..., None]
        zone_weight = weights.sum(dim=1)
        zone_mean = (weights * common_opl[..., None]).sum(dim=1) / zone_weight.clip(
            min=1e-30
        )
        period = common_opl.new_tensor(
            self.diffraction_order * self.design_wavelength * 1e-6
        )
        # 论文分支等式：c_i = Lbar_(i+1) - Lbar_i - M lambda_0 = 0。
        constraints = zone_mean[..., 1:] - zone_mean[..., :-1] - period
        # A missing zone makes the discrete topology invalid.  Keep the vector
        # length fixed for jacfwd and return a finite, unmistakable violation.
        adjacent_valid = (zone_weight[..., 1:] > 0) & (zone_weight[..., :-1] > 0)
        return constraints.where(adjacent_valid, period.abs() + 1.0)


class HDOEBranchResiduals(HDOEBranchConstraintResiduals):
    """Soft least-squares form of the complete-OPL branch residual."""

    name = "hdoe_branch"
    constraint = False


class CoherentWavefrontResiduals(Residuals):
    """最大化公共像点处与 piston 无关的相干集中度。

    加权中心化单位相量的平方范数严格等于
    ``1-|sum(w_j exp(i phi_j))|^2``。它跨整数波分支连续，不需要相位展开。
    """

    name = "coherent_wavefront"
    constraint = False
    data_keys = ("opl",)

    def __init__(
        self,
        weight: float | None,
        field_indices: tuple[int, ...] | list[int] | None = None,
        wavelength_weights: tuple[float, ...] | list[float] | None = None,
        reduce_to_scalar: bool = False,
        name: str | None = None,
    ):
        super().__init__(weight, reduce_to_scalar, name=name)
        self.field_indices = field_indices
        self.wavelength_weights = wavelength_weights

    def compute_loss(
        self,
        opl: torch.Tensor,
        xy: torch.Tensor,
        direction: torch.Tensor,
        image_distance: torch.Tensor,
        wavelengths: list[float] | tuple[float, ...] | torch.Tensor,
        wavelength_weights: torch.Tensor | None = None,
        pupil_weights: torch.Tensor | None = None,
    ):
        """Return a real coherent residual on a common reference sphere.

        Args:
            opl: OPL at every trace event [field, ray, wavelength, lens, event].
            xy: Final image-plane coordinates [2, field, ray, wavelength, lens].
            direction: Final directions [3, field, ray, wavelength, lens].
            image_distance: Final nominal propagation distance [lens].
            wavelengths: Traced wavelengths in nm.
            wavelength_weights: Relative spectral weights [wavelength].
            pupil_weights: Relative pupil quadrature weights [ray].
        """
        opl = opl[..., -1]
        valid = opl.isfinite() & xy.isfinite().all(dim=0) & direction.isfinite().all(dim=0)
        n_rays = opl.shape[1]
        if pupil_weights is None:
            pupil_weights = opl.new_ones(n_rays)
        else:
            pupil_weights = torch.as_tensor(pupil_weights).to(opl)
        if pupil_weights.numel() != n_rays:
            raise ValueError("Pupil weights must match the traced ray count.")

        safe_xy = xy.where(valid[None], 0.0)
        target_weights = pupil_weights.view(1, -1, 1, 1) * valid
        target_xy = (
            safe_xy * target_weights[None]
        ).sum(dim=2, keepdim=True) / target_weights.sum(
            dim=1, keepdim=True
        )[None].clip(min=1e-30)

        image_distance = image_distance.view(1, 1, 1, -1)
        safe_direction = direction.where(valid[None], 0.0)
        back_distance = image_distance / safe_direction[2].where(valid, 1.0)
        reference_xy = safe_xy - safe_direction[:2] * back_distance[None]
        opl_reference = torch.nan_to_num(opl) - back_distance
        common_distance = (
            (reference_xy - target_xy).square().sum(dim=0)
            + image_distance.square()
        ).sqrt()
        common_opl = opl_reference + common_distance

        if self.field_indices is not None:
            field_indices = torch.as_tensor(
                self.field_indices, device=opl.device, dtype=torch.long
            )
            common_opl = common_opl.index_select(0, field_indices)
            valid = valid.index_select(0, field_indices)

        n_wavelengths = common_opl.shape[2]
        if self.wavelength_weights is not None:
            wavelength_weights = self.wavelength_weights
        if wavelength_weights is None:
            wavelength_weights = common_opl.new_ones(n_wavelengths)
        else:
            wavelength_weights = torch.as_tensor(wavelength_weights).to(common_opl)
        if wavelength_weights.numel() != n_wavelengths:
            raise ValueError("Wavelength weights must match the traced wavelengths.")
        wavelength_weights = wavelength_weights / wavelength_weights.sum()

        ray_weights = pupil_weights.view(1, -1, 1, 1) * valid
        ray_weights = ray_weights / ray_weights.sum(dim=1, keepdim=True).clip(min=1e-30)
        # 减去公共 piston 不改变中心化相量，同时缩小三角函数自变量，
        # 让 float64 自动微分更稳定。
        piston = (ray_weights * common_opl).sum(dim=1, keepdim=True)
        wavelength_mm = torch.as_tensor(wavelengths).to(common_opl) * 1e-6
        phase = 2 * np.pi * (common_opl - piston) / wavelength_mm.view(1, 1, -1, 1)
        phasor = torch.polar(torch.ones_like(phase), phase)
        mean_phasor = (ray_weights * phasor).sum(dim=1, keepdim=True)
        centered = phasor - mean_phasor
        scale = (
            ray_weights.sqrt()
            * wavelength_weights.sqrt().view(1, 1, -1, 1)
        )
        real = (scale * centered.real).where(valid, 0.0)
        imag = (scale * centered.imag).where(valid, 0.0)
        return torch.stack((real, imag), dim=0)


class GlassVariableResiduals(Residuals):
    """Residuals for limiting distance between optimized materials and closest catalog materials."""

    name = "glass_variable"
    constraint = False
    data_keys = ()

    def compute_loss(
        self,
        nd: torch.Tensor,
        vd: torch.Tensor,
        dpgf: torch.Tensor,
        glass_model: param.GlassModel,
        freeze_mask: torch.Tensor,
    ):
        """Forward pass.

        Args:
            nd: Refractive index (shape: [n_glasses, 1]).
            vd: Abbe number (shape: [n_glasses, 1]).
            dpgf: Deviation from normal partial dispersion (shape: [n_glasses, 1]).
            glass_model: Glass model.
            freeze_mask: Mask for frozen glasses (shape: [n_glasses, 1, 2]).
        """
        catalog_g = glass_model.catalog_g
        assert catalog_g is not None, "Glass catalog not available."
        g = glass_model.g_from_nd_vd_dpgf(nd, vd, dpgf)
        g = g[~freeze_mask.any(dim=-1)]
        dist = glass_model.glass_distance(g[..., None, :], glass_model.catalog_g)
        residuals = dist.min(dim=-1)[0]
        return residuals


class GlassMeshDistanceResiduals(Residuals):
    """Residuals for limiting distance between optimized materials and the 2D/3D mesh of the glass catalog."""

    name = "glass_mesh_distance"
    constraint = False
    data_keys = ()

    def __init__(
        self,
        weight: float,
        threshold: float = 0.0,
        max_simplex_size: float = float("inf"),
    ):
        """Constructor.

        Args:
            weight: Weight on residuals.
            threshold: Allowed distance to the mesh.
            max_simplex_size: Maximum size of simplices in the mesh.
        """
        super().__init__(weight)
        self.threshold = threshold
        self.max_simplex_size = max_simplex_size
        self.coefficients = None

    def compute_loss(
        self,
        nd: torch.Tensor,
        vd: torch.Tensor,
        dpgf: torch.Tensor,
        glass_model: param.GlassModel,
        freeze_mask: torch.Tensor,
    ):
        """Forward pass.

        Args:
            nd: Refractive index (shape: [n_glasses, 1]).
            vd: Abbe number (shape: [n_glasses, 1]).
            dpgf: Deviation from normal partial dispersion (shape: [n_glasses, 1]).
            glass_model: Glass model.
            freeze_mask: Mask for frozen glasses (shape: [n_glasses, 1, 2]).
        """
        g = glass_model.g_from_nd_vd_dpgf(nd, vd, dpgf) * glass_model.eigenvalues.sqrt()
        g = g[~freeze_mask.any(dim=-1)]
        if g.numel() == 0:
            return torch.tensor([], device=g.device)

        # Postprocess glass mesh and remove large simplices
        delaunay = glass_model.glass_mesh
        points = torch.as_tensor(delaunay.points, device=g.device)
        selected = self.filter_simplices(delaunay, points)
        selected_simplices = torch.tensor(delaunay.simplices).to(points.device)[
            selected
        ]

        # Compute distance
        selected_vertices = points[selected_simplices, :]
        residuals = (
            point_to_mesh_distance(g, selected_vertices) - self.threshold
        ).clip(min=0)
        return residuals.view(-1)

    def filter_simplices(self, delaunay: scipy.spatial.Delaunay, points: torch.Tensor):
        """Filter simplices based on the maximum size of their edges.

        Args:
            delaunay: Delaunay triangulation.
            points: Points in the triangulation.
        """
        # Calculate maximum distances between simplex vertices
        vertices = points[delaunay.simplices, :]
        vertices_distance = torch.cdist(vertices, vertices)
        max_simplex_size = vertices_distance.flatten(1).max(dim=1).values

        # For each vertex, we make sure to preserve the simplex connecting to it with the smallest size
        vertex_to_simplex = [[] for _ in range(len(delaunay.points))]
        for i, indices in enumerate(delaunay.simplices):
            for j in indices:
                vertex_to_simplex[j].append(i)
        keep_simplices = torch.zeros(
            len(delaunay.simplices), dtype=torch.bool, device=points.device
        )
        for simplex_indices in vertex_to_simplex:
            criteria = max_simplex_size[simplex_indices]
            if len(criteria) > 0:
                keep_simplices[simplex_indices[criteria.argmin()]] = True

        return (max_simplex_size <= self.max_simplex_size) | keep_simplices


class FocalLengthResiduals(Residuals):
    """Residuals to constrain lens focal length, using its reciprocal the lens power."""

    name = "focal_length"
    constraint = False
    data_keys = ()

    def compute_loss(self, power: torch.Tensor, target_power: float):
        """Forward pass.

        Args:
            power: Lens power (shape: [*]).
            target_power: Target lens power.
        """
        residuals = power - target_power
        return residuals


class TotalTrackLengthResiduals(Residuals):
    """Residuals for bounding the total track length of the lens.

    Accurate results are only guaranteed for lenses with a convex or flat first surface.
    """

    name = "total_track_length"
    constraint = False
    data_keys = ()

    def __init__(self, weight: float | None, target: float):
        """Constructor.

        Args:
            weight: Weight on residuals.
            target: Target total track length.
        """
        super().__init__(weight)
        self.target = target

    def compute_loss(self, s: torch.Tensor):
        """Forward pass.

        Args:
            s: Spacings between surfaces (shape: [n_surfaces, 2]).
        """
        # Compute cumulative length from image plane and select maximum value
        # This prevents an erroneous TTL when the aperture stop is on the first surface with negative spacing
        ttl = s.flip(0).cumsum(dim=0)[-1]
        residuals = (ttl - self.target).clip(min=0)
        return residuals


class ImageHeightResiduals(Residuals):
    """Residuals for bounding the image height at the image plane."""

    name = "image_height"
    constraint = False
    data_keys = ()

    def __init__(self, weight: float | None, target: float):
        """Constructor.

        Args:
            weight: Weight on residuals.
            target: Target image height.
        """
        super().__init__(weight)
        self.target = target

    def compute_loss(self, max_y_centroid: torch.Tensor):
        """Forward pass.

        Args:
            max_y_centroid: Maximum y centroid at the image plane (shape: [*]).
        """
        residuals = max_y_centroid - self.target
        return residuals


class DistortionResiduals(Residuals):
    """Residuals for bounding distortion."""

    name = "distortion"
    constraint = False
    data_keys = ()

    def __init__(self, weight: float | None, threshold: float):
        """Constructor.

        Args:
            weight: Weight on residuals.
            threshold: Maximum distortion.
        """
        super().__init__(weight)
        self.threshold = threshold

    def compute_loss(self, lens: optics.Lens, y_centroid: torch.Tensor, hfov: float):
        """Forward pass.

        Args:
            lens: Lens.
            y_centroid: Ray centroid coordinates at the image plane (shape: [n_fields, 1]).
            hfov: Horizontal field of view in degrees.
        """
        # Compute reference paraxial value for every field
        n_fields = y_centroid.shape[0]
        fields = torch.linspace(0, np.deg2rad(hfov), n_fields)
        y_ref = lens.evaluate_paraxial_heights_at_image_plane(fields).squeeze()

        # Compute distortion
        distortion = (y_centroid.squeeze() - y_ref) / y_ref[-1]

        # Penalize
        residuals = (distortion.abs() - self.threshold).clip(min=0) / np.sqrt(n_fields)
        return residuals[residuals > 0]


class RimmerRelativeIlluminationResiduals(Residuals):
    """Residuals for bounding relative illumination using the Rimmer method."""

    name = "rimmer_relative_illumination"
    constraint = False
    data_keys = ()

    def __init__(self, weight: float | None, threshold: float):
        """Constructor.

        Args:
            weight: Weight on residuals.
            threshold: Minimum relative illumination.
        """
        super().__init__(weight)
        self.threshold = threshold

    def compute_loss(self, lens: optics.Lens, ray_initialization: ri.RayInitialization):
        """Forward pass.

        Args:
            lens: Lens.
            ray_initialization: Ray initialization.
        """
        r0, d0 = ray_initialization(lens, pupil_sampling_mode="tee")
        wavelengths = torch.tensor(ray_initialization.wavelengths).to(lens.c)
        r, d, ray_status, _ = next(lens.trace_rays(r0, d0, wavelengths, yield_on="end"))
        alpha_u = d[1, :, 0]
        alpha_l = d[1, :, 1]
        beta = -d[0, :, -1]
        beta_on_axis = beta[0]
        relative_illumination = (alpha_u - alpha_l) * beta / (2 * beta_on_axis**2)

        n_values = relative_illumination.numel()
        residuals = (self.threshold - relative_illumination).clip(min=0)
        residuals = residuals[residuals > 0] / np.sqrt(n_values)
        return residuals


class RelativeIlluminationResiduals(Residuals):
    """Residuals for bounding relative illumination using all rays from spot diagrams."""

    name = "relative_illumination"
    constraint = False
    data_keys = ()

    def __init__(self, weight: float | None, threshold: float):
        """Constructor.

        Args:
            weight: Weight on residuals.
            threshold: Minimum relative illumination.
        """
        super().__init__(weight)
        self.threshold = threshold

    def compute_loss(
        self, d: torch.Tensor, wavelength_weights: torch.Tensor | None = None
    ):
        """Forward pass.

        Args:
            d: Ray direction vectors at the image plane (shape: [3, n_fields, n_rays, n_wavelengths, 1]).
            wavelength_weights: Weights for wavelengths (shape: [n_wavelengths, 1]).
        """
        # Generalization of Rimmer relative illumination residuals
        # We use all traced rays instead of only marginal and sagittal rays
        # This prevents the scenario where the optimization attempts to correct only these rays
        # TODO: Remove invalid rays from computation
        dx, dy = d[:2]
        if wavelength_weights is None:
            wavelength_weights = 1 / d.shape[3]
        dy_mean = dy.mean(dim=(1, 3), keepdim=True)
        std_x = (dx**2 * wavelength_weights).sum(dim=(1, 2, 3)).sqrt()
        std_y = ((dy - dy_mean) ** 2 * wavelength_weights).sum(dim=(1, 2, 3)).sqrt()
        relative_illumination = std_x * std_y / (std_x[0:1] * std_y[0:1])
        n_values = relative_illumination.numel()
        residuals = (self.threshold - relative_illumination).clip(min=0)
        residuals = residuals[residuals > 0] / np.sqrt(n_values)
        return residuals


class GroupDelayResiduals(Residuals):
    """Residuals for bounding group delay of dispersion-engineered metasurfaces."""

    name = "group_delay"
    constraint = False
    data_keys = ("group_delay",)

    def __init__(self, weight: float | None, cutoff: float | tuple[float, float]):
        """Constructor.

        Args:
            weight: Weight on residuals.
            cutoff: Cutoff for group delay in fs.
        """
        super().__init__(weight)
        if isinstance(cutoff, (float, int)):
            self.cutoffs = (cutoff, cutoff)
        else:
            assert len(cutoff) == 2, "Cutoff must be a float or a tuple of two floats."
            self.cutoffs = cutoff

    def compute_loss(self, group_delay: torch.Tensor):
        """Forward pass.

        Args:
            group_delay: Group delay (shape: [*]).
        """
        if group_delay.numel() == 0:
            return torch.tensor([], device=group_delay.device)
        n_rays = group_delay.numel()
        min_cutoff, max_cutoff = self.cutoffs
        residuals = torch.maximum(
            group_delay - max_cutoff, min_cutoff - group_delay
        ).clip(min=0)
        residuals = residuals[residuals > 0] / np.sqrt(n_rays)
        return residuals


def point_to_mesh_distance(p: torch.Tensor, vertices: torch.Tensor):
    """Return the distance of a point to the mesh.

    Args:
        p: Points to compute the distance for (shape: [*, d]).
        vertices: Vertices of the alpha shape (shape: [n_simplices, d+1, d]),
                  where d is the dimensionality of the space and the simplex.

    Returns:
        Distance of the points to the mesh (shape: [*]).
    """
    d = vertices.shape[-1]
    assert vertices.shape[-2] == d + 1, (
        "The number of vertices must be d+1 to define a simplex."
    )
    assert p.shape[-1] == d, (
        "The dimensionality of the points must match the dimensionality of the space."
    )

    if d == 2:
        min_distance_to_simplices = point_to_triangle_distance(
            p[..., None, :], vertices
        )
    elif d == 3:
        min_distance_to_simplices = point_to_tetrahedron_distance(
            p[..., None, :], vertices
        )
    else:
        raise ValueError(
            "This function currently supports only 2D triangles and 3D tetrahedra."
        )

    # Return the minimum distance to the alpha shape
    return min_distance_to_simplices.min(dim=-1)[0]


def point_to_tetrahedron_distance(
    p: torch.Tensor, vertices: torch.Tensor, eps: float = 1e-6
):
    """Compute the minimum distance from a batch of points to a tetrahedron (3D).

    Args:
        p: Points (shape: [*, 3]).
        vertices: Vertices of the tetrahedron (shape: [*, 4, d]).
        eps: Small value for numerical stability.

    Returns:
        Minimum distance from the points to the tetrahedron (shape: [*]).
    """
    assert vertices.shape[-2] == 4, (
        "Four vertices are required to define a tetrahedron."
    )

    # Compute barycentric coordinates
    bary_coords = compute_barycentric_coordinates(p, vertices)

    # Check if projected points are inside the tetrahedron
    inside = (bary_coords > -eps).all(dim=-1) & (bary_coords.sum(dim=-1) < 1 + eps)

    # Compute distances to the faces for points outside the triangle
    triangle_vertices = torch.stack(
        (
            vertices[..., [0, 1, 2], :],
            vertices[..., [0, 1, 3], :],
            vertices[..., [0, 2, 3], :],
            vertices[..., [1, 2, 3], :],
        ),
        dim=0,
    )
    distance_to_faces = point_to_triangle_distance(p[..., None, :], triangle_vertices)
    min_face_distance = distance_to_faces.min(dim=-2).values

    return min_face_distance.where(~inside, 0.0)


def point_to_triangle_distance(
    p: torch.Tensor, vertices: torch.Tensor, eps: float = 1e-6
):
    """Compute the minimum distance from a batch of points to a triangle (2D or more).

    Args:
        p: Points (shape: [*, d]).
        vertices: Vertices of the triangle (shape: [*, 3, d]).
        eps: Small value for numerical stability.

    Returns:
        Minimum distance from the points to the triangle (shape: [*]).
    """
    d = p.shape[-1]
    assert vertices.shape[-2] == 3, "Three vertices are required to define a triangle."

    # Compute barycentric coordinates
    bary_coords = compute_barycentric_coordinates(p, vertices)

    # Check if projected points are inside the triangle
    inside = (bary_coords > -eps).all(dim=-1) & (bary_coords.sum(dim=-1) < 1 + eps)

    if d == 2:
        # In 2D, the distance to the plane (line) is always 0 for points inside the triangle
        distance_to_plane = 0.0
    else:
        # Compute distances to the plane for points inside the triangle (for d > 2)
        v0 = vertices[..., 0:1, :]  # shape: [*, 1, d]
        v = vertices[..., 1:, :] - v0  # shape: [*, 2, d]
        normal = torch.cross(v[..., 0, :], v[..., 1, :], dim=-1)
        normal_norm = normal.norm(dim=-1, keepdim=True)
        normal = normal / normal_norm
        distance_to_plane = ((p - v0.squeeze(dim=-2)) * normal).sum(dim=-1).abs()

    # Compute distances to the edges for points outside the triangle
    distance_to_edges = point_to_line_segment_distance(
        p[..., None, :], vertices, vertices.roll(shifts=-1, dims=-2)
    )
    min_edge_distance = distance_to_edges.min(dim=-1).values

    # Combine the results
    return min_edge_distance.where(~inside, distance_to_plane)


def point_to_line_segment_distance(p: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor):
    """Compute the distance from a point to a line segment.

    Args:
        p: Point (shape: [*, d]).
        v0: Start of the line segment (shape: [*, d]).
        v1: End of the line segment (shape: [*, d]).

    Returns:
        Distance from the point to the line segment (shape: [*]).
    """
    line_vec = v1 - v0
    p_vec = p - v0
    line_len_sq = (line_vec * line_vec).sum(dim=-1, keepdim=True)
    proj = (p_vec * line_vec).sum(dim=-1, keepdim=True) / line_len_sq
    proj_clamped = proj.clip(0, 1)
    closest_point = v0 + proj_clamped * line_vec
    return (p - closest_point).norm(dim=-1)


def check_inside_simplex(x: torch.Tensor, vertices: torch.Tensor):
    """Check if a point is inside a simplex.

    Args:
        x: Points to check (shape: [*, d]).
        vertices: Vertices of the simplex (shape: [*, d+1, d]),
                  where n is the dimensionality of the simplex and d is the dimensionality of the space.

    Returns:
        Boolean tensor indicating whether each point is inside the simplex (shape: [*]).
    """
    # Compute barycentric coordinates
    bary_coords = compute_barycentric_coordinates(x, vertices)

    # Check if the barycentric coordinates are valid (i.e., all >= 0 and sum <= 1)
    inside = (bary_coords >= 0).all(dim=-1) & (bary_coords.sum(dim=-1) <= 1)

    return inside


def compute_barycentric_coordinates(p: torch.Tensor, vertices: torch.Tensor):
    """Compute the barycentric coordinates of a point relative to a simplex.

    Args:
        p: Points to check (shape: [*, d]).
        vertices: Vertices of the simplex (shape: [*, d+1, d]),
                  where n is the dimensionality of the simplex and d is the dimensionality of the space.

    Returns:
        Barycentric coordinates of the point relative to the simplex (shape: [*, d+1]).
    """
    d = vertices.shape[-1]

    # Ensure the input shapes are compatible
    assert p.shape[-1] == d, (
        "Dimensionality of points p must match the simplex vertices."
    )

    # Subtract the first vertex from all vertices and points x
    v0 = vertices[..., 0:1, :]  # shape: [*, 1, d]
    v = vertices[..., 1:, :] - v0  # shape: [*, d, d]
    x_diff = p - v0.squeeze(dim=-2)  # shape: [*, d]

    # Solve the linear system to find the barycentric coordinates
    a = torch.einsum("...ij,...kj->...ik", v, v)  # Gram matrix, shape: [*, d, d]
    b = torch.einsum("...ij,...j->...i", v, x_diff)  # shape: [*, d]
    bary_coords = torch.linalg.solve(a, b[..., :, None]).squeeze(
        dim=-1
    )  # shape: [*, d]

    # Append the coordinate for the first vertex
    last_coord = 1 - bary_coords.sum(dim=-1, keepdim=True)
    bary_coords = torch.cat([last_coord, bary_coords], dim=-1)  # shape: [*, d+1]

    return bary_coords
