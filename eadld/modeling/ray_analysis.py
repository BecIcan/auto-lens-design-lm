import torch


def evaluate_mean_ray_height(
    y: torch.Tensor,
    ray_valid: torch.Tensor,
    reduce_dims: tuple[int, ...],
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    """Return mean ray height by properly handling ray failures.

    Args:
        y: "y" ray coordinates (shape: [*]).
        ray_valid: Boolean mask, whether rays should be considered (shape: [*]).
        reduce_dims: Dimensions corresponding to the statistics used to compute the ray height.
        weights: Weights for computing the centroid (shape: [*]).
        eps: Small value to avoid division by zero.
    """
    # Fill invalid configs with zeros
    y = y.where(ray_valid, 0.0)

    # Make sure rays have a weight of 1 on average
    if weights is not None:
        weights = weights / weights.mean()
    else:
        weights = 1

    y_weighted = y * weights
    ray_valid_weighted = ray_valid * weights
    # Compute the mean of rays from every field angle from valid configs only
    # Mean x is 0 for rotationally symmetric systems
    y_mean = y_weighted.sum(dim=reduce_dims, keepdim=True) / ray_valid_weighted.sum(
        dim=reduce_dims, keepdim=True
    ).clip(min=eps)
    return y_mean


def evaluate_transverse_ray_aberrations(
    x: torch.Tensor,
    y: torch.Tensor,
    ray_valid: torch.Tensor,
    reduce_dims: tuple[int, ...],
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    """Return transverse ray aberrations in a way that properly handles ray failures.

    Args:
        x: "x" ray coordinates at image plane (shape: [*]).
        y: "y" ray coordinates at image plane (shape: [*]).
        ray_valid: Boolean mask, whether rays should be considered (shape: [*]).
        weights: Weights for computing the centroid (shape: [*]).
        reduce_dims: Dimensions corresponding to the statistics used to compute the spot size.
        eps: Small value to avoid division by zero.
    """
    y_centroid = evaluate_mean_ray_height(y, ray_valid, reduce_dims, weights, eps=eps)

    # Center rays and set invalid rays to 0
    y = y - y_centroid
    # ``inf * 0`` is NaN; failed rays must be replaced rather than multiplied out.
    x = x.where(ray_valid, 0.0)
    y = y.where(ray_valid, 0.0)

    return x, y


def compute_rms_spot_size(
    x: torch.Tensor,
    y: torch.Tensor,
    ray_valid: torch.Tensor,
    reduce_dims: tuple[int, ...],
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    """Return RMS spot size for every ray bundle.

    Args:
        x: "x" ray coordinates at image plane (shape: [*]).
        y: "y" ray coordinates at image plane (shape: [*]).
        ray_valid: Boolean mask, whether rays should be considered (shape: [*]).
        reduce_dims: Dimensions corresponding to the statistics used to compute the spot size.
        weights: Quadrature and/or spectral weights, broadcastable to ``x`` (shape: [*]).
        eps: Small value to avoid square root of 0.
    """
    x, y = evaluate_transverse_ray_aberrations(
        x, y, ray_valid, reduce_dims, weights=weights, eps=eps
    )
    if weights is None:
        weights = torch.ones_like(x)
    else:
        weights = torch.broadcast_to(weights, x.shape)
    valid_weights = weights * ray_valid
    denominator = valid_weights.sum(dim=reduce_dims).clip(min=eps)
    x_var = (x**2 * weights).sum(dim=reduce_dims) / denominator
    y_var = (y**2 * weights).sum(dim=reduce_dims) / denominator
    rms = (x_var + y_var).clip(min=eps**2).sqrt()
    return rms


def compute_psfs(
    x: torch.Tensor,
    y: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    kernel_type: str = "linear",
    sigma_rel: float = 1.0,
    normalize: bool = False,
    eps: float = 1e-6,
):
    """Compute PSFs from spot diagrams using kernel density estimation.

    With either linear or cosine kernel, each ray contributes equally to the energy.
    With Gaussian kernel, the energy contribution of a ray depends on where it hits.
    Note that with the Gaussian kernel, unlike the other two, the energy of each ray is not conserved.
    Assume symmetry in the x-axis.

    Args:
        x: Centered spot diagram coordinates in "x" (shape: [*, n_channels, n_rays]).
        y: Centered spot diagram coordinates in "y" (shape: [*, n_channels, n_rays]).
        x_grid: Centered coordinates in "x" (shape: [*, n_x]).
        y_grid: Centered coordinates in "y" (shape: [*, n_y]).
        kernel_type: Type of kernel to use ('gaussian', 'linear', or 'cosine').
        sigma_rel: Multiplier for the kernel sigma (if 1, sigma is equal to the size of a bin).
        normalize: Whether to normalize the PSFs to have unit area regardless of missed rays.
        eps: Small value to avoid division by zero.
    """
    # Find grid span
    span_x = x_grid.max(dim=-1)[0] - x_grid.min(dim=-1)[0]
    span_y = y_grid.max(dim=-1)[0] - y_grid.min(dim=-1)[0]

    # Compute distance
    dist_x = x[..., None] - x_grid[..., None, None, :]
    dist_y = y[..., None, :] - y_grid[..., None]

    # Compute separable kernels
    bin_width = span_x / (x_grid.shape[-1] - 1)
    bin_height = span_y / (y_grid.shape[-1] - 1)
    sigma_x = sigma_rel * bin_width
    sigma_y = sigma_rel * bin_height
    if kernel_type == "gaussian":
        kernel_x = (-(dist_x**2 / sigma_x**2) / 2).exp()
        kernel_y = (-(dist_y**2 / sigma_y**2) / 2).exp()
    elif kernel_type == "linear":
        kernel_x = (1 - dist_x.abs() / sigma_x).clip(min=0) / sigma_rel
        kernel_y = (1 - dist_y.abs() / sigma_y).clip(min=0) / sigma_rel
    elif kernel_type == "cosine":
        arg_x = torch.pi / 2 * (dist_x / sigma_x).clip(min=-1 - eps, max=1 + eps)
        arg_y = torch.pi / 2 * (dist_y / sigma_y).clip(min=-1 - eps, max=1 + eps)
        kernel_x = arg_x.cos().clip(min=0.0) ** 2 / sigma_rel
        kernel_y = arg_y.cos().clip(min=0.0) ** 2 / sigma_rel
    else:
        raise ValueError(f'Unknown kernel type "{kernel_type}"')

    # Apply kernel density estimation
    kernels = kernel_y @ kernel_x

    # Apply symmetry over the x-axis
    kernels = kernels + kernels.flip(dims=(-1,))

    # Normalize
    if normalize:
        # Normalize to have unit area
        kernels = kernels / kernels.sum(dim=(-1, -2), keepdims=True).clip(min=eps)
    else:
        # Normalize according to the number of rays
        kernels = kernels / (x.shape[-1] * 2)

    return kernels


def define_psf_grid(
    n_x_bins: int, n_y_bins: int, x_increment: torch.Tensor, y_increment: torch.Tensor
):
    """Return grid coordinates for PSF sampling.

    Args:
        n_x_bins: Number of bins in x.
        n_y_bins: Number of bins in y.
        x_increment: Physical size of a bin in x (shape: [*]).
        y_increment: Physical size of a bin in y (shape: [*]).
    """
    # Compute the pixel center coordinates
    # Grid coords in y are upside down to follow image coordinate convention
    x_grid = (torch.arange(n_x_bins).to(x_increment) + 0.5 - n_x_bins / 2) * x_increment
    y_grid = (n_y_bins / 2 - torch.arange(n_y_bins).to(y_increment) - 0.5) * y_increment

    return x_grid, y_grid
