"""
Functions related to paraxial ray tracing, based on ABCD matrices
"""

import torch


def reduce_abcd(abcd: torch.Tensor):
    """Reduce the ABCD matrices efficiently.

    This function computes the product of the ABCD matrices in a divide-and-conquer fashion.

    Args:
        abcd: ABCD matrices (shape: [N, *, 2, 2]).
    """
    while abcd.shape[0] > 1:
        if abcd.shape[0] % 2 == 0:
            abcd = abcd[1::2] @ abcd[::2]
        else:
            abcd = torch.cat((abcd[1::2] @ abcd[:-1:2], abcd[-1:]), dim=0)

    return abcd.squeeze(0)


def reduce_abcd_cumulative(abcd: torch.Tensor):
    """Reduce the ABCD matrices one component at a time, and return all intermediate results.

    Args:
        abcd: ABCD matrices (shape: [N, *, 2, 2]).
    """
    progressive_abcd = torch.zeros_like(abcd)
    reduced_abcd = torch.eye(2, 2).to(abcd)[None, ...]
    for i, m in enumerate(abcd.unbind(dim=0)):
        reduced_abcd = m @ reduced_abcd
        progressive_abcd[i] = reduced_abcd

    return progressive_abcd


def interface_abcd(mu: torch.Tensor, c: torch.Tensor):
    """Return the ABCD matrix of a spherical interface.

    Args:
        mu: Refractive index (shape: [*]).
        c: Curvature (shape: [*]).
    """
    assert mu.shape == c.shape
    abcd = torch.eye(2).broadcast_to(*(*c.shape, 2, 2)).to(c).contiguous()
    abcd[..., 1, 0] = c * (mu - 1)
    abcd[..., 1, 1] = mu
    return abcd


def propagation_abcd(d: torch.Tensor):
    """Return the ABCD matrix of a propagation event.

    Args:
        d: Distance (shape: [*]).
    """
    abcd = torch.eye(2).broadcast_to(*(*d.shape, 2, 2)).to(d).contiguous()
    abcd[..., 0, 1] = d
    return abcd


def interface_propagation_abcd(c: torch.Tensor, t: torch.Tensor, mu: torch.Tensor):
    """Return the combined ABCD matrix of a spherical interface followed by a propagation.

    Args:
        c: Curvature (shape: [*]).
        t: Thickness (shape: [*]).
        mu: Refractive index (shape: [*]).
    """
    assert mu.shape[-1] == c.shape[-1] == t.shape[-1]

    D = mu  # D = n / n_prime
    C = c * (D - 1)  # C = c * (n - n_prime) / n_prime
    A = 1 + C * t  # A = 1 + t * c * (n - n_prime) / n_prime
    B = D * t  # B = t * n / n_prime

    abcd = torch.stack((A, B, C, D), dim=-1).view(mu.shape[0], -1, 2, 2)
    return abcd


def diffraction_abcd(p: torch.Tensor, wavelength_ratio: float):
    """Return the ABCD matrix of a diffractive interface.

    We assume a phase profile phi(r) = (2pi/lambda_0) (a0 * r^2 + ...),
        such that delta u = 2a0 * (lambda / lambda_0) * r + ...
    We only consider the a0 term for paraxial computations.

    Args:
        p: Diffractive phase profile (shape: [*]).
        wavelength_ratio: Wavelength ratio (lambda / lambda_0).
    """
    abcd = torch.eye(2).broadcast_to(*(*p.shape, 2, 2)).to(p).contiguous()
    abcd[..., 1, 0] = 2 * wavelength_ratio * p
    return abcd
