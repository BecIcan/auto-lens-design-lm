import torch
import numpy as np
import math


class MiscSurfaceModel(torch.nn.Module):
    """Base class for all miscellaneous optical surface models."""

    def __init__(self):
        super().__init__()

    def get_tabular_parameters(self, p: torch.Tensor):
        """Get the parameters in tabular form for exporting.

        Args:
            p: Parameters of the miscellaneous surface (shape: [n_lens, n_parameters]).
        """
        return NotImplementedError


class DispersionEngineeredMetasurface(MiscSurfaceModel):
    """Idealized model to explore dispersion-engineered metasurfaces."""

    def __init__(
        self,
        normalization_radius: float = 1.0,
        w0: float = 550.0,
        bezier_parameterization: bool = False,
    ):
        """Constructor.

        Args:
            normalization_radius: Normalization radius in mm.
            w0: Design wavelength in nm.
            bezier_parameterization: Whether to use a Bézier parameterization for the metasurface parameters.
        """
        super().__init__()
        self.normalization_radius = normalization_radius
        self.w0 = w0
        self.bezier_parameterization = bezier_parameterization

    def forward(
        self,
        r: torch.Tensor,
        d: torch.Tensor,
        p: torch.Tensor,
        w: torch.Tensor,
        eps: float = 1e-6,
    ):
        """Return the ray directions and intermediate information after passing through the metasurface(s).

        Args:
            r: Ray position vectors (shape: [3, *, n_wavelengths, n_lens]).
            d: Ray direction vectors in normalized form (shape: [3, *, n_wavelengths, n_lens]).
            p: Parameters of the metasurface (shape: [n_lens, n_parameters]).
            w: Wavelength in nm (shape: [n_wavelengths]).
            eps: Small number to avoid division by zero.
        """
        coefficients = self.retrieve_phase_coefficients(p, w)
        phase_gradient = self.compute_phase_gradient(r, coefficients)

        rho_squared = (r[:2] ** 2).sum(dim=0)
        xi = rho_squared / self.normalization_radius**2
        powers = torch.arange(coefficients.shape[-1], device=p.device) + 1
        phase = torch.sum(coefficients * xi[..., None] ** powers, dim=-1)
        phase_opd = w.view(-1, 1) * 1e-6 * phase / (2 * np.pi)

        w = w.view(-1, 1) * 1e-6  # Convert wavelength from nm to mm
        lm = d[:2] + w / (2 * np.pi) * phase_gradient

        # Check for failures (ray directions going backward)
        # n corresponds to the cosine of the angle of "refraction"
        n_square = 1 - (lm**2).sum(dim=0)
        failures = n_square - eps < 0

        n = n_square.clip(min=eps).sqrt()

        d = torch.cat((lm, n[None, ...]), dim=0)

        return_dict = {
            "group_delay": self.group_delay(r[:2].norm(dim=0), p) * 1e15,
            "phase_opd": phase_opd,
        }
        return d, failures, n_square, return_dict

    def get_coefficients(self, p: torch.Tensor):
        """Retrieve coefficients from metasurface parameters.

        Args:
            p: Parameters of the metasurface (shape: [n_lens, n_parameters]).
        """
        n_parameters = p.shape[-1]
        split = n_parameters // 2
        nominal_phase_coefficients = p[..., :split]
        group_delay_coefficients = p[..., split:]
        if self.bezier_parameterization:
            nominal_phase_coefficients = self.apply_bezier_parameterization(
                nominal_phase_coefficients
            )
            group_delay_coefficients = self.apply_bezier_parameterization(
                group_delay_coefficients
            )
        return nominal_phase_coefficients, group_delay_coefficients

    def retrieve_phase_coefficients(self, p: torch.Tensor, w: torch.Tensor):
        """Retrieve phase profile coefficients from metasurface parameters.

        Args:
            p: Parameters of the metasurface (shape: [n_lens, n_parameters]).
            w: Wavelength in nm (shape: [n_wavelengths]).
        """
        nominal_phase_coefficients, group_delay_coefficients = self.get_coefficients(p)
        coefficients = nominal_phase_coefficients + group_delay_coefficients * (
            self.w0 / w.view(-1, 1, 1) - 1
        )
        return coefficients

    def apply_bezier_parameterization(self, p: torch.Tensor):
        """Apply Bézier transformation to the metasurface parameters.

        Args:
            p: Parameters of the metasurface (shape: [n_lens, n_parameters]).
        """
        n_params = p.shape[-1]
        n = n_params - 1
        bezier_matrix = np.zeros((n_params,) * 2)
        for i in range(n_params):
            for j in range(n_params):
                if i >= j:
                    bezier_matrix[i, j] = (
                        math.comb(n, j)
                        * math.comb(n - j, i - j)
                        * (((i - j) % 2) * -2 + 1)
                    )
        bezier_matrix = torch.tensor(bezier_matrix).to(p)
        p = (bezier_matrix @ p.T).T
        return p

    def compute_phase_gradient(self, r: torch.Tensor, p: torch.Tensor):
        """Return the phase gradient.

        The expression for the phase is:
            phi(rho) = p_0 * (rho/rho_max) ** 2 + p_1 * (rho/rho_max) ** 4 + ...

        We make the substitution xi = (rho/rho_max) ** 2.
            phi(xi) = p_0 * xi + p_1 * xi ** 2 + ...
            d phi/d xi = p_0 + 2 * p_1 * xi + 3 * p_2 * xi ** 2 + ...
            d phi/d rho^2 = d phi/d xi / rho_max ** 2

        For numerical stability, we use the following form:
            (d phi/d rho) x/rho = (d phi/d rho^2) 2x
            (d phi/d rho) y/rho = (d phi/d rho^2) 2y

        Args:
            r: Ray position vectors (shape: [3, *, n_wavelengths, n_lens]).
            p: Parameters of the metasurface (shape: [n_lens, n_parameters]).
        """
        rho_squared = (r[:2] ** 2).sum(dim=0)
        xi = rho_squared / self.normalization_radius**2
        n_parameters = p.shape[-1]
        powers = torch.arange(n_parameters, device=p.device)  # 0, 1, 2, ...
        coeffs = powers + 1
        raised_xi = xi[..., None].expand(*xi.shape, len(powers)) ** powers
        terms = coeffs * p * raised_xi
        derivative = torch.sum(terms, dim=-1) / self.normalization_radius**2
        phase_gradient = 2 * r[:2] * derivative
        return phase_gradient

    def get_abcd(self, p: torch.Tensor, w: float):
        """Compute the ABCD matrix of the metasurface(s).

        Necessary for first-order computations, e.g., to accurately compute the focal length.

        Args:
            p: Parameters of the metasurface(s) (shape: [n_metasurfaces, n_lens, n_parameters]).
            w: Wavelength in nm.
        """
        w = torch.tensor(w).to(p).view(-1)
        shape = p.shape
        n_surfaces, n_lens, n_parameters = shape

        coefficients = self.retrieve_phase_coefficients(
            p.view(-1, n_parameters), w
        ).view(*shape[:-1], -1)
        first_order_coefficients = coefficients[..., 0]
        scaled_coefficients = first_order_coefficients / self.normalization_radius**2
        c_component = scaled_coefficients * w * 1e-6 / np.pi

        ones = torch.ones_like(c_component)
        zeros = torch.zeros_like(c_component)
        abcd = torch.stack((ones, zeros, c_component, ones), dim=-1)
        return abcd.view(*abcd.shape[:-1], 2, 2)

    def phase_at_design_wavelength(self, rho: torch.Tensor, p: torch.Tensor):
        """Return the phase of the metasurface(s) at the design wavelength.

        The expression for the phase is:
            phi(rho) = p_0 * (rho/rho_max) ** 2 + p_1 * (rho/rho_max) ** 4 + ...

        We make the substitution xi = (rho/rho_max) ** 2.
            phi(xi) = p_0 * xi + p_1 * xi ** 2 + ...

        Args:
            rho: Radial ray coordinates (shape: [*, n_lens]).
            p: Parameters of the metasurface (shape: [n_lens, n_parameters]).
        """
        # Retrieve nominal phase coefficients
        nominal_phase_coefficients, _ = self.get_coefficients(p)

        # Compute phase at design wavelength
        powers = (
            torch.arange(nominal_phase_coefficients.shape[-1], device=p.device) + 1
        )  # 1, 2, 3, ...
        xi = rho**2 / self.normalization_radius**2
        raised_xi = xi[..., None].expand(*rho.shape, len(powers)) ** powers
        phase = torch.sum(nominal_phase_coefficients * raised_xi, dim=-1)
        return phase

    def group_delay(self, rho: torch.Tensor, p: torch.Tensor):
        """Return the phase delay of the metasurface(s).

        The expression for the group delay is:
            d phi(rho) / d w = p_0 * (rho/rho_max) ** 2 + p_1 * (rho/rho_max) ** 4 + ...

        We make the substitution xi = (rho/rho_max) ** 2.
            phi(xi) = p_0 * xi + p_1 * xi ** 2 + ...

        Args:
            rho: Radial ray coordinates (shape: [*, n_lens]).
            p: Parameters of the metasurface (shape: [n_lens, n_parameters]).
        """
        # Retrieve group delay coefficients
        _, group_delay_coefficients = self.get_coefficients(p)

        # Compute group delay
        powers = (
            torch.arange(group_delay_coefficients.shape[-1], device=p.device) + 1
        )  # 1, 2, 3, ...
        xi = rho**2 / self.normalization_radius**2
        raised_xi = xi[..., None].expand(*rho.shape, len(powers)) ** powers
        group_delay = torch.sum(group_delay_coefficients * raised_xi, dim=-1)
        omega0 = 2 * np.pi * 3e8 / (self.w0 * 1e-9)
        return group_delay / omega0

    def get_tabular_parameters(self, p: torch.Tensor):
        """Return the scaled diffractive surface coefficients.

        The scaled coefficients are such that:
            phi(rho, p') = 2pi / w0 * (p'_0 * rho ** 2 + p'_1 * rho ** 4 + ...)

        In the default model, we have
            phi(rho, p) = p_0 * (rho/rho_max) ** 2 + p_1 * (rho/rho_max) ** 4 + ...

        As such,
            p_i' = w0 / 2pi / rho_max ** (2 (i + 1)) * p_i

        Group delay coefficients are ignored.

        Args:
            p: Parameters of the metasurface (shape: [n_lens, n_parameters]).
        """
        nominal_phase_coefficients, _ = self.get_coefficients(p)
        powers = (
            torch.arange(nominal_phase_coefficients.shape[-1], device=p.device) + 1
        )  # 1, 2, 3, ...
        scale_factors = self.normalization_radius ** (2 * powers)
        w0 = self.w0 * 1e-6
        scaled_coefficients = (
            w0 / (2 * np.pi) / scale_factors * nominal_phase_coefficients
        )
        return scaled_coefficients


class GeneralizedSnellModel(MiscSurfaceModel):
    """Base class for all optical surfaces using the generalized law of refraction and rotational symmetry."""

    def __init__(self):
        """Constructor."""
        super().__init__()

        # Compile the gradient functions
        self.compute_scaled_phase_gradient = torch.func.grad(
            lambda *inputs: self.compute_phase(*inputs).sum(), 0
        )
        self.compute_c_component = torch.func.grad(
            lambda *inputs: self.u_from_y(*inputs).sum(), 0
        )

    def forward(
        self,
        r: torch.Tensor,
        d: torch.Tensor,
        p: torch.Tensor,
        w: torch.Tensor,
        eps: float = 1e-6,
    ):
        """Return the ray directions and intermediate information after passing through the surface(s).

        Generalized law of refraction:
            n' sin(i') - n sin(i) = (lambda/2pi) (d phi/d rho)
            where rho^2 = x^2 + y^2

        With a flat interface surrounded by air, the direction cosines are updated following:
            l' = l + (lambda/2pi) (d phi/d rho) x/rho
            m' = m + (lambda/2pi) (d phi/d rho) y/rho

        Args:
            r: Ray position vectors (shape: [3, *, n_wavelengths, n_lens]).
            d: Ray direction vectors in normalized form (shape: [3, *, n_wavelengths, n_lens]).
            p: Parameters of the surface (shape: [n_lens, n_parameters]).
            w: Wavelength in nm (shape: [n_wavelengths]).
            eps: Small number to avoid division by zero.
        """
        rho_squared = (r[:2] ** 2).sum(dim=0)
        phase = self.compute_phase(rho_squared, d[2], p, w)
        phase_opd = w.view(-1, 1) * 1e-6 * phase / (2 * np.pi)
        phase_gradient = self.compute_phase_gradient(r, d, p, w)
        w = w.view(-1, 1) * 1e-6  # Convert wavelength from nm to mm
        lm = d[:2] + w / (2 * np.pi) * phase_gradient

        # Check for failures (ray directions going backward)
        # n corresponds to the cosine of the angle of "refraction"
        n_square = 1 - (lm**2).sum(dim=0)
        failures = n_square - eps < 0

        n = n_square.clip(min=eps).sqrt()

        d = torch.cat((lm, n[None, ...]), dim=0)

        return d, failures, n_square, {"phase_opd": phase_opd}

    def compute_phase_gradient(
        self, r: torch.Tensor, d: torch.Tensor, p: torch.Tensor, w: torch.Tensor
    ):
        """Return phase gradients (d phi/d rho) x/rho and (d phi/d rho) y/rho.

        For numerical stability, we use the following form:
            (d phi/d rho) x/rho = (d phi/d rho^2) 2x
            (d phi/d rho) y/rho = (d phi/d rho^2) 2y

        Args:
            r: Ray position vectors (shape: [3, *, n_wavelengths, n_lens]).
            d: Ray direction vectors in normalized form (shape: [3, *, n_wavelengths, n_lens]).
            p: Parameters of the surface (shape: [n_lens, n_parameters]).
            w: Wavelength in nm (shape: [n_wavelengths]).
        """
        rho_squared = (r[:2] ** 2).sum(dim=0)
        cos_i = d[2]

        with torch.inference_mode(False):
            scaled_phase_gradient = self.compute_scaled_phase_gradient(
                rho_squared, cos_i, p, w.clone()
            )
        phase_gradient = 2 * r[:2] * scaled_phase_gradient
        return phase_gradient

    def compute_phase(
        self,
        rho_squared: torch.Tensor,
        cos_i: torch.Tensor,
        p: torch.Tensor,
        w: torch.Tensor,
    ):
        """Return the phase of the surface(s).

        Args:
            rho_squared: Radial ray coordinates squared (shape: [*, n_wavelengths, n_lens]).
            cos_i: Cosine of the angle of incidence (shape: [*, n_wavelengths, n_lens]).
            p: Parameters of the surface (shape: [n_lens, n_parameters]).
            w: Wavelength in nm (shape: [n_wavelengths]).
        """
        return NotImplementedError

    def get_abcd(self, p: torch.Tensor, w: float):
        """Compute the ABCD matrix of the surface(s).

        Necessary for first-order computations, e.g., to accurately compute the focal length.

        Args:
            p: Parameters of the surface(s) (shape: [n_surfaces, n_lens, n_parameters]).
            w: Wavelength in nm.
        """
        w = torch.tensor(w).to(p)
        shape = p.shape
        n_surfaces, n_lens, n_parameters = shape

        y = p.new_zeros((1, 1, 1))
        with torch.inference_mode(False):
            c_component = self.compute_c_component(y, p.view(-1, n_parameters), w).view(
                *shape[:-1]
            )

        ones = torch.ones_like(c_component)
        zeros = torch.zeros_like(c_component)
        abcd = torch.stack((ones, zeros, c_component, ones), dim=-1)
        return abcd.view(*abcd.shape[:-1], 2, 2)

    def u_from_y(self, y: torch.Tensor, p: torch.Tensor, w: torch.Tensor):
        """Return direction cosine in "y" based on the ray height "y".

        To be used with torch.func.grad() to compute the C component of the ABCD matrix.

        Args:
            y: Ray height (shape: [*, n_wavelengths, n_lens]).
            p: Parameters of the surface(s) (shape: [n_surfaces, n_lens, n_parameters]).
            w: Wavelength in nm (shape: [n_wavelengths]).
        """
        r = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), dim=0)
        d = torch.stack(
            (torch.zeros_like(y), torch.zeros_like(y), torch.ones_like(y)), dim=0
        )
        d_prime = self.forward(r, d, p, w)[0]
        u = d_prime[1]
        return u


class DiffractiveOpticalElement(GeneralizedSnellModel):
    """Implementation for an ideal diffractive optical element."""

    def __init__(self, w0: float = 550, normalization_radius: float = 1.0):
        """Constructor.

        Args:
            w0: Reference wavelength in nm.
            normalization_radius: Normalization radius in mm.
        """
        super().__init__()
        self.w0 = w0  # Reference wavelength in nm
        self.normalization_radius = normalization_radius

    def compute_phase(
        self,
        rho_squared: torch.Tensor,
        cos_i: torch.Tensor,
        p: torch.Tensor,
        w: torch.Tensor,
    ):
        """Return the phase of the diffractive optical element.

        The phase is given by:
            phi(r) = 2pi/lambda_0 * (p_0 r^2 + p_1 r^4 + p_2 r^6 + ...)

        Args:
            rho_squared: Radial ray coordinates squared (shape: [*, n_wavelengths, n_lens]).
            cos_i: Cosine of the angle of incidence (shape: [*, n_wavelengths, n_lens]).
            p: Parameters of the diffractive optical element (shape: [n_lens, n_parameters]).
            w: Wavelength in nm (shape: [n_wavelengths]).
        """
        phase = 0
        rho_squared_normalized = rho_squared / self.normalization_radius**2
        for i, pp in enumerate(p.unbind(dim=-1)):
            phase = phase + pp * rho_squared_normalized ** (i + 1)
        phase = phase * 2 * np.pi / (self.w0 * 1e-6)
        return phase
