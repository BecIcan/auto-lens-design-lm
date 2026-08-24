import math

import torch
import numpy as np
import pandas as pd
import scipy.spatial

from eadld.modeling import optics, misc_surfaces as ms


class LensParameterization(torch.nn.Module):
    """Parameterization of a lens system for optimization purposes."""

    def __init__(
        self,
        # Lens initialization
        lens_sequence: str,
        s: list[float],
        nd: list[float],
        vd: list[float],
        c: list[float],
        a: list[list[float]],
        d: list[list[float]],
        m: list[list[float]],
        nominal_wavelength: float,
        dpgf: list[float] | None,
        z: list[list[list[float]]] | None = None,
        misc_surface_model: ms.MiscSurfaceModel | None = None,
        bezier_aspherics: bool = False,
        # Parameterization tricks
        target_efl: float | None = None,
        solve_type: str = "focal_length",
        solve_idx: int | None = None,
        paraxial_image_solve: bool = False,
        total_track_length_solve: float | None = None,
        qc_vars: bool = False,
        glass_file: str | None = None,
        freeze: dict[str, bool | list | dict] | None = None,
        scale_factor: float = 1,
    ):
        """Constructor.

        Args:
            lens_sequence: Sequence of optical components in the lens, from object space to image space.
            s: Initial spacings.
            nd: Initial refractive indices.
            vd: Initial Abbe numbers.
            c: Initial curvatures.
            a: Initial aspherical coefficients (including the conic constant).
            d: Initial diffractive optical element parameters.
            m: Initial miscellaneous surface parameters.
            nominal_wavelength: Used for normalization and for paraxial computations (EFL, BFL, etc.).
            dpgf: Initial deviation from normal partial dispersion.
            z: Initial zonal surface parameters [delta_A1, A2, delta_Z, Rmax]
                (shape: [n_zonal, n_zones, 4]).
            misc_surface_model: MiscSurfaceModel object to allow ray tracing through miscellaneous surfaces.
            bezier_aspherics: Whether to use Bezier parameterization for the aspherical coefficients.
            target_efl: Desired EFL (in mm); can be enforced using the "solve_idx" parameter;
                if None, compute automatically using the lens parameters provided.
            solve_type: Type of solve ('focal_length' or 'image_height') to enforce the target EFL;
                with image_height, an equivalent paraxial image height is obtained from the nominal target EFL.
            solve_idx: Index of the refractive surface whose curvature will be computed to enforce the target EFL;
                if None, there is no solve on curvatures.
            paraxial_image_solve: Whether to use the paraxial image solve.
            total_track_length_solve: If not None, total track length (mm) is enforced by solving the last spacing.
            qc_vars: Whether to use quantized continuous glass variables;
                used to enforce the use of glass materials found in glass_file.
            glass_file: Path of the .csv glass file containing the available glass materials.
            freeze: Dict to specify whether variables should be frozen.
            scale_factor: Scale factor for the variables (in mm); EPD / 2 is usually a good starting point.
        """
        super().__init__()

        # Optimization tricks
        self.solve_idx = solve_idx
        self.paraxial_image_solve = paraxial_image_solve
        self.total_track_length_solve = total_track_length_solve
        assert not (
            self.paraxial_image_solve and self.total_track_length_solve is not None
        ), "Cannot use both paraxial image solve and total track length solve."

        # Refractive materials
        self.glass_model = GlassModel(glass_file, assume_normal_dispersion=dpgf is None)
        if self.glass_model.catalog_g is None:
            assert qc_vars is False, "No glass catalog available."
        self.qc_vars = qc_vars

        # Lens initialization
        self.sequence = optics.LensSequence(lens_sequence)
        self.parameter_keys = ("s", "nd", "vd", "dpgf", "c", "a", "d", "m", "z")
        if dpgf is None:
            dpgf = [0.0] * len(nd)
        parameters = (s, nd, vd, dpgf, c, a, d, m)
        initial_parameters = {
            k: torch.tensor(v) for k, v in zip(self.parameter_keys[:-1], parameters)
        }
        n_lens = initial_parameters["s"].shape[1] if initial_parameters["s"].ndim > 1 else 1
        if z is None or len(z) == 0:
            z_tensor = initial_parameters["s"].new_empty((0, n_lens, 0, 4))
        else:
            z_tensor = torch.tensor(z)
            if z_tensor.ndim == 3:
                if z_tensor.shape[-1] != 4:
                    raise ValueError(
                        "Zonal surface parameters must contain "
                        "[delta_A1, A2, delta_Z, Rmax]."
                    )
                z_tensor = z_tensor[:, None].expand(-1, n_lens, -1, -1).clone()
            elif z_tensor.ndim == 4:
                if z_tensor.shape[-1] != 4 or z_tensor.shape[1] not in (1, n_lens):
                    raise ValueError(
                        "Zonal surface parameters must have shape "
                        "[n_zonal, n_zones, 4] or "
                        "[n_zonal, n_lens, n_zones, 4]."
                    )
                if z_tensor.shape[1] == 1 and n_lens > 1:
                    z_tensor = z_tensor.expand(-1, n_lens, -1, -1).clone()
            else:
                raise ValueError(
                    "Zonal surface parameters must have shape "
                    "[n_zonal, n_zones, 4]."
                )
        if z_tensor.shape[0] != self.sequence.n_zonal:
            raise ValueError(
                f"Expected {self.sequence.n_zonal} zonal surfaces, "
                f"got {z_tensor.shape[0]}."
            )
        initial_parameters["z"] = z_tensor
        self.w0 = nominal_wavelength
        self.misc_surface_model = misc_surface_model

        # Initialize the Lens object for paraxial computations
        lens = optics.Lens(
            self.sequence,
            w0=self.w0,
            misc_surface_model=self.misc_surface_model,
            **{k: v for k, v in initial_parameters.items() if k in self.parameter_keys},
        )
        if target_efl is None:
            self.target_efl = lens.efl
        else:
            self.target_efl = target_efl

        # If total track length solve, modify the last spacing directly to enforce the desired total track length
        if self.total_track_length_solve is not None:
            self.update_last_spacing_inplace(lens, self.total_track_length_solve)
        # If curvature solve, modify the curvature directly to enforce the desired focal length
        self.solve_type = solve_type
        if solve_idx is not None:
            self.update_curvature_inplace(lens)

        if bezier_aspherics:
            self.register_bezier_aspherics(initial_parameters)

        # Get the normalized variables
        self.variable_keys = ("s", "g", "c", "a", "d", "m", "z")
        self.scale_factor = scale_factor
        bezier_matrix_inv = self.bezier_matrix_inv if bezier_aspherics else None
        lens_variable_dict = get_normalized_variables(
            lens,
            self.scale_factor,
            self.glass_model,
            self.paraxial_image_solve,
            bezier_matrix_inv,
        )
        initial_variable_list = torch.cat(
            [lens_variable_dict[k].view(-1) for k in self.variable_keys]
        )

        # Register which variables are frozen
        self.variable_shape_dict = {k: v.shape for k, v in lens_variable_dict.items()}
        self.freeze_dict = parse_freeze_dict(freeze, lens_variable_dict)
        if self.freeze_dict["z"].numel() > 0:
            # Zone boundaries define the table topology and cannot be optimized.
            self.freeze_dict["z"][..., 3] = True
            # The first zone is the reference for cumulative A1 and Z offsets.
            # delta_A1 of the first zone is only degenerate with the
            # base curvature of the surface it modifies.  When that curvature is
            # frozen -- a flat plate whose power must come from the zones -- the
            # degeneracy is gone and forcing the gauge would keep the central
            # zone flat for ever.  delta_Z stays frozen either way: it is
            # degenerate with a global piston, which is never observable.
            zonal_curvatures = [
                event["c"]
                for event in self.sequence.events
                if event["type"] == "r" and "z" in event
            ]
            for index, curvature in enumerate(zonal_curvatures):
                if bool(self.freeze_dict["c"].view(-1)[curvature]):
                    continue
                self.freeze_dict["z"][index, :, 0, 0] = True
            self.freeze_dict["z"][:, :, 0, 2] = True
            # Non-positive Rmax marks a padded zone; freeze the complete row.
            padding = lens_variable_dict["z"][..., 3] <= 0
            self.freeze_dict["z"] |= padding[..., None]
        optimization_mask = torch.cat(
            [~self.freeze_dict[k].view(-1) for k in self.variable_keys]
        )

        # Register the variables
        self._lens_variables = torch.nn.Parameter(
            initial_variable_list[optimization_mask]
        )
        self.register_buffer("initial_variables", initial_variable_list)
        self.register_buffer("optimization_mask", optimization_mask)

    @property
    def lens_variables(self):
        """Lens variable dict the combines the frozen and optimized variables."""
        lens_variable_list = self.initial_variables.clone()
        lens_variable_list[self.optimization_mask] = self._lens_variables
        i = 0
        lens_variable_dict = {}
        for k in self.variable_keys:
            shape = self.variable_shape_dict[k]
            n_items = shape.numel()
            lens_variable_dict[k] = lens_variable_list[i : i + n_items].view(shape)
            i += n_items
        return lens_variable_dict

    @property
    def lens(self):
        """Lens object constructed from the current lens variables."""
        lens_parameters = self.scale_lens_parameters()
        lens = self.generate_lens_from_lens_parameters(lens_parameters)
        apply_tolerancing = False
        if apply_tolerancing:
            import tolerancing

            tolerancing_model = tolerancing.TolerancingModel(self.sequence)
            lens = tolerancing_model.apply_tolerancing(lens)
        return lens

    def generate_lens_from_lens_parameters(
        self, lens_parameters: dict[str, torch.Tensor]
    ):
        """Return a Lens object from the lens parameters.

        Args:
            lens_parameters: Dictionary of lens parameters.
        """
        lens = optics.Lens(
            sequence=self.sequence,
            w0=self.w0,
            misc_surface_model=self.misc_surface_model,
            **lens_parameters,
        )

        if self.total_track_length_solve is not None:
            # If total track length solve, modify the last spacing directly
            self.update_last_spacing_inplace(lens, self.total_track_length_solve)

        if self.solve_idx is not None:
            # If curvature solve, modify the curvatures directly
            self.update_curvature_inplace(lens)

        if self.paraxial_image_solve:
            # If paraxial image solve, modify the last spacings directly
            self.update_last_spacing_inplace(lens)

        return lens

    def register_bezier_aspherics(self, initial_parameters: dict[str, torch.Tensor]):
        """Register the Bezier matrix and its inverse for aspherical coefficients.

        Args:
            initial_parameters: Initial lens parameters.
        """
        # See https://doi.org/10.1080/16583655.2019.1601913
        n_aspherical_params = max(initial_parameters["a"].shape[-1] - 1, 0)
        n = n_aspherical_params - 1

        a = np.zeros((n_aspherical_params,) * 2)
        for i in range(n_aspherical_params):
            for j in range(n_aspherical_params):
                if i >= j:
                    a[i, j] = (
                        math.comb(n, j)
                        * math.comb(n - j, i - j)
                        * (((i - j) % 2) * -2 + 1)
                    )
        a_inv = np.linalg.inv(a)
        self.register_buffer("bezier_matrix", torch.tensor(a))
        self.register_buffer("bezier_matrix_inv", torch.tensor(a_inv))

    def update_curvature_inplace(self, lens: optics.Lens):
        """Update the curvature of the refractive surface at index "solve_idx" to enforce the target EFL.

        Args:
            lens: Lens object.
        """
        solve_idx = self.solve_idx
        if solve_idx < 0:
            solve_idx = lens.sequence.n_interfaces + solve_idx
        new_c = lens.curvature_solve(solve_idx, self.target_efl, self.solve_type)
        lens.c = torch.cat(
            (lens.c[:solve_idx], new_c[None, ...], lens.c[solve_idx + 1 :]), dim=0
        )

    def update_last_spacing_inplace(
        self, lens: optics.Lens, target_ttl: float | None = None
    ):
        """Update the last spacing, either for paraxial image solve or total track length solve.

        Args:
            lens: Lens object.
            target_ttl: Target total track length (mm).
        """
        if target_ttl is None:
            # Paraxial image solve
            new_spacing = lens.s[-1:] + lens.bfl
        else:
            # Total track length solve
            ttl_minus_last_spacing = (
                lens.s[:-1].flip(0).cumsum(dim=0).max(dim=0, keepdims=True)[0]
            )
            new_spacing = target_ttl - ttl_minus_last_spacing
        index = torch.tensor([lens.s.shape[0] - 1]).to(new_spacing.device)
        lens.s = torch.index_copy(lens.s, 0, index, new_spacing)

    def scale_lens_parameters(
        self, lens_variables: dict[str, torch.Tensor] | None = None
    ):
        """Scale the lens parameters to the correct units.

        Args:
            lens_variables: Lens variables to scale (if None, use the current lens variables).
        """
        if lens_variables is None:
            lens_variables = self.lens_variables
        if hasattr(self, "bezier_matrix"):
            a = lens_variables["a"][..., 1:].permute(0, 2, 1)
            a = (self.bezier_matrix.to(a) @ a).permute(0, 2, 1)
            a = torch.cat((lens_variables["a"][..., 0:1], a), dim=-1)
        else:
            a = lens_variables["a"]
        a_exp = torch.tensor(
            [0] + [2 * (i + 2) for i in range(lens_variables["a"].shape[-1] - 1)]
        ).to(lens_variables["s"].device)
        d_exp = torch.tensor(
            [2 * i + 2 for i in range(lens_variables["d"].shape[-1])]
        ).to(lens_variables["s"].device)
        z_exp = lens_variables["z"].new_tensor([1.0, 3.0, -1.0, -1.0])
        nd, vd, dpgf = self.get_nd_vd_dpgf(lens_variables["g"])
        return {
            "s": lens_variables["s"] * self.scale_factor,
            "c": lens_variables["c"] / self.scale_factor,
            "nd": nd,
            "vd": vd,
            "dpgf": dpgf,
            "a": a / self.scale_factor**a_exp,
            "d": lens_variables["d"] / self.scale_factor**d_exp,
            "m": lens_variables["m"],
            "z": lens_variables["z"] / self.scale_factor**z_exp,
        }

    def get_nd_vd_dpgf(self, g: torch.Tensor, qc_vars: bool | None = None):
        """Return refractive indices, Abbe numbers, and dpgf from normalized glass g.

        Under "quantized continuous glass variables", use gradient pass-through trick.

        Args:
            g: Normalized glass variables.
            qc_vars: Whether to use quantized continuous glass variables.
        """
        if qc_vars is None:
            # If None, use the class attribute (auto)
            qc_vars = self.qc_vars
        if qc_vars:
            # Only consider glass variables that are optimized; the rest will stay as is
            qc_vars_mask = torch.ones(g.shape[0], dtype=torch.bool, device=g.device)
            qc_vars_mask[self.freeze_dict["g"].any(dim=-1).squeeze(dim=-1)] = False
        else:
            qc_vars_mask = None
        return self.glass_model.get_nd_vd_dpgf(g, qc_vars_mask)

    def freeze_glass_variables(
        self, indices: tuple[int], bind_to_catalog: bool = False, unfreeze: bool = False
    ):
        """Freeze the glass variables at the given indices and optionally bind them to the catalog glasses.

        Args:
            indices: Indices of the optimized glass variables to bind to the catalog glasses.
            bind_to_catalog: Whether to bind the glass variables to the catalog glasses.
            unfreeze: Whether to unfreeze the glass variables (not compatible with bind_to_catalog).
        """
        lens_variables = self.lens_variables
        scaled_lens_variables = self.scale_lens_parameters(lens_variables)
        lens = self.generate_lens_from_lens_parameters(scaled_lens_variables)
        freeze_mask = self.freeze_dict["g"].clone()
        if unfreeze:
            assert not bind_to_catalog, (
                "Cannot unfreeze glass materials and bind to catalog at the same time."
            )
            freeze_mask[indices, :] = False
        else:
            freeze_mask[indices, :] = True

        if bind_to_catalog:
            # Update the glass variables to the closest catalog glasses
            # Frozen materials are left as is
            nd_bind, vd_bind, dpgf_bind = self.get_nd_vd_dpgf(
                self.lens_variables["g"], qc_vars=True
            )
            scaled_lens_variables["nd"] = nd_bind.where(
                freeze_mask.any(dim=-1).to(nd_bind.device), scaled_lens_variables["nd"]
            )
            scaled_lens_variables["vd"] = vd_bind.where(
                freeze_mask.any(dim=-1).to(nd_bind.device), scaled_lens_variables["vd"]
            )
            scaled_lens_variables["dpgf"] = dpgf_bind.where(
                freeze_mask.any(dim=-1).to(nd_bind.device),
                scaled_lens_variables["dpgf"],
            )

            # Update the lens once again to apply solves
            lens = self.generate_lens_from_lens_parameters(scaled_lens_variables)

        # Get normalized lens parameters
        bezier_matrix_inv = (
            self.bezier_matrix_inv if hasattr(self, "bezier_matrix") else None
        )
        lens_variable_dict = get_normalized_variables(
            lens,
            self.scale_factor,
            self.glass_model,
            self.paraxial_image_solve,
            bezier_matrix_inv,
        )
        initial_variable_list = torch.cat(
            [lens_variable_dict[k].view(-1) for k in self.variable_keys]
        )

        # Update all members
        self.initial_variables = initial_variable_list
        self.freeze_dict["g"] = freeze_mask
        self.optimization_mask = torch.cat(
            [~self.freeze_dict[k].view(-1) for k in self.variable_keys]
        )
        self._lens_variables.set_(initial_variable_list[self.optimization_mask])

    @torch.inference_mode()
    def log_lens_parameterization(self, lens: optics.Lens):
        """Return a dictionary of lens parameters for logging purposes.

        Args:
            lens: Lens object.
        """
        key2category = {
            "s": "vars-spacings",
            "s'": "vars-spacings",
            "g'": "vars-glass",
            "nd": "vars-glass",
            "vd": "vars-glass",
            "dpgf": "vars-glass",
            "c": "vars-curvatures",
            "c'": "vars-curvatures",
            "a": "vars-aspherical",
            "a'": "vars-aspherical",
            "d": "vars-diffractive",
            "d'": "vars-diffractive",
            "m": "vars-miscsurface",
            "m'": "vars-miscsurface",
            "z": "vars-zonal",
            "z'": "vars-zonal",
        }
        lens_parameters = {
            **{
                k: getattr(lens, k).squeeze(1).cpu().numpy()
                for k in self.parameter_keys
            },
            **{
                k + "'": v.squeeze(1).cpu().numpy()
                for k, v in self.lens_variables.items()
            },
        }
        logs = {}
        for k, v in lens_parameters.items():
            for index, value in np.ndenumerate(v):
                logs[f"{key2category[k]}/{k}{'-'.join([str(id) for id in index])}"] = (
                    value
                )
        return logs


class GlassModel(torch.nn.Module):
    """Representation of glass material parameterization and normalization.

    The glass model is based on at least two parameters: refractive index (nd) and Abbe number (vd).
    Optionally, the model can also include the deviation from normal partial dispersion (dpgf).
    """

    def __init__(
        self,
        glass_file: str | None = None,
        assume_normal_dispersion: bool = True,
    ):
        """Constructor.

        Args:
            glass_file: Path of the .csv glass file containing the available glass materials.
            assume_normal_dispersion: Whether to assume normal dispersion for the glass materials;
                the number of normalized glass variables is 2 if True and 3 if False.
        """
        super().__init__()
        self.assume_normal_dispersion = assume_normal_dispersion

        if glass_file is not None:
            # Load the glass properties data, ignoring initial spaces in the CSV
            glass_data = pd.read_csv(glass_file, skipinitialspace=True)

            # Filter out rows marked for ignoring
            if "ignore" in glass_data.columns:
                glass_data = glass_data[~glass_data["ignore"]]

            if "name" in glass_data.columns:
                self.catalog_glass_names = glass_data["name"].to_numpy()

            w, w_inv, mean, eigenvalues, catalog_g = self.fit_pca(glass_data)
        else:
            # Set identity transformation
            if assume_normal_dispersion:
                w = w_inv = [[1.0, 0.0], [0.0, 1.0]]
                mean = [0.0, 0.0]
                eigenvalues = [1.0, 1.0]
            else:
                w = w_inv = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                mean = [0.0, 0.0, 0.0]
                eigenvalues = [1.0, 1.0, 1.0]
            catalog_g = None
            self.catalog_glass_names = None

        # Store results in class attributes for further use
        self.register_buffer("w", torch.tensor(w))
        self.register_buffer("w_inv", torch.tensor(w_inv))
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("eigenvalues", torch.tensor(eigenvalues))
        self.register_buffer(
            "catalog_g", torch.tensor(catalog_g) if catalog_g is not None else None
        )

        self.glass_mesh = None
        if catalog_g is not None:
            self.glass_mesh = self.build_glass_mesh()

    def fit_pca(self, glass_data: pd.DataFrame):
        """Fit a PCA model to the glass data.

        Args:
            glass_data: Glass data in a pandas DataFrame format (columns: 'nd', 'vd', 'dpgf').
        """
        # Extract refractive index (nd) and Abbe number (vd) as numpy arrays
        refractive_index = glass_data["nd"].to_numpy()
        abbe_number = glass_data["vd"].to_numpy()
        variables = (refractive_index, abbe_number)
        if not self.assume_normal_dispersion:
            assert "dpgf" in glass_data.columns, (
                "As normal dispersion is not assumed, dpgf must be provided in the glass catalog file."
            )
            dpgf = glass_data["dpgf"].to_numpy()
            variables += (dpgf,)

        # Prepare data for PCA: stack glass variables
        pca_data = np.stack(variables, axis=1)

        # Normalize data: center and scale
        pca_data_normalized = (pca_data - pca_data.mean(axis=0)) @ np.diag(
            1 / pca_data.std(axis=0).clip(min=1e-6)
        )

        # We want materials in low-density regions to have more influence
        # Calculate Mahalanobis distance matrix and square it (gives better fit)
        dist_matrix = (
            scipy.spatial.distance.cdist(
                pca_data_normalized, pca_data_normalized, metric="mahalanobis"
            )
            ** 2
        )
        avg_dist = np.mean(dist_matrix, axis=1)
        weights = avg_dist / avg_dist.mean()

        # Normalize again according to the new weights
        weighted_mean = np.average(pca_data, axis=0, weights=weights)
        weighted_std = np.sqrt(
            np.average((pca_data - weighted_mean) ** 2, axis=0, weights=weights)
        )
        pca_weighted_data_normalized = (pca_data - weighted_mean) / weighted_std

        # Calculate covariance matrix and perform eigen decomposition
        covariance_matrix_weighted = np.cov(
            pca_weighted_data_normalized, rowvar=False, aweights=weights, bias=True
        )
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix_weighted)

        # Sort eigenvectors by descending eigenvalues
        sorted_indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Compute transformation matrices
        transformation_matrix = (
            np.diag(1 / weighted_std) @ eigenvectors @ np.diag(1 / np.sqrt(eigenvalues))
        )
        inverse_transformation_matrix = (
            np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T @ np.diag(weighted_std)
        )

        # Transform data using PCA
        transformed_data = (pca_data - weighted_mean) @ transformation_matrix
        return (
            transformation_matrix,
            inverse_transformation_matrix,
            weighted_mean,
            eigenvalues,
            transformed_data,
        )

    def get_nd_vd_dpgf(self, g: torch.Tensor, qc_vars: torch.Tensor | None = None):
        """Return refractive indices, Abbe numbers, and dpgf from normalized glass g.

        Under "quantized continuous glass variables", use gradient pass-through trick.

        Args:
            g: Normalized glass variables.
            qc_vars: Whether to use quantized continuous glass variables.
        """
        if qc_vars is not None:
            # Gradient pass through trick
            qc_vars = qc_vars.view(-1, 1, 1)
            g = g.where(~qc_vars, g + (self.closest_catalog_glasses(g) - g).detach())
        nd, vd, dpgf = self.nd_vd_dpgf_from_g(g)
        return nd, vd, dpgf

    def get_catalog_glass_indices(self, g: torch.Tensor):
        """Return the index of the closest catalog glass counterpart of each optimized glass.

        Args:
            g: Normalized glass variables.
        """
        assert self.catalog_g is not None, "No glass catalog available."
        dist = self.glass_distance(g[..., None, :], self.catalog_g)
        min_dist_idx = dist.argmin(dim=-1)
        return min_dist_idx

    def glass_distance(self, g1: torch.Tensor, g2: torch.Tensor):
        """Return the distance between two glasses.

        Args:
            g1: Normalized glass variables of the first glass (shape: [*, 2|3]).
            g2: Normalized glass variables of the second glass (shape: [*, 2|3]).
        """
        return (((g1 - g2) ** 2) * self.eigenvalues).sum(dim=-1).sqrt()

    def closest_catalog_glasses(self, g):
        """Return the closest catalog glass counterpart of each optimized glass.

        Args:
            g: Normalized glass variables.
        """
        indices = self.get_catalog_glass_indices(g)
        return self.catalog_g[indices]

    def g_from_nd_vd_dpgf(
        self, nd: torch.Tensor, vd: torch.Tensor, dpgf: torch.Tensor | None = None
    ):
        """Return normalized glass variables from refractive indices, Abbe numbers, and dpgf (optional).

        Args:
            nd: Refractive indices.
            vd: Abbe numbers.
            dpgf: Deviation from partial dispersion.
        """
        w = self.w
        mean = self.mean
        variables = (nd, vd)
        if not self.assume_normal_dispersion:
            variables += (dpgf,)
        g = (torch.stack(variables, dim=-1) - mean) @ w
        return g

    def nd_vd_dpgf_from_g(self, g: torch.Tensor):
        """Return refractive indices, Abbe numbers, and dpgf from normalized glass variables.

        Args:
            g: Normalized glass variables (shape: [*, 2|3]).
        """
        w_inv = self.w_inv
        mean = self.mean
        if g.shape[-1] == 2:
            nd_prime, vd_prime = torch.unbind(g @ w_inv + mean, dim=-1)
            dpgf = torch.zeros_like(nd_prime)
        elif g.shape[-1] == 3:
            nd_prime, vd_prime, dpgf = torch.unbind(g @ w_inv + mean, dim=-1)
        else:
            raise ValueError("Tensor g must have 2 or 3 columns.")

        return nd_prime, vd_prime, dpgf

    def build_glass_mesh(self):
        """Return the glass mesh."""
        assert self.catalog_g is not None, "Glass catalog not available."
        scaled_g = self.catalog_g * self.eigenvalues.sqrt()
        return scipy.spatial.Delaunay(scaled_g.cpu().detach().numpy())


def parse_freeze_dict(
    freeze_dict: dict[str, bool | list | dict] | None,
    lens_variable_dict: dict[str, torch.Tensor],
):
    """Parse the freeze dictionary to create a mask for the frozen variables.

    Args:
        freeze_dict: Dictionary specifying which variables should be frozen.
        lens_variable_dict: Dictionary of normalized lens variables.
    """
    freeze_dict = {} if freeze_dict is None else freeze_dict
    unknown_keys = set(freeze_dict) - set(lens_variable_dict)
    if unknown_keys:
        raise ValueError(f"Unknown freeze keys: {sorted(unknown_keys)}.")

    freeze_tensor_dict = {}
    for k, lens_variable in lens_variable_dict.items():
        v = freeze_dict.get(k, False)
        broadcast_shape = lens_variable_dict[k][:, 0, ...].shape
        if isinstance(v, (list, bool)):
            freeze_tensor = torch.tensor(v)
            try:
                freeze_mask = freeze_tensor.broadcast_to(broadcast_shape).contiguous()
            except RuntimeError:
                raise ValueError(f"Invalid freeze shape for {k}.")
        elif isinstance(v, dict):
            default = v["default"]
            freeze_mask = (
                torch.tensor(default).broadcast_to(broadcast_shape).contiguous()
            )
            for toggle_idx in v["toggle_row_col_list"]:
                if isinstance(toggle_idx, int):
                    row = toggle_idx
                    col = None
                elif isinstance(toggle_idx, (list, tuple)):
                    row, col = toggle_idx
                else:
                    raise ValueError(f"Invalid toggle index for {k}.")
                if col is None:
                    freeze_mask[row] = not default
                else:
                    if row is None:
                        row = slice(None)
                    freeze_mask[row, col] = not default
        else:
            raise ValueError(f"Invalid freeze type for {k}.")
        freeze_tensor_dict[k] = (
            freeze_mask[:, None, ...].expand_as(lens_variable).contiguous()
        )
    return freeze_tensor_dict


def distance_from_point_to_line(
    x: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
):
    """Compute the distance from a point to a line segment.

    Args:
        x: Point coordinates of interest (shape: [*, 2]).
        x1: Coordinates of first point on the line (shape: [*, 2]).
        x2: Coordinates of second point on the line (shape: [*, 2]).
        eps: Small value to avoid division by zero.
    """
    v = x2 - x1
    w = x - x1
    u = x2 - x
    c1 = (v * w).sum(dim=-1, keepdim=True)
    c2 = (v**2).sum(dim=-1, keepdim=True)
    b = c1 / c2.clip(eps)
    pb = x1 + v * b
    distance = (x - pb).norm(dim=-1)

    inside_line = (b >= 0) & (b <= 1)
    p = 2
    distance_to_points = torch.minimum(w.norm(p=p, dim=-1), u.norm(p=p, dim=-1))
    distance = distance.where(inside_line.squeeze(dim=-1), distance_to_points)

    return distance


def get_normalized_variables(
    lens: optics.Lens,
    scale_factor: float,
    glass_model: GlassModel,
    paraxial_image_solve: bool,
    aspherics_mat: torch.Tensor | None = None,
):
    """Return normalized lens variables that will act as the optimized variables.

    Args:
        lens: Lens object.
        scale_factor: Scale factor for the variables (in mm).
        glass_model: GlassModel object.
        paraxial_image_solve: Whether to use the paraxial image solve.
        aspherics_mat: Aspherical matrix for Bezier aspherics.
    """
    # Use lens wrapper to compute EFL
    a_exp = torch.tensor([0] + [2 * (i + 2) for i in range(lens.a.shape[-1] - 1)]).to(
        lens.a.device
    )
    a = lens.a * scale_factor**a_exp
    if aspherics_mat is not None:
        a = (aspherics_mat.to(lens.a) @ a[..., 1:].permute(0, 2, 1)).permute(0, 2, 1)
        a = torch.cat((lens.a[..., 0:1], a), dim=-1)
    d_exp = torch.tensor([2 * i + 2 for i in range(lens.d.shape[-1])]).to(lens.d.device)
    z_exp = lens.z.new_tensor([1.0, 3.0, -1.0, -1.0])
    s = lens.s
    if paraxial_image_solve:
        new_last_s = lens.s[-1:] - lens.bfl
        index = torch.tensor([lens.s.shape[0] - 1]).to(new_last_s.device)
        s = torch.index_copy(lens.s, 0, index, new_last_s)
    lens_parameters = {
        "s": s / scale_factor,
        "g": glass_model.g_from_nd_vd_dpgf(lens.nd, lens.vd, lens.dpgf),
        "c": lens.c * scale_factor,
        "a": a,
        "d": lens.d * scale_factor**d_exp,
        "m": lens.m,
        "z": lens.z * scale_factor**z_exp,
    }
    return lens_parameters
