import torch
import numpy as np

from eadld.modeling import (
    paraxial_ray_tracing as prt,
    ray_tracing as rt,
    misc_surfaces as ms,
)


class LensSequence:
    """Representation of the sequence of elements in a lens system for sequential ray tracing."""

    def __init__(self, sequence: str):
        """Constructor.

        Args:
            sequence: String representing the lens sequence.
        """
        self.events = []
        self.sequence = sequence
        assert sequence.count("s") == 1
        self.parse_sequence()

    def __len__(self):
        """Return the number of ray-tracing events."""
        return len(self.events)

    @property
    def n_refractive(self):
        """Number of refractive elements."""
        return sum(
            [1 for item in self.events if item["type"] == "p" and "refractive" in item]
        )

    @property
    def n_aspherical(self):
        """Number of aspherical surfaces."""
        return sum([1 for item in self.events if item["type"] == "r" and "a" in item])

    @property
    def n_zonal(self):
        """Number of zonal surfaces."""
        return sum([1 for item in self.events if item["type"] == "r" and "z" in item])

    @property
    def n_diffractive(self):
        """Number of diffractive surfaces."""
        return sum([1 for item in self.events if item["type"] == "d"])

    @property
    def n_misc_surfaces(self):
        """Number of miscellaneous surfaces."""
        return sum([1 for item in self.events if item["type"] == "m"])

    @property
    def n_interfaces(self):
        """Number of refractive interfaces."""
        return sum([1 for item in self.events if item["type"] == "r"])

    @property
    def n_propagations(self):
        """Number of propagation events."""
        return (
            max(
                [
                    item["s"] if isinstance(item["s"], int) else -1
                    for item in self.events
                    if "s" in item
                ]
            )
            + 1
        )

    @property
    def stop_idx(self):
        """Index of the aperture stop."""
        for i, event in enumerate(self.events):
            if event["type"] == "s":
                return i

    def parse_sequence(self):
        """Generate a list of ray-tracing events with the relevant lens variable indices."""
        r_idx = 0
        c_idx = 0
        a_idx = 0
        z_idx = 0
        s_idx = 0
        d_idx = 0
        m_idx = 0
        events = []
        current_propagation = {"type": "p", "s": None}
        next_refraction = {"n1": None}

        # Loop over lens sequence, register events, and associate them to the proper variables
        for char in self.sequence:
            if char == "R":  # Refractive element
                # Update current propagation
                current_propagation["c"] = c_idx
                if current_propagation["s"] is None:
                    events.append(current_propagation)
                # Register first refraction
                next_refraction = {
                    **next_refraction,
                    "type": "r",
                    "n2": r_idx,
                    "c": c_idx,
                    "s": s_idx,
                }
                events.append(next_refraction)
                c_idx += 1
                # Register propagation
                current_propagation = {
                    "type": "p",
                    "s": s_idx,
                    "c": c_idx,
                    "refractive": True,
                }
                events.append(current_propagation)
                s_idx += 1
                # Prepare next refraction
                next_refraction = {"type": "r", "n1": r_idx, "c": c_idx, "s": s_idx}
                r_idx += 1
            elif char == "-":  # Air gap
                if "c" in next_refraction:  # Close refraction
                    next_refraction["n2"] = None
                    c_idx += 1
                    events.append(next_refraction)
                current_propagation = {
                    "type": "p",
                    "s": s_idx,
                }
                events.append(current_propagation)
                s_idx += 1
                next_refraction = {"n1": None}
            elif char == "a":  # Aspherical surface
                assert "z" not in current_propagation and "z" not in next_refraction, (
                    "A surface cannot be both aspherical and zonal."
                )
                current_propagation["a"] = a_idx
                next_refraction["a"] = a_idx
                a_idx += 1
            elif char == "z":  # Zonal surface
                assert "a" not in current_propagation and "a" not in next_refraction, (
                    "A surface cannot be both aspherical and zonal."
                )
                assert "z" not in current_propagation and "z" not in next_refraction, (
                    "A zonal surface can only be declared once."
                )
                current_propagation["z"] = z_idx
                next_refraction["z"] = z_idx
                z_idx += 1
            elif char == "d":  # Diffractive surface
                if current_propagation["s"] is None:
                    events.append(current_propagation)
                    current_propagation = {"type": "p", "s": None}
                events.append(
                    {
                        "type": "d",
                        "d": d_idx,
                    }
                )
                d_idx += 1
            elif char == "m":  # Miscellaneous surface
                if current_propagation["s"] is None:
                    events.append(current_propagation)
                    current_propagation = {"type": "p", "s": None}
                events.append(
                    {
                        "type": "m",
                        "m": m_idx,
                    }
                )
                m_idx += 1
            elif char == "s":  # Aperture stop
                if current_propagation["s"] is None:
                    events.append(current_propagation)
                    current_propagation = {"type": "p", "s": None}
                events.append({"type": "s"})

        # Finalize
        if "c" in next_refraction:  # Close refraction
            next_refraction["n2"] = None
            events.append(next_refraction)

        assert z_idx == sum(
            [1 for event in events if event["type"] == "r" and "z" in event]
        ), "Zonal surfaces must modify refractive interfaces."
        self.events = events


class Lens:
    """Representation of a lens based on its sequence of elements and lens parameters.

    Supports multiple lenses provided they share the same sequence.
    """

    def __init__(
        self,
        sequence: str | LensSequence,
        s: torch.Tensor,
        c: torch.Tensor,
        nd: torch.Tensor,
        vd: torch.Tensor,
        dpgf: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor,
        m: torch.Tensor,
        w0: float,
        misc_surface_model: ms.MiscSurfaceModel | None = None,
        z: torch.Tensor | None = None,
    ):
        """Constructor.

        Args:
            sequence: str or LensSequence object.
            s: Spacing between elements (shape: [n_spacings, n_lens]).
            c: Curvatures for refractive surfaces (shape: [n_interfaces, n_lens]).
            nd: Refractive indices (shape: [n_refractive, n_lens]).
            vd: Abbe numbers (shape: [n_refractive, n_lens]).
            dpgf: Partial dispersion deviation (shape: [n_refractive, n_lens]).
            a: Aspherical coefficients starting with conic constant (shape: [n_aspherical, n_lens, n_coefficients]).
            d: Polynomial coefficients for diffractive surfaces (shape: [n_diffractive, n_lens, n_coefficients]).
            m: Miscellaneous surface parameters (shape: [n_surfaces, n_lens, n_parameters]).
            w0: Design wavelength (used for paraxial computations and diffractive surface modeling).
            misc_surface_model: Miscellaneous surface model (used for diffractive surface modeling).
            z: Zonal surface parameters [delta_A1, A2, delta_Z, Rmax]
                (shape: [n_zonal, n_lens, n_zones, 4]).
        """
        if isinstance(sequence, str):
            self.sequence = LensSequence(sequence)
        else:
            self.sequence = sequence

        n_lens = s.shape[1] if len(s.shape) > 1 else 1
        self.s = s.view(self.sequence.n_propagations, n_lens)
        self.c = c.view(self.sequence.n_interfaces, n_lens)
        self.nd = nd.view(self.sequence.n_refractive, n_lens)
        self.vd = vd.view(self.sequence.n_refractive, n_lens)
        self.dpgf = dpgf.view(self.sequence.n_refractive, n_lens)
        self.a = a.view(self.sequence.n_aspherical, n_lens, a.shape[-1])
        self.d = d.view(self.sequence.n_diffractive, n_lens, d.shape[-1])
        self.m = m.view(self.sequence.n_misc_surfaces, n_lens, m.shape[-1])
        if z is None or z.numel() == 0:
            assert self.sequence.n_zonal == 0, (
                "Zonal surface parameters are required by the lens sequence."
            )
            self.z = self.c.new_empty((0, n_lens, 0, 4))
        else:
            assert z.shape[-1] == 4, (
                "Zonal surface parameters must contain [delta_A1, A2, delta_Z, Rmax]."
            )
            self.z = z.view(self.sequence.n_zonal, n_lens, -1, 4)
        # Verify that the number of lenses is the same
        assert (
            len(
                set(
                    (
                        item.shape[1]
                        for item in (
                            self.s,
                            self.c,
                            self.nd,
                            self.vd,
                            self.dpgf,
                            self.a,
                            self.d,
                            self.m,
                            self.z,
                        )
                    )
                )
            )
            == 1
        )

        # Nominal design wavelength (used to define variables and for paraxial computations
        self.w0 = w0

        # Miscellaneous surface model
        if self.sequence.n_misc_surfaces > 0:
            assert misc_surface_model is not None, (
                "A model must be provided if there are miscellaneous surfaces."
            )
        self.misc_surface_model = misc_surface_model

    def __len__(self):
        """Return number of lenses in the object."""
        return self.s.shape[1]

    @property
    def attributes(self):
        """Return the list of attributes."""
        return (
            self.sequence,
            self.s,
            self.c,
            self.nd,
            self.vd,
            self.dpgf,
            self.a,
            self.d,
            self.m,
            self.w0,
            self.misc_surface_model,
            self.z,
        )

    def detach(self):
        """Return a copy of the lens with detached variables."""

        def detach_item(item):
            return item.detach() if hasattr(item, "detach") else item

        return Lens(*map(detach_item, self.attributes))

    def clone(self):
        """Return a copy of the lens with cloned variables."""

        def clone_item(item):
            return item.clone() if hasattr(item, "clone") else item

        return Lens(*map(clone_item, self.attributes))

    @property
    def efl(self):
        """Effective focal length.

        Given by -1 / C for the ABCD matrix of the lens.
        """
        return -1 / self.get_abcd(reduce=True)[..., 1, 0]

    @property
    def bfl(self):
        """Back focal length.

        Given by A / C for the ABCD matrix representing the lens up to the last optical surface.
        """
        abcd = self.get_abcd(idx=slice(None, -1), reduce=True)
        return -abcd[..., 0, 0] / abcd[..., 1, 0]

    @property
    def pupil_position(self):
        """Position of the entrance pupil w.r.t. the leftmost surface (virtual or physical) of the lens.

        Given by B / A for the ABCD matrix up to the aperture stop.
        """
        if self.sequence.stop_idx == 0:
            return torch.zeros(1)
        else:
            abcd_stop = self.get_abcd(idx=slice(self.sequence.stop_idx), reduce=True)
            return abcd_stop[:, 0, 1] / abcd_stop[:, 0, 0]

    def get_abcd(
        self, idx: slice | None = None, reduce: bool = False, wavelength: float = None
    ):
        """Return ABCD matrix of the lens.

        Args:
            idx: Slice to specify all ray-tracing events that should be taken into account.
            reduce: If False, return the ABCD matrix individually for each ray-tracing event.
            wavelength: Wavelength used to compute ABCD matrix (default: w0).
        """
        if wavelength is None:
            wavelength = self.w0
        abcd = (
            torch.eye(2)
            .to(self.c)
            .view((1, 1, 2, 2))
            .expand((len(self.sequence), len(self), -1, -1))
            .contiguous()
        )
        propagation_abcd = prt.propagation_abcd(self.s)
        propagation_mask = [
            item["type"] == "p" and item["s"] is not None
            for item in self.sequence.events
        ]
        abcd[torch.tensor(propagation_mask)] = propagation_abcd

        if self.sequence.n_refractive > 0:
            mu = self.get_mu(torch.tensor(wavelength).to(abcd))
            mu = mu.squeeze(dim=-1)
            refraction_abcd = prt.interface_abcd(mu, self.c)
            abcd[
                torch.tensor([item["type"] == "r" for item in self.sequence.events])
            ] = refraction_abcd

        if self.sequence.n_diffractive > 0:
            diffraction_abcd = prt.diffraction_abcd(
                self.d.clone()[..., 0], wavelength / self.w0
            )
            abcd[
                torch.tensor([item["type"] == "d" for item in self.sequence.events])
            ] = diffraction_abcd

        if self.sequence.n_misc_surfaces > 0:
            misc_surface_abcd = self.misc_surface_model.get_abcd(self.m, wavelength)
            abcd[
                torch.tensor([item["type"] == "m" for item in self.sequence.events])
            ] = misc_surface_abcd

        if idx is not None:
            abcd = abcd[idx]

        if reduce:
            abcd = prt.reduce_abcd(abcd)

        return abcd

    def trace_rays(
        self,
        r: torch.Tensor,
        d: torch.Tensor,
        wavelengths: list[float] | tuple[float, ...] | torch.Tensor,
        yield_on: str = "end",
    ):
        """Trace rays surface by surface and yield updated ray positions and directions and intermediate information.

        With yield_on = 'all', the updated "r" and "d" are returned (yielded) after each ray-tracing event.

        Args:
            r: Ray position vectors (shape: [3 (xyz), *, n_wavelengths, n_lens]).
            d: Ray direction cosines (shape: [3 (lmn), *, n_wavelengths, n_lens]).
            wavelengths: In nm (shape: [n_wavelengths]).
            yield_on (optional, string): When to yield (end/position/stop/all).
        """
        # Only keep xyz X n_rays X n_wavelengths X n_lens
        assert r.shape == d.shape
        shape = r.shape
        r = r.reshape(3, -1, len(wavelengths), len(self))
        d = d.reshape(3, -1, len(wavelengths), len(self))

        cum_z_distance = 0
        cos_theta = 0
        cos2_theta = 0
        ray_status = torch.zeros_like(r[0], dtype=torch.int)

        if not isinstance(wavelengths, torch.Tensor):
            wavelengths = torch.tensor(wavelengths).to(r)
        wavelength_ratios = wavelengths / self.w0
        # Eikonal of the incident plane wave, with the field-wise constant
        # z term removed.  This preserves the pupil tilt for off-axis fields.
        opl = (r[:2] * d[:2]).sum(dim=0)
        current_n = r.new_ones((len(wavelengths), len(self)))

        for i, event in enumerate(self.sequence.events):
            event_info = {}
            if event["type"] == "r":  # Refraction; update "d"
                c = self.c[event["c"]]
                n1idx, n2idx = event["n1"], event["n2"]

                def get_refractive_index(idx):
                    if idx is None:
                        return r.new_ones((len(self), len(wavelengths)))
                    return hartmann_dispersion(
                        wavelengths, self.nd[idx], self.vd[idx], self.dpgf[idx]
                    )

                n1 = get_refractive_index(n1idx)
                n2 = get_refractive_index(n2idx)
                mu = (n1 / n2).T
                r0 = rt.shift_rays(r, cum_z_distance)
                if "z" in event:
                    z = self.z[event["z"]]
                    radius = r0[:2].square().sum(dim=0).sqrt()
                    valid_zones = (z != 0).any(dim=-1)
                    zone_matches = (radius[..., None] <= z[..., 3]) & valid_zones
                    zonal_index = zone_matches.to(torch.int64).argmax(dim=-1)
                    event_info["zonal_index"] = zonal_index.view(*shape[1:])
                    d, tir, backward, cos2_prime, cos2_theta, cos_n = (
                        rt.apply_snell_zonal(r0, d, c, z, mu)
                    )
                    event_info["cos_n"] = cos_n.view(*shape[1:])
                elif "a" in event:
                    a = self.a[event["a"]]
                    d, tir, backward, cos2_prime, cos2_theta, cos_n = (
                        rt.apply_snell_aspherical(r0, d, c, a, mu)
                    )
                    event_info["cos_n"] = cos_n.view(*shape[1:])
                else:
                    d, tir, backward, cos2_prime = rt.apply_snell_spherical(
                        r0, d, c, mu, cos_theta
                    )
                event_info["cos2_theta"] = cos2_theta.view(*shape[1:])
                event_info["cos2_prime"] = cos2_prime.view(*shape[1:])
                ray_status = ray_status.where(~backward, 3)
                ray_status = ray_status.where(~tir, 2)
                r, d = rt.reset_bad_rays(r, d, ray_status < 2, normalize=True)
                current_n = n2.T
            elif event["type"] == "m":  # Miscellaneous surface; update "d"
                p = self.m[event["m"]]
                d, backward, cos2_prime, return_dict = self.misc_surface_model(
                    r, d, p, wavelengths
                )
                event_info["cos2_prime"] = cos2_prime.view(*shape[1:])
                for tensor in return_dict:
                    event_info[tensor] = return_dict[tensor].view(*shape[1:])
                if "phase_opd" in return_dict:
                    candidate_opl = opl + return_dict["phase_opd"]
                    opl = torch.where(ray_status == 0, candidate_opl, opl)
                ray_status = ray_status.where(~backward, 3)
                r, d = rt.reset_bad_rays(r, d, ray_status < 2, normalize=True)
            elif event["type"] == "d":  # Diffraction; update "d"
                p = self.d[event["d"]]
                d, backward, cos2_prime, phase_opd = rt.apply_phase_shift(
                    r, d, p, wavelength_ratios
                )
                event_info["cos2_prime"] = cos2_prime.view(*shape[1:])
                event_info["phase_opd"] = phase_opd.view(*shape[1:])
                candidate_opl = opl + phase_opd
                opl = torch.where(ray_status == 0, candidate_opl, opl)
                ray_status = ray_status.where(~backward, 3)
                r, d = rt.reset_bad_rays(r, d, ray_status < 2, normalize=True)
            elif event["type"] == "p":  # Propagation; update "r"
                s = r.new_zeros(1) if event["s"] is None else self.s[event["s"]]
                cum_z_distance = cum_z_distance + s
                r0 = rt.shift_rays(r, cum_z_distance)
                if "c" in event:
                    # Propagate up to the next spherical interface
                    c = self.c[event["c"]]
                    # Find r relative to the next surface
                    if "z" in event:
                        z = self.z[event["z"]]
                        distance, miss = rt.find_marching_distance_zonal(r0, d, c, z)
                    elif "a" in event:
                        a = self.a[event["a"]]
                        distance, miss = rt.find_marching_distance_aspherical(
                            r0, d, c, a
                        )
                    else:
                        distance, miss, cos_theta, cos2_theta = (
                            rt.find_marching_distance_spherical(r0, d, c)
                        )
                    ray_status = ray_status.where(~miss, 4)
                else:
                    # Propagate up to the next flat surface
                    delta_z = -r0[2]
                    distance = delta_z / d[2]
                r, delta_z = rt.update_ray_coordinates(r, d, distance)
                r, d = rt.reset_bad_rays(r, d, ray_status < 2, normalize=False)
                delta_z = delta_z.where(ray_status < 2, float("nan"))
                if i > 0:  # Exclude first propagation from the entrance pupil
                    event_info["delta_z"] = delta_z.view(*shape[1:])
                    backtrack = delta_z < 0
                    ray_status = ray_status.where(~backtrack | (ray_status > 0), 1)
                candidate_opl = opl + current_n * distance
                opl = torch.where(ray_status == 0, candidate_opl, opl)

            event_info["opl"] = opl.where(ray_status == 0, float("nan")).view(
                *shape[1:]
            )

            if any(
                (
                    yield_on == "all",
                    yield_on == "position" and event["type"] == "p",
                    yield_on == "stop" and event["type"] == "s",
                    yield_on == "end" and i == len(self.sequence.events) - 1,
                )
            ):
                yield (
                    r.view(*shape),
                    d.view(*shape),
                    ray_status.view(*shape[1:]),
                    event_info,
                )

    def curvature_solve(
        self, c_idx: int, target_efl: float | torch.Tensor, solve_type: str
    ):
        """Return the curvature required to fulfill the target EFL.

        For solve_type == 'focal_length':
        The appropriate curvature corresponding to c_idx is found such that, in the resultant ABCD matrix:
            -1 / C = EFL

        For solve_type == 'image_height':
        In contrast to solve_type == 'focal_length', converts target efl to paraxial chief ray height.
        The appropriate curvature corresponding to c_idx is found such that, in the resultant ABCD matrix:
            B - A * pupil_position = EFL

        Note that when A = 0 (imaging system), we have B = EFL since AD - BC = 1.
        In this case, the two solve types are perfectly equivalent.

        Args:
            c_idx: Index of refractive interface.
            target_efl: Desired effective focal length (shape: [n_lens]).
            solve_type: Either 'focal_length' or 'image_height'.
        """
        # Find the index of the refractive surface
        # TODO: handle case where the refractive surface is the first or last of the system
        assert 0 <= c_idx < self.sequence.n_interfaces, "Solve index out of bounds."
        surface_idx = 0
        stop_encountered = False
        for i, event in enumerate(self.sequence.events):
            if event["type"] == "s":
                stop_encountered = True
            if event["type"] == "r":
                if surface_idx == c_idx:
                    break
                else:
                    surface_idx += 1
        else:
            raise ValueError("No refractive surface at index {}".format(c_idx))

        # Compute the ABCD matrix of the refractive surface and the rest of the optical system on both sides
        abcd = self.get_abcd()
        solve_abcd = abcd[i]
        abcd_left = prt.reduce_abcd(abcd[:i])
        abcd_right = prt.reduce_abcd(abcd[i + 1 :])
        a1, b1, c1, d1 = abcd_left.view(*abcd_left.shape[:-2], 4).unbind(-1)
        a2, b2, c2, d2 = abcd_right.view(*abcd_right.shape[:-2], 4).unbind(-1)
        mu = solve_abcd[..., 1, 1]

        if solve_type == "focal_length":
            c = (1 / d2 / target_efl / a1 + c2 / d2 + mu * c1 / a1) / (1 - mu)
        elif solve_type == "image_height":
            z = self.pupil_position
            c = -(
                a1 * a2 * z - a2 * b1 + b2 * c1 * mu * z - b2 * d1 * mu + target_efl
            ) / (b2 * (mu - 1) * (a1 * z - b1))
            if not stop_encountered:
                raise ValueError(
                    "Image height solve requires the solved curvature to be past the aperture stop."
                )
        else:
            raise ValueError("Invalid solve type {}".format(solve_type))
        return c

    def evaluate_paraxial_heights_at_image_plane(self, field_angles: torch.Tensor):
        """Return the paraxial heights at the image plane for a given set of field angles.

        We consider the height of a paraxial chief ray (that hits the entrance pupil at the middle).
        The height is proportional to the tangent of the field angle, and is given by:
            h = tan(theta) * (B - A * pupil_position),
        where A and B are the elements of the ABCD matrix of the lens.

        Args:
            field_angles: Field angles (shape: [*]).
        """
        pupil_position = self.pupil_position
        abcd = self.get_abcd(reduce=True)
        a, b = abcd[:, 0, 0], abcd[:, 0, 1]
        b_prime = b - a * pupil_position

        angles = field_angles.view(1, -1).to(a)
        heights = angles.tan() * b_prime
        return heights

    def get_mu(self, wavelengths: torch.Tensor):
        """Return n1/n2 ratios at each refractive interface.

        Args:
            wavelengths: In nm (shape: [n_wavelengths]).
        """
        n = self.get_n(wavelengths)
        n1 = self.s.new_ones(self.sequence.n_interfaces, *n.shape[1:])
        n2 = self.s.new_ones(self.sequence.n_interfaces, *n.shape[1:])
        mask1 = np.array(
            [
                item["n1"] is not None
                for item in self.sequence.events
                if item["type"] == "r"
            ]
        )
        mask2 = np.array(
            [
                item["n2"] is not None
                for item in self.sequence.events
                if item["type"] == "r"
            ]
        )
        n1_idx = [
            item["n1"]
            for item in self.sequence.events
            if item["type"] == "r" and item["n1"] is not None
        ]
        n2_idx = [
            item["n2"]
            for item in self.sequence.events
            if item["type"] == "r" and item["n2"] is not None
        ]
        n1[mask1] = n[n1_idx]
        n2[mask2] = n[n2_idx]
        mu = n1 / n2
        return mu

    def get_n(self, wavelengths: torch.Tensor):
        """Return refractive indices for each glass.

        Args:
            wavelengths: In nm (shape: [n_wavelengths]).
        """
        n = hartmann_dispersion(wavelengths, self.nd, self.vd, self.dpgf)
        return n

    def return_geometry(self):
        """Yield geometry information on the lens.

        Useful for tracing the lens layout.
        """
        cum_z_distance = self.c.new_zeros(1)

        for i, event in enumerate(self.sequence.events):
            sag_fn = None
            if event["type"] == "r":
                c = self.c[event["c"]]
                a = self.a[event["a"]] if "a" in event else None
                z = self.z[event["z"]] if "z" in event else None

                # 绑定当前表面的参数；几何列表化后也不能引用到后续表面。
                def sag_fn(y, c=c, a=a, z=z):
                    rho = y.view(-1, 1) ** 2
                    if z is not None:
                        return rt.evaluate_zonal_profile(rho, c.to(y), z.to(y))[0][:, 0]
                    else:
                        return rt.evaluate_aspherical_profile(
                            rho, c.to(y), None if a is None else a.to(y)
                        )[0][:, 0]
            elif event["type"] == "p":
                s = self.c.new_zeros(1) if event["s"] is None else self.s[event["s"]]
                cum_z_distance = cum_z_distance + s
                is_refractive = "refractive" in event
            if (
                event["type"] in ("r", "s", "d", "m")
                or i == len(self.sequence.events) - 1
            ):
                # 光阑可能与玻璃后表面位于同一轴向位置。它不是介质界面，
                # 不能提前消耗随后真实表面的玻璃闭合标记。
                closes_glass = is_refractive and event["type"] != "s"
                yield event["type"], cum_z_distance, sag_fn, closes_glass
                if event["type"] != "s":
                    is_refractive = False

    def estimate_diameters(
        self, r: torch.Tensor, d: torch.Tensor, wavelengths: list[float] | torch.Tensor
    ):
        """Estimate maximum diameters at each interface based on input rays.

        Args:
            r: Ray position vectors (shape: [3 (xyz), n_rays, n_wavelengths, n_lens]).
            d: Ray direction cosines (shape: [3 (lmn), n_rays, n_wavelengths, n_lens]).
            wavelengths: In nm (shape: [n_wavelengths]).
        """
        xy = torch.stack(
            [
                r[:2].view(2, -1, len(self))
                for r, d, *_ in self.trace_rays(r, d, wavelengths, yield_on="position")
            ]
        )
        return 2 * xy.norm(dim=1).max(dim=1)[0]

    def as_tabular(self):
        """Return a tabular representation of the lens.

        Only works if there is a single lens.
        """
        tabular_data = []
        current_entry = {}
        for event in self.sequence.events:
            if event["type"] == "p":  # Propagation
                if event["s"] is not None:
                    current_entry["s"] = (
                        0 if event["s"] is None else self.s[event["s"]].item()
                    )
                    tabular_data.append(current_entry)
                current_entry = {}
            if event["type"] == "r":  # Refraction
                current_entry["c"] = self.c[event["c"]].item()
                if "a" in event:
                    current_entry["a"] = self.a[event["a"]][0].tolist()
                if "z" in event:
                    current_entry["z"] = self.z[event["z"]][0].tolist()
                if event["n2"] is not None:
                    current_entry["nd"] = self.nd[event["n2"]].item()
                    current_entry["vd"] = self.vd[event["n2"]].item()
                    current_entry["dpgf"] = self.dpgf[event["n2"]].item()
            if event["type"] == "s":  # Aperture stop
                current_entry["stop"] = True
            if event["type"] == "d":  # Diffraction
                current_entry["d"] = self.d[event["d"]][0].tolist()
            if event["type"] == "m":  # Miscellaneous surface
                try:
                    current_entry["m"] = self.misc_surface_model.get_tabular_parameters(
                        self.m[event["m"]]
                    )[0]
                except NotImplementedError:
                    pass
        return tabular_data


def cauchy_dispersion(wavelengths: torch.Tensor, nd: torch.Tensor, vd: torch.Tensor):
    """Interpolate refractive indices at the desired wavelengths using the Cauchy model.

    The refractive index is given by:
        n(w) = A + B / w**2 + C / w**4

    We assume C=0.
    A and B are recovered from the refractive index at the "d" wavelength and the Abbe number.
    See "End-to-End Complex Lens Design with Differentiable Ray Tracing" (Sun et al., 2021).

    Args:
        wavelengths: Wavelengths in nm (shape: [n_wavelengths]).
        nd: Refractive indices at "d" line (shape: [*]).
        vd: Abbe numbers at "d" line (shape: [*]).
    """
    wc = 656.3
    wd = 587.6
    wf = 486.1
    b = (nd - 1) / (vd * (wf**-2 - wc**-2))
    a = nd - b / wd**2
    n = a[..., None] + b[..., None] / wavelengths**2
    return n


def hartmann_dispersion(
    wavelengths: torch.Tensor,
    nd: torch.Tensor,
    vd: torch.Tensor,
    dpgf: torch.Tensor | float = 0.0,
):
    """Interpolate refractive indices at the desired wavelengths using the Hartmann (1926) three-parameter model.

    The refractive index is given by:
        n(w) = A + C / (w - B)

    B is recovered from deviation from normal partial dispersion,
    as per Schott (see TIE-29 Refractive Index and Dispersion):
        (n(wg) - n(wF)) / (n(wF) - n(wC)) = - 0.001682 vd + 0.6438 + dpgf

    C is recovered from the definition of the Abbe number and B:
        vd = (nd - 1) / (n(wF) - n(wC))

    A is recovered from the definition of the refractive index at the "d" line, B, and C:
        nd = n(wd)

    Args:
        wavelengths: Wavelengths in nm (shape: [n_wavelengths]).
        nd: Refractive indices at "d" line (shape: [*]).
        vd: Abbe numbers at "d" line (shape: [*]).
        dpgf: Partial dispersion deviation (shape: [*]).
    """
    wc = 656.3e-3
    wd = 587.6e-3
    wf = 486.1e-3
    wg = 435.8e-3
    m = -0.001682
    o = 0.6438

    wavelengths = wavelengths * 1e-3

    pgf = m * vd + o + dpgf
    b = (pgf * (wc * wg - wf * wg) - wc * wf + wc * wg) / (pgf * (wc - wf) - wf + wg)
    c = (b - wc) * (b - wf) * (nd - 1) / (vd * (wc - wf))
    a = (b * nd + c - nd * wd) / (b - wd)
    n = a[..., None] + c[..., None] / (wavelengths - b[..., None])
    return n
