"""Generate an N-element lens from a specification and optimise it.

add by cjy.  The pipeline adds no new physics; it wires up machinery the
framework already has, in the order that keeps the problem well posed:

  1. Paraxial start.  Power is split evenly over N equiconvex elements.  A flat
     start does NOT work here.  It works for the annular singlet only because
     zone coefficients are local, so the Jacobian is near block diagonal.
     Continuous surfaces are global and strongly coupled, and LM from zero
     power walks into a degenerate configuration and stalls.
  2. Project solves.  "target_efl"/"solve_idx" re-solve one curvature every
     step so the EFL never drifts, and "paraxial_image_solve" keeps the image
     plane at focus.  Without them the EFL ran to 179 mm on a 100 mm target.
  3. Geometric constraints.  RayPathResiduals bounds air gaps and glass
     thickness along every ray; SurfaceNormalResiduals and RayAngleResiduals
     bound surface steepness and incidence.  Without them the optimiser
     produced a -2.8 mm air gap -- the two elements interpenetrating -- and
     2.4 mm radii on a 12.5 mm pupil, which score well and mean nothing.
  4. Staged release.  Curvatures and aspheres first with glass frozen, then
     glass.  Releasing glass from a poor start collapses the trace.

add by cjy.  With a fold order the first element becomes annular and carries its
share of the power in the fold rather than in a curved face.  The one thing that
has to be got right is where that power is written down, because get_abcd()
builds the refraction matrix from lens.c alone.

The zonal sag is (c/2 + A1)*rho + A2*rho^2 + Zoff, so the fold envelope can be
expressed either way: c = 0 with A1 = -c_eff/2, or c = -c_eff with A1 = 0.  The
two trace identically -- verified, 57.17/57.32/57.90 um both ways -- but only the
second is visible to the paraxial code, which reads 105.3 mm against the first
form's 199.3 mm (element 2 alone, the fold contributing nothing).  Writing the
fold into c therefore keeps "solve_type" and "paraxial_image_solve" working
unchanged, and A1 stays free as the per-zone correction on top.

The alternative -- solves off, ImageHeightResiduals holding the focal length --
does not work: the dispatch hands that residual xy_centroid[1, ..., -1].squeeze(),
which under this ray initialisation is every field, not the outer one.  LM duly
drove the 0.5 and 1.0 degree centroids to the same 1.7455 mm height.

    python configs/multi_element/demo.py [n_elements] [fold|sph|M]
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eisoptx.modeling import optics  # noqa: E402
from eisoptx.modeling import ray_initialization as ri  # noqa: E402
from eisoptx.modeling import ray_tracing as rt  # noqa: E402

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------- specification
# add by cjy: named specifications, selected with SPEC=... in the environment.
# Each carries the project's own defaults file, so the residual set, the LM
# settings and the log directory come from the project rather than being
# restated here -- c_mount in particular is already tuned for a fast, wide
# design (DistortionResiduals at 2%, GlassMeshDistanceResiduals, lm_parameter
# 1e6), and its four-element spherical design is the baseline to beat.
SPECS = {
    "default": dict(
        defaults="configs/multi_element/defaults.yml", logs="multi_element",
        efl=100.0, f_number=8.0, field=1.0, n_fields=3,
        wavelengths=[486.1, 550.0, 656.3], design=550.0,
        glass_file="configs/moldable_materials.csv",
        glass={"nd": 1.6, "vd": 35.0}, stop_gap=0.5, thickness=3.0, air_gap=20.0,
        max_track=400.0,
        own_residuals=False),
    "c_mount": dict(
        defaults="configs/c_mount/defaults.yml", logs="logs_fab",
        # add by cjy: 11 is the project's own field sampling for this spec, and
        # its RGBPSFs visualization indexes into it -- 5 fields raised
        # IndexError: index 6 is out of bounds for axis 0 with size 5.
        efl=28.0, f_number=2.0, field=15.88, n_fields=11,
        wavelengths=[486.1, 587.6, 656.3], design=587.6,
        glass_file="configs/recommended_ohara_glass.csv",
        glass={"nd": 1.75, "vd": 45.0}, stop_gap=0.25, thickness=2.0, air_gap=2.0,
        # the project 4p design measures 39.3 mm; hold the generator to it.
        max_track=39.3,
        own_residuals=True),
    "c_mount_easy": dict(
        defaults="configs/c_mount/defaults.yml", logs="logs_fab",
        efl=28.0, f_number=8.0, field=2.0, n_fields=3,
        wavelengths=[486.1, 587.6, 656.3], design=587.6,
        glass_file="configs/recommended_ohara_glass.csv",
        glass={"nd": 1.75, "vd": 45.0}, stop_gap=0.25, thickness=2.0, air_gap=2.0,
        max_track=39.3,
        own_residuals=True),
    "c_mount_medium": dict(
        defaults="configs/c_mount/defaults.yml", logs="logs_fab",
        efl=28.0, f_number=4.0, field=8.0, n_fields=5,
        wavelengths=[486.1, 587.6, 656.3], design=587.6,
        glass_file="configs/recommended_ohara_glass.csv",
        glass={"nd": 1.75, "vd": 45.0}, stop_gap=0.25, thickness=2.0, air_gap=2.0,
        max_track=39.3,
        own_residuals=True),
    "c_mount_step1": dict(
        defaults="configs/c_mount/defaults.yml", logs="logs_fab",
        efl=28.0, f_number=6.3, field=3.0, n_fields=3,
        wavelengths=[486.1, 587.6, 656.3], design=587.6,
        glass_file="configs/recommended_ohara_glass.csv",
        glass={"nd": 1.75, "vd": 45.0}, stop_gap=0.25, thickness=2.0, air_gap=2.0,
        max_track=39.3,
        own_residuals=True),
    "c_mount_step2": dict(
        defaults="configs/c_mount/defaults.yml", logs="logs_fab",
        efl=28.0, f_number=4.5, field=5.0, n_fields=5,
        wavelengths=[486.1, 587.6, 656.3], design=587.6,
        glass_file="configs/recommended_ohara_glass.csv",
        glass={"nd": 1.75, "vd": 45.0}, stop_gap=0.25, thickness=2.0, air_gap=2.0,
        max_track=39.3,
        own_residuals=True),
    "c_mount_step3": dict(
        defaults="configs/c_mount/defaults.yml", logs="logs_fab",
        efl=28.0, f_number=3.0, field=9.0, n_fields=7,
        wavelengths=[486.1, 587.6, 656.3], design=587.6,
        glass_file="configs/recommended_ohara_glass.csv",
        glass={"nd": 1.75, "vd": 45.0}, stop_gap=0.25, thickness=2.0, air_gap=2.0,
        max_track=39.3,
        own_residuals=True),
    "doublet_challenge": dict(
        defaults="configs/multi_element/defaults.yml", logs="multi_element",
        efl=100.0, f_number=8.0, field=1.5, n_fields=5,
        wavelengths=[486.1, 550.0, 656.3], design=550.0,
        glass_file="configs/moldable_materials.csv",
        glass={"nd": 1.6, "vd": 35.0}, stop_gap=0.5, thickness=3.0, air_gap=12.0,
        max_track=220.0,
        own_residuals=False),
    "cooke_triplet": dict(
        defaults="configs/multi_element/defaults.yml", logs="multi_element",
        # add by cjy: formal Cooke benchmark.  Do not relax this to the f/8
        # diagnostic case: Kodak's Cooke-triplet Example 6 is explicitly f/2.8
        # over +/-5 degrees.  EFL only fixes the scale, so retain the project's
        # convenient 100 mm normalization while matching that relative aperture
        # and field.
        efl=100.0, f_number=2.8, field=5.0, n_fields=5,
        wavelengths=[486.1, 550.0, 656.3], design=550.0,
        glass_file="configs/moldable_materials.csv",
        glass={"nd": 1.6, "vd": 45.0}, stop_gap=0.5, thickness=3.0, air_gap=3.0,
        max_track=250.0,
        own_residuals=False),
}
SPEC = SPECS[os.environ.get("SPEC", "default")]

DEFAULTS, LOG_ROOT = SPEC["defaults"], SPEC["logs"]
EFL, F_NUMBER = SPEC["efl"], SPEC["f_number"]
WAVELENGTHS, DESIGN = SPEC["wavelengths"], SPEC["design"]
FIELD_DEG, N_FIELDS = SPEC["field"], SPEC["n_fields"]
GLASS = SPEC["glass"]  # start; released in stage 2
STOP_GAP, THICKNESS, AIR_GAP = SPEC["stop_gap"], SPEC["thickness"], SPEC["air_gap"]
N_ASPHERE = 5  # conic + r^4..r^10; the conic stays frozen
# add by cjy: overridable so a control can be given the same total budget as
# the annular flow, which runs four stages against the refractive two.
STAGE_STEPS = int(os.environ.get("STAGE_STEPS", 300))
POWER_STEPS = int(os.environ.get("POWER_STEPS", max(200, STAGE_STEPS // 6)))
# add by cjy: the fast Cooke pupil needs higher orders to keep the annular
# topology below the same manufacturing/compute limits.  M remains an outer
# discrete choice; LM never differentiates through it or through the zone count.
ORDERS = (5, 10, 20, 30, 45, 60, 90, 120, 180, 240)
ROUNDS = int(os.environ.get("ROUNDS", 3))  # power stages, each followed by a fresh M search

# add by cjy: manufacturing limits, carried over from the singlet.
MIN_ZONE_WIDTH = 0.020  # mm
MAX_ZONES = 96

HERE = Path(__file__).resolve().parent
RADIUS = EFL / (2 * F_NUMBER)
LAM0 = DESIGN * 1e-6


# add by cjy: which element carries the fold, 1-based.  It used to be pinned to
# the first, which is wrong when the point is to save glass: at the c_mount
# specification element 1 is the smallest of the four at 0.325 cm3 while element
# 4 is 1.045 cm3, 40% of the total.  Folding the largest element is where the
# volume is.
FOLD_AT = int(os.environ.get("FOLD_AT", 1))
# add by cjy: which face of it.  Geometrically the two are the same problem --
# fresnel_zones already takes the fold direction from copysign(step, curvature),
# and a step of +M or -M waves is equally invisible at the design line -- so only
# the sequence letter and the surface index change.
FOLD_FACE = os.environ.get("FOLD_FACE", "back")
RUN_TAG = os.environ.get("RUN_TAG", "").strip()
COOKE_START = bool(os.environ.get("COOKE_START"))


def sequence(n_elements, order=None):
    """Stop in front, then n elements; one of them is annular when a fold is asked for.

    The "-" after the stop matters: RayPathResiduals assumes exactly one
    propagation without a spacing (object space), and a bare leading "s"
    creates a second one, which leaves its cutoff vector one entry short.

    The fold goes on the *back* face of its element, so the front asphere still
    corrects the coma one shaped surface cannot reach -- the same placement as
    the singlet's s-aRz-.
    """
    parts = ["aRa-"] * n_elements
    if order:
        parts[FOLD_AT - 1] = "zRa-" if FOLD_FACE == "front" else "aRz-"
    return "s-" + "".join(parts)


def fold_surface(n_elements):
    """Index into lens.c of the folded face of element FOLD_AT."""
    return 2 * (FOLD_AT - 1) + (0 if FOLD_FACE == "front" else 1)


def element_focal(n_elements):
    """Focal length of one element under the even power split."""
    return n_elements * EFL


# add by cjy: zone topology for the folded element, derived from whatever power
# that element currently has -- never from a planned share of the system EFL.
# The scaffold IS the power: pinning Rmax to the fold radii and Zoff to k*step
# forces sag(r_k) = k*step, which is the envelope equation.  Measured on the
# converged singlet: the scaffold implies a 100.00 mm envelope and the centre
# zone came out at 100.17 mm, while the outer zones drift to 132 mm as aspheric
# correction on top.  So laying the scaffold before the optimiser has chosen the
# power would be choosing it for the optimiser -- which is what forbade the
# negative first element the refractive solution wants (f1 = -169 mm, Vd 18.5).
def zone_footprint(lens):
    """Outer radius the zone table has to span, measured on the face before the fold.

    add by cjy.  The table used to stop at RADIUS, the entrance-pupil semi-diameter.
    That is only right when the stop sits on the element.  At the c_mount
    specification the stop ends up 8.1 mm in front and the field is 15.88 deg, so
    the chief ray walks 2.3 mm out and the footprint on the folded face is 8.46 mm
    against a 7.0 mm table -- every ray past the last Rmax has no zone and comes
    back as status 4, 21% of the pupil on axis and 36% at the outer field.  The
    front face is one thin element away from the fold, so its footprint bounds the
    fold's closely; the margin covers the rest.
    """
    rays = ri.RayInitialization(
        aperture=2 * RADIUS, aperture_type="epd", hfov=FIELD_DEG, n_fields=N_FIELDS,
        wavelengths=WAVELENGTHS, pupil_sampling_mode="skew_uniform",
        pupil_sampling_kwargs={"n_r": 24, "n_theta": 16}, ray_aiming_steps=0)
    r0, d0 = rays(lens)
    # the face just before the fold bounds its footprint; folding the very first
    # surface leaves nothing in front of it, so fall back to the pupil bound
    k = fold_surface(len(lens.c.reshape(-1)) // 2)
    if k == 0:
        return RADIUS + 1.08 * float(lens.s.reshape(-1)[0]) * math.tan(
            math.radians(FIELD_DEG))
    front = [i for i, e in enumerate(lens.sequence.events) if e["type"] == "r"][k - 1]
    for i, (r, _, status, _) in enumerate(lens.trace_rays(r0, d0, WAVELENGTHS, yield_on="all")):
        if i == front:
            radius = r[:2].norm(dim=0)
            alive = status == 0
            if alive.any():
                return 1.08 * float(radius[alive].max())
            break
    return RADIUS


def zone_radii(order, focal, outer=None):
    """Fold boundaries for an element of this focal length; |focal| sets the scale."""
    outer = RADIUS if outer is None else max(outer, RADIUS)
    edges, index = [], 1
    while True:
        radius = math.sqrt(2 * abs(focal) * order * LAM0 * index)
        if radius >= outer:
            break
        edges.append(radius)
        index += 1
    # add by cjy: the aperture almost never lands on a fold boundary, so the
    # outermost zone is a partial one of arbitrary width.  Merge it into the zone
    # below when it is much narrower than its neighbour, not just when it breaks
    # the absolute limit.  A 27.8 um sliver next to 62 um neighbours passed the
    # 20 um test and then took almost no rays during fitting -- 4 samples per
    # zone over 6.25 mm puts about one ray in it -- so LM gave it an
    # unconstrained A1/A2, and at evaluation density its 8 rays landed 711 um
    # off axis and pushed the 1 degree RMS from 11.9 to 14.2 um.
    if edges and outer - edges[-1] < max(
        MIN_ZONE_WIDTH, 0.5 * (edges[-1] - (edges[-2] if len(edges) > 1 else 0.0))
    ):
        edges.pop()
    return edges + [outer]


def widths(edges):
    return [b - a for a, b in zip([0.0] + edges[:-1], edges)]


def start_from(path, n_elements):
    """Read the complete physical state from an existing design.

    add by cjy.  The even power split is a neutral start, not a good one: at the
    c_mount specification it walks LM into a basin with a +0.122 / -0.183 diopter
    front pair on an 8 mm semi-aperture, a 14 mm air gap and a 56 mm track, and
    6000 steps only reach 26.7 um against the project design's 12.2.  Starting
    from a structure that is already right and adding surface freedom is what a
    designer does, and it is the only way to ask what the fold is worth on top of
    a competitive design rather than on top of my own shortfall.
    """
    init = yaml.safe_load(Path(path).read_text())["model"]["lens_parameterization"]["init_args"]
    spacings = [float(v) for v in init["s"]]
    curvatures = [float(v) for v in init["c"]]
    assert len(curvatures) == 2 * n_elements, (
        f"{path} has {len(curvatures)} surfaces, not {2 * n_elements}")
    aspheres = [[float(x) for x in row] for row in (init.get("a") or [])]
    zonal = [
        [[float(x) for x in zone] for zone in surface]
        for surface in (init.get("z") or [])
    ]
    return (spacings, curvatures, [float(v) for v in init["nd"]],
            [float(v) for v in init["vd"]], aspheres, zonal)


def paraxial_start(n_elements, order=None):
    """Even power split, every element equiconvex -- except a folded first one."""
    curvature = 1 / (2 * (GLASS["nd"] - 1) * element_focal(n_elements))
    spacings = [STOP_GAP]
    for _ in range(n_elements):
        spacings += [THICKNESS, AIR_GAP]
    spacings[-1] = EFL  # back focal distance
    curvatures = [curvature, -curvature] * n_elements
    if order:
        # Flat front (the asphere shapes it); the fold envelope is the back face.
        k = fold_surface(n_elements)
        curvatures[k - 1], curvatures[k] = 0.0, -fold_curvature(n_elements, GLASS)
    return spacings, curvatures


def cooke_start():
    """Deterministic positive-negative-positive Cooke-form paraxial seed."""
    powers = (1.25 / EFL, -1.50 / EFL, 1.25 / EFL)
    nd = [1.62, 1.72, 1.62]
    vd = [60.0, 30.0, 60.0]
    curvatures = []
    for power, index in zip(powers, nd):
        curvature = power / (2 * (index - 1))
        curvatures.extend((curvature, -curvature))
    spacings = [STOP_GAP, 4.0, 3.0, 2.5, 3.0, 4.0, EFL]
    return spacings, curvatures, nd, vd


def design_index(glass):
    return float(
        optics.hartmann_dispersion(
            *(torch.tensor([v]) for v in (DESIGN, glass["nd"], glass["vd"], 0.0))
        )
    )


def fold_curvature(n_elements, glass):
    """Envelope curvature of the folded face: the lens the staircase reproduces."""
    return 1 / ((design_index(glass) - 1) * element_focal(n_elements))


def fresnel_zones(order, curvature, index, outer=None):
    """Zone table for a folded face of this base curvature: [A1, A2, Zoff, Rmax].

    add by cjy.  Everything is derived from the curvature the optimiser has
    arrived at -- nothing is planned.  Zoff_k = -(c/2) * r_k^2 cancels the
    envelope exactly at every fold radius, which reduces to k * M * lambda0 /
    (n - 1); copysign carries the sign, so a negative element folds the other
    way and is handled without a special case.

    A1 and A2 start at zero: they are the per-zone aspheric correction, and the
    power is already in c.  Left with no curvature and no A1 the element is a
    plane-parallel plate -- a flat facet has no power however the Zoff steps are
    stacked -- which started the two-element system 100 mm out of focus with a
    10 mm spot that LM never recovered from.
    """
    step = -math.copysign(order * LAM0 / (index - 1), curvature)
    edges = zone_radii(order, -1 / ((index - 1) * curvature), outer)
    return [[[0.0, 0.0, k * step, edge] for k, edge in enumerate(edges)]], edges, step


def fold_options(curvature, index, outer=None):
    """Every fold order that meets the manufacturing limits at this element power."""
    out = []
    for order in ORDERS:
        zones, edges, step = fresnel_zones(order, curvature, index, outer)
        if min(widths(edges)) < MIN_ZONE_WIDTH or len(edges) > MAX_ZONES:
            continue
        out.append((order, zones, edges, step))
    return out


def fit_centered_wrapped_asphere(order, curvature, asphere, index, outer):
    """Fold an asphere with a centered nearest-period wrapping convention.

    add by cjy: a fast Cooke surface is not its paraxial curvature.  Building
    zones from ``c`` alone discarded millimetres of high-order sag at f/2.8 and
    turned a 32 um starting point into a 3.3 mm spot.  Instead sample the full
    EISOPTX asphere, remove the nearest integer multiple of M*lambda/(n-1), and
    least-squares fit each resulting branch to the native
    ``(c/2+dA1)*rho + A2*rho^2 + dZ`` form.

    The first branch keeps dA1=dZ=0 so the existing ABCD curvature and the
    parameterization's centre-zone gauge remain identical.  Its jumps occur at
    half-integer periods, so this capability initializer is intentionally not
    the exact integer-OPD topology used by ``optimization.hdoe`` and the paper's
    constrained-M experiments.
    """
    period = order * LAM0 / (index - 1)
    rho = torch.linspace(0.0, outer**2, 200001)
    sag, inside = rt.evaluate_aspherical_profile(
        rho, torch.tensor(curvature), torch.tensor(asphere))
    if not bool(inside.all()):
        raise ValueError("wrapped source asphere is undefined inside the required footprint")
    branch = torch.round(sag / period).to(torch.int64)
    changes = torch.where(branch[1:] != branch[:-1])[0] + 1
    edges = [math.sqrt(float((rho[i - 1] + rho[i]) / 2)) for i in changes]
    edges.append(outer)

    zones = []
    r_min = 0.0
    for zone_index, r_max in enumerate(edges):
        local_rho = torch.linspace(r_min**2, r_max**2, 48)
        target, local_inside = rt.evaluate_aspherical_profile(
            local_rho, torch.tensor(curvature), torch.tensor(asphere))
        if not bool(local_inside.all()):
            raise ValueError("wrapped source asphere fit crossed its valid domain")
        mid = (r_min**2 + r_max**2) / 2
        mid_sag, _ = rt.evaluate_aspherical_profile(
            torch.tensor(mid), torch.tensor(curvature), torch.tensor(asphere))
        k = int(torch.round(mid_sag / period))
        residual = target - k * period - curvature * local_rho / 2
        if zone_index == 0:
            # Centre gauge: only rho^2 is needed for the first high-order term.
            denom = (local_rho**4).sum().clip(min=1e-30)
            delta_a1, a2, delta_z = 0.0, float((local_rho**2 * residual).sum() / denom), 0.0
        else:
            matrix = torch.stack((local_rho, local_rho**2, torch.ones_like(local_rho)), -1)
            delta_a1, a2, delta_z = torch.linalg.lstsq(matrix, residual).solution.tolist()
        zones.append([delta_a1, a2, delta_z, r_max])
        r_min = r_max
    return [zones], edges, period


def wrapped_asphere_options(curvature, asphere, index, outer):
    """Manufacturable fixed-topology candidates for folding an existing face."""
    out = []
    for order in ORDERS:
        zones, edges, period = fit_centered_wrapped_asphere(
            order, curvature, asphere, index, outer)
        if min(widths(edges)) < MIN_ZONE_WIDTH or len(edges) > MAX_ZONES:
            continue
        out.append((order, zones, edges, period))
    return out


def choose_fold(curvature, index, fixed=None, outer=None):
    """Re-lay the scaffold at the current element power and pick a fold order.

    add by cjy.  Run after every stage that moves the power, which is the point:
    the scaffold follows the optimiser instead of constraining it.

    Selection is on manufacturability alone, because the spot does not depend on
    M -- measured 35.66 um at M=5 (36 zones) against 35.63 um at M=60 (3 zones),
    with identical glasses, spacings and radii either side.  Relief goes as
    |step| ~ M, so the smallest admissible order gives the thinnest part; the
    price is more zones to cut, which is why the whole admissible set is printed.
    """
    options = fold_options(curvature, index, outer)
    if not options:
        raise SystemExit(
            f"no fold order is manufacturable at f = "
            f"{-1 / ((index - 1) * curvature):.1f} mm"
        )
    if fixed is not None:
        options = [o for o in options if o[0] == fixed] or options[:1]
        return options[0], options
    # add by cjy: the fit samples 4 rays per zone with a floor of 96, so the zone
    # count sets the cost -- 87 zones ran at 0.21 steps/s against 1.05 for the
    # refractive path, four stages of 3000 steps projecting to 15.8 hours.  The
    # spot does not depend on M (35.66 um at M=5/36 zones against 35.63 at
    # M=60/3 zones, identical glasses and spacings), so prefer an order that
    # stays under the floor and only pay for more zones when none does.
    cheap = [o for o in options if len(o[2]) * 4 <= 96]
    if cheap:
        return cheap[0], options          # smallest order that is already free
    return min(options, key=lambda o: len(o[2])), options   # otherwise the fewest zones


def flatten(values):
    """Snapshots store scalars as [v] per lens; the configs want plain floats."""
    if isinstance(values[0], list):
        values = [row[0] for row in values]
    return [float(v) for v in values]


def write_design(path, n_elements, spacings, curvatures, nd, vd, aspheres,
                 order=None, zones=None, enable_solves=True):
    # add by cjy: with the fold written into the base curvature the paraxial ABCD
    # is correct, so the solves stay on -- except for a folded singlet, where
    # every curvature is either flat or the frozen fold and there is nothing left
    # for a focal-length solve to size.  The image solve still works there.
    solved = enable_solves and (order is None or n_elements > 1)
    config = {
        "model": {
            "lens_parameterization": {
                "class_path": "eisoptx.optimization.parameterization.LensParameterization",
                "init_args": {
                    "lens_sequence": sequence(n_elements, order),
                    "s": spacings,
                    "c": curvatures,
                    "nd": nd,
                    "vd": vd,
                    "a": aspheres,
                    "d": [],
                    "m": [],
                    "z": zones or [],
                    "nominal_wavelength": DESIGN,
                    "target_efl": EFL,
                    "solve_type": "focal_length" if solved else None,
                    "solve_idx": 2 * n_elements - 1 if solved else None,
                    "paraxial_image_solve": solved,
                    "total_track_length_solve": None,
                    "qc_vars": False,
                    "scale_factor": RADIUS,
                    "bezier_aspherics": False,
                    "glass_file": SPEC["glass_file"],
                    "misc_surface_model": None,
                },
            }
        }
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def write_stage(path, n_elements, freeze_glass, order=None, n_zones=0, spherical=False,
                freeze_zones=False):
    """Freeze mask and constraint set for one LM stage."""
    n_spacings = 1 + 2 * n_elements
    freeze = {
        # The image distance is set by the paraxial solve.
        "s": [False] * (n_spacings - 1) + [True],
        "c": False,
        "g": freeze_glass,
        "m": True,
        "d": True,
        # Conic frozen: LM drives it to ~3e2 and the polynomial
        # terms then carry almost none of the correction.
        # add by cjy: freezing every term as well leaves a plain sphere -- the
        # cheapest spherical control, with no second sequence to maintain.
        "a": [[True] * (N_ASPHERE if spherical else 1)
              + [False] * (0 if spherical else N_ASPHERE - 1)],
    }
    if order:
        # add by cjy: two kinds of stage, and they must not overlap.  A1 and the
        # base curvature both contribute a rho term, so freeing them together
        # lets LM split the element's power arbitrarily between them and the
        # scaffold can no longer be read off c.
        #
        #   freeze_zones: the whole zone table is frozen and c is free, so the
        #       element's power lands in c alone and choose_fold() can re-lay the
        #       scaffold from it.  The frozen steps go stale while c moves; the
        #       re-lay at the end of the stage restores them to M waves exactly.
        #   otherwise:   c is frozen at the scaffold it was just laid for, and
        #       per-zone A1/A2 build the aspheric correction on top.  Zoff and
        #       Rmax stay frozen -- releasing Zoff on the singlet produced +41
        #       and -137 um steps in alternating directions for no gain.
        #
        # The front face is flat in both; its asphere does the shaping.
        k = fold_surface(n_elements)
        freeze["c"] = [i == k and not freeze_zones for i in range(2 * n_elements)]
        freeze["z"] = True if freeze_zones else [[[False, False, True, True]]]
    config = {
        "model": {
            "ray_initialization": {
                "init_args": {
                    "pupil_sampling_mode": "skew_uniform_zonal" if order else "skew_uniform",
                    # add by cjy: a zone no ray lands in is invisible to the
                    # residual, so radial sampling follows the zone count.  Set
                    # for both variants: the spec's own file may carry a
                    # different scheme (c_mount uses jittered n_r=16), and the
                    # comparison is only controlled if both are sampled alike.
                    "pupil_sampling_kwargs": {
                        "n_r": max(96, 4 * n_zones) if order else 96, "n_theta": 32},
                }
            },
            "lens_parameterization": {"init_args": {"freeze": freeze}},
            # add by cjy: only supplied when the spec does not bring its own.
            "residuals": [] if SPEC["own_residuals"] else [
                {
                    "class_path": "eisoptx.optimization.residuals.TransverseRayAberrationResiduals",
                    "init_args": {"weight": 1.0},
                },
                {
                    "class_path": "eisoptx.optimization.residuals.GlassVariableResiduals",
                    # Keeps glass inside the catalog hull.  At 1e-2 it dominates
                    # stage 2: the start sits far from the catalog centre, so LM
                    # spends the stage pulling glass inward and the spot grew
                    # from 52 to 57 um.  1e-3 leaves the spot in charge.
                    "init_args": {"weight": 1.0e-3},
                },
                {
                    "class_path": "eisoptx.optimization.residuals.RayPathResiduals",
                    "init_args": {
                        "weight": 10.0,
                        "min_cutoff": 0.5,  # air gap along every ray
                        "max_cutoff": 40.0,  # keeps the system compact
                        # Image space is set by the solve, so leave it alone.
                        "other_max_cutoffs": [[-1, float("inf")]],
                        "min_cutoff_refractive": 1.5,  # centre and edge thickness
                        "max_cutoff_refractive": 15.0,
                        # These must stay finite.  They default to -/+ inf, and
                        # the cutoff is built as spacing * relative: once a
                        # spacing goes negative that product flips sign to
                        # -/+ inf, the residual evaluates to inf, and the
                        # isfinite() filter inside the residual drops it -- so
                        # the constraint switches itself off exactly when a
                        # thickness has gone unphysical.  Observed directly:
                        # -47 mm and -64 mm glass thicknesses with ray_path
                        # logging 0.0 for the whole run.  0 and 100 are
                        # chosen so the relative bounds never bind for a
                        # physical spacing -- clip() returns the absolute
                        # cutoff above -- while staying finite for a
                        # negative one, which is what keeps the penalty on.
                        "min_cutoff_refractive_relative": 0.0,
                        "max_cutoff_refractive_relative": 100.0,
                    },
                },
                {
                    # add by cjy: nothing else in the set constrains distortion, and
                    # both variants ran pincushion -- 9.23% on the annular, 12.39%
                    # on the aspheric control, at a 1 degree field.  The residual is
                    # a ramp above the threshold, so it costs nothing once inside.
                    # Its paraxial reference is honest here for the same reason the
                    # solves are: the fold's power lives in lens.c, so
                    # evaluate_paraxial_heights_at_image_plane sees it.
                    "class_path": "eisoptx.optimization.residuals.DistortionResiduals",
                    "init_args": {"weight": 10.0, "threshold": 0.01},
                },
                {
                    "class_path": "eisoptx.optimization.residuals.SurfaceNormalResiduals",
                    "init_args": {"weight": 10.0, "max_angle": 30.0},
                },
                {
                    "class_path": "eisoptx.optimization.residuals.RayAngleResiduals",
                    "init_args": {"weight": 10.0, "max_angle": 60.0},
                },
            ],
            "lens_optimizer": {
                "class_path": "eisoptx.optimization.optimizers.LMOptimizer",
                "init_args": {
                    "lm_parameter": 1.0,
                    "damped_term_min": 1.0e-6,
                    "tolerance": 1.0,
                    "lam_increase_factor": 4.0,
                    "lam_decrease_factor": 2.0,
                    "lam_eps": 1.0e-10,
                    "beta": 0.95,
                },
            },
        },
        "trainer": {
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": None,
            "callbacks": [
                {
                    "class_path": "eisoptx.utils.callbacks.ConfigFileCallback",
                    "init_args": {"every_n_steps": 25},
                },
                "eisoptx.main.CustomProgressBar",
            ]
        },
    }
    if SPEC["own_residuals"]:
        # add by cjy: the project's own file already carries a tuned residual set,
        # an optimiser and a callback list; do not restate or override any of
        # them.  Overriding the callbacks had been dropping c_mount's
        # IncreaseGlassVariableResidualsWeightCallback, which is what lets the
        # glass constraint tighten as the design settles -- without it nd drifted
        # to 2.0068 against a 2.003 catalog maximum.  ConfigFileCallback is
        # already in that list, so the snapshots still get written.
        config["model"].pop("residuals", None)
        config["model"].pop("lens_optimizer", None)
        # The callback list cannot simply be inherited: c_mount's
        # VisualizationCallback is wired to its own 11 wavelengths and 11 fields
        # and indexes straight into them, so a 3-wavelength run dies on
        # "index 10 is out of bounds for axis 0 with size 3".  Restate the ones
        # that shape the optimisation -- the glass schedule is what stops nd
        # drifting to 2.0068 against a 2.003 catalog maximum -- and leave the
        # review plots to a separate test run.
        config["trainer"]["callbacks"] = [
            "eisoptx.utils.callbacks.ExtendedLoggingCallback",
            {"class_path": "eisoptx.utils.callbacks.CodeVSeqFileCallback",
             "init_args": {"use_private_catalog": True}},
            {"class_path": "eisoptx.utils.callbacks.ConfigFileCallback",
             "init_args": {"every_n_steps": 250}},
            "eisoptx.main.CustomProgressBar",
            {"class_path": "ModelCheckpoint",
             "init_args": {"every_n_train_steps": 100}},
            {"class_path": "eisoptx.utils.callbacks.IncreaseGlassVariableResidualsWeightCallback",
             "init_args": {"initial_step": 0.0, "final_step": 0.5, "n_increments": 25,
                           "initial_weight": 0.00001525878, "final_weight": 512}},
            {"class_path": "eisoptx.utils.callbacks.ToggleGlassOptimizationCallback",
             "init_args": {"initial_step": 0.0, "final_step": 0.5, "n_cycles": 25}},
            {"class_path": "eisoptx.utils.callbacks.BindMaterialsCallback",
             "init_args": {"step": 0.5}},
        ]
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def lens_from_parameters(path, n_elements, order=None):
    """Rebuild a lens from a runtime lens_parameters snapshot."""
    p = yaml.safe_load(Path(path).read_text())
    kw = {"dtype": torch.float64}
    column = lambda key: torch.tensor(flatten(p[key]), **kw).reshape(-1, 1)  # noqa: E731
    a = torch.tensor(p["a"], **kw)
    z = torch.tensor(p["z"], **kw) if order else None
    return optics.Lens(
        sequence=sequence(n_elements, order),
        s=column("s"),
        c=column("c"),
        nd=column("nd"),
        vd=column("vd"),
        dpgf=column("dpgf"),
        a=a.reshape(a.shape[0], 1, a.shape[-1]),
        d=torch.empty((0, 1, 0), **kw),
        m=torch.empty((0, 1, 0), **kw),
        z=z.reshape(1, 1, z.shape[-2], 4) if order else torch.empty((0, 1, 0, 4), **kw),
        w0=DESIGN,
    )


def spot_radii(lens, n_r=96, n_theta=64, mode="skew_uniform"):
    """Polychromatic RMS spot radius (um) per field, and the usable-ray fraction.

    A ray counts as usable when its status is below 2, which is the rule
    OpticsSimulator applies before the residuals see it: status 1 is a
    backtracking warning, 2 and above are TIR, reversal and misses.
    """
    rays = ri.RayInitialization(
        aperture=2 * RADIUS,
        aperture_type="epd",
        hfov=FIELD_DEG,
        n_fields=N_FIELDS,
        wavelengths=WAVELENGTHS,
        pupil_sampling_mode=mode,
        pupil_sampling_kwargs={"n_r": n_r, "n_theta": n_theta},
        ray_aiming_steps=0,
    )
    r0, d0 = rays(lens)
    r, _, status, _ = list(lens.trace_rays(r0, d0, WAVELENGTHS, yield_on="end"))[-1]
    xy = r[:2].where(status.unsqueeze(0) == 0, torch.tensor(float("inf")))
    if mode == "skew_uniform_zonal":
        pupil_weights = ri.zonal_pupil_weights(
            n_r=n_r,
            n_theta=n_theta,
            zone_edges=ri.zone_edges_from_lens(lens, 2 * RADIUS),
        ).view(-1, 1, 1)
    else:
        pupil_weights = torch.ones(xy.shape[2], dtype=xy.dtype).view(-1, 1, 1)
    out, centres = [], []
    for field in range(N_FIELDS):
        valid = status[field] == 0
        weights = pupil_weights.expand_as(valid) * valid
        denominator = weights.sum().clip(min=torch.finfo(weights.dtype).eps)
        points = xy[:, field].where(valid[None], 0.0)
        centre = (points * weights[None]).sum(dim=(1, 2, 3)) / denominator
        centres.append(float(centre[1]))
        radius2 = (points - centre.view(2, 1, 1, 1)).square().sum(0)
        out.append(float((radius2 * weights).sum().div(denominator).sqrt()) * 1e3)
    # add by cjy: distortion at the outer field, against the lens's own paraxial
    # chief-ray height.  Use the project's reference, not EFL * tan(theta):
    # the paraxial height is tan(theta) * (B - A * pupil_position), and with the
    # stop in front of the lens those differ badly.  Reading it off EFL made
    # these designs look like 9-12% pincushion when the real figure is 1.5%.
    # It is the same reference DistortionResiduals penalises against, so the
    # reported number and the optimised quantity are now the same thing.
    fields = torch.linspace(0.0, math.radians(FIELD_DEG), N_FIELDS)
    reference = lens.evaluate_paraxial_heights_at_image_plane(fields).squeeze()
    distortion = 100 * (centres[-1] - float(reference[-1])) / float(reference[-1])
    usable = float(
        (pupil_weights.expand_as(status) * (status == 0)).sum()
        / (status.shape[0] * status.shape[2] * status.shape[3])
    )
    return out, usable, distortion


# add by cjy: how deep the folded face has to be cut, peak to valley.  This is
# the number the fold buys: a continuous surface of the same power needs the
# full sag, the staircase needs one step plus the facet.
def relief(lens):
    rho = torch.linspace(0.0, RADIUS**2, 4001).reshape(-1, 1)
    k = fold_surface(len(lens.c.reshape(-1)) // 2)
    sag, inside = rt.evaluate_zonal_profile(
        rho, lens.c.reshape(-1)[k:k + 1], lens.z[0, 0].reshape(1, -1, 4)
    )
    sag = sag[inside]
    return float(sag.max() - sag.min())


def run_stage(design, stage, n_elements, steps, order=None):
    """Run one LM stage and return (final snapshot, run directory).

    The run is identified by differencing version directories: callbacks keep
    writing after a run finishes, so mtime does not track creation order.
    """
    root = ROOT / "logs" / LOG_ROOT / sequence(n_elements, order)
    before = {p.name for p in root.glob("version_*")} if root.exists() else set()
    result = subprocess.run(
        [
            sys.executable, "-m", "eisoptx.main", "fit",
            "-c", DEFAULTS,
            "-c", str(design.relative_to(ROOT)).replace("\\", "/"),
            "-c", str(stage.relative_to(ROOT)).replace("\\", "/"),
            f"--trainer.max_steps={steps}",
            f"--data.init_args.n_samples={steps}",
            # add by cjy: the defaults file pins the specification's aperture and
            # field; the generator owns those, so override them here.
            "--model.ray_initialization.init_args.aperture_type=epd",
            f"--model.ray_initialization.init_args.aperture={EFL / F_NUMBER}",
            f"--model.ray_initialization.init_args.hfov={FIELD_DEG}",
            f"--model.ray_initialization.init_args.n_fields={N_FIELDS}",
            "--model.ray_initialization.init_args.wavelengths="
            + json.dumps(WAVELENGTHS),
            # add by cjy: the weights must be overridden with the wavelengths --
            # c_mount ships 11 of each, and leaving 11 weights against 3 lines
            # makes RayInitialization reject the whole config.
            "--model.ray_initialization.init_args.wavelength_weights="
            + json.dumps([1.0] * len(WAVELENGTHS)),
            # add by cjy: the simulator carries its own [3 channels x n_wavelengths]
            # matrix, and c_mount ships an 11-column one.  Its RGBPSFs
            # visualization asserts the widths match, so it has to follow the
            # wavelengths too.  Rows are R, G, B against WAVELENGTHS in order.
            "--model.optics_simulator.init_args.wavelength_weights="
            + json.dumps([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            # add by cjy: pin the image circle to the sensor.  target_efl only
            # fixes the paraxial EFL (-1/C); the image height at a field angle is
            # tan(theta) * (B - A * pupil_position), and nothing tied B to EFL --
            # the four-element designs came out imaging 15.88 deg at 8.80 mm
            # against a 7.965 mm sensor semi-diagonal, so they covered 14.4 deg
            # of the specified 15.88.  Appended rather than written into the
            # stage file so the spec's own residual list survives intact.
            "--model.residuals+=" + json.dumps({
                "class_path": "eisoptx.optimization.residuals.ImageHeightResiduals",
                "init_args": {"weight": 10.0,
                              "target": EFL * math.tan(math.radians(FIELD_DEG))}}),
            # add by cjy: nothing bounded the track, and it ran to 56 mm against
            # the project design's 39.3.  A ramp above the target, so it costs
            # nothing while the design stays inside it.
            "--model.residuals+=" + json.dumps({
                "class_path": "eisoptx.optimization.residuals.TotalTrackLengthResiduals",
                "init_args": {"weight": 10.0, "target": SPEC["max_track"]}}),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    # add by cjy: check=True hides the traceback behind a bare CalledProcessError.
    if result.returncode:
        raise RuntimeError(result.stderr[-2500:])
    created = [p for p in root.glob("version_*") if p.name not in before]
    if len(created) != 1:
        raise RuntimeError(f"expected one new run directory, got {created}")
    return sorted((created[0] / "lens_parameters").glob("*.yml"))[-1], created[0]


def report_fold(order, edges, step, curvature, index, options):
    focal = -1 / ((index - 1) * curvature)
    print(f"    fold: element focal {focal:+8.2f} mm -> M={order}, {len(edges)} zones, "
          f"narrowest {min(widths(edges)) * 1e3:.1f} um, step {abs(step) * 1e3:.1f} um")
    print("          admissible " + ", ".join(
        f"M={o}({len(e)}z,{min(widths(e)) * 1e3:.0f}um)" for o, _, e, _ in options))


def main():
    n_elements = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    # add by cjy: "fold" searches the fold order every round, an integer pins it
    # to one order, "sph" is the spherical control, nothing is the aspheric one.
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    spherical = arg == "sph"
    fixed = None if arg in (None, "sph", "fold") else int(arg)
    annular = arg == "fold" or fixed is not None

    # add by cjy: START=<design.yml> begins from an existing structure instead.
    started = None
    started_z = []
    fold_source_asphere = None
    flat_input = None
    if os.environ.get("START"):
        spacings, curvatures, nd, vd, started, started_z = start_from(
            os.environ["START"], n_elements
        )
        if started:
            # add by cjy: keep the surface shape the start already has -- the point
            # of folding an optimised design is to ask what the fold costs on top
            # of it, not to re-derive it.  The folded face's row goes away with the
            # asphere it replaces.
            target_count = 2 * n_elements - (1 if annular else 0)
            if len(started) == target_count:
                # The source already has the same annular sequence.  Removing
                # another row here would shift every downstream asphere.
                aspheres = started
            elif annular and len(started) == 2 * n_elements:
                fold_source_asphere = started[fold_surface(n_elements)]
                aspheres = [
                    row for i, row in enumerate(started)
                    if i != fold_surface(n_elements)
                ]
            else:
                raise ValueError(
                    f"START contains {len(started)} aspheres; expected "
                    f"{target_count} for {sequence(n_elements, annular)}"
                )
        else:
            aspheres = [[0.0] * N_ASPHERE for _ in range(
                2 * n_elements - (1 if annular else 0)
            )]

    else:
        if COOKE_START:
            if n_elements != 3 or annular:
                raise ValueError("COOKE_START is only valid for the three-element baseline")
            spacings, curvatures, nd, vd = cooke_start()
        else:
            spacings, curvatures = paraxial_start(n_elements, annular)
            nd, vd = [GLASS["nd"]] * n_elements, [GLASS["vd"]] * n_elements
        if os.environ.get("FLAT_START"):
            flat_input = (
                list(spacings),
                [0.0] * (2 * n_elements),
                list(nd),
                list(vd),
                [[0.0] * N_ASPHERE for _ in range(2 * n_elements)],
            )
    if started is None:
        # add by cjy: one asphere per refractive face, minus the one the fold replaces.
        aspheres = [[0.0] * N_ASPHERE for _ in range(2 * n_elements - (1 if annular else 0))]

    if os.environ.get("RECORD_FLAT"):
        flat_input = (
            list(spacings),
            [0.0] * (2 * n_elements),
            list(nd),
            list(vd),
            [[0.0] * N_ASPHERE for _ in range(2 * n_elements)],
        )

    tag = f"{n_elements}p" + ("_fold" if annular else "_sph" if spherical else "")
    if RUN_TAG:
        tag += f"_{RUN_TAG}"
    design = HERE / "designs" / f"generated_{tag}.yml"
    design.parent.mkdir(parents=True, exist_ok=True)

    if flat_input is not None:
        flat_design = HERE / "designs" / f"generated_{tag}_flat.yml"
        write_design(
            flat_design,
            n_elements,
            *flat_input,
            order=None,
            zones=None,
            enable_solves=False,
        )
        print(f"Strict flat input recorded at {flat_design.relative_to(ROOT)}")

    print(
        f"Specification: EFL {EFL} mm, f/{F_NUMBER:g}, {WAVELENGTHS} nm, "
        f"+/-{FIELD_DEG} deg, {n_elements} element(s), "
        f"sequence {sequence(n_elements, annular)}."
    )

    # add by cjy: the schedule.  For the annular variant the fold order is not an
    # input -- every stage that can move the element's power is followed by a
    # fresh scaffold and a fresh M search, so the topology tracks the optimiser
    # instead of being planned in front of it.  The final stage freezes the
    # scaffold and spends its budget on the per-zone correction, so what comes
    # out is consistent: every step is exactly M waves at the design line.
    if annular:
        # add by cjy: a power round only moves curvatures, spacings and glass with
        # the zone table frozen, and settles in a few hundred steps; the shape
        # round is the one that needs the budget.
        if fold_source_asphere is not None:
            # The wrapped fit already inherits the continuous solution's power,
            # spacing, glass and high-order shape.  Do not erase that seed with
            # the generic paraxial power rounds.
            plan = [("zone shape", False, False, STAGE_STEPS)]
        else:
            plan = [(f"power {r + 1}", r > 0, True, POWER_STEPS)
                    for r in range(ROUNDS)]
            plan.append(("zone shape", False, False, STAGE_STEPS))
    else:
        plan = [("1: shape, glass frozen", False, False, STAGE_STEPS),
                ("2: glass released", True, False, STAGE_STEPS)]
        # add by cjy: a benchmark can deliberately stop after its validated
        # shape stage.  The fast Cooke material-release trial increased the
        # independently measured mean RMS from 32.04 to 32.87 um, so the
        # reproducible recipe records one stage instead of retaining that
        # rejected continuation.
        plan = plan[: int(os.environ.get("BASELINE_STAGES", len(plan)))]

    index = design_index({"nd": nd[FOLD_AT - 1], "vd": vd[FOLD_AT - 1]})
    order, zones, edges, step, options = None, None, [], 0.0, []
    if annular:
        # add by cjy: no lens to trace yet, so bound the chief-ray walk from the
        # start geometry; every later round measures it instead.
        # Conservative chief-ray walk to the selected face.  The old bound only
        # reached the first element and clipped later folded elements.
        outer = RADIUS + 1.08 * sum(
            spacings[: 1 + fold_surface(n_elements)]
        ) * math.tan(math.radians(FIELD_DEG)) + 2.0
        fold_c = curvatures[fold_surface(n_elements)]
        if started_z:
            if len(started_z) != 1 or len(started_z[0]) < 2:
                raise ValueError("An annular continuation needs one surface with at least two zones")
            old = [zone[:] for zone in started_z[0]]
            step = old[1][2] - old[0][2]
            inferred = int(round(abs(step) * (index - 1) / LAM0))
            if inferred not in ORDERS:
                raise ValueError(f"Could not infer a supported M from the existing step: {inferred}")
            order = fixed if fixed is not None else inferred
            fresh, fresh_edges, fresh_step = fresnel_zones(order, fold_c, index, outer)
            if outer > old[-1][3]:
                common = len(old) - 1
                if len(fresh_edges) <= common or any(
                    not math.isclose(old[i][3], fresh_edges[i], rel_tol=2e-4, abs_tol=2e-5)
                    for i in range(common)
                ):
                    raise ValueError("Existing annular boundaries do not match the current c/M scaffold")
                # The old last edge was an aperture cutoff, not a physical fold
                # boundary.  Extend that fitted zone to its true boundary, then
                # append only the newly illuminated outer zones.
                old[-1][3] = fresh_edges[common]
                old.extend(zone[:] for zone in fresh[0][len(old):])
            zones = [old]
            edges = [zone[3] for zone in old]
            step = fresh_step
            options = fold_options(fold_c, index, outer)
        elif fold_source_asphere is not None:
            options = wrapped_asphere_options(
                fold_c, fold_source_asphere, index, outer)
            if not options:
                raise SystemExit("no wrapped-asphere M meets the zone-count/width limits")
            if fixed is not None:
                selected = [o for o in options if o[0] == fixed]
                order, zones, edges, step = (selected or options[:1])[0]
            else:
                order, zones, edges, step = options[0]
        else:
            (order, zones, edges, step), options = choose_fold(
                fold_c, index, fixed, outer)
        print()
        report_fold(order, edges, step, fold_c, index, options)

    write_design(design, n_elements, spacings, curvatures, nd, vd, aspheres,
                 order, zones)

    print()
    print(f"{'stage':<14} {'spot um: axis/mid/edge':>26} {'':>13} {'EFL':>8} {'dist%':>7} {'usable':>7}"
          + ("  fold" if annular else ""))

    mode = "skew_uniform_zonal" if annular else "skew_uniform"
    lens, run, dist = None, None, 0.0
    for label, release_glass, freeze_zones, steps in plan:
        stage_name = label.split()[0].rstrip(":")
        stage = HERE / f"stage_{stage_name}_{tag}.yml"
        write_stage(stage, n_elements, not release_glass, order, len(edges),
                    spherical, freeze_zones)
        snapshot, run = run_stage(design, stage, n_elements, steps, order)
        p = yaml.safe_load(snapshot.read_text())
        spacings, curvatures = flatten(p["s"]), flatten(p["c"])
        nd, vd = flatten(p["nd"]), flatten(p["vd"])
        aspheres = [[float(x) for x in row] for row in p["a"]]
        zones = [[[float(x) for x in z] for z in p["z"][0]]] if annular else None

        lens = lens_from_parameters(snapshot, n_elements, order)
        spots, usable, dist = spot_radii(
            lens, n_r=max(96, 8 * len(edges)) if annular else 96, mode=mode)
        # add by cjy: 11 fields is too wide to print; show the ends and the mean.
        shown = spots if len(spots) <= 5 else [spots[0], spots[len(spots) // 2], spots[-1]]
        print(f"{label:<14} " + "  ".join(f"{v:7.2f}" for v in shown)
              + f" | mean {sum(spots) / len(spots):7.2f}"
              + f" {float(lens.efl):8.2f} {dist:7.2f} {usable:7.3f}"
              + (f"  M={order}, {len(edges)}z" if annular else ""))

        # add by cjy: the power just moved, so re-lay the scaffold on it.  A1/A2
        # are reset with it -- they were fitted to the old zone partition and do
        # not transfer -- and the next stage rebuilds them.
        if annular and freeze_zones:
            index = design_index({"nd": nd[FOLD_AT - 1], "vd": vd[FOLD_AT - 1]})
            (order, zones, edges, step), options = choose_fold(
                curvatures[fold_surface(n_elements)], index, fixed, zone_footprint(lens))
            report_fold(
                order,
                edges,
                step,
                curvatures[fold_surface(n_elements)],
                index,
                options,
            )

        write_design(design, n_elements, spacings, curvatures, nd, vd, aspheres,
                     order, zones)

    s = lens.s.reshape(-1).tolist()
    print(f"\ntotal track {sum(s):.1f} mm, EFL {float(lens.efl):.2f} mm, "
          f"distortion {dist:.2f}%")
    print("  spacings  " + "  ".join(f"{v:7.3f}" for v in s))
    print("  radii     " + "  ".join(
        f"{1 / v:7.1f}" if abs(v) > 1e-9 else "    inf" for v in lens.c.reshape(-1).tolist()
    ))
    print("  nd/vd     " + "  ".join(
        f"{float(a):.4f}/{float(b):.1f}"
        for a, b in zip(lens.nd.reshape(-1), lens.vd.reshape(-1))
    ))
    for k in range(n_elements):
        n = design_index({"nd": float(lens.nd.reshape(-1)[k]),
                          "vd": float(lens.vd.reshape(-1)[k])})
        c = lens.c.reshape(-1)
        d = float(lens.s.reshape(-1)[1 + 2 * k])
        f_e = 1 / ((n - 1) * (c[2 * k] - c[2 * k + 1])
                   + (n - 1) ** 2 * d * c[2 * k] * c[2 * k + 1] / n)
        print(f"  element {k + 1}  f = {f_e:+9.2f} mm   power {1 / f_e:+.6f}"
              + ("   <- folded" if annular and k == FOLD_AT - 1 else ""))
    if annular:
        print(f"  zones     M={order}, {len(edges)}, relief {relief(lens) * 1e3:.1f} um")
    print(f"run directory: {run.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
