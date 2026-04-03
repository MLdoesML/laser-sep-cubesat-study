from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

from aether_traj.models import EARTH_RADIUS_KM, Vec3


MOON_GRAVITY_MODEL_PATH = Path("data/gravity/grgm1200a_degree8_pa.json")
EARTH_J2 = 1.08262668e-3
EARTH_J3 = -2.5326564853322355e-6
EARTH_J4 = -1.619621591367e-6
SUN_RADIUS_KM = 695_700.0


@dataclass(frozen=True)
class SphericalHarmonicModel:
    model_name: str
    frame: str
    normalization: str
    reference_radius_km: float
    gm_km3_s2: float
    max_degree: int
    max_order: int
    source_url: str | None
    cbar: tuple[tuple[float, ...], ...]
    sbar: tuple[tuple[float, ...], ...]

    def truncated(self, max_degree: int, max_order: int) -> "SphericalHarmonicModel":
        clipped_degree = min(max_degree, self.max_degree)
        clipped_order = min(max_order, self.max_order)
        c_rows: list[tuple[float, ...]] = []
        s_rows: list[tuple[float, ...]] = []
        for degree in range(clipped_degree + 1):
            c_row = []
            s_row = []
            for order in range(clipped_degree + 1):
                if order <= degree and order <= clipped_order:
                    c_row.append(self.cbar[degree][order])
                    s_row.append(self.sbar[degree][order])
                else:
                    c_row.append(0.0)
                    s_row.append(0.0)
            c_rows.append(tuple(c_row))
            s_rows.append(tuple(s_row))
        return SphericalHarmonicModel(
            model_name=self.model_name,
            frame=self.frame,
            normalization=self.normalization,
            reference_radius_km=self.reference_radius_km,
            gm_km3_s2=self.gm_km3_s2,
            max_degree=clipped_degree,
            max_order=clipped_order,
            source_url=self.source_url,
            cbar=tuple(c_rows),
            sbar=tuple(s_rows),
        )


def _empty_coefficients(max_degree: int) -> tuple[list[float], ...]:
    return tuple([0.0] * (max_degree + 1) for _ in range(max_degree + 1))


def _normalized_zonal_from_jn(jn: float, degree: int) -> float:
    return -jn / math.sqrt(2.0 * degree + 1.0)


@lru_cache(maxsize=1)
def earth_gravity_model() -> SphericalHarmonicModel:
    max_degree = 4
    c_rows = [list(row) for row in _empty_coefficients(max_degree)]
    s_rows = [list(row) for row in _empty_coefficients(max_degree)]
    c_rows[0][0] = 1.0
    c_rows[2][0] = _normalized_zonal_from_jn(EARTH_J2, 2)
    c_rows[3][0] = _normalized_zonal_from_jn(EARTH_J3, 3)
    c_rows[4][0] = _normalized_zonal_from_jn(EARTH_J4, 4)
    return SphericalHarmonicModel(
        model_name="Earth zonals through J4",
        frame="J2000_equatorial",
        normalization="fully_normalized_4pi",
        reference_radius_km=EARTH_RADIUS_KM,
        gm_km3_s2=398_600.4418,
        max_degree=max_degree,
        max_order=0,
        source_url=None,
        cbar=tuple(tuple(row) for row in c_rows),
        sbar=tuple(tuple(row) for row in s_rows),
    )


@lru_cache(maxsize=1)
def moon_gravity_model() -> SphericalHarmonicModel:
    payload = json.loads(MOON_GRAVITY_MODEL_PATH.read_text(encoding="utf-8"))
    max_degree = int(payload["max_degree"])
    c_rows = [list(row) for row in _empty_coefficients(max_degree)]
    s_rows = [list(row) for row in _empty_coefficients(max_degree)]
    c_rows[0][0] = 1.0
    for coefficient in payload["coefficients"]:
        degree = int(coefficient["degree"])
        order = int(coefficient["order"])
        if degree > max_degree or order > max_degree:
            continue
        c_rows[degree][order] = float(coefficient["c"])
        s_rows[degree][order] = float(coefficient["s"])
    return SphericalHarmonicModel(
        model_name=str(payload["model"]),
        frame=str(payload["frame"]),
        normalization=str(payload["normalization"]),
        reference_radius_km=float(payload["reference_radius_km"]),
        gm_km3_s2=float(payload["gm_km3_s2"]),
        max_degree=max_degree,
        max_order=int(payload["max_order"]),
        source_url=payload.get("source_url"),
        cbar=tuple(tuple(row) for row in c_rows),
        sbar=tuple(tuple(row) for row in s_rows),
    )


@lru_cache(maxsize=32)
def normalization_factors(max_degree: int) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for degree in range(max_degree + 1):
        row: list[float] = []
        for order in range(max_degree + 1):
            if order > degree:
                row.append(0.0)
                continue
            numerator = (2.0 - (1.0 if order == 0 else 0.0)) * (2.0 * degree + 1.0)
            numerator *= math.factorial(degree - order)
            denominator = math.factorial(degree + order)
            row.append(math.sqrt(numerator / denominator))
        rows.append(tuple(row))
    return tuple(rows)


def associated_legendre_normalized_python(
    sin_latitude: float,
    max_degree: int,
    max_order: int,
) -> tuple[tuple[float, ...], ...]:
    unclipped = max(-1.0, min(1.0, sin_latitude))
    one_minus_s2 = max(0.0, 1.0 - unclipped * unclipped)
    cos_latitude = math.sqrt(one_minus_s2)
    unnormalized = [list(row) for row in _empty_coefficients(max_degree)]
    unnormalized[0][0] = 1.0
    for order in range(1, max_order + 1):
        unnormalized[order][order] = (2 * order - 1) * cos_latitude * unnormalized[order - 1][order - 1]
    for order in range(max_order + 1):
        if order < max_degree:
            unnormalized[order + 1][order] = (2 * order + 1) * unclipped * unnormalized[order][order]
        for degree in range(order + 2, max_degree + 1):
            unnormalized[degree][order] = (
                (2 * degree - 1) * unclipped * unnormalized[degree - 1][order]
                - (degree + order - 1) * unnormalized[degree - 2][order]
            ) / (degree - order)
    norms = normalization_factors(max_degree)
    return tuple(
        tuple(
            norms[degree][order] * unnormalized[degree][order] if order <= degree and order <= max_order else 0.0
            for order in range(max_degree + 1)
        )
        for degree in range(max_degree + 1)
    )


def spherical_harmonic_acceleration_python(
    position_body_km: tuple[float, float, float],
    model: SphericalHarmonicModel,
    max_degree: int,
    max_order: int,
) -> tuple[float, float, float]:
    degree_limit = min(max_degree, model.max_degree)
    order_limit = min(max_order, model.max_order)
    x, y, z = position_body_km
    r2 = max(1.0e-18, x * x + y * y + z * z)
    r = math.sqrt(r2)
    rho2 = max(1.0e-18, x * x + y * y)
    rho = math.sqrt(rho2)
    sin_latitude = max(-1.0, min(1.0, z / r))
    longitude = math.atan2(y, x)
    pbar = associated_legendre_normalized_python(sin_latitude, degree_limit, order_limit)
    delta = 1.0e-8
    pbar_plus = associated_legendre_normalized_python(min(1.0, sin_latitude + delta), degree_limit, order_limit)
    pbar_minus = associated_legendre_normalized_python(max(-1.0, sin_latitude - delta), degree_limit, order_limit)
    dbar_ds = tuple(
        tuple((pbar_plus[n][m] - pbar_minus[n][m]) / (2.0 * delta) for m in range(degree_limit + 1))
        for n in range(degree_limit + 1)
    )
    radial_sum = 0.0
    ds_sum = 0.0
    dlambda_sum = 0.0
    radius_ratio = model.reference_radius_km / r
    for degree in range(2, degree_limit + 1):
        degree_scale = radius_ratio**degree
        degree_term = 0.0
        degree_ds = 0.0
        degree_dlambda = 0.0
        order_cap = min(degree, order_limit)
        for order in range(order_cap + 1):
            cosine = math.cos(order * longitude)
            sine = math.sin(order * longitude)
            coeff = model.cbar[degree][order] * cosine + model.sbar[degree][order] * sine
            degree_term += pbar[degree][order] * coeff
            degree_ds += dbar_ds[degree][order] * coeff
            degree_dlambda += order * pbar[degree][order] * (
                -model.cbar[degree][order] * sine + model.sbar[degree][order] * cosine
            )
        scaled_term = degree_scale * degree_term
        radial_sum += (degree + 1) * scaled_term
        ds_sum += degree_scale * degree_ds
        dlambda_sum += degree_scale * degree_dlambda
    d_u_dr = -model.gm_km3_s2 / (r * r) * (1.0 + radial_sum)
    d_u_ds = model.gm_km3_s2 / r * ds_sum
    d_u_dlambda = model.gm_km3_s2 / r * dlambda_sum
    return (
        d_u_dr * (x / r) + d_u_ds * (-x * z / (r**3)) + d_u_dlambda * (-y / rho2),
        d_u_dr * (y / r) + d_u_ds * (-y * z / (r**3)) + d_u_dlambda * (x / rho2),
        d_u_dr * (z / r) + d_u_ds * ((rho * rho) / (r**3)),
    )


def third_body_point_mass_acceleration(position: Vec3, source_position: Vec3, mu: float) -> Vec3:
    delta = position - source_position
    delta_norm = delta.norm()
    source_norm = source_position.norm()
    if delta_norm <= 0.0 or source_norm <= 0.0:
        return Vec3(0.0, 0.0, 0.0)
    return delta * (-mu / (delta_norm**3)) - source_position * (mu / (source_norm**3))


def earth_point_mass_acceleration(position: Vec3, mu: float) -> Vec3:
    radius = position.norm()
    if radius <= 0.0:
        return Vec3(0.0, 0.0, 0.0)
    return position * (-mu / (radius**3))


def earth_j2_acceleration(position: Vec3, mu: float, enabled: bool = True) -> Vec3:
    if not enabled:
        return Vec3(0.0, 0.0, 0.0)
    x, y, z = position.as_tuple()
    r2 = x * x + y * y + z * z
    if r2 <= 0.0:
        return Vec3(0.0, 0.0, 0.0)
    r = math.sqrt(r2)
    factor = 1.5 * EARTH_J2 * mu * (EARTH_RADIUS_KM**2) / (r**5)
    z2_r2 = (z * z) / r2
    return Vec3(
        factor * x * (5.0 * z2_r2 - 1.0),
        factor * y * (5.0 * z2_r2 - 1.0),
        factor * z * (5.0 * z2_r2 - 3.0),
    )


def angular_radius(radius_km: float, distance_km: float) -> float:
    ratio = min(1.0, max(0.0, radius_km / max(distance_km, 1.0e-12)))
    return math.asin(ratio)


def eclipse_fraction_python(
    sun_vector_km: tuple[float, float, float],
    occulting_vector_km: tuple[float, float, float],
    occulting_radius_km: float,
) -> float:
    sun_distance = math.sqrt(sum(component * component for component in sun_vector_km))
    occulting_distance = math.sqrt(sum(component * component for component in occulting_vector_km))
    if sun_distance <= 0.0 or occulting_distance <= 0.0:
        return 1.0
    sun_radius = angular_radius(SUN_RADIUS_KM, sun_distance)
    occ_radius = angular_radius(occulting_radius_km, occulting_distance)
    dot = sum(s * o for s, o in zip(sun_vector_km, occulting_vector_km))
    separation = math.acos(max(-1.0, min(1.0, dot / (sun_distance * occulting_distance))))
    if separation >= sun_radius + occ_radius:
        return 1.0
    if separation <= abs(occ_radius - sun_radius):
        if occ_radius >= sun_radius:
            return 0.0
        overlap_ratio = (occ_radius * occ_radius) / max(sun_radius * sun_radius, 1.0e-12)
        return max(0.0, 1.0 - overlap_ratio)
    sun_r2 = sun_radius * sun_radius
    occ_r2 = occ_radius * occ_radius
    part1 = sun_r2 * math.acos(
        max(-1.0, min(1.0, (separation * separation + sun_r2 - occ_r2) / max(2.0 * separation * sun_radius, 1.0e-12)))
    )
    part2 = occ_r2 * math.acos(
        max(-1.0, min(1.0, (separation * separation + occ_r2 - sun_r2) / max(2.0 * separation * occ_radius, 1.0e-12)))
    )
    part3 = 0.5 * math.sqrt(
        max(
            0.0,
            (-separation + sun_radius + occ_radius)
            * (separation + sun_radius - occ_radius)
            * (separation - sun_radius + occ_radius)
            * (separation + sun_radius + occ_radius),
        )
    )
    overlap_area = part1 + part2 - part3
    return max(0.0, min(1.0, 1.0 - overlap_area / max(math.pi * sun_r2, 1.0e-12)))
