from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

from aether_traj.models import SECONDS_PER_DAY


AU_KM = 149_597_870.7
EARTH_MOON_DISTANCE_KM = 384_400.0
ECLIPTIC_OBLIQUITY_RAD = math.radians(23.439291)

REFERENCE_YEAR = 2026
REFERENCE_MONTH = 3
REFERENCE_DAY = 28
REFERENCE_EPOCH_UTC = f"{REFERENCE_YEAR:04d}-{REFERENCE_MONTH:02d}-{REFERENCE_DAY:02d}T00:00:00"
REFERENCE_JULIAN_DAY = 2_461_125.5

ANALYTIC_TABLE_ID = "analytic_low_precision_2026a"
ANALYTIC_HIACC_TABLE_ID = "analytic_hiacc_2026a"
DE440_TABLE_ID = "de440_local_2026a"
DE440_HIACC_TABLE_ID = "de440_hiacc_v2_2026a"
TABLE_START_DAYS = -60.0
TABLE_END_DAYS = 400.0
LEGACY_TABLE_STEP_SECONDS = 3_600.0
HIACC_TABLE_STEP_SECONDS = 600.0
EPHEMERIS_CACHE_VERSION = "hiacc_v2"

DE440_KERNEL_PATH = Path("data/kernels/de440s.bsp")
NAIF_TLS_PATH = Path("data/kernels/naif0012.tls")
PCK_PATH = Path("data/kernels/pck00011.tpc")
MOON_BPC_PATH = Path("data/kernels/moon_pa_de440_200625.bpc")
MOON_FRAME_PATH = Path("data/kernels/moon_de440_250416.tf")
DE440_CACHE_PATH = Path("data/ephemeris/de440_local_2026a.npz")
DE440_HIACC_CACHE_PATH = Path("data/ephemeris/de440_hiacc_v2_2026a_600s.npz")


@dataclass(frozen=True)
class EphemerisVector:
    x: float
    y: float
    z: float

    def __add__(self, other: "EphemerisVector") -> "EphemerisVector":
        return EphemerisVector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "EphemerisVector") -> "EphemerisVector":
        return EphemerisVector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> "EphemerisVector":
        return EphemerisVector(self.x * scale, self.y * scale, self.z * scale)

    __rmul__ = __mul__

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class EphemerisState:
    position: EphemerisVector
    velocity: EphemerisVector


@dataclass(frozen=True)
class EphemerisTable:
    table_id: str
    start_seconds: float
    step_seconds: float
    times_seconds: tuple[float, ...]
    moon_positions_km: tuple[tuple[float, float, float], ...]
    moon_velocities_kms: tuple[tuple[float, float, float], ...]
    sun_positions_km: tuple[tuple[float, float, float], ...]
    sun_velocities_kms: tuple[tuple[float, float, float], ...]
    moon_j2000_to_pa: tuple[tuple[float, ...], ...] | None = None
    source: str = "unknown"
    orientation_source: str | None = None
    orientation_frame: str | None = None
    kernel_paths: tuple[str, ...] = ()
    cache_version: str = "legacy_v1"

    @property
    def end_seconds(self) -> float:
        return self.times_seconds[-1]


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def campaign_bounds_seconds() -> tuple[float, float]:
    return TABLE_START_DAYS * SECONDS_PER_DAY, TABLE_END_DAYS * SECONDS_PER_DAY


def _rotate_ecliptic_to_equatorial(vector: EphemerisVector) -> EphemerisVector:
    cosine = math.cos(ECLIPTIC_OBLIQUITY_RAD)
    sine = math.sin(ECLIPTIC_OBLIQUITY_RAD)
    return EphemerisVector(
        vector.x,
        cosine * vector.y - sine * vector.z,
        sine * vector.y + cosine * vector.z,
    )


def _sun_position_analytic(days_since_j2000: float) -> EphemerisVector:
    mean_anomaly_deg = 357.52911 + 0.98560028 * days_since_j2000
    mean_longitude_deg = 280.46646 + 0.98564736 * days_since_j2000
    mean_anomaly = math.radians(mean_anomaly_deg % 360.0)
    mean_longitude = math.radians(mean_longitude_deg % 360.0)
    ecliptic_longitude = mean_longitude + math.radians(
        1.914602 * math.sin(mean_anomaly)
        + 0.019993 * math.sin(2.0 * mean_anomaly)
        + 0.000289 * math.sin(3.0 * mean_anomaly)
    )
    radius_au = (
        1.000140612
        - 0.016708617 * math.cos(mean_anomaly)
        - 0.000139589 * math.cos(2.0 * mean_anomaly)
    )
    ecliptic = EphemerisVector(
        radius_au * AU_KM * math.cos(ecliptic_longitude),
        radius_au * AU_KM * math.sin(ecliptic_longitude),
        0.0,
    )
    return _rotate_ecliptic_to_equatorial(ecliptic)


def _moon_position_analytic(days_since_j2000: float) -> EphemerisVector:
    l0 = math.radians((218.316 + 13.176396 * days_since_j2000) % 360.0)
    m_moon = math.radians((134.963 + 13.064993 * days_since_j2000) % 360.0)
    m_sun = math.radians((357.529 + 0.98560028 * days_since_j2000) % 360.0)
    d = math.radians((297.850 + 12.190749 * days_since_j2000) % 360.0)
    f = math.radians((93.272 + 13.229350 * days_since_j2000) % 360.0)
    longitude = l0 + math.radians(
        6.289 * math.sin(m_moon)
        + 1.274 * math.sin(2.0 * d - m_moon)
        + 0.658 * math.sin(2.0 * d)
        + 0.214 * math.sin(2.0 * m_moon)
        + 0.110 * math.sin(d)
    )
    latitude = math.radians(
        5.128 * math.sin(f)
        + 0.280 * math.sin(m_moon + f)
        + 0.277 * math.sin(m_moon - f)
        + 0.173 * math.sin(2.0 * d - f)
        + 0.055 * math.sin(2.0 * d + f - m_moon)
        + 0.046 * math.sin(2.0 * d - f - m_moon)
        + 0.033 * math.sin(2.0 * d + f)
        + 0.017 * math.sin(2.0 * m_moon + f)
    )
    radius_km = (
        385_000.56
        - 20_905.0 * math.cos(m_moon)
        - 3_699.0 * math.cos(2.0 * d - m_moon)
        - 2_956.0 * math.cos(2.0 * d)
        - 570.0 * math.cos(2.0 * m_moon)
        + 246.0 * math.cos(2.0 * m_moon - 2.0 * d)
        - 205.0 * math.cos(m_sun - 2.0 * d)
        - 171.0 * math.cos(m_moon + 2.0 * d)
    )
    ecliptic = EphemerisVector(
        radius_km * math.cos(latitude) * math.cos(longitude),
        radius_km * math.cos(latitude) * math.sin(longitude),
        radius_km * math.sin(latitude),
    )
    return _rotate_ecliptic_to_equatorial(ecliptic)


def _position_and_velocity(position_fn, days_since_j2000: float, delta_seconds: float = 300.0) -> EphemerisState:
    delta_days = delta_seconds / SECONDS_PER_DAY
    position = position_fn(days_since_j2000)
    previous_position = position_fn(days_since_j2000 - delta_days)
    next_position = position_fn(days_since_j2000 + delta_days)
    velocity = (next_position - previous_position) * (0.5 / delta_seconds)
    return EphemerisState(position=position, velocity=velocity)


def _seconds_to_days_since_j2000(seconds_from_reference: float) -> float:
    reference_days_since_j2000 = REFERENCE_JULIAN_DAY - 2_451_545.0
    return reference_days_since_j2000 + seconds_from_reference / SECONDS_PER_DAY


def _matrix_tuple(matrix: object) -> tuple[float, ...]:
    rows = []
    for row in matrix:
        rows.extend(float(component) for component in row)
    return tuple(rows)


def _orthonormalize_matrix(flat_matrix: tuple[float, ...]) -> tuple[tuple[float, float, float], ...]:
    import numpy as np

    matrix = np.array(flat_matrix, dtype=np.float64).reshape(3, 3)
    row0 = matrix[0] / max(np.linalg.norm(matrix[0]), 1.0e-15)
    row1_trial = matrix[1] - row0 * float(np.dot(row0, matrix[1]))
    row1 = row1_trial / max(np.linalg.norm(row1_trial), 1.0e-15)
    row2 = np.cross(row0, row1)
    row2 = row2 / max(np.linalg.norm(row2), 1.0e-15)
    row1 = np.cross(row2, row0)
    return (
        (float(row0[0]), float(row0[1]), float(row0[2])),
        (float(row1[0]), float(row1[1]), float(row1[2])),
        (float(row2[0]), float(row2[1]), float(row2[2])),
    )


@lru_cache(maxsize=8)
def analytic_ephemeris_table(step_seconds: float) -> EphemerisTable:
    total_seconds = (TABLE_END_DAYS - TABLE_START_DAYS) * SECONDS_PER_DAY
    sample_count = int(round(total_seconds / step_seconds)) + 1
    times_seconds: list[float] = []
    moon_positions: list[tuple[float, float, float]] = []
    moon_velocities: list[tuple[float, float, float]] = []
    sun_positions: list[tuple[float, float, float]] = []
    sun_velocities: list[tuple[float, float, float]] = []
    for index in range(sample_count):
        seconds = TABLE_START_DAYS * SECONDS_PER_DAY + index * step_seconds
        moon_state = sample_ephemeris_state_direct("moon", seconds, prefer_analytic=True)
        sun_state = sample_ephemeris_state_direct("sun", seconds, prefer_analytic=True)
        times_seconds.append(seconds)
        moon_positions.append(moon_state.position.as_tuple())
        moon_velocities.append(moon_state.velocity.as_tuple())
        sun_positions.append(sun_state.position.as_tuple())
        sun_velocities.append(sun_state.velocity.as_tuple())
    table_id = ANALYTIC_HIACC_TABLE_ID if step_seconds <= HIACC_TABLE_STEP_SECONDS else ANALYTIC_TABLE_ID
    return EphemerisTable(
        table_id=table_id,
        start_seconds=times_seconds[0],
        step_seconds=step_seconds,
        times_seconds=tuple(times_seconds),
        moon_positions_km=tuple(moon_positions),
        moon_velocities_kms=tuple(moon_velocities),
        sun_positions_km=tuple(sun_positions),
        sun_velocities_kms=tuple(sun_velocities),
        source="analytic ephemeris series",
        cache_version="analytic_v1",
    )


def _save_ephemeris_cache(path: Path, table: EphemerisTable) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "table_id": table.table_id,
        "start_seconds": table.start_seconds,
        "step_seconds": table.step_seconds,
        "times_seconds": np.array(table.times_seconds, dtype=np.float64),
        "moon_positions_km": np.array(table.moon_positions_km, dtype=np.float64),
        "moon_velocities_kms": np.array(table.moon_velocities_kms, dtype=np.float64),
        "sun_positions_km": np.array(table.sun_positions_km, dtype=np.float64),
        "sun_velocities_kms": np.array(table.sun_velocities_kms, dtype=np.float64),
        "source": np.array(table.source),
        "orientation_source": np.array("" if table.orientation_source is None else table.orientation_source),
        "orientation_frame": np.array("" if table.orientation_frame is None else table.orientation_frame),
        "kernel_paths": np.array(table.kernel_paths, dtype=object),
        "cache_version": np.array(table.cache_version),
    }
    if table.moon_j2000_to_pa is not None:
        payload["moon_j2000_to_pa"] = np.array(table.moon_j2000_to_pa, dtype=np.float64)
    np.savez_compressed(path, **payload)


def _load_ephemeris_cache(path: Path) -> EphemerisTable:
    import numpy as np

    payload = np.load(path, allow_pickle=True)
    moon_rotation = None
    if "moon_j2000_to_pa" in payload.files:
        moon_rotation = tuple(
            tuple(float(component) for component in row)
            for row in payload["moon_j2000_to_pa"].tolist()
        )
    return EphemerisTable(
        table_id=str(payload["table_id"]),
        start_seconds=float(payload["start_seconds"]),
        step_seconds=float(payload["step_seconds"]),
        times_seconds=tuple(float(value) for value in payload["times_seconds"].tolist()),
        moon_positions_km=tuple(tuple(float(component) for component in row) for row in payload["moon_positions_km"].tolist()),
        moon_velocities_kms=tuple(tuple(float(component) for component in row) for row in payload["moon_velocities_kms"].tolist()),
        sun_positions_km=tuple(tuple(float(component) for component in row) for row in payload["sun_positions_km"].tolist()),
        sun_velocities_kms=tuple(tuple(float(component) for component in row) for row in payload["sun_velocities_kms"].tolist()),
        moon_j2000_to_pa=moon_rotation,
        source=str(payload["source"]) if "source" in payload.files else "unknown",
        orientation_source=(str(payload["orientation_source"]) or None) if "orientation_source" in payload.files else None,
        orientation_frame=(str(payload["orientation_frame"]) or None) if "orientation_frame" in payload.files else None,
        kernel_paths=tuple(str(value) for value in payload["kernel_paths"].tolist()) if "kernel_paths" in payload.files else (),
        cache_version=str(payload["cache_version"]) if "cache_version" in payload.files else "legacy_v1",
    )


@lru_cache(maxsize=1)
def _reference_et() -> float:
    import spiceypy

    _load_spice_kernels()
    return float(spiceypy.str2et(REFERENCE_EPOCH_UTC))


@lru_cache(maxsize=1)
def _load_spice_kernels() -> tuple[str, ...]:
    import spiceypy

    kernels = (
        str(NAIF_TLS_PATH),
        str(PCK_PATH),
        str(MOON_FRAME_PATH),
        str(MOON_BPC_PATH),
        str(DE440_KERNEL_PATH),
    )
    for kernel in kernels:
        if not Path(kernel).is_file():
            raise FileNotFoundError(f"missing SPICE kernel: {kernel}")
        spiceypy.furnsh(kernel)
    return kernels


def _spice_et_from_seconds(seconds_from_reference: float) -> float:
    return _reference_et() + seconds_from_reference


def sample_ephemeris_state_direct(body: str, seconds: float, prefer_analytic: bool = False) -> EphemerisState:
    if prefer_analytic:
        days_since_j2000 = _seconds_to_days_since_j2000(seconds)
        if body == "sun":
            return _position_and_velocity(_sun_position_analytic, days_since_j2000)
        if body == "moon":
            return _position_and_velocity(_moon_position_analytic, days_since_j2000)
        raise ValueError(f"unsupported body: {body}")
    import spiceypy

    _load_spice_kernels()
    et = _spice_et_from_seconds(seconds)
    target = "MOON" if body == "moon" else "SUN"
    state, _ = spiceypy.spkezr(target, et, "J2000", "NONE", "EARTH")
    return EphemerisState(
        position=EphemerisVector(float(state[0]), float(state[1]), float(state[2])),
        velocity=EphemerisVector(float(state[3]), float(state[4]), float(state[5])),
    )


def sample_moon_j2000_to_pa_direct(seconds: float) -> tuple[tuple[float, float, float], ...]:
    import spiceypy

    _load_spice_kernels()
    et = _spice_et_from_seconds(seconds)
    return tuple(tuple(float(component) for component in row) for row in spiceypy.pxform("J2000", "MOON_PA", et))


def _build_spice_table(step_seconds: float, include_orientation: bool) -> EphemerisTable:
    total_seconds = (TABLE_END_DAYS - TABLE_START_DAYS) * SECONDS_PER_DAY
    sample_count = int(round(total_seconds / step_seconds)) + 1
    times_seconds: list[float] = []
    moon_positions: list[tuple[float, float, float]] = []
    moon_velocities: list[tuple[float, float, float]] = []
    sun_positions: list[tuple[float, float, float]] = []
    sun_velocities: list[tuple[float, float, float]] = []
    moon_rotations: list[tuple[float, ...]] = []
    for index in range(sample_count):
        seconds = TABLE_START_DAYS * SECONDS_PER_DAY + index * step_seconds
        moon_state = sample_ephemeris_state_direct("moon", seconds)
        sun_state = sample_ephemeris_state_direct("sun", seconds)
        times_seconds.append(seconds)
        moon_positions.append(moon_state.position.as_tuple())
        moon_velocities.append(moon_state.velocity.as_tuple())
        sun_positions.append(sun_state.position.as_tuple())
        sun_velocities.append(sun_state.velocity.as_tuple())
        if include_orientation:
            moon_rotations.append(_matrix_tuple(sample_moon_j2000_to_pa_direct(seconds)))
    table_id = DE440_HIACC_TABLE_ID if include_orientation or step_seconds <= HIACC_TABLE_STEP_SECONDS else DE440_TABLE_ID
    return EphemerisTable(
        table_id=table_id,
        start_seconds=times_seconds[0],
        step_seconds=step_seconds,
        times_seconds=tuple(times_seconds),
        moon_positions_km=tuple(moon_positions),
        moon_velocities_kms=tuple(moon_velocities),
        sun_positions_km=tuple(sun_positions),
        sun_velocities_kms=tuple(sun_velocities),
        moon_j2000_to_pa=tuple(moon_rotations) if include_orientation else None,
        source="SPICE-derived Earth-Moon-Sun cache",
        orientation_source="SPICE PCK/BPC direct sample" if include_orientation else None,
        orientation_frame="MOON_PA" if include_orientation else None,
        kernel_paths=_load_spice_kernels(),
        cache_version=EPHEMERIS_CACHE_VERSION if include_orientation else "legacy_v1",
    )


@lru_cache(maxsize=1)
def de440_local_ephemeris_table() -> EphemerisTable:
    if DE440_CACHE_PATH.is_file():
        cached = _load_ephemeris_cache(DE440_CACHE_PATH)
        if cached.table_id == DE440_TABLE_ID:
            return cached
    table = _build_spice_table(LEGACY_TABLE_STEP_SECONDS, include_orientation=False)
    _save_ephemeris_cache(DE440_CACHE_PATH, table)
    return table


@lru_cache(maxsize=4)
def de440_hiacc_ephemeris_table(step_seconds: float = HIACC_TABLE_STEP_SECONDS) -> EphemerisTable:
    cache_path = DE440_HIACC_CACHE_PATH if abs(step_seconds - HIACC_TABLE_STEP_SECONDS) < 1.0e-9 else Path(
        f"data/ephemeris/de440_hiacc_v2_2026a_{int(step_seconds)}s.npz"
    )
    if cache_path.is_file():
        cached = _load_ephemeris_cache(cache_path)
        if cached.table_id == DE440_HIACC_TABLE_ID and abs(cached.step_seconds - step_seconds) < 1.0e-9:
            return cached
    table = _build_spice_table(step_seconds, include_orientation=True)
    _save_ephemeris_cache(cache_path, table)
    return table


def get_ephemeris_table(table_id: str = DE440_HIACC_TABLE_ID, step_seconds: float | None = None) -> EphemerisTable:
    if table_id == DE440_HIACC_TABLE_ID:
        return de440_hiacc_ephemeris_table(step_seconds or HIACC_TABLE_STEP_SECONDS)
    if table_id == DE440_TABLE_ID:
        return de440_local_ephemeris_table()
    if table_id == ANALYTIC_TABLE_ID:
        return analytic_ephemeris_table(step_seconds or LEGACY_TABLE_STEP_SECONDS)
    if table_id == ANALYTIC_HIACC_TABLE_ID:
        return analytic_ephemeris_table(step_seconds or HIACC_TABLE_STEP_SECONDS)
    raise ValueError(f"unknown ephemeris table id: {table_id}")


def _hermite_basis(tau: float) -> tuple[float, float, float, float]:
    tau2 = tau * tau
    tau3 = tau2 * tau
    return (
        2.0 * tau3 - 3.0 * tau2 + 1.0,
        tau3 - 2.0 * tau2 + tau,
        -2.0 * tau3 + 3.0 * tau2,
        tau3 - tau2,
    )


def _hermite_basis_derivative(tau: float) -> tuple[float, float, float, float]:
    tau2 = tau * tau
    return (
        6.0 * tau2 - 6.0 * tau,
        3.0 * tau2 - 4.0 * tau + 1.0,
        -6.0 * tau2 + 6.0 * tau,
        3.0 * tau2 - 2.0 * tau,
    )


def sample_ephemeris_state(table: EphemerisTable, body: str, seconds: float) -> EphemerisState:
    if seconds < table.start_seconds or seconds > table.end_seconds:
        raise ValueError(
            f"ephemeris table {table.table_id} does not cover t={seconds / SECONDS_PER_DAY:.2f} days; "
            f"coverage is {table.start_seconds / SECONDS_PER_DAY:.2f} to {table.end_seconds / SECONDS_PER_DAY:.2f} days"
        )
    index_float = (seconds - table.start_seconds) / table.step_seconds
    lower = int(math.floor(index_float))
    if lower >= len(table.times_seconds) - 1:
        lower = len(table.times_seconds) - 2
        tau = 1.0
    else:
        tau = clamp(index_float - lower, 0.0, 1.0)
    if body == "moon":
        positions = table.moon_positions_km
        velocities = table.moon_velocities_kms
    elif body == "sun":
        positions = table.sun_positions_km
        velocities = table.sun_velocities_kms
    else:
        raise ValueError(f"unsupported ephemeris body: {body}")
    p0 = positions[lower]
    p1 = positions[lower + 1]
    v0 = velocities[lower]
    v1 = velocities[lower + 1]
    h00, h10, h01, h11 = _hermite_basis(tau)
    dh00, dh10, dh01, dh11 = _hermite_basis_derivative(tau)
    dt = table.step_seconds
    position = EphemerisVector(
        h00 * p0[0] + h10 * dt * v0[0] + h01 * p1[0] + h11 * dt * v1[0],
        h00 * p0[1] + h10 * dt * v0[1] + h01 * p1[1] + h11 * dt * v1[1],
        h00 * p0[2] + h10 * dt * v0[2] + h01 * p1[2] + h11 * dt * v1[2],
    )
    velocity = EphemerisVector(
        (dh00 * p0[0] + dh10 * dt * v0[0] + dh01 * p1[0] + dh11 * dt * v1[0]) / dt,
        (dh00 * p0[1] + dh10 * dt * v0[1] + dh01 * p1[1] + dh11 * dt * v1[1]) / dt,
        (dh00 * p0[2] + dh10 * dt * v0[2] + dh01 * p1[2] + dh11 * dt * v1[2]) / dt,
    )
    return EphemerisState(position=position, velocity=velocity)


def sample_moon_j2000_to_pa(table: EphemerisTable, seconds: float) -> tuple[tuple[float, float, float], ...]:
    if table.moon_j2000_to_pa is None:
        raise ValueError(f"ephemeris table {table.table_id} does not contain Moon orientation samples")
    if seconds < table.start_seconds or seconds > table.end_seconds:
        raise ValueError(f"orientation table {table.table_id} does not cover t={seconds / SECONDS_PER_DAY:.2f} days")
    index_float = (seconds - table.start_seconds) / table.step_seconds
    lower = int(math.floor(index_float))
    if lower >= len(table.times_seconds) - 1:
        lower = len(table.times_seconds) - 2
        tau = 1.0
    else:
        tau = clamp(index_float - lower, 0.0, 1.0)
    left = table.moon_j2000_to_pa[lower]
    right = table.moon_j2000_to_pa[lower + 1]
    blended = tuple((1.0 - tau) * left[index] + tau * right[index] for index in range(9))
    return _orthonormalize_matrix(blended)


def table_metadata(table: EphemerisTable) -> dict[str, object]:
    return {
        "table_id": table.table_id,
        "source": table.source,
        "cache_version": table.cache_version,
        "kernel_paths": list(table.kernel_paths),
        "orientation_source": table.orientation_source,
        "orientation_frame": table.orientation_frame,
        "reference_epoch_utc": f"{REFERENCE_EPOCH_UTC}Z",
        "start_days": table.start_seconds / SECONDS_PER_DAY,
        "end_days": table.end_seconds / SECONDS_PER_DAY,
        "step_seconds": table.step_seconds,
        "sample_count": len(table.times_seconds),
        "moon_orientation_sample_count": 0 if table.moon_j2000_to_pa is None else len(table.moon_j2000_to_pa),
    }
