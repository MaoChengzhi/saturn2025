"""Cassini location queries and exports from the local NAIF kernel archive."""

from contextlib import contextmanager
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np
import pandas as pd
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceSPKINSUFFDATA

from data_process import DATA_DIR, LOCATION_DIR, _parse_time_range


SPICE_ROOT = DATA_DIR / "raw" / "spice_kernels" / "spice-cassini"
SATURN_RADIUS_KM = 60_268.0
POSITION_COLUMNS = ["x_Rs", "y_Rs", "z_Rs"]
MISSION_START_UTC = "2004-01-01T00:00:00Z"
MISSION_STOP_UTC = "2017-09-15T10:40:00Z"  # Exclusive, as in the original exports.
LOCATION_FILES = {
    "CASSINI_KRTP": LOCATION_DIR / "cassini_krtp_2004_2017_60s.pkl",
    "J2000": LOCATION_DIR / "cassini_j2000_saturn_centered_2004_2017_60s.pkl",
    "CASSINI_KSM": LOCATION_DIR / "cassini_ksm_2004_2017_60s.pkl",
    "ENIS": LOCATION_DIR / "cassini_enis_2004_2017_60s.pkl",
}
_SPICE_LOCK = Lock()


def latest_tm_for_year(year):
    """Return the highest-version local meta-kernel for a calendar year."""
    manifests = list((SPICE_ROOT / "extras" / "mk").glob(f"cas_{year}_v[0-9]*.tm"))
    if not manifests:
        raise FileNotFoundError(f"No Cassini meta-kernel for {year} under {SPICE_ROOT}")
    return max(manifests, key=lambda path: int(path.stem.rsplit("_v", 1)[1]))


@contextmanager
def _load_position_kernels(manifest):
    # This archive maps $KERNELS to ./data. Absolute paths avoid changing cwd.
    entries = re.findall(
        r"'\$KERNELS/((?:lsk|pck|fk|spk)/[^']+)'",
        manifest.read_text(encoding="latin-1"),
    )
    if not entries:
        raise ValueError(f"No position kernels listed in {manifest}")
    paths = [SPICE_ROOT / "data" / entry for entry in dict.fromkeys(entries)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Kernel listed in {manifest.name} is missing: {path}")

    with _SPICE_LOCK:
        loaded = {
            Path(spice.kdata(i, "ALL")[0]).resolve()
            for i in range(spice.ktotal("ALL"))
        }
        added = []
        try:
            for path in paths:
                if path not in loaded:
                    spice.furnsh(str(path))
                    added.append(path)
            yield
        finally:
            for path in reversed(added):
                spice.unload(str(path))


def _time_grid(start_time, end_time, step_seconds, *, include_end):
    bounds = _parse_time_range(start_time, end_time)
    if bounds is None or (not include_end and bounds[0] == bounds[1]):
        raise ValueError("Provide an ordered time range")
    step = pd.to_timedelta(step_seconds, unit="s")
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("step_seconds must be positive")
    start, end = (time.tz_localize("UTC") for time in bounds)
    times = pd.date_range(
        start, end, freq=step, tz="UTC", inclusive="both" if include_end else "left",
    )
    if include_end and times[-1] != end:
        times = times.append(pd.DatetimeIndex([end]))
    return times


def _ephemeris_times(epochs):
    # UTC has already been normalized. SPICE accepts ISO strings without an offset.
    return spice.str2et([time.isoformat() for time in epochs.tz_localize(None)])


def read_saturn_positions(start_time, end_time, *, frame="CASSINI_KSM", step_seconds=60):
    """Return geometric positions of Cassini and three moons in Saturn radii.

    Both UTC endpoints are included. The supported frames are CASSINI_KSM and
    J2000. Temporary text kernel loads can reset values set directly in the pool.
    """
    frame = frame.upper()
    if frame == "KSM":
        frame = "CASSINI_KSM"
    if frame not in ("CASSINI_KSM", "J2000"):
        raise ValueError("frame must be 'CASSINI_KSM' or 'J2000'")
    epochs = _time_grid(start_time, end_time, step_seconds, include_end=True)
    frames = []
    for year in epochs.year.unique():
        times = epochs[epochs.year == year]
        with _load_position_kernels(latest_tm_for_year(year)):
            et = _ephemeris_times(times)
            for body in ("Cassini", "Enceladus", "Mimas", "Tethys"):
                try:
                    xyz_km, _ = spice.spkpos(body.upper(), et, frame, "NONE", "SATURN")
                except SpiceSPKINSUFFDATA as error:
                    raise ValueError(
                        f"Incomplete local SPICE coverage for {body} between "
                        f"{times[0]} and {times[-1]}"
                    ) from error
                positions = pd.DataFrame(xyz_km / SATURN_RADIUS_KM, columns=POSITION_COLUMNS)
                positions.insert(0, "body", body)
                positions.insert(0, "utc", times)
                frames.append(positions)
    result = pd.concat(frames, ignore_index=True)
    result.attrs.update(frame=frame, observer="SATURN", abcorr="NONE", radius_km=SATURN_RADIUS_KM)
    return result


def _state_table(utc, et, frame):
    states, _ = spice.spkezr("CASSINI", et, frame, "NONE", "SATURN")
    position, velocity = states[:, :3], states[:, 3:]
    df = pd.DataFrame({"utc": utc, "et": et})
    if frame == "CASSINI_KRTP":
        # CASSINI_KRTP is a Cartesian basis whose X axis is radial.
        df[["R_km", "Theta_km", "Phi_km"]] = position
        df[["vR_km_s", "vTheta_km_s", "vPhi_km_s"]] = velocity
        df["speed_km_s"] = np.linalg.norm(velocity, axis=1)
        return df

    radius_km = spice.bodvrd("SATURN", "RADII", 3)[1][0]
    xyz_rs = position / radius_km
    if frame == "J2000":
        df[["x_km", "y_km", "z_km"]] = position
        df[["vx_km_s", "vy_km_s", "vz_km_s"]] = velocity
        df["r_km"] = np.linalg.norm(position, axis=1)
        df["speed_km_s"] = np.linalg.norm(velocity, axis=1)
        df[["x_Rs", "y_Rs", "z_Rs"]] = xyz_rs
        df["r_Rs"] = df["r_km"] / radius_km
        return df

    x_rs, y_rs, z_rs = xyz_rs.T
    radius_rs = np.linalg.norm(xyz_rs, axis=1)
    latitude = np.arcsin(z_rs / radius_rs)
    df[["x_ksm_km", "y_ksm_km", "z_ksm_km"]] = position
    df[["x_ksm_Rs", "y_ksm_Rs", "z_ksm_Rs"]] = xyz_rs
    df["r_Rs"] = radius_rs
    df["L_approx"] = radius_rs / np.cos(latitude)**2
    df["MLT_hr"] = (12 + np.degrees(np.arctan2(y_rs, x_rs)) / 15) % 24
    df["MLAT_approx_deg"] = np.degrees(latitude)
    df[["vx_ksm_km_s", "vy_ksm_km_s", "vz_ksm_km_s"]] = velocity
    df["speed_km_s"] = np.linalg.norm(velocity, axis=1)
    return df


def _unit_rows(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _enis_table(utc, et):
    """Use the original Enceladus-centered, right-handed ENIS convention."""
    saturn, _ = spice.spkpos("SATURN", et, "J2000", "NONE", "ENCELADUS")
    sun, _ = spice.spkpos("SUN", et, "J2000", "NONE", "ENCELADUS")
    cassini, _ = spice.spkezr("CASSINI", et, "J2000", "NONE", "ENCELADUS")
    north = np.array([spice.pxform("IAU_SATURN", "J2000", epoch)[:, 2] for epoch in et])
    y_hat = _unit_rows(saturn)
    x_hat = _unit_rows(np.cross(north, -saturn))
    z_hat = _unit_rows(np.cross(x_hat, y_hat))
    y_hat = _unit_rows(np.cross(z_hat, x_hat))
    axes = np.stack((x_hat, y_hat, z_hat), axis=1)

    position = (axes @ cassini[:, :3, None])[:, :, 0]
    # Resolve inertial relative velocity onto the instantaneous ENIS axes.
    velocity = (axes @ cassini[:, 3:, None])[:, :, 0]
    distance = np.linalg.norm(position, axis=1)
    radius_km = spice.bodvrd("ENCELADUS", "RADII", 3)[1].mean()
    position_re = position / radius_km
    df = pd.DataFrame({"utc": utc, "et": et})
    df[["x_enis_km", "y_enis_km", "z_enis_km"]] = position
    df[["x_enis_RE", "y_enis_RE", "z_enis_RE"]] = position_re
    df["distance_km"] = distance
    df["distance_RE"] = distance / radius_km
    df["altitude_km"] = distance - radius_km
    df["local_time_hr"] = (np.degrees(np.arctan2(position_re[:, 0], position_re[:, 1])) / 15) % 24
    df[["vx_enis_km_s", "vy_enis_km_s", "vz_enis_km_s"]] = velocity
    df["speed_km_s"] = np.linalg.norm(velocity, axis=1)
    for name, xyz in (("saturn", saturn), ("sun", sun)):
        projected = (axes @ xyz[:, :, None])[:, :, 0]
        df[[f"{name}_{axis}_enis_km" for axis in "xyz"]] = projected
        df[[f"{name}_{axis}_enis_RE" for axis in "xyz"]] = projected / radius_km
    return df


def cassini_locations_at(times, *, frame="KSM"):
    """Query exact UTC timestamps in annual batches, preserving input order and duplicates.

    Naive timestamps are UTC. Supports J2000, KSM, KRTP, and ENIS. Positions
    and derived columns use the original pickle schemas. Temporary text kernel
    loads can reset values set directly in the SPICE pool.
    """
    frame = frame.upper()
    frame = {"KSM": "CASSINI_KSM", "KRTP": "CASSINI_KRTP"}.get(frame, frame)
    if frame not in LOCATION_FILES:
        raise ValueError("frame must be 'J2000', 'KSM', 'KRTP' or 'ENIS'")
    epochs = pd.DatetimeIndex(pd.to_datetime(list(times), utc=True, format="mixed"))
    if not len(epochs) or epochs.hasnans:
        raise ValueError("Provide at least one valid UTC timestamp")

    frames = []
    manifests = []
    for year in epochs.year.unique():
        indices = np.flatnonzero(epochs.year == year)
        utc = epochs[indices]
        manifest = latest_tm_for_year(year)
        manifests.append(manifest.name)
        with _load_position_kernels(manifest):
            et = _ephemeris_times(utc)
            try:
                table = _enis_table(utc, et) if frame == "ENIS" else _state_table(utc, et, frame)
            except SpiceSPKINSUFFDATA as error:
                raise ValueError(
                    f"Incomplete local SPICE coverage between {utc.min()} and {utc.max()}"
                ) from error
        table.index = indices
        frames.append(table)

    result = pd.concat(frames).sort_index(ignore_index=True)
    result.attrs.update(
        target="CASSINI", observer="ENCELADUS" if frame == "ENIS" else "SATURN",
        frame="custom_ENIS_official_definition" if frame == "ENIS" else frame,
        abcorr="NONE", meta_kernels=tuple(manifests),
    )
    if frame == "CASSINI_KSM":
        result.attrs["MLT_convention"] = "+X noon, +Y dusk, -X midnight, -Y dawn"
        result.attrs["L_note"] = "L_approx = r_Rs / cos(MLAT)^2, using KSM z for approximate latitude."
    elif frame == "ENIS":
        result.attrs["axis_definition"] = "+X corotation, +Y Saturn-facing, +Z northward"
        result.attrs["local_time_convention"] = "+Y 0 hr, +X 6 hr, -Y 12 hr, -X 18 hr"
        result.attrs["velocity_definition"] = "Inertial relative velocity resolved into instantaneous ENIS axes."
    return result


def generate_cassini_locations(start_time, stop_time, *, frame="KSM", step_seconds=60):
    """Make a location table on a regular UTC grid with an exclusive stop."""
    times = _time_grid(start_time, stop_time, step_seconds, include_end=False)
    result = cassini_locations_at(times, frame=frame)
    result.attrs.update(
        step_seconds=step_seconds, start_utc=times[0].isoformat(),
        stop_utc=pd.to_datetime(stop_time, utc=True).isoformat(),
    )
    return result


def save_cassini_locations(
    out_file, start_time, stop_time, *, frame="KSM", step_seconds=60, overwrite=False,
):
    """Generate and save a location pickle. Replacing a file requires overwrite=True."""
    out_file = Path(out_file)
    if out_file.exists() and not overwrite:
        raise FileExistsError(out_file)
    data = generate_cassini_locations(start_time, stop_time, frame=frame, step_seconds=step_seconds)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_file.parent) as temp_dir:
        temp_path = Path(temp_dir) / out_file.name
        data.to_pickle(temp_path, protocol=5)
        if overwrite:
            temp_path.replace(out_file)
        else:
            out_file.hardlink_to(temp_path)
    return out_file
