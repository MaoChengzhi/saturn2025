"""Read local Juno MAG (FGM) files and return field-aligned (FAC) data for the shared wave code.

Sources under data/raw/PDS/juno/ (downloaded by jupiter/juno_download.py):

    resolution="sec"    mag_pc_1s/      1-second daily files
    resolution="full"   mag_pc_native/  native daily files (8-64 samples/s)
    resolution="flyby"  mag_pc_flyby/   native moon-flyby files (about 1 h each)
    resolution="auto"   1 s when it samples the chosen ion's gyrofrequency well
                        enough, otherwise native, otherwise a flyby file

Daily files give B and Juno's position in the planetocentric (PC) frame:
Jupiter-centred, rotating with Jupiter (IAU_JUPITER). Flyby files give both
in the moon's body-fixed frame (IAU_IO, IAU_EUROPA, IAU_GANYMEDE; checked
against SPICE to about 2 km). Flyby B is rotated into IAU_JUPITER with SPICE,
and Juno's Jupiter-centred position comes from SPICE.

Each sample is then expressed in Jupiter-centred spherical components
(BR, BTHETA, BPHI: radial, southward, eastward), the same columns as the
Cassini KRTP data, so data_process.fac_decompose_dB applies unchanged.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from data_process import FAC_COLUMNS, _parse_time_range, fac_decompose_dB

JUPITER_RADIUS_KM = 71_492.0
JUNO_PDS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "PDS" / "juno"
SOURCE_DIRS = {
    "sec": JUNO_PDS_DIR / "mag_pc_1s",
    "full": JUNO_PDS_DIR / "mag_pc_native",
    "flyby": JUNO_PDS_DIR / "mag_pc_flyby",
}
MOON_FRAMES = {"io": ("IO", "IAU_IO"), "eu": ("EUROPA", "IAU_EUROPA"),
               "ga": ("GANYMEDE", "IAU_GANYMEDE")}
RAW_COLUMNS = ["YEAR", "DOY", "HOUR", "MINUTE", "SECOND", "MSEC", "DDAY",
               "BX", "BY", "BZ", "RANGE", "X_km", "Y_km", "Z_km"]
JUNO_COLUMNS = ["TIME", "BR", "BTHETA", "BPHI", "BTOTAL", "NPTS", "RANGE",
                "r_RJ", "lat_spin_deg", "lon_east_deg"]
_NAME = re.compile(r"fgm_jno_l3_(?P<day>\d{7})pc(?:_(?P<moon>io|eu|ga))?"
                   r"(?:_(?P<cad>r1s))?_v(?P<ver>\d\d)\.sts")
_DATA_LINE = re.compile(rb"^\s+\d{4}\s+\d{1,3}\s+\d{1,2}\s")
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def _cyclotron_hz(field_nT, mass_amu):
    return E_CHARGE * field_nT * 1e-9 / (2 * np.pi * mass_amu * AMU)


def _label_time(path, key):
    """START_TIME or STOP_TIME from a .lbl file as a naive UTC Timestamp."""
    text = path.with_suffix(".lbl").read_text(errors="replace")
    value = re.search(rf"^{key}\s*=\s*(\S+)", text, re.M)[1]
    return pd.to_datetime(value, format="%Y-%jT%H:%M:%S.%f")


def list_local_files(resolution):
    """Local files for one source: DataFrame of path, start, stop, moon (highest version per product)."""
    folder = SOURCE_DIRS[resolution]
    best = {}
    for path in folder.glob("fgm_jno_l3_*.sts"):
        match = _NAME.fullmatch(path.name)
        if not match or (resolution == "sec") != bool(match["cad"]):
            continue
        key = (match["day"], match["moon"])
        if key not in best or int(match["ver"]) > int(best[key][1]["ver"]):
            best[key] = (path, match)
    rows = []
    for (day, moon), (path, _) in best.items():
        if moon:
            if not path.with_suffix(".lbl").is_file():
                continue
            start, stop = _label_time(path, "START_TIME"), _label_time(path, "STOP_TIME")
        else:
            start = pd.to_datetime(day, format="%Y%j")
            stop = start + pd.Timedelta(days=1)
        rows.append({"path": path, "start": start, "stop": stop, "moon": moon})
    columns = ["path", "start", "stop", "moon"]
    return pd.DataFrame(rows, columns=columns).sort_values("start", ignore_index=True)


def read_sts(path):
    """Read one .sts file: UTC TIME, B and position columns as stored (frame depends on file)."""
    path = Path(path)
    with open(path, "rb") as handle:
        offset = 0
        for line in handle:
            if _DATA_LINE.match(line):
                break
            offset += len(line)
        handle.seek(offset)
        raw = pd.read_csv(handle, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="c")
    year_start = pd.to_datetime(raw["YEAR"].astype(str), format="%Y")
    raw.insert(0, "TIME", year_start + pd.to_timedelta(
        (raw["DOY"] - 1) * 86_400 + raw["HOUR"] * 3600 + raw["MINUTE"] * 60 + raw["SECOND"]
        + raw["MSEC"] / 1000, unit="s"))
    return raw.drop(columns=["YEAR", "DOY", "HOUR", "MINUTE", "SECOND", "MSEC", "DDAY"])


def regularize_times(times, max_shift_s=1e-3):
    """Put each continuous run of samples on an exact uniform grid when rounding explains the jitter.

    Juno times are printed to 1 ms, so 64 Hz samples (15.625 ms apart) appear
    as 15 and 16 ms steps, which the shared FFT checks reject as irregular.
    A run ends at a gap or a cadence change (a step more than 30% from the
    local median step). Within a run, times are replaced by the least-squares
    grid t0 + k * dt only if no time moves by more than max_shift_s; otherwise
    the run keeps its original times. Field values are not changed.
    Returns the new times and one record per run.
    """
    times = pd.DatetimeIndex(times)
    n = len(times)
    if n < 3:
        return times, []
    t = (times - times[0]).total_seconds().to_numpy()
    steps = np.diff(t)
    local = pd.Series(steps).rolling(9, center=True, min_periods=1).median().to_numpy()
    breaks = np.flatnonzero(np.abs(steps - local) > 0.3 * local) + 1
    new_t = t.copy()
    runs = []
    for left, right in zip(np.r_[0, breaks], np.r_[breaks, n]):
        count = right - left
        record = {"start": times[left], "stop": times[right - 1], "samples": int(count),
                  "spacing_s": float(np.median(steps[left:right - 1])) if count > 1 else np.nan,
                  "regularized": False, "max_shift_s": 0.0}
        if count >= 3:
            k = np.arange(count)
            dt, t0 = np.polyfit(k, t[left:right], 1)
            fitted = t0 + dt * k
            shift = float(np.max(np.abs(fitted - t[left:right])))
            if shift <= max_shift_s:
                new_t[left:right] = fitted
                record.update(spacing_s=float(dt), regularized=True, max_shift_s=shift)
        runs.append(record)
    return times[0] + pd.to_timedelta(new_t, unit="s"), runs


def _moon_to_jupiter(raw, moon):
    """Rotate flyby B from the moon's body-fixed frame into IAU_JUPITER; add SPICE positions."""
    import spiceypy as spice
    from jupiter import juno_spice

    body, frame = MOON_FRAMES[moon]
    times = pd.DatetimeIndex(raw["TIME"]).tz_localize("UTC")
    step = pd.Timedelta(seconds=10)
    grid = pd.date_range(times.min().floor(step) - step, times.max().ceil(step) + step, freq=step)
    et = juno_spice._ephemeris_times(grid)
    matrices = np.array([spice.pxform(frame, "IAU_JUPITER", value) for value in et])
    offsets = (times - grid[0]).total_seconds()
    grid_offsets = (grid - grid[0]).total_seconds()
    rotation = np.stack([np.interp(offsets, grid_offsets, matrices[:, i, j])
                         for i in range(3) for j in range(3)], axis=1).reshape(-1, 3, 3)
    B_moon = raw[["BX", "BY", "BZ"]].to_numpy(dtype=float)
    B_jup = np.einsum("nij,nj->ni", rotation, B_moon)
    result = raw.rename(columns={"X_km": "x_moon_km", "Y_km": "y_moon_km", "Z_km": "z_moon_km"})
    result[["BX", "BY", "BZ"]] = B_jup
    result[["X_km", "Y_km", "Z_km"]] = juno_spice.juno_position_at(times, "IAU_JUPITER")
    result["moon"] = body
    result["r_moon_km"] = np.linalg.norm(raw[["X_km", "Y_km", "Z_km"]].to_numpy(dtype=float), axis=1)
    return result


def _to_spherical(raw):
    """Add Jupiter-centred spherical B components and position from Cartesian PC values."""
    xyz = raw[["X_km", "Y_km", "Z_km"]].to_numpy(dtype=float)
    B = raw[["BX", "BY", "BZ"]].to_numpy(dtype=float)
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:, 2] / r, -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    out = raw.copy()
    out["BR"] = B[:, 0] * st * cp + B[:, 1] * st * sp + B[:, 2] * ct
    out["BTHETA"] = B[:, 0] * ct * cp + B[:, 1] * ct * sp - B[:, 2] * st
    out["BPHI"] = -B[:, 0] * sp + B[:, 1] * cp
    out["BTOTAL"] = np.linalg.norm(B, axis=1)
    out["NPTS"] = 1
    out["r_RJ"] = r / JUPITER_RADIUS_KM
    out["lat_spin_deg"] = 90 - np.degrees(theta)
    out["lon_east_deg"] = np.degrees(phi) % 360
    return out


def select_df(start_time, end_time, resolution="sec"):
    """Juno samples in [start, end] with KRTP-like spherical columns (no FAC yet)."""
    if resolution not in SOURCE_DIRS:
        raise ValueError(f"resolution must be one of {list(SOURCE_DIRS)} or 'auto'")
    bounds = _parse_time_range(start_time, end_time)
    empty = pd.DataFrame(columns=JUNO_COLUMNS)
    if bounds is None:
        return empty
    start, end = bounds
    files = list_local_files(resolution)
    files = files.loc[files["start"].le(end) & files["stop"].ge(start)]
    if resolution == "flyby" and len(files) > 1:
        files = files.iloc[:1]
    frames = []
    for row in files.itertuples(index=False):
        raw = read_sts(row.path)
        raw = raw.loc[raw["TIME"].between(start, end)]
        if raw.empty:
            continue
        if row.moon:
            raw = _moon_to_jupiter(raw, row.moon)
        frames.append(_to_spherical(raw))
    if not frames:
        empty.attrs.update(resolution=resolution, files=[])
        return empty
    df = (pd.concat(frames, ignore_index=True).drop_duplicates("TIME")
          .sort_values("TIME", ignore_index=True))
    keep = [c for c in JUNO_COLUMNS + ["x_moon_km", "y_moon_km", "z_moon_km", "moon", "r_moon_km"]
            if c in df]
    df = df[keep].copy()
    df["TIME"], runs = regularize_times(df["TIME"])
    longest = max(runs, key=lambda run: run["samples"]) if runs else None
    # attrs hold plain records: pandas compares attrs on concat and cannot compare DataFrames.
    df.attrs.update(resolution=resolution, files=[p.name for p in files["path"]],
                    sampling_hz=float(1 / longest["spacing_s"]) if longest else np.nan,
                    cadence_segments=runs)
    return df


def _meets_sampling(df, start, end, ion_mass_amu, nyquist_multiplier):
    """Same rule as the Cassini auto mode, for the chosen ion instead of H2O+."""
    valid = df["BTOTAL"].gt(0) if not df.empty else pd.Series(dtype=bool)
    if not valid.any():
        return False
    required_hz = nyquist_multiplier * 2 * _cyclotron_hz(df.loc[valid, "BTOTAL"].max(), ion_mass_amu)
    times = df.loc[valid, "TIME"].sort_values()
    spacing = max(times.diff().max(), times.iloc[0] - start, end - times.iloc[-1])
    return 1 / max(spacing.total_seconds(), 1e-9) >= required_hz


def read_and_process_data(start_time, end_time, resolution="auto", *, ion_mass_amu=32.0,
                          nyquist_multiplier=2, background_s=1200, trim_s=600):
    """Read Juno MAG and return the shared FAC result (same columns as the Cassini pipeline).

    "auto" uses 1 s data when it gives at least nyquist_multiplier x 2 x the
    gyrofrequency of ion_mass_amu (default S+, 32) everywhere in the interval,
    including edge gaps, then native daily files, then a flyby file.
    background_s is the centred FAC background window in seconds and trim_s
    the time removed from each end (the read should be padded by trim_s).
    """
    bounds = _parse_time_range(start_time, end_time)
    if bounds is None:
        raise ValueError("Provide valid UTC timestamps")
    start, end = bounds
    if resolution == "auto":
        df = select_df(start, end, "sec")
        chosen = "sec"
        if not _meets_sampling(df, start, end, ion_mass_amu, nyquist_multiplier):
            for candidate in ("full", "flyby"):
                trial = select_df(start, end, candidate)
                if not trial.empty:
                    df, chosen = trial, candidate
                    break
            else:
                warnings.warn("No native or flyby file covers this interval; using 1 s data, "
                              "which may not resolve the chosen ion's band.", stacklevel=2)
        df.attrs["resolution"] = chosen
    else:
        df = select_df(start, end, resolution)
    attrs = dict(df.attrs)
    if df.empty:
        result = pd.DataFrame(columns=JUNO_COLUMNS + FAC_COLUMNS)
        result.attrs.update(attrs)
        return result
    # Juno 1 s files have gaps, so every source uses the time-based background window.
    result = fac_decompose_dB(df.copy(), start, end, resolution="full",
                              background_s=background_s, trim_s=trim_s)
    result.attrs.update(attrs, background_s=background_s, trim_s=trim_s)
    segments = attrs.get("cadence_segments") or []
    if len(segments) > 1:
        rates = ", ".join(sorted({f"{1 / s['spacing_s']:.3g} Hz" for s in segments}))
        print(f"Note · {len(segments)} cadence segments or gaps in the read ({rates}); "
              "the classifier needs one uniform cadence in the event window.")
    return result
