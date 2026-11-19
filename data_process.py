import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET


DATA_DIR = Path(__file__).resolve().parent / "data"
SEC_KRTP_INDEX = DATA_DIR / "interim" / "sec-krtp-file_time_index.csv"
FULL_KRTP_INDEX = DATA_DIR / "interim" / "full_krtp_file_time_index.csv"
SEC_KRTP_DIR = DATA_DIR / "raw" / "PDS" / "_cassini-mag-cal-sec-krtp"
FULL_KRTP_DIR = DATA_DIR / "raw" / "PDS" / "_cassini-mag-cal_full_krtp"
LOCATION_DIR = DATA_DIR / "interim" / "spice-location-ready"
WAVE_LABEL_DIR = DATA_DIR / "processed" / "check_wave"
FULL_KRTP_COLUMNS = ["TIME", "BR", "BTHETA", "BPHI"]
SEC_KRTP_COLUMNS = FULL_KRTP_COLUMNS + ["BTOTAL", "NPTS"]
FAC_COLUMNS = [
    "B0_R", "B0_THETA", "B0_PHI", "B0_mag",
    "dB_R", "dB_THETA", "dB_PHI", "dB_mag",
    "b_hat_R", "b_hat_THETA", "b_hat_PHI",
    "dB_para", "dB_perp_mag", "dB_perp_1", "dB_perp_2",
    "fac_reference_axis",
]
_FAC_REFERENCE_DTYPE = pd.CategoricalDtype(["R", "THETA"])
_FAC_RADIAL_ALIGNMENT_LIMIT = 0.999  # About 2.56 degrees from either radial direction.
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_ATOMIC_MASS_UNIT_KG = 1.66053906660e-27


# Shared helpers

def _parse_time_range(start_time, end_time):
    """Return ordered, timezone-naive UTC bounds, or None for invalid input."""
    bounds = []
    for value in (start_time, end_time):
        if not isinstance(value, (str, date, np.datetime64)):
            return None
        try:
            bound = pd.Timestamp(value)
            if pd.isna(bound):
                return None
            if bound.tzinfo is not None:
                bound = bound.tz_convert(None)
            bounds.append(bound.as_unit("ns"))
        except (TypeError, ValueError, OverflowError):
            return None
    start, end = bounds
    return (start, end) if start <= end else None


def _empty_frame(columns):
    return pd.DataFrame(columns=columns, dtype="float64").astype({
        "TIME": "datetime64[ns]",
        "NPTS": "int64",
    })


# File time indexes

def _read_sec_krtp_bounds(tab_path):
    # The one-second XML labels report .005Z for sample times ending in .5.
    with tab_path.open("rb") as table:
        first_record = table.readline().split(maxsplit=1)
        # Records are 70 bytes, so the tail includes the complete last record.
        table.seek(max(0, tab_path.stat().st_size - 1024))
        tail = table.read().rstrip()

    if not first_record or not tail:
        raise ValueError("Missing first or last record")

    last_record = tail.splitlines()[-1].split(maxsplit=1)
    return first_record[0].decode("ascii"), last_record[0].decode("ascii")


def _read_full_krtp_bounds(tab_path):
    label = ET.parse(tab_path.with_suffix(".xml")).getroot()
    file_name = label.findtext("./{*}File_Area_Observational/{*}File/{*}file_name")
    if file_name != tab_path.name:
        raise ValueError(f"XML label names {file_name!r}, expected {tab_path.name!r}")

    times = label.find("./{*}Observation_Area/{*}Time_Coordinates")
    if times is None:
        raise ValueError("XML label is missing Time_Coordinates")
    return times.findtext("{*}start_date_time"), times.findtext("{*}stop_date_time")


def build_krtp_index(resolution):
    """Write the UTC file time index CSV for "sec" or "full" and return its rows."""
    if resolution == "sec":
        source_dir, pattern = SEC_KRTP_DIR, "*_FGM_KRTP_1S.TAB"
        read_bounds, index_path = _read_sec_krtp_bounds, SEC_KRTP_INDEX
    elif resolution == "full":
        source_dir, pattern = FULL_KRTP_DIR, "*_FGM_KRTP.TAB"
        read_bounds, index_path = _read_full_krtp_bounds, FULL_KRTP_INDEX
    else:
        raise ValueError("resolution must be 'sec' or 'full'")

    tab_files = sorted(path for path in source_dir.glob(pattern) if path.is_file())
    if not tab_files:
        raise FileNotFoundError(f"No KRTP tables found in {source_dir}")

    records = []
    for tab_path in tab_files:
        try:
            start, end = pd.to_datetime(read_bounds(tab_path), format="ISO8601", utc=True)
            if pd.isna(start) or pd.isna(end) or start > end:
                raise ValueError("Missing or reversed time bounds")
        except (OSError, ET.ParseError, ValueError) as error:
            raise ValueError(f"Cannot index {tab_path}: {error}") from error
        records.append((tab_path.name, start, end))

    index_df = pd.DataFrame(records, columns=["file_name", "start_time", "end_time"])
    index_df = index_df.sort_values(["start_time", "file_name"], ignore_index=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=index_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / index_path.name
        index_df.to_csv(temp_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
        temp_path.replace(index_path)
    return index_df


def _read_krtp_index(index_path):
    """Reads a KRTP index CSV into a DataFrame with start_time and end_time parsed as naive UTC datetimes."""
    index_df = pd.read_csv(index_path)
    time_columns = ["start_time", "end_time"]
    # 
    index_df[time_columns] = index_df[time_columns].apply(
        lambda column: pd.to_datetime(column, format="ISO8601", utc=True).dt.tz_convert(None)
    )
    return index_df


def read_sec_krtp_index():
    """Load the one-second CSV index with timezone-naive UTC timestamps."""
    return _read_krtp_index(SEC_KRTP_INDEX)


def read_full_krtp_index():
    """Load the full-resolution CSV index with timezone-naive UTC timestamps."""
    return _read_krtp_index(FULL_KRTP_INDEX)


def _overlapping_krtp_file_names(index_df, start_time, end_time):
    overlap = (index_df["end_time"] >= start_time) & (index_df["start_time"] <= end_time)
    return index_df.loc[overlap, "file_name"]


# Raw KRTP samples

def read_sec_krtp(file_name):
    """Read a Cassini 1-second table using its filename from the index."""
    return pd.read_csv(
        SEC_KRTP_DIR / Path(file_name).name,
        sep=r"\s+",
        names=SEC_KRTP_COLUMNS,
    )


def read_full_krtp(file_name):
    """Read the timestamp and three field components from a native-resolution table."""
    return pd.read_csv(
        FULL_KRTP_DIR / Path(file_name).name,
        sep=r"\s+",
        names=FULL_KRTP_COLUMNS,
    )


def _select_full_krtp(start_time, end_time):
    file_names = _overlapping_krtp_file_names(read_full_krtp_index(), start_time, end_time)
    frames = []
    for file_name in file_names:
        try:
            frame = read_full_krtp(file_name)
        except FileNotFoundError:
            # An indexed daily table may be absent from a partial local archive.
            continue
        frame["TIME"] = pd.to_datetime(frame["TIME"])
        selected = frame.loc[frame["TIME"].between(start_time, end_time)]
        if not selected.empty:
            frames.append(selected)

    if not frames:
        return _empty_frame(SEC_KRTP_COLUMNS)
    df = pd.concat(frames, ignore_index=True).sort_values("TIME", ignore_index=True)
    df["BTOTAL"] = np.linalg.norm(df[["BR", "BTHETA", "BPHI"]], axis=1)
    df["NPTS"] = 1
    return df


def _select_sec_krtp(start_time, end_time):
    file_names = _overlapping_krtp_file_names(read_sec_krtp_index(), start_time, end_time)
    if file_names.empty:
        return _empty_frame(SEC_KRTP_COLUMNS)

    df = pd.concat((read_sec_krtp(file_name) for file_name in file_names), ignore_index=True)
    df["TIME"] = pd.to_datetime(df["TIME"])
    mask = df["TIME"].between(start_time, end_time)
    return df.loc[mask].reset_index(drop=True)


# Automatic sampling selection

def _sampling_info(df, start_time, end_time, nyquist_multiplier):
    field = df["BTOTAL"]
    valid = np.isfinite(field) & field.gt(0)
    field_nT = h2o_plus_hz = required_sampling_hz = None
    sampling_hz = max_spacing_s = meets_requirement = None

    if valid.any():
        field_nT = float(field[valid].max())
        h2o_plus_hz = (
            _ELEMENTARY_CHARGE_C * field_nT * 1e-9 / (2 * np.pi * 18 * _ATOMIC_MASS_UNIT_KG)
        )
        nyquist_rate_hz = 2 * h2o_plus_hz
        required_sampling_hz = nyquist_multiplier * nyquist_rate_hz

        times = df.loc[valid, "TIME"].drop_duplicates().sort_values()
        if len(times) >= 2:
            steps = times.diff().iloc[1:]
            second = pd.Timedelta(seconds=1)
            sampling_hz = float(second / steps.median())
            # Include interior gaps and uncovered edges when checking the margin.
            max_spacing = max(steps.max(), times.iloc[0] - start_time, end_time - times.iloc[-1])
            max_spacing_s = float(max_spacing / second)
            conservative_rate_hz = 1 / max_spacing_s
            meets_requirement = bool(
                conservative_rate_hz >= required_sampling_hz
                # Reject the exact Nyquist boundary despite floating-point rounding.
                and conservative_rate_hz > nyquist_rate_hz * (1 + 8 * np.finfo(float).eps)
            )

    return {
        "field_nT": field_nT,
        "h2o_plus_hz": h2o_plus_hz,
        "nyquist_multiplier": nyquist_multiplier,
        "required_sampling_hz": required_sampling_hz,
        "sampling_hz": sampling_hz,
        "nyquist_hz": None if sampling_hz is None else sampling_hz / 2,
        "max_spacing_s": max_spacing_s,
        "meets_requirement": meets_requirement,
    }


def _select_auto_krtp(start_time, end_time, nyquist_multiplier):
    try:
        nyquist_multiplier = float(nyquist_multiplier)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("nyquist_multiplier must be finite and at least 1") from error
    if not np.isfinite(nyquist_multiplier) or nyquist_multiplier < 1:
        raise ValueError("nyquist_multiplier must be finite and at least 1")

    resolution = "sec"
    df = _select_sec_krtp(start_time, end_time)
    info = _sampling_info(df, start_time, end_time, nyquist_multiplier)
    if not info["meets_requirement"]:
        resolution = "full"
        df = _select_full_krtp(start_time, end_time)
        info = _sampling_info(df, start_time, end_time, nyquist_multiplier)

    df.attrs.update(info, resolution=resolution)
    return df


def select_df(start_time, end_time, resolution="sec", *, nyquist_multiplier=2):
    """Return raw Cassini samples within the inclusive time range.

    Invalid bounds and ranges with no samples return an empty raw DataFrame.
    Naive timestamps are interpreted as UTC. Use "sec" for one-second averages
    or "full" for the native cadence. For full data, BTOTAL is the vector
    magnitude and NPTS is 1.

    "auto" tries one-second data first and uses native data if it needs more
    coverage or a higher rate. It requires at least nyquist_multiplier times
    the Nyquist rate (2 * cyclotron frequency), using the maximum positive
    finite BTOTAL and singly charged H2O+ (18 u).
    The multiplier defaults to 2, or four samples per cycle of H2O+.

    Auto results carry the chosen resolution and sampling diagnostics in attrs.
    The median timestamp step estimates sampling_hz and nyquist_hz. The
    largest spacing, including gaps at the requested edges, determines
    meets_requirement. None means there were too few usable samples to assess.
    """
    if resolution not in ("sec", "full", "auto"):
        raise ValueError("resolution must be 'sec', 'full' or 'auto'")
    bounds = _parse_time_range(start_time, end_time)
    if bounds is None:
        return _empty_frame(SEC_KRTP_COLUMNS)
    if resolution == "full":
        return _select_full_krtp(*bounds)
    if resolution == "sec":
        return _select_sec_krtp(*bounds)
    return _select_auto_krtp(*bounds, nyquist_multiplier)


def krtp_sampling_info(time, *, nyquist_multiplier=2):
    """Return auto-resolution diagnostics over the one second on either side of time."""
    bounds = _parse_time_range(time, time)
    if bounds is None:
        raise ValueError("time must be a valid UTC timestamp")
    center = bounds[0]
    second = pd.Timedelta(seconds=1)
    return select_df(
        center - second, center + second, resolution="auto",
        nyquist_multiplier=nyquist_multiplier,
    ).attrs


# Field-aligned coordinate (FAC) decomposition

def fac_decompose_dB(df, start_time, end_time, resolution="sec"):
    """Add FAC columns and trim 10 minutes from each requested endpoint.

    The background mean uses 1,200 samples for "sec" and a centered
    1,200-second window over native samples for "full".

    The per-sample fac_reference_axis is THETA when abs(b_hat_R) exceeds
    _FAC_RADIAL_ALIGNMENT_LIMIT, R otherwise, and missing when the background
    field has no valid direction.
    Column suffix "_mag" denotes a nonnegative scalar magnitude. dB_para,
    dB_perp_1 and dB_perp_2 are signed scalar projections onto the FAC axes.
    Suffixes "_R", "_THETA" and "_PHI" identify scalar KRTP components.
    Local arrays ending in "_vec" hold three-component vectors, while "_hat"
    identifies unit vectors.
    "auto" uses the resolution recorded by select_df in the input attrs.
    """
    if resolution == "auto":
        resolution = df.attrs.get("resolution")
        if resolution not in ("sec", "full") and not df.empty:
            raise ValueError("FAC auto mode requires samples from select_df(..., resolution='auto')")
    elif resolution not in ("sec", "full"):
        raise ValueError("resolution must be 'sec', 'full' or 'auto'")
    bounds = _parse_time_range(start_time, end_time)
    if df.empty or bounds is None:
        result = _empty_frame(SEC_KRTP_COLUMNS + FAC_COLUMNS).astype(
            {"fac_reference_axis": _FAC_REFERENCE_DTYPE}
        )
        result.attrs.update(df.attrs)
        return result
    start_time, end_time = bounds

    components = df[["BR", "BTHETA", "BPHI"]]
    B_vec = components.to_numpy()
    if resolution == "full":
        components = components.set_axis(pd.DatetimeIndex(df["TIME"]))
        background = components.rolling("1200s", center=True).mean()
        missing = components.isna().rolling("1200s", center=True).sum()
        background = background.mask(missing > 0)
    else:
        background = components.rolling(window=1200, center=True).mean()
    B0_vec = background.to_numpy()
    df[['B0_R','B0_THETA','B0_PHI']] = B0_vec

    B0_mag = np.linalg.norm(B0_vec, axis=1)
    df['B0_mag'] = B0_mag

    dB_vec = B_vec - B0_vec

    df[['dB_R','dB_THETA','dB_PHI']] = dB_vec
    df['dB_mag'] = np.linalg.norm(dB_vec, axis=1)

    has_direction = np.isfinite(B0_mag) & (B0_mag > 0)
    b_hat = np.divide(
        B0_vec, B0_mag[:, None], out=np.full_like(B0_vec, np.nan),
        where=has_direction[:, None],
    )
    df[['b_hat_R','b_hat_THETA','b_hat_PHI']]=b_hat

    dB_para = np.sum(dB_vec * b_hat, axis=1)
    df['dB_para'] = dB_para

    dB_perp_vec = dB_vec - dB_para[:, None] * b_hat
    df['dB_perp_mag'] = np.linalg.norm(dB_perp_vec, axis=1)

    use_theta = np.abs(b_hat[:, 0]) > _FAC_RADIAL_ALIGNMENT_LIMIT
    r_hat = np.array([1.0, 0.0, 0.0])
    theta_hat = np.array([0.0, 1.0, 0.0])
    reference_hat = np.where(use_theta[:, None], theta_hat, r_hat)

    # Gram–Schmidt: v = reference - (reference · b) b, so v · b = 0.
    perp_basis_vec = reference_hat - np.sum(reference_hat * b_hat, axis=1, keepdims=True) * b_hat
    e_perp_1_hat = perp_basis_vec / np.linalg.norm(perp_basis_vec, axis=1, keepdims=True)
    e_perp_2_hat = np.cross(b_hat, e_perp_1_hat)

    dB_perp_1 = np.sum(dB_vec * e_perp_1_hat, axis=1)
    dB_perp_2 = np.sum(dB_vec * e_perp_2_hat, axis=1)
    df['dB_perp_1'] = dB_perp_1
    df['dB_perp_2'] = dB_perp_2
    reference_codes = use_theta.astype(np.int8)
    reference_codes[~has_direction] = -1  # Missing categorical value.
    df["fac_reference_axis"] = pd.Categorical.from_codes(
        reference_codes, dtype=_FAC_REFERENCE_DTYPE,
    )

    # delete the first and last 10 minutes to avoid nan values
    df=df[(df['TIME']>=start_time+pd.Timedelta(minutes=10)) & (df['TIME']<=end_time-pd.Timedelta(minutes=10))].reset_index(drop=True)
    return df

def read_and_process_data(start_time, end_time, resolution="sec", *, nyquist_multiplier=2):
    """Read Cassini data with the selected resolution and return the field-aligned result."""
    df = select_df(
        start_time, end_time, resolution=resolution, nyquist_multiplier=nyquist_multiplier,
    )
    return fac_decompose_dB(df, start_time, end_time, resolution=resolution)
