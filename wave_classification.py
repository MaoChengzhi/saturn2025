"""MAG-only ICW and mirror-like candidate screening for Saturn occurrence pilots."""

import numpy as np
import pandas as pd
from scipy import fft, signal


CLASSIFIER_VERSION = "mag-pilot-v1"
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
_WAVE_COLUMNS = ("TIME", "BTOTAL", "dB_para", "dB_perp_1", "dB_perp_2")


def _cyclotron_freq_hz(field_nT, mass_amu):
    return E_CHARGE * field_nT * 1e-9 / (2 * np.pi * mass_amu * AMU)


def _prepare_wave_window(df, min_points, min_duration_s):
    """Validate an FFT window and return time-ordered values, timing, and any failure."""
    if df is None or not set(_WAVE_COLUMNS).issubset(df.columns):
        return None, {}, ("bad_data", "missing_required_columns")

    count = len(df)
    timing = {"n": count}
    if count < max(4, min_points):
        return None, timing, ("too_short", "too_few_points")

    times = pd.to_datetime(df["TIME"], errors="coerce", utc=True)
    if times.isna().any():
        return None, timing, ("bad_cadence", "invalid_or_duplicate_timestamps")
    order = times.argsort(kind="stable").to_numpy()
    times = times.iloc[order]
    steps = times.diff().iloc[1:].dt.total_seconds().to_numpy()
    median_dt = np.median(steps)
    if not np.isfinite(steps).all() or np.min(steps) <= 0:
        return None, timing, ("bad_cadence", "invalid_or_duplicate_timestamps")
    dt = float(np.mean(steps))
    timing.update(dt_s=dt, max_spacing_s=float(np.max(steps)))
    if np.max(np.abs(steps - median_dt)) > 0.05 * median_dt:
        return None, timing, ("bad_cadence", "gap_or_irregular_sampling")

    offsets = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    if np.max(np.abs(offsets - dt * np.arange(count))) > 0.1 * dt:
        return None, timing, ("bad_cadence", "drift_from_uniform_grid")

    duration_s = count * dt
    timing["duration_s"] = duration_s
    if duration_s < min_duration_s - 0.05 * dt:
        return None, timing, ("too_short", "insufficient_duration")

    data = df.iloc[order]
    try:
        values = data[list(_WAVE_COLUMNS[1:])].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return None, timing, ("bad_data", "invalid_magnetic_samples")
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any():
        return None, timing, ("bad_data", "invalid_magnetic_samples")
    if "fac_reference_axis" in data and data["fac_reference_axis"].nunique(dropna=False) > 1:
        return None, timing, ("bad_data", "fac_reference_axis_changed")

    return values, timing, None


def _one_sided_spectra(values, dt):
    """Return frequency, complex FFT and one-sided PSD for each nT component."""
    count = len(values)
    window = np.hanning(count)
    detrended = signal.detrend(values, axis=0, type="linear")
    transform = fft.rfft(detrended * window[:, None], axis=0)
    psd = np.abs(transform)**2 * (2 * dt / np.sum(window**2))
    psd[0] *= 0.5
    if count % 2 == 0:
        psd[-1] *= 0.5
    return fft.rfftfreq(count, d=dt), transform, psd


def _band_masks(freqs, band, resolved_min, usable_max):
    lower, upper = band
    if not (np.isfinite(lower) and np.isfinite(upper)) or upper <= lower:
        return None, None, "empty_or_invalid_band"
    if lower < resolved_min:
        return None, None, "lower_edge_below_resolution"
    if upper > usable_max:
        return None, None, "upper_edge_above_usable_band"

    in_band = (freqs >= lower) & (freqs <= upper)
    flank = (
        ((freqs >= max(resolved_min, 0.5 * lower)) & (freqs < lower))
        | ((freqs > upper) & (freqs <= min(usable_max, 2 * upper)))
    )
    if np.count_nonzero(in_band) < 3:
        return None, None, "too_few_band_bins"
    if np.count_nonzero(flank) < 3:
        return None, None, "too_few_background_bins"
    return in_band, flank, "eligible"


def _peak_diagnostics(freqs, transform, psd, in_band, flank):
    peak_index = np.argmax(np.where(in_band, psd, -np.inf))
    peak_psd = psd[peak_index]
    background_psd = np.median(psd[flank])
    if background_psd > 0:
        prominence = peak_psd / background_psd
    else:
        prominence = np.inf if peak_psd > 0 else 0.0

    x, y = transform[peak_index, 2:4]
    transverse_energy = abs(x)**2 + abs(y)**2
    helicity = (
        2 * np.imag(x * np.conj(y)) / transverse_energy
        if transverse_energy > 0 else np.nan
    )
    return freqs[peak_index], prominence, helicity


def _icw_svd_polarization(freqs, transform, in_band, peak_frequency):
    """Magnetic SVD of Santolík et al. (2003), section 2, equations 2 and 8–13."""
    in_band_indices = np.flatnonzero(in_band)
    peak_index = np.searchsorted(freqs, peak_frequency)
    peak_position = np.searchsorted(in_band_indices, peak_index)
    count = min(5, len(in_band_indices))
    first = int(np.clip(peak_position - count // 2, 0, len(in_band_indices) - count))
    bins = in_band_indices[first:first + count]

    # Order the complex FFT as a right-handed (perp1, perp2, parallel) FAC basis.
    # Averaging B_i B_j* over nearby bins precedes SVD. The common PSD scale cancels.
    harmonics = transform[bins][:, [2, 3, 1]]
    spectral = harmonics.T @ harmonics.conj() / count
    stacked = np.vstack((spectral.real, -spectral.imag))
    _, singular, right = np.linalg.svd(stacked, full_matrices=False)
    largest, middle, smallest = singular

    diagnostics = {
        "icw_svd_status": "no_magnetic_power",
        "icw_svd_fmin_Hz": freqs[bins[0]],
        "icw_svd_fmax_Hz": freqs[bins[-1]],
        "icw_svd_n_bins": count,
        "icw_svd_ellipticity": np.nan,
        "icw_svd_planarity": np.nan,
        "icw_svd_normal_gap": np.nan,
        "icw_wave_normal_deg": np.nan,
        "icw_wave_normal_perp1": np.nan,
        "icw_wave_normal_perp2": np.nan,
        "icw_wave_normal_parallel": np.nan,
    }
    if largest <= 0:
        return diagnostics

    gap = (middle - smallest) / largest
    diagnostics.update(
        icw_svd_ellipticity=middle / largest,
        icw_svd_planarity=1 - np.sqrt(smallest / largest),
        icw_svd_normal_gap=gap,
    )
    if gap <= 64 * np.finfo(float).eps:
        diagnostics["icw_svd_status"] = "degenerate_wave_normal"
        return diagnostics

    # MAG constrains an axis. Choose its +B0 hemisphere as a display convention.
    normal = right[-1]
    if normal[2] < 0:
        normal = -normal
    diagnostics.update(
        icw_svd_status="estimated",
        icw_wave_normal_deg=np.degrees(np.arccos(np.clip(normal[2], 0, 1))),
        icw_wave_normal_perp1=normal[0],
        icw_wave_normal_perp2=normal[1],
        icw_wave_normal_parallel=normal[2],
    )
    return diagnostics


def check_ICW_mirror(
    df,
    ion_mass_amu=18.0,
    icw_band=(0.5, 1.2),
    icw_cb_max=0.2,
    icw_ellipticity_min=0.5,
    icw_peak_prominence_min=2.0,
    mirror_band_fraction_of_fc=(0.02, 0.3),
    mirror_band_hz=None,
    mirror_cb_min=0.5,
    mirror_fractional_rms_absB_min=0.01,
    mirror_peak_prominence_min=1.5,
    min_points=64,
    return_details=False,
    *,
    min_duration_s=600.0,
    equatorial_field_nT=None,
):
    """Classify one continuous window and optionally return eligibility diagnostics.

    Inputs are TIME, BTOTAL (nT), and the signed FAC components dB_para,
    dB_perp_1 and dB_perp_2 (nT). ICW uses the local mean magnitude unless
    the caller supplies a modeled equatorial magnitude in nT. The mirror
    band uses local field, or an explicit spacecraft-frame band in Hz.

    The pilot requires at least three cycles at its lower resolved frequency
    and four samples per cycle at the upper edge. The default mirror envelope
    is evaluated above the three-cycle floor, with the effective band reported.
    The eligibility flags determine which false candidate flags count as
    non-detections. Window length measures data support, not event duration.

    The legacy icw_ellipticity_min parameter thresholds the absolute reduced
    magnetic helicity. ICW details also include an unsigned magnetic-SVD
    ellipse axis ratio, planarity and a wave-normal axis in FAC coordinates.
    """

    def fail(label, reason, **diagnostics):
        details = {
            "classifier_version": CLASSIFIER_VERSION,
            "reason": reason,
            "icw_eligible": False,
            "mirror_eligible": False,
            "icw_status": reason,
            "icw_svd_status": "icw_unobservable",
            "mirror_status": reason,
            "is_ICW": False,
            "is_mirror": False,
            "label": label,
            **diagnostics,
        }
        result = (False, False, label)
        return (*result, details) if return_details else result

    values, timing, failure = _prepare_wave_window(df, min_points, min_duration_s)
    if failure is not None:
        return fail(*failure, **timing)
    dt = timing["dt_s"]
    duration_s = timing["duration_s"]

    local_field = values[:, 0].mean()
    if not np.isfinite(ion_mass_amu) or ion_mass_amu <= 0:
        return fail("bad_fc", "invalid_ion_mass", **timing)
    local_fc = _cyclotron_freq_hz(local_field, ion_mass_amu)
    if not np.isfinite(local_fc) or local_fc <= 0:
        return fail("bad_fc", "invalid_local_gyrofrequency", **timing)

    icw_reference = "local" if equatorial_field_nT is None else "equatorial"
    icw_fc = (
        local_fc if equatorial_field_nT is None
        else _cyclotron_freq_hz(equatorial_field_nT, ion_mass_amu)
    )
    resolved_min = 3 / duration_s
    usable_max = 1 / (4 * dt)
    freqs, transform, psd = _one_sided_spectra(values, dt)
    if not np.isfinite(psd).all():
        return fail("psd_failed", "nonfinite_power", **timing)
    psd_absB, psd_parallel, psd_p1, psd_p2 = psd.T
    psd_perp = psd_p1 + psd_p2
    df_hz = freqs[1] - freqs[0]

    icw_limits = tuple(edge * icw_fc for edge in icw_band)
    icw_mask, icw_flank, icw_status = _band_masks(freqs, icw_limits, resolved_min, usable_max)
    if not np.isfinite(icw_fc) or icw_fc <= 0:
        icw_status = "invalid_equatorial_gyrofrequency"
    icw_eligible = icw_status == "eligible"

    mirror_nominal = (
        tuple(edge * local_fc for edge in mirror_band_fraction_of_fc)
        if mirror_band_hz is None else mirror_band_hz
    )
    mirror_limits = (max(mirror_nominal[0], resolved_min), mirror_nominal[1])
    mirror_mask, mirror_flank, mirror_status = _band_masks(
        freqs, mirror_limits, resolved_min, usable_max,
    )
    mirror_eligible = mirror_status == "eligible"

    def band_power(component, mask):
        return float(np.sum(component[mask]) * df_hz)

    icw_peak = icw_prominence = icw_helicity = np.nan
    icw_parallel = icw_perp = icw_cb = icw_transverse_ratio = np.nan
    is_icw = False
    if icw_eligible:
        icw_parallel = band_power(psd_parallel, icw_mask)
        icw_perp = band_power(psd_perp, icw_mask)
        if icw_parallel + icw_perp > 0:
            icw_cb = icw_parallel / (icw_parallel + icw_perp)
            icw_transverse_ratio = icw_perp / icw_parallel if icw_parallel > 0 else np.inf
        icw_peak, icw_prominence, icw_helicity = _peak_diagnostics(
            freqs, transform, psd_perp, icw_mask, icw_flank,
        )
        is_icw = (
            icw_cb <= icw_cb_max
            and abs(icw_helicity) >= icw_ellipticity_min
            and icw_prominence >= icw_peak_prominence_min
        )

    mirror_peak = mirror_prominence = mirror_helicity = np.nan
    mirror_absB = mirror_parallel = mirror_perp = mirror_cb = np.nan
    mirror_absB_ratio = mirror_fractional_rms = np.nan
    is_mirror = False
    if mirror_eligible:
        mirror_absB = band_power(psd_absB, mirror_mask)
        mirror_parallel = band_power(psd_parallel, mirror_mask)
        mirror_perp = band_power(psd_perp, mirror_mask)
        if mirror_parallel + mirror_perp > 0:
            mirror_cb = mirror_parallel / (mirror_parallel + mirror_perp)
            mirror_absB_ratio = mirror_absB / (mirror_parallel + mirror_perp)
        mirror_fractional_rms = np.sqrt(mirror_absB) / local_field
        mirror_peak, mirror_prominence, mirror_helicity = _peak_diagnostics(
            freqs, transform, psd_absB, mirror_mask, mirror_flank,
        )
        is_mirror = (
            mirror_cb >= mirror_cb_min
            and mirror_fractional_rms >= mirror_fractional_rms_absB_min
            and mirror_prominence >= mirror_peak_prominence_min
        )

    if is_icw and is_mirror:
        label = "mixed"
    elif is_icw:
        label = "ICW"
    elif is_mirror:
        label = "mirror_like"
    else:
        label = "no_candidate" if icw_eligible and mirror_eligible else "unobservable"

    result = (bool(is_icw), bool(is_mirror), label)
    if not return_details:
        return result

    icw_polarization = {"icw_svd_status": "icw_unobservable"}
    if icw_eligible:
        icw_polarization = _icw_svd_polarization(freqs, transform, icw_mask, icw_peak)

    details = {
        "classifier_version": CLASSIFIER_VERSION,
        **timing,
        "nyquist_Hz": 0.5 / dt,
        "resolved_min_Hz": resolved_min,
        "usable_max_Hz": usable_max,
        "B0_nT": local_field,
        "fc_Hz_H2Oplus": local_fc,
        "fc_period_s": 1 / local_fc,
        "equatorial_field_nT": equatorial_field_nT,
        "icw_reference": icw_reference,
        "icw_reference_fc_Hz": icw_fc,
        "icw_eligible": icw_eligible,
        "icw_status": icw_status,
        "icw_fmin_Hz": icw_limits[0],
        "icw_fmax_Hz": icw_limits[1],
        "icw_f_peak_Hz": icw_peak,
        "icw_f_peak_over_fc": icw_peak / local_fc,
        "icw_f_peak_over_reference": icw_peak / icw_fc if icw_eligible else np.nan,
        "icw_P_parallel": icw_parallel,
        "icw_P_perp": icw_perp,
        "icw_C_B": icw_cb,
        "icw_transverse_ratio": icw_transverse_ratio,
        "icw_ellipticity": icw_helicity,
        "icw_abs_ellipticity": abs(icw_helicity),
        "icw_reduced_helicity": icw_helicity,
        "icw_abs_reduced_helicity": abs(icw_helicity),
        **icw_polarization,
        "icw_peak_prominence": icw_prominence,
        "mirror_eligible": mirror_eligible,
        "mirror_status": mirror_status,
        "mirror_nominal_fmin_Hz": mirror_nominal[0],
        "mirror_fmin_Hz": mirror_limits[0],
        "mirror_fmax_Hz": mirror_limits[1],
        "mirror_f_peak_Hz": mirror_peak,
        "mirror_f_peak_over_fc": mirror_peak / local_fc,
        "mirror_P_absB": mirror_absB,
        "mirror_P_parallel": mirror_parallel,
        "mirror_P_perp": mirror_perp,
        "mirror_C_B": mirror_cb,
        "mirror_absB_compressibility": mirror_absB_ratio,
        "mirror_fractional_rms_absB": mirror_fractional_rms,
        "mirror_ellipticity": mirror_helicity,
        "mirror_abs_ellipticity": abs(mirror_helicity),
        "mirror_peak_prominence": mirror_prominence,
        "is_ICW": result[0],
        "is_mirror": result[1],
        "label": label,
    }
    return (*result, details)
