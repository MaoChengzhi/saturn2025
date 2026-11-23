"""MAG-only ICW and mirror-like candidate screening for Saturn occurrence pilots."""

import warnings

import numpy as np
import pandas as pd
from scipy import fft, signal


CLASSIFIER_VERSION = "mag-icw-v1"
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
DIPOLE_G10_NT = 21141.0  # Cassini 11+, Cao et al. (2020), Table 5.
DEFAULT_ICW_FC_REFERENCE = "dipole_mapped"
_DIPOLE_POSITION_COLUMNS = ("r_RS", "lat_spin_deg")
_DIPOLE_REFERENCES = ("dipole", "dipole_mapped")
_WAVE_COLUMNS = ("TIME", "BTOTAL", "dB_para", "dB_perp_1", "dB_perp_2")
_ICW_POLARIZATION_DEFAULTS = {
    "icw_svd_status": "icw_unobservable",
    "icw_svd_fmin_Hz": np.nan,
    "icw_svd_fmax_Hz": np.nan,
    "icw_svd_n_bins": 0,
    "icw_svd_ellipticity": np.nan,
    "icw_svd_planarity": np.nan,
    "icw_svd_normal_gap": np.nan,
    "icw_wave_normal_deg": np.nan,
    "icw_wave_normal_perp1": np.nan,
    "icw_wave_normal_perp2": np.nan,
    "icw_wave_normal_parallel": np.nan,
    "icw_stokes_I": np.nan,
    "icw_stokes_Q": np.nan,
    "icw_stokes_U": np.nan,
    "icw_stokes_V": np.nan,
    "icw_polarization_degree": np.nan,
    "icw_ellipticity_angle_deg": np.nan,
    "icw_orientation_deg": np.nan,
    "icw_rotation_sense": "undetermined",
}
_ICW_EQUATORIAL_DEFAULTS = dict.fromkeys((
    "B_eq_nT", "B_eq_min_nT", "B_eq_max_nT", "r_RS_mean", "lat_spin_deg_mean",
    "L_dipole_mean", "B_dip_local_nT", "B_meas_over_B_dip",
), np.nan)


def _cyclotron_freq_hz(field_nT, mass_amu):
    return E_CHARGE * field_nT * 1e-9 / (2 * np.pi * mass_amu * AMU)


def dipole_L_shell(r_RS, lat_deg):
    """Return L = r / cos²(latitude) for a centered spin-aligned dipole.

    Radius is in Saturn radii and planetocentric spin latitude is in degrees.
    The equator is the spin equator. The magnetic offset is ignored.
    """
    return np.asarray(r_RS, dtype=float) / np.cos(np.deg2rad(lat_deg))**2


def dipole_field_nT(r_RS, lat_deg):
    """Return g10/r³ sqrt(1 + 3 sin²(latitude)) in nT for the same dipole.

    Inputs are radius in Saturn radii and planetocentric spin latitude in
    degrees. This centered internal dipole omits the offset and higher terms.
    """
    return DIPOLE_G10_NT / np.asarray(r_RS, dtype=float)**3 * np.sqrt(
        1 + 3 * np.sin(np.deg2rad(lat_deg))**2,
    )


def equatorial_field_nT_from_dipole(r_RS, lat_deg, B_meas_nT=None, method=DEFAULT_ICW_FC_REFERENCE):
    """Estimate spin-equatorial field magnitude for a centered aligned dipole.

    With L = r / cos²(latitude), `dipole` returns g10/L³ in nT and ignores
    the measurement. `dipole_mapped` returns B_meas cos⁶(latitude) /
    sqrt(1 + 3 sin²(latitude)), assuming B_meas/B_dip is constant along the
    field line. Radius is in Saturn radii, latitude in degrees, and measured
    magnitude in nT. The ~0.05 Saturn-radii northward offset is ignored.
    """
    if method == "dipole":
        return DIPOLE_G10_NT / dipole_L_shell(r_RS, lat_deg)**3
    if method != "dipole_mapped":
        raise ValueError("method must be 'dipole' or 'dipole_mapped'")
    if B_meas_nT is None:
        raise ValueError("dipole_mapped requires B_meas_nT")
    latitude = np.deg2rad(lat_deg)
    return np.asarray(B_meas_nT, dtype=float) * np.cos(latitude)**6 / np.sqrt(
        1 + 3 * np.sin(latitude)**2,
    )


def _with_dipole_position(df, *, icw_fc_reference=DEFAULT_ICW_FC_REFERENCE, equatorial_field_nT=None):
    """Populate omitted dipole coordinates, or mark them unavailable after a lookup failure."""
    if (df is None or df.empty or equatorial_field_nT is not None
            or icw_fc_reference not in _DIPOLE_REFERENCES):
        return df
    missing = [column for column in _DIPOLE_POSITION_COLUMNS if column not in df]
    if not missing:
        return df

    try:
        from cassini_spice.locations import add_dipole_position
        from spiceypy.utils.exceptions import SpiceyError
    except ImportError as error:
        position_error = error
    else:
        try:
            positions = add_dipole_position(df)
        except (OSError, ValueError, SpiceyError) as error:
            position_error = error
        else:
            return df.assign(**{column: positions[column].to_numpy() for column in missing})

    warnings.warn(f"Dipole positions unavailable: {position_error}", stacklevel=3)
    # Keep the failed lookup visible to later slices without retrying it.
    return df.assign(**dict.fromkeys(missing, np.nan))


def _icw_reference_field(df, reference, explicit_field, local_field):
    """Return the reference field, its diagnostics, and any unobservable status."""
    diagnostics = {**_ICW_EQUATORIAL_DEFAULTS, "icw_reference": reference}
    if explicit_field is not None:
        diagnostics.update(icw_reference="equatorial", B_eq_nT=explicit_field,
                           B_eq_min_nT=explicit_field, B_eq_max_nT=explicit_field)
        return explicit_field, diagnostics, None
    if reference == "local":
        return local_field, diagnostics, None
    if reference not in _DIPOLE_REFERENCES:
        raise ValueError("icw_fc_reference must be 'local', 'dipole', or 'dipole_mapped'")

    missing_position = (np.nan, diagnostics, "missing_position_for_equatorial_field")
    if not set(_DIPOLE_POSITION_COLUMNS).issubset(df.columns):
        return missing_position
    try:
        radius, latitude = df[list(_DIPOLE_POSITION_COLUMNS)].to_numpy(dtype=float, na_value=np.nan).T
    except (TypeError, ValueError):
        return missing_position
    if not (np.isfinite(radius) & np.isfinite(latitude) & (radius > 0) & (abs(latitude) < 90)).all():
        return missing_position

    measured = df["BTOTAL"].to_numpy(dtype=float)
    dipole_local = dipole_field_nT(radius, latitude)
    eq_field = equatorial_field_nT_from_dipole(radius, latitude, measured, method=reference)
    diagnostics.update(
        B_eq_nT=eq_field.mean(), B_eq_min_nT=eq_field.min(), B_eq_max_nT=eq_field.max(),
        r_RS_mean=radius.mean(), lat_spin_deg_mean=latitude.mean(),
        L_dipole_mean=dipole_L_shell(radius, latitude).mean(), B_dip_local_nT=dipole_local.mean(),
        B_meas_over_B_dip=(measured / dipole_local).mean(),
    )
    return diagnostics["B_eq_nT"], diagnostics, None


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


def _icw_svd_polarization(freqs, transform, in_band, peak_frequency, n_bins=5):
    """Magnetic SVD of Santolík et al. (2003), section 2, equations 2 and 8–13.

    Stokes parameters (Born & Wolf and Rybicki & Lightman §2.4) use S0 = I,
    S1 = I p cos(2 psi) cos(2 chi), S2 = I p sin(2 psi) cos(2 chi), and
    S3 = I p sin(2 chi). I, Q, U and V share the averaged FFT cross-power
    normalization. Psi is measured from the first in-plane SVD basis vector.
    """
    in_band_indices = np.flatnonzero(in_band)
    peak_index = np.searchsorted(freqs, peak_frequency)
    peak_position = np.searchsorted(in_band_indices, peak_index)
    count = min(n_bins, len(in_band_indices))
    first = int(np.clip(peak_position - count // 2, 0, len(in_band_indices) - count))
    bins = in_band_indices[first:first + count]

    # Order the complex FFT as a right-handed (perp1, perp2, parallel) FAC basis.
    # The right-handedness is shown in data_process.py :     perp_basis_vec = reference_hat - np.sum(reference_hat * b_hat, axis=1, keepdims=True) * b_hat.          e_perp_1_hat = perp_basis_vec / np.linalg.norm(perp_basis_vec, axis=1, keepdims=True)               e_perp_2_hat = np.cross(b_hat, e_perp_1_hat)
    # Averaging B_i B_j* over nearby bins precedes SVD. The common PSD scale cancels.
    harmonics = transform[bins][:, [2, 3, 1]]
    spectral = harmonics.T @ harmonics.conj() / count
    stacked = np.vstack((spectral.real, -spectral.imag))
    _, singular, right = np.linalg.svd(stacked, full_matrices=False)
    largest, middle, smallest = singular

    diagnostics = {
        **_ICW_POLARIZATION_DEFAULTS,
        "icw_svd_status": "no_magnetic_power",
        "icw_svd_fmin_Hz": freqs[bins[0]],
        "icw_svd_fmax_Hz": freqs[bins[-1]],
        "icw_svd_n_bins": count,
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
    major, minor, normal = right
    if normal[2] < 0:
        normal = -normal
    if np.dot(np.cross(major, minor), normal) < 0:
        minor = -minor
    angle = np.degrees(np.arccos(np.clip(normal[2], 0, 1)))
    diagnostics.update(
        icw_svd_status="estimated",
        icw_wave_normal_deg=angle,
        icw_wave_normal_perp1=normal[0],
        icw_wave_normal_perp2=normal[1],
        icw_wave_normal_parallel=normal[2],
    )

    basis = np.stack((major, minor, normal))
    wave_spectral = basis @ spectral @ basis.T
    intensity = (wave_spectral[0, 0] + wave_spectral[1, 1]).real
    q = (wave_spectral[0, 0] - wave_spectral[1, 1]).real
    u = 2 * wave_spectral[0, 1].real
    v = 2 * wave_spectral[0, 1].imag
    diagnostics.update(icw_stokes_I=intensity, icw_stokes_Q=q, icw_stokes_U=u, icw_stokes_V=v)
    if intensity > 0:
        polarized = np.hypot(np.hypot(q, u), v)
        diagnostics["icw_polarization_degree"] = np.clip(polarized / intensity, 0, 1)
        if polarized > np.finfo(float).eps * intensity:
            diagnostics["icw_ellipticity_angle_deg"] = np.degrees(
                0.5 * np.arcsin(np.clip(v / polarized, -1, 1)),
            )
            diagnostics["icw_orientation_deg"] = np.degrees(0.5 * np.arctan2(u, q))
        if angle <= 80 and abs(v) / intensity >= 0.1:
            # Verified against (cos wt, -sin wt) in the right-handed FAC basis.
            diagnostics["icw_rotation_sense"] = "left" if v < 0 else "right"
    return diagnostics


def check_ICW_mirror(
    df,
    ion_mass_amu=18.0,
    icw_band=(0.5, 1.2),
    icw_cb_max=0.2,
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
    icw_fc_reference=DEFAULT_ICW_FC_REFERENCE,
    icw_svd_ellipticity_min=0.5,
    icw_planarity_min=0.7,
    icw_wave_normal_max_deg=30.0,
    icw_require_left_handed=True,
    icw_svd_bins=5,
    icw_polarization_min=0.9,
    icw_max_L=10.0,
    ion_label="H2O+",
):
    """Classify one continuous window and optionally return eligibility diagnostics.

    Inputs are TIME, BTOTAL (nT), and the signed FAC components dB_para,
    dB_perp_1 and dB_perp_2 (nT). ICW defaults to the mean of the per-sample
    dipole-mapped fields. Both dipole references use r_RS and lat_spin_deg
    for every sample and query local SPICE if either column is absent. Failed
    position lookups warn and make ICW unobservable. Select "local" to use mean
    BTOTAL. Both dipole references mark ICW unobservable when mean L exceeds
    icw_max_L (None disables the limit). An explicit equatorial_field_nT
    overrides the reference and bypasses this limit.
    The mirror band uses local field, or an explicit spacecraft-frame band in Hz.

    The pilot requires at least three cycles at its lower resolved frequency
    and four samples per cycle at the upper edge. The default mirror envelope
    is evaluated above the three-cycle floor, with the effective band reported.
    The eligibility flags determine which false candidate flags count as
    non-detections. Window length measures data support, not event duration.

    The SVD axis ratio, planarity, wave-normal angle and left-handed rotation
    criteria are provisional pilot choices pending a literature citation.
    Rotation is the spacecraft-frame sense about +B0. The signed
    icw_fac_circular_polarization is single-bin Stokes V/I in the plane
    perpendicular to B0. This diagnostic and its absolute value are excluded
    from the decision.

    Polarization degree p = sqrt(Q^2 + U^2 + V^2) / I comes from the
    wave-plane spectral matrix averaged over up to icw_svd_bins bins
    (Taubenschuss & Santolík 2019, Surveys in Geophysics, eq. 14). With few
    bins, noise can appear partly polarized (pilot median p about 0.77 with
    five bins). The 0.9 noise-rejection threshold is provisional and tied to
    icw_svd_bins=5.

    ion_label names the ion whose mass-to-charge ratio is ion_mass_amu (H2O+
    at Saturn; for example S+ or SO2+ at Jupiter). Details report the local
    gyrofrequency as fc_Hz_local, and also as fc_Hz_H2Oplus for H2O+.
    """
    if not isinstance(icw_svd_bins, (int, np.integer)) or icw_svd_bins < 1:
        raise ValueError("icw_svd_bins must be a positive integer")

    def fail(label, reason, **diagnostics):
        details = {
            "classifier_version": CLASSIFIER_VERSION,
            "reason": reason,
            "icw_eligible": False,
            "mirror_eligible": False,
            "icw_status": reason,
            **_ICW_POLARIZATION_DEFAULTS,
            **_ICW_EQUATORIAL_DEFAULTS,
            "icw_fac_circular_polarization": np.nan,
            "icw_abs_fac_circular_polarization": np.nan,
            "icw_fail_reasons": [reason],
            "mirror_status": reason,
            "mirror_fail_reasons": [reason],
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

    df = _with_dipole_position(df, icw_fc_reference=icw_fc_reference, equatorial_field_nT=equatorial_field_nT)
    icw_field, icw_field_diagnostics, reference_failure = _icw_reference_field(
        df, icw_fc_reference, equatorial_field_nT, local_field,
    )
    icw_fc = _cyclotron_freq_hz(icw_field, ion_mass_amu)
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
    if reference_failure is not None:
        icw_status = reference_failure
    elif (equatorial_field_nT is None and icw_fc_reference in _DIPOLE_REFERENCES
          and icw_max_L is not None and icw_field_diagnostics["L_dipole_mean"] > icw_max_L):
        icw_status = "outside_dipole_mapping_range"
    elif not np.isfinite(icw_fc) or icw_fc <= 0:
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

    icw_peak = icw_prominence = icw_fac_circular_polarization = np.nan
    icw_parallel = icw_perp = icw_cb = icw_transverse_ratio = np.nan
    icw_polarization = _ICW_POLARIZATION_DEFAULTS
    icw_fail_reasons = [icw_status]
    is_icw = False
    if icw_eligible:
        icw_parallel = band_power(psd_parallel, icw_mask)
        icw_perp = band_power(psd_perp, icw_mask)
        if icw_parallel + icw_perp > 0:
            icw_cb = icw_parallel / (icw_parallel + icw_perp)
            icw_transverse_ratio = icw_perp / icw_parallel if icw_parallel > 0 else np.inf
        icw_peak, icw_prominence, icw_fac_circular_polarization = _peak_diagnostics(
            freqs, transform, psd_perp, icw_mask, icw_flank,
        )
        icw_polarization = _icw_svd_polarization(
            freqs, transform, icw_mask, icw_peak, n_bins=icw_svd_bins,
        )
        icw_checks = {
            "compressibility": icw_cb <= icw_cb_max,
            "peak_prominence": icw_prominence >= icw_peak_prominence_min,
            "svd_status": icw_polarization["icw_svd_status"] == "estimated",
            "svd_ellipticity": icw_polarization["icw_svd_ellipticity"] >= icw_svd_ellipticity_min,
            "svd_planarity": icw_polarization["icw_svd_planarity"] >= icw_planarity_min,
            "polarization_degree": icw_polarization["icw_polarization_degree"] >= icw_polarization_min,
            "wave_normal_angle": icw_polarization["icw_wave_normal_deg"] <= icw_wave_normal_max_deg,
            "rotation_sense": not icw_require_left_handed or icw_polarization["icw_rotation_sense"] == "left",
        }
        icw_fail_reasons = [criterion for criterion, passed in icw_checks.items() if not passed]
        is_icw = not icw_fail_reasons

    mirror_peak = mirror_prominence = mirror_helicity = np.nan
    mirror_absB = mirror_parallel = mirror_perp = mirror_cb = np.nan
    mirror_absB_ratio = mirror_fractional_rms = np.nan
    mirror_fail_reasons = [mirror_status]
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
        mirror_checks = {
            "compressibility": mirror_cb >= mirror_cb_min,
            "absB_amplitude": mirror_fractional_rms >= mirror_fractional_rms_absB_min,
            "peak_prominence": mirror_prominence >= mirror_peak_prominence_min,
        }
        mirror_fail_reasons = [criterion for criterion, passed in mirror_checks.items() if not passed]
        is_mirror = not mirror_fail_reasons

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

    details = {
        "classifier_version": CLASSIFIER_VERSION,
        **timing,
        "nyquist_Hz": 0.5 / dt,
        "resolved_min_Hz": resolved_min,
        "usable_max_Hz": usable_max,
        "B0_nT": local_field,
        **({"fc_Hz_H2Oplus": local_fc} if ion_label == "H2O+" else {}),
        "ion_label": ion_label,
        "ion_mass_amu": ion_mass_amu,
        "fc_Hz_local": local_fc,
        "fc_period_s": 1 / local_fc,
        "equatorial_field_nT": equatorial_field_nT,
        **icw_field_diagnostics,
        "icw_reference_fc_Hz": icw_fc,
        "icw_eligible": icw_eligible,
        "icw_status": icw_status,
        "icw_fail_reasons": icw_fail_reasons,
        "icw_fmin_Hz": icw_limits[0],
        "icw_fmax_Hz": icw_limits[1],
        "icw_f_peak_Hz": icw_peak,
        "icw_f_peak_over_fc": icw_peak / local_fc,
        "icw_f_peak_over_reference": icw_peak / icw_fc if icw_eligible else np.nan,
        "icw_P_parallel": icw_parallel,
        "icw_P_perp": icw_perp,
        "icw_C_B": icw_cb,
        "icw_transverse_ratio": icw_transverse_ratio,
        "icw_fac_circular_polarization": icw_fac_circular_polarization,
        "icw_abs_fac_circular_polarization": abs(icw_fac_circular_polarization),
        **icw_polarization,
        "icw_peak_prominence": icw_prominence,
        "mirror_eligible": mirror_eligible,
        "mirror_status": mirror_status,
        "mirror_fail_reasons": mirror_fail_reasons,
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


def _icw_welch_coherence(window, dt, peak):
    """Estimate transverse magnitude-squared coherence near the whole-window peak."""
    transverse = window[["dB_perp_1", "dB_perp_2"]].to_numpy(dtype=float).T
    # Round half the window down to an even length so a 50% overlap fits three segments.
    segment_points = 2 * (len(window) // 4)
    overlap_points = segment_points // 2
    if segment_points < 4:
        return np.nan, np.nan
    if (np.ptp(transverse, axis=1) == 0).any():
        return np.nan, np.nan

    settings = {
        "fs": 1 / dt,
        "window": "hann",
        "nperseg": segment_points,
        "noverlap": overlap_points,
        "detrend": "linear",
    }
    freqs, autos = signal.welch(transverse, **settings)
    _, cross = signal.csd(*transverse, **settings)
    index = np.argmin(np.abs(freqs - peak))
    denominator = autos[0, index] * autos[1, index]
    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(cross[index]):
        return np.nan, np.nan

    coherence = abs(cross[index])**2 / denominator
    return float(np.clip(coherence, 0, 1)), float(freqs[index])


def icw_window_diagnostics(df, *, window_s=120.0, step_s=20.0, **classifier_options):
    """Return diagnostics at the centers of complete sliding FAC windows.

    The half-open UTC grid starts at the first processed sample. Unobservable
    windows retain their timestamps and classifier status. The RMS values are
    square roots of the integrated ICW-band powers in nT². Coherence is a
    descriptive transverse Welch estimate at the bin nearest the ICW peak,
    with linearly detrended half-window Hann segments and 50% segment overlap.
    Sliding classifications use window_s as their minimum duration. Any needed
    dipole positions are looked up once for the interval before slicing it.
    """
    window_s, step_s = float(window_s), float(step_s)
    if not all(np.isfinite(value) and value > 0 for value in (window_s, step_s)):
        raise ValueError("window_s and step_s must be finite positive seconds")
    duration = pd.to_timedelta(window_s, unit="s")
    step = pd.to_timedelta(step_s, unit="s")
    if min(duration, step) <= pd.Timedelta(0):
        raise ValueError("window_s and step_s must be at least one nanosecond")

    required_columns = (
        "time", "interval_start", "interval_end", "label", "is_ICW", "is_mirror",
        "icw_eligible", "icw_status", "icw_fail_reasons", "icw_f_peak_Hz", "icw_C_B",
        "icw_fac_circular_polarization", "icw_abs_fac_circular_polarization",
        *_ICW_POLARIZATION_DEFAULTS, *_ICW_EQUATORIAL_DEFAULTS, "icw_perp_rms_nT", "icw_parallel_rms_nT",
        "icw_coherence", "icw_coherence_frequency_Hz",
        "mirror_eligible", "mirror_status", "mirror_fail_reasons",
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=required_columns)
    if "TIME" not in df:
        raise ValueError("Sliding diagnostics require a TIME column")

    times = pd.to_datetime(df["TIME"], errors="coerce", utc=True)
    data = df.assign(TIME=times).sort_values("TIME", kind="stable")
    dated = data.loc[data["TIME"].notna()]
    undated = data.loc[data["TIME"].isna()]
    timestamps = pd.DatetimeIndex(dated["TIME"])
    spacing = timestamps.unique().to_series().diff().median()
    if pd.isna(spacing):
        return pd.DataFrame(columns=required_columns)

    # The last sample supports one nominal spacing; do not schedule trailing partial windows.
    starts = pd.date_range(timestamps[0], timestamps[-1] + spacing - duration, freq=step)
    if len(starts) and undated.empty:
        dated = _with_dipole_position(
            dated, icw_fc_reference=classifier_options.get("icw_fc_reference", DEFAULT_ICW_FC_REFERENCE),
            equatorial_field_nT=classifier_options.get("equatorial_field_nT"),
        )
    options = {**classifier_options, "return_details": True, "min_duration_s": window_s}
    rows = []
    for start in starts:
        end = start + duration
        left, right = timestamps.searchsorted([start, end])
        selected = dated.iloc[left:right]
        # An undated sample cannot be assigned to a window, so retain the cadence failure.
        window = pd.concat([selected, undated]) if not undated.empty else selected
        _, _, _, details = check_ICW_mirror(window, **options)
        coherence = coherence_frequency = np.nan
        if details["icw_eligible"]:
            coherence, coherence_frequency = _icw_welch_coherence(
                window, details["dt_s"], details["icw_f_peak_Hz"],
            )
        rows.append({
            "time": start + duration / 2,
            "interval_start": start,
            "interval_end": end,
            **details,
            "icw_perp_rms_nT": np.sqrt(details.get("icw_P_perp", np.nan)),
            "icw_parallel_rms_nT": np.sqrt(details.get("icw_P_parallel", np.nan)),
            "icw_coherence": coherence,
            "icw_coherence_frequency_Hz": coherence_frequency,
        })

    diagnostics = pd.DataFrame(rows)
    return diagnostics.reindex(columns=diagnostics.columns.union(required_columns, sort=False))
