"""Time-series and FFT panels for field-aligned Cassini MAG data."""

from datetime import timezone

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wave_classification import _one_sided_spectra, _prepare_wave_window, check_ICW_mirror


_PARALLEL_COLOR = "#3b4556"
_PERP_1_COLOR = "#1c71ae"
_PERP_2_COLOR = "#ba551c"
_PEAK_COLOR = "#8f2f63"


def _select_interval(df, start_time, end_time):
    required = {"TIME", "BTOTAL", "dB_para", "dB_perp_1", "dB_perp_2"}
    missing = required.difference(df)
    if missing:
        raise ValueError(f"Field-aligned data are missing columns: {', '.join(sorted(missing))}")

    times = pd.to_datetime(df["TIME"], errors="coerce", utc=True)
    start = None if start_time is None else pd.to_datetime(start_time, utc=True)
    end = None if end_time is None else pd.to_datetime(end_time, utc=True)
    if (start is not None and pd.isna(start)) or (end is not None and pd.isna(end)):
        raise ValueError("Use valid UTC bounds")
    if start is not None and end is not None and start >= end:
        raise ValueError("end_time must be later than start_time")

    selected = times.notna()
    if start is not None:
        selected &= times.ge(start)
    if end is not None:
        selected &= times.lt(end)
    if not selected.any():
        raise ValueError("No dated samples in the requested interval")
    # Keep undated rows so they cannot silently become an apparently valid FFT.
    return df.assign(TIME=times).loc[selected | times.isna()].sort_values("TIME", kind="stable")


def _draw_timeseries(axes, data):
    dated = data.loc[data["TIME"].notna()]
    times = dated["TIME"]
    steps = times.diff().dt.total_seconds()
    cadence = steps[steps > 0].median()
    gaps = np.flatnonzero(steps.gt(1.5 * cadence) | steps.eq(0))
    plot_times = np.insert(times.to_numpy(dtype="datetime64[ns]"), gaps, times.iloc[gaps])

    def plot_component(ax, column, label, color, style="-"):
        values = pd.to_numeric(dated[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        ax.plot(plot_times, np.insert(values, gaps, np.nan), style, color=color, lw=1, label=label)

    field, parallel, transverse = axes
    plot_component(field, "BTOTAL", r"$|B|$", _PARALLEL_COLOR)
    if "B0_mag" in data:
        plot_component(field, "B0_mag", r"$|B_0|$ (rolling background)", _PERP_1_COLOR, "--")
    plot_component(parallel, "dB_para", r"$\delta B_\parallel$", _PARALLEL_COLOR)
    plot_component(transverse, "dB_perp_1", r"$\delta B_{\perp 1}$", _PERP_1_COLOR)
    plot_component(transverse, "dB_perp_2", r"$\delta B_{\perp 2}$", _PERP_2_COLOR)

    field.set_ylabel("Field (nT)")
    parallel.set_ylabel(r"$\delta B_\parallel$ (nT)")
    transverse.set_ylabel(r"$\delta B_\perp$ (nT)")
    first, last = (
        time.tz_localize(None).isoformat(sep=" ", timespec="milliseconds").rstrip("0").rstrip(".")
        for time in (times.iloc[0], times.iloc[-1])
    )
    field.set_title(f"Cassini MAG   {first} to {last} UTC", loc="left")
    for ax in axes:
        ax.grid(alpha=0.18)
    for ax in (field, parallel):
        ax.tick_params(axis="x", labelbottom=False)
    field.legend(loc="best", frameon=False, fontsize="small")
    transverse.legend(loc="upper right", frameon=False, ncols=2, fontsize="small")
    if len(times) > 1 and times.iloc[0] < times.iloc[-1]:
        transverse.set_xlim(times.iloc[0], times.iloc[-1])
    locator = mdates.AutoDateLocator(tz=timezone.utc, minticks=5, maxticks=10, interval_multiples=False)
    transverse.xaxis.set_major_locator(locator)
    transverse.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    transverse.set_xlabel("Time (UTC)")


def _draw_spectrum(ax, data, classifier_options):
    ax.set_title("FFT power spectral density", loc="left")
    values, timing, failure = _prepare_wave_window(data, min_points=4, min_duration_s=0)
    if failure is None:
        dt = timing["dt_s"]
        freqs, _, psd = _one_sided_spectra(values, dt)
        if not np.isfinite(psd).all():
            failure = ("psd_failed", "nonfinite_power")
    if failure is not None:
        reason = failure[1].replace("_", " ")
        ax.text(0.5, 0.5, f"FFT unavailable\n{reason.capitalize()}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    parallel_psd = psd[1:, 1]
    transverse_psd = psd[1:, 2] + psd[1:, 3]
    ax.plot(freqs[1:], np.where(parallel_psd > 0, parallel_psd, np.nan),
            color=_PARALLEL_COLOR, lw=1.2, label=r"$S_\parallel$")
    ax.plot(freqs[1:], np.where(transverse_psd > 0, transverse_psd, np.nan),
            color=_PERP_1_COLOR, lw=1.2, label=r"$S_\perp=S_{\perp 1}+S_{\perp 2}$")
    ax.axvspan(1 / (4 * dt), freqs[-1], color="#728094", alpha=0.15,
               zorder=0, label="Above pilot cutoff")

    options = {**classifier_options, "return_details": True}
    is_icw, _, _, info = check_ICW_mirror(data, **options)
    low = info.get("icw_fmin_Hz", np.nan)
    high = info.get("icw_fmax_Hz", np.nan)
    if np.isfinite(low) and np.isfinite(high) and low < high:
        ax.axvspan(low, high, color=_PERP_1_COLOR, alpha=0.10, zorder=0, label="ICW search band")
    reference = info.get("icw_reference_fc_Hz", np.nan)
    if freqs[1] <= reference <= freqs[-1]:
        reference_field = info["icw_reference"]
        ax.axvline(reference, ls="--", lw=1, color="#386957",
                   label=rf"$f_c$ ({reference_field})")
    peak = info.get("icw_f_peak_Hz", np.nan)
    if np.isfinite(peak):
        ax.axvline(peak, ls=":", lw=1.3, color=_PEAK_COLOR, label=f"Band peak {peak:.4f} Hz")

    if info["icw_eligible"]:
        status = "ICW candidate" if is_icw else "Eligible · no ICW candidate"
        if is_icw and info["icw_svd_status"] == "estimated":
            status += (
                f" · WNA {info['icw_wave_normal_deg']:.1f}°"
                f" · SVD |ε| {info['icw_svd_ellipticity']:.2f}"
                f" · planarity {info['icw_svd_planarity']:.2f}"
            )
    else:
        reason = info["icw_status"].replace("_", " ")
        status = f"ICW unobservable · {reason}"
    ax.set_title(status, loc="right", fontsize="small")
    ax.set(xscale="log", xlim=(freqs[1], freqs[-1]), xlabel="Frequency (Hz)",
           ylabel=r"PSD (nT$^2$/Hz)")
    if np.any(parallel_psd > 0) or np.any(transverse_psd > 0):
        ax.set_yscale("log")
    ax.grid(which="major", alpha=0.18)
    ax.legend(loc="lower left", frameon=False, fontsize="small", ncols=2)


def plot_wave_timeseries(df, *, start_time=None, end_time=None):
    """Return a Matplotlib figure for an optional half-open interval in UTC."""
    data = _select_interval(df, start_time, end_time)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10.5, 5.4), layout="constrained")
    _draw_timeseries(axes, data)
    return fig


def plot_wave_panel(df, *, start_time=None, end_time=None, **classifier_options):
    """Return a time-series and FFT-PSD panel from an already processed FAC frame.

    Supply optional half-open UTC bounds to select within the available samples.
    Extra keyword options configure the ICW annotation, e.g. equatorial_field_nT.
    """
    data = _select_interval(df, start_time, end_time)
    fig = plt.figure(figsize=(10.5, 8.4), layout="constrained")
    grid = fig.add_gridspec(4, 1, height_ratios=(1, 1, 1, 2.15))
    field = fig.add_subplot(grid[0])
    parallel = fig.add_subplot(grid[1], sharex=field)
    transverse = fig.add_subplot(grid[2], sharex=field)
    spectrum = fig.add_subplot(grid[3])
    _draw_timeseries((field, parallel, transverse), data)
    _draw_spectrum(spectrum, data, classifier_options)
    return fig
