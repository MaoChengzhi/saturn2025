"""Configurable event panels for field-aligned Cassini MAG data."""

from dataclasses import dataclass
from datetime import timezone
from inspect import signature
from textwrap import fill

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
import numpy as np
import pandas as pd
from scipy.signal import detrend

from data_process import _parse_time_range, read_and_process_data
from do_wavelet import wavelet
from wave_classification import (
    DEFAULT_ICW_FC_REFERENCE, _with_dipole_position,
    _one_sided_spectra, _prepare_wave_window, check_ICW_mirror, icw_window_diagnostics,
)


_PARALLEL_COLOR = "#3b4556"
_PERP_1_COLOR = "#1c71ae"
_PERP_2_COLOR = "#ba551c"
_PEAK_COLOR = "#8f2f63"
_MIRROR_COLOR = "#956023"
_INACTIVE_COLOR = "#777777"
_REFERENCE_COLOR = "#386957"
_FAC_SWITCH_COLOR = "#b44455"
_HODOGRAM_SPAN_COLOR = "#339a9c"
_RAW_ROWS = ("field", "parallel", "transverse")
_DETAIL_ROWS = ("fft", "hodogram", "orbit")

# Change the order of this tuple to change the default panel order.
DEFAULT_WAVE_PANELS = (
    *_RAW_ROWS, "wavelet", "amplitude", "peak_frequency", "polarization", "wave_normal",
    "compressibility", "hodogram", "fft",
)
_MIRROR_WAVE_PANELS = (*_RAW_ROWS, "wavelet", "compressibility", "mirror_amplitude", "fft", "orbit")
_ROW_HEIGHTS = {
    "field": 1.0, "parallel": 1.0, "transverse": 1.0, "wavelet": 2.3,
    "amplitude": 1.2, "peak_frequency": 1.2, "polarization": 1.35, "wave_normal": 1.2,
    "compressibility": 1.15, "mirror_amplitude": 1.15,
    "fft": 3.6, "hodogram": 3.0, "orbit": 3.0,
}
_TIMELINE_ROWS = {
    "amplitude": ("RMS amplitude", "RMS (nT)", (
        ("icw_perp_rms_nT", r"$\perp$ RMS", _PERP_1_COLOR, 1),
        ("icw_parallel_rms_nT", r"$\parallel$ RMS", _PARALLEL_COLOR, 1),
    )),
    "peak_frequency": ("Transverse peak", "Frequency (Hz)", (
        ("icw_f_peak_Hz", "Band peak", _PEAK_COLOR, 1),
    )),
    "polarization": ("Polarization and coherence", "Ratio", (
        ("icw_polarization_degree", "p (gate)", _PERP_2_COLOR, 1),
        ("icw_svd_ellipticity", "SVD |ε| (gate)", _PEAK_COLOR, 1),
        ("icw_coherence", "Transverse coherence", _PERP_1_COLOR, 1),
    )),
    "wave_normal": ("Wave-normal angle", "Angle (°)", (
        ("icw_wave_normal_deg", "Wave-normal angle", _PEAK_COLOR, 1),
    )),
    "compressibility": ("Band compressibility", r"$C_B$", (
        ("icw_C_B", "ICW band", _PERP_1_COLOR, 1),
        ("mirror_C_B", "Mirror band", _PERP_2_COLOR, 1),
    )),
    "mirror_amplitude": ("Field-magnitude RMS", r"RMS / mean $|B|$ (%)", (
        ("mirror_fractional_rms_absB", "Mirror-band amplitude", _PERP_2_COLOR, 100),
    )),
}
_TIMELINE_GATES = {
    "polarization": (
        ("icw_polarization_min", "p ≥ {value:g}", 1, _PERP_2_COLOR),
        ("icw_svd_ellipticity_min", "|ε| ≥ {value:g}", 1, _PEAK_COLOR),
    ),
    "wave_normal": (("icw_wave_normal_max_deg", "θ ≤ {value:g}°", 1, _PEAK_COLOR),),
    "compressibility": (
        ("icw_cb_max", r"ICW $C_B$ ≤ {value:g}", 1, _PERP_1_COLOR),
        ("mirror_cb_min", r"Mirror $C_B$ ≥ {value:g}", 1, _PERP_2_COLOR),
    ),
    "mirror_amplitude": (("mirror_fractional_rms_absB_min", "RMS / B₀ ≥ {value:g}%", 100, _PERP_2_COLOR),),
}


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


def _cadence_breaks(times):
    steps = times.diff().dt.total_seconds()
    cadence = steps[steps > 0].median()
    return (steps.gt(1.5 * cadence) | steps.le(0)).to_numpy()


def _fac_references(data):
    if "fac_reference_axis" not in data:
        return np.full(len(data), "", dtype="U5")
    labels = data["fac_reference_axis"]
    return labels.where(labels.isin(("R", "THETA"))).astype("string").fillna("").to_numpy(dtype="U5")


def _fac_switches(references):
    changes = np.zeros(len(references), dtype=bool)
    changes[1:] = ((references[1:] != "") & (references[:-1] != "")
                   & (references[1:] != references[:-1]))
    return changes


def _draw_timeseries(axes, dated, references):
    times = dated["TIME"].to_numpy(dtype="datetime64[ns]")
    gaps = _cadence_breaks(dated["TIME"])
    reference_breaks = np.r_[False, references[1:] != references[:-1]]

    def plot_component(ax, column, label, color, style="-", breaks=gaps):
        values = pd.to_numeric(dated[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        indices = np.flatnonzero(breaks)
        ax.plot(np.insert(times, indices, times[indices]), np.insert(values, indices, np.nan),
                style, color=color, lw=1, label=label)

    if "field" in axes:
        field = axes["field"]
        plot_component(field, "BTOTAL", r"$|B|$", _PARALLEL_COLOR)
        if "B0_mag" in dated:
            plot_component(field, "B0_mag", r"$|B_0|$ (rolling background)", _PERP_1_COLOR, "--")
        field.set_ylabel("Field (nT)")
        field.legend(loc="best", frameon=False, fontsize="small")
    if "parallel" in axes:
        parallel = axes["parallel"]
        plot_component(parallel, "dB_para", r"$\delta B_\parallel$", _PARALLEL_COLOR)
        parallel.set_ylabel(r"$\delta B_\parallel$ (nT)")
    if "transverse" in axes:
        transverse = axes["transverse"]
        breaks = gaps | reference_breaks
        plot_component(transverse, "dB_perp_1", r"$\delta B_{\perp 1}$", _PERP_1_COLOR, breaks=breaks)
        plot_component(transverse, "dB_perp_2", r"$\delta B_{\perp 2}$", _PERP_2_COLOR, breaks=breaks)
        transverse.set_ylabel(r"$\delta B_\perp$ (nT)")
        handles, labels = transverse.get_legend_handles_labels()
        transverse.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.01),
                          borderaxespad=0, frameon=False, ncols=2, fontsize="small")
    for ax in axes.values():
        ax.grid(alpha=0.18)


def _mark_fac_switches(ax, times, references, switches):
    """Mark the first observed sample in each new, known FAC reference."""
    for time in times.iloc[np.flatnonzero(switches)]:
        ax.axvline(time, color=_FAC_SWITCH_COLOR, ls="--", lw=1, zorder=4,
                   gid="fac_reference_switch")
    if switches.any():
        ax.set_title(f"FAC R ↔ Θ switches · {switches.sum()}", loc="right", fontsize="small",
                     color=_FAC_SWITCH_COLOR)
    elif not (references != "").all():
        ax.set_title("FAC reference unknown for some samples", loc="right", fontsize="small")


def _hodogram_selection(times, interval):
    if interval is None:
        midpoint = times.iloc[0] + (times.iloc[-1] - times.iloc[0]) / 2
        half_span = pd.Timedelta(seconds=20)
        lower, upper = midpoint - half_span, midpoint + half_span
    else:
        start, stop = interval
        lower = pd.to_datetime(start, utc=True) if start is not None else None
        upper = pd.to_datetime(stop, utc=True) if stop is not None else None
        if ((lower is not None and pd.isna(lower)) or (upper is not None and pd.isna(upper))
                or (lower is not None and upper is not None and lower >= upper)):
            raise ValueError("hodogram_interval needs ordered UTC bounds or None endpoints")
    selected = np.ones(len(times), dtype=bool)
    if lower is not None:
        selected &= times.ge(lower).to_numpy()
    if upper is not None:
        selected &= times.lt(upper).to_numpy()
    return selected


def _peak_rms_hodogram_interval(diagnostics):
    if diagnostics.empty:
        return None
    rms = diagnostics["icw_perp_rms_nT"]
    supported = rms[np.isfinite(rms)]
    if supported.empty:
        return None
    strongest = diagnostics.loc[supported.idxmax()]
    peak = strongest["icw_f_peak_Hz"]
    if not np.isfinite(peak) or peak <= 0:
        return None
    half_span = pd.Timedelta(seconds=max(40, 6 / peak) / 2)
    return strongest["time"] - half_span, strongest["time"] + half_span


def _draw_hodogram(ax, data, references, gaps, *, is_icw=False):
    ax.set(xlabel=r"$\delta B_{\perp 1}$ (nT)", ylabel=r"$\delta B_{\perp 2}$ (nT)", aspect="equal")
    title = "Transverse hodogram · ICW candidate" if is_icw else "Transverse hodogram"
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.2)
    points = data[["dB_perp_1", "dB_perp_2"]].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan,
    )
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        ax.text(0.5, 0.5, "No transverse samples in this interval", ha="center", va="center",
                transform=ax.transAxes, wrap=True)
        return

    times = data["TIME"]
    elapsed = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    known = references != ""
    # Never draw a chord between projections in different (or unknown) bases.
    connected = (finite[:-1] & finite[1:] & known[:-1] & known[1:]
                 & (references[:-1] == references[1:]) & ~gaps[1:])
    segments = np.stack((points[:-1][connected], points[1:][connected]), axis=1)
    norm = Normalize(0, max(elapsed[-1], 1))
    path = LineCollection(segments, cmap="viridis", norm=norm, linewidth=1.25, gid="hodogram_path")
    path.set_array((elapsed[:-1][connected] + elapsed[1:][connected]) / 2)
    ax.add_collection(path)
    ax.scatter(*points[finite].T, c=elapsed[finite], cmap="viridis", norm=norm, s=9, zorder=3)
    ax.autoscale_view()

    labels = " / ".join(dict.fromkeys(references[known])).replace("THETA", "Θ")
    reference_text = f" · FAC {labels}" if labels else " · FAC reference unknown"
    if labels and not known.all():
        reference_text += " + unknown"
    if _fac_switches(references).any():
        reference_text += " · split"
    ax.set_title(f"{title}\n"
                 f"{times.iloc[0]:%H:%M:%S}–{times.iloc[-1]:%H:%M:%S} UTC{reference_text}",
                 loc="left", fontsize="medium")
    if elapsed[-1] > 0:
        bar = ax.figure.colorbar(path, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08,
                                label="Elapsed time (s)")
        bar.ax.set_label("hodogram_colorbar")


def _event_title(times):
    first, last = (
        time.tz_localize(None).isoformat(sep=" ", timespec="milliseconds").rstrip("0").rstrip(".")
        for time in (times.iloc[0], times.iloc[-1])
    )
    return f"Cassini MAG   {first} to {last} UTC"


def _label_time_axes(axes, times):
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    bottom = axes[-1]
    if len(times) > 1 and times.iloc[0] < times.iloc[-1]:
        bottom.set_xlim(times.iloc[0], times.iloc[-1])
    locator = mdates.AutoDateLocator(tz=timezone.utc, minticks=5, maxticks=10, interval_multiples=False)
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    bottom.set_xlabel("Time (UTC)")


def _draw_wavelet(ax, data, window, info, *, component="transverse"):
    if component == "parallel":
        title, columns = "Parallel (compressional) Morlet wavelet power", slice(1, 2)
    else:
        title, columns = "Transverse Morlet wavelet power", slice(2, 4)
    ax.set_title(title, loc="left")
    values, timing, failure = window
    if failure is not None:
        _wavelet_unavailable(ax, failure[1].replace("_", " "))
        return None

    dt = timing["dt_s"]
    fmin, fmax = 3 / timing["duration_s"], 1 / (4 * dt)
    if fmin >= fmax:
        _wavelet_unavailable(ax, "insufficient frequency coverage")
        return None

    freqs = np.geomspace(fmin, fmax, num=64)
    components = detrend(values[:, columns], axis=0, type="linear").T
    coefficients, _, _, coi_period = wavelet(components, dt, freq=freqs)
    power = np.sum(np.abs(coefficients)**2, axis=1)
    if not np.isfinite(power).all() or not np.any(power > 0):
        _wavelet_unavailable(ax, f"no finite positive {component} power")
        return None

    vmax = power.max()
    vmin = max(np.percentile(power[power > 0], 5), vmax * 1e-6)
    if vmin >= vmax:
        vmin = vmax / 10
    times = mdates.date2num(data["TIME"].to_numpy(dtype="datetime64[ns]"))
    mesh = ax.pcolormesh(
        times, freqs, np.ma.masked_less_equal(power, 0),
        shading="auto", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    coi_freq = np.divide(1, coi_period, out=np.full_like(coi_period, np.inf), where=coi_period > 0)
    ax.fill_between(times, fmin, np.clip(coi_freq, fmin, fmax), color="white", alpha=0.45)
    boundary = np.where((coi_freq >= fmin) & (coi_freq <= fmax), coi_freq, np.nan)
    ax.plot(times, boundary, "--", color="#f6f8fa", lw=1)
    ax.set_title("Shaded region affected by edges", loc="right", fontsize="small")
    ax.set(yscale="log", ylim=(fmin, fmax), ylabel="Frequency (Hz)")
    if component == "parallel":
        references = (("mirror_fmin_Hz", "Mirror low"), ("mirror_fmax_Hz", "Mirror high"))
    else:
        references = (("icw_fmin_Hz", "ICW low"), ("icw_reference_fc_Hz", r"$f_c$"),
                      ("icw_fmax_Hz", "ICW high"))
    outline = [withStroke(linewidth=2.0, foreground="#222222")]
    for key, label in references:
        frequency = info.get(key, np.nan)
        if fmin <= frequency <= fmax:
            ax.axhline(frequency, color="white", ls="--", lw=0.8, zorder=3,
                       path_effects=outline, gid="wavelet_reference")
            near_top = np.log(fmax / frequency) < 0.045 * np.log(fmax / fmin)
            ax.annotate(label, xy=(1, frequency), xycoords=ax.get_yaxis_transform(),
                        xytext=(-4, -2 if near_top else 2), textcoords="offset points",
                        ha="right", va="top" if near_top else "bottom",
                        color="white", fontsize="x-small", path_effects=outline, clip_on=True)
    return mesh


def _wavelet_unavailable(ax, reason):
    ax.text(0.5, 0.5, f"Wavelet unavailable\n{reason.capitalize()}", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_yticks([])


def _is_candidate(info, mode):
    return info["is_ICW" if mode == "icw" else "is_mirror"]


def _mode_status(info, mode, *, show_failures=False):
    if _is_candidate(info, mode):
        return "candidate"
    if not info[f"{mode}_eligible"]:
        reason = info[f"{mode}_status"].replace("_", " ")
        return f"not applicable ({reason})"
    if show_failures:
        reasons = ", ".join(reason.replace("absB", "field magnitude").replace("svd", "SVD").replace("_", " ")
                            for reason in info[f"{mode}_fail_reasons"])
        return f"no candidate (fails: {reasons})"
    return "not applicable (no candidate)"


def _draw_classification_header(fig, title, info):
    label = {"ICW": "ICW", "mirror_like": "mirror-like", "mixed": "mixed",
             "no_candidate": "no candidate", "no_data": "no data"}.get(info["label"], "unobservable")
    lines = [f"Classification: {label}"]
    if label != "no data":
        lines.extend(f"{name}: {_mode_status(info, mode, show_failures=True)}"
                     for name, mode in (("ICW", "icw"), ("Mirror", "mirror")))
    width = fig.get_figwidth()
    title = fill(title, width=max(18, int(5.5 * width)), break_on_hyphens=False)
    text = "\n".join(fill(line, width=max(24, int(7.5 * width)), subsequent_indent="    ") for line in lines)
    height = fig.get_figheight()
    header_top = 1 - 0.12 / height
    if fig.axes:
        header_height = max(0.22 * (title.count("\n") + 1), 0.15 * (text.count("\n") + 1))
        axes_top = header_top - (header_height + 0.14) / height
        fig.get_layout_engine().set(rect=(0, 0, 1, max(0.2, axes_top)))
    fig.suptitle(title, x=0.015, y=header_top, ha="left", va="top", fontsize=12)
    fig.text(0.49, header_top, text, ha="left", va="top", fontsize=9, gid="classification_result",
             bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f1f4f7", "edgecolor": "#c8d0d9"})


def _draw_timeline(ax, name, diagnostics, info, limits, *, note=None):
    title, ylabel, lines = _TIMELINE_ROWS[name]
    mode = "mirror" if name in ("compressibility", "mirror_amplitude") else "icw"
    mode_name = "mirror" if mode == "mirror" else "ICW"
    candidate = _is_candidate(info, mode)
    if candidate:
        status = f"{mode_name} candidate"
    elif info[f"{mode}_eligible"]:
        status = f"not applicable: no {mode_name} candidate"
    else:
        status = "not applicable: " + info[f"{mode}_status"].replace("_", " ")
    ax.set_title(f"{title}\n{note}" if note else title, loc="left", fontsize="medium")
    ax.set_title(status, loc="right", fontsize="small")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)

    gate_labels = {}
    for key, label, multiplier, color in _TIMELINE_GATES.get(name, ()):
        value = limits[key] * multiplier
        if np.isfinite(value):
            ax.axhline(value, ls="--", lw=0.8, color=color, alpha=0.65, zorder=1)
            gate_labels.setdefault(value, []).append(label.format(value=value))

    has_values = False
    for index, (column, label, color, multiplier) in enumerate(lines):
        if column not in diagnostics:
            continue
        values = diagnostics[column].to_numpy(dtype=float, na_value=np.nan) * multiplier
        if np.isfinite(values).any():
            style = "-" if candidate else ("-", "--", "-.")[index]
            is_coherence = column == "icw_coherence"
            ax.plot(diagnostics["time"], values, marker=None if is_coherence else ".", linestyle=style,
                    color=color if candidate else _INACTIVE_COLOR, alpha=1 if candidate else 0.55,
                    lw=0.65 if is_coherence else 1, ms=4, label=label)
            has_values = True

    if name == "peak_frequency":
        band_columns = ["icw_fmin_Hz", "icw_fmax_Hz"]
        if set(band_columns).issubset(diagnostics):
            lower, upper = diagnostics[band_columns].to_numpy(dtype=float, na_value=np.nan).T
            supported = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
            if supported.any():
                times = diagnostics["time"]
                ax.fill_between(times, lower, upper, where=supported,
                                color=_PERP_1_COLOR, alpha=0.15, linewidth=0, label="ICW search band", zorder=0)
                isolated = supported & ~(np.r_[False, supported[:-1]] | np.r_[supported[1:], False])
                if isolated.any():
                    ax.vlines(times[isolated], lower[isolated], upper[isolated],
                              color=_PERP_1_COLOR, alpha=0.15, linewidth=4, zorder=0)
        candidates = diagnostics.loc[diagnostics["is_ICW"].eq(True)]
        if not candidates.empty:
            ax.scatter(candidates["time"], candidates["icw_f_peak_Hz"], facecolors="none",
                       edgecolors=_PEAK_COLOR if candidate else _INACTIVE_COLOR,
                       alpha=1 if candidate else 0.55, s=32, lw=0.8, label="Window passes ICW screen")

    if name in ("polarization", "compressibility"):
        ax.set(ylim=(-0.02, 1.04), yticks=[0, 0.5, 1])
    elif name == "wave_normal":
        ax.set(ylim=(0, 90), yticks=[0, 30, 60, 90])
    elif name == "mirror_amplitude" and gate_labels:
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.12 * max(gate_labels)))
    elif name in ("amplitude", "mirror_amplitude"):
        ax.set_ylim(bottom=0)
    bottom, top = ax.get_ylim()
    for value, labels in gate_labels.items():
        near_bottom = value - bottom < 0.24 * (top - bottom)
        ax.annotate(" · ".join(labels), (1, value), xycoords=ax.get_yaxis_transform(),
                    xytext=(-3, 2 if near_bottom else -4), textcoords="offset points",
                    ha="right", va="bottom" if near_bottom else "top", fontsize="x-small",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.3))

    if not has_values:
        message = "No complete diagnostic window" if diagnostics.empty else "No supported estimates"
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
        if not gate_labels:
            ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    if handles and (len(lines) > 1 or name == "peak_frequency"):
        # Keep the right edge free for the inline gate labels.
        legend_anchor = (0.84, 1) if gate_labels else None
        ax.legend(handles, labels, loc="upper right", frameon=False, ncols=len(handles),
                  bbox_to_anchor=legend_anchor, fontsize="small", handlelength=1.5, columnspacing=0.9)


def _draw_orbit(ax, times, frame):
    try:
        from spiceypy.utils.exceptions import SpiceyError
        from cassini_spice.plots import draw_wave_orbit_context
    except ImportError:
        reason = "SPICE plotting dependencies unavailable"
    else:
        try:
            draw_wave_orbit_context(ax, times.iloc[0], times.iloc[-1], frame=frame)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
                      frameon=False, fontsize="small")
            return
        except (OSError, ValueError, SpiceyError) as error:
            message = str(error).strip().splitlines()
            reason = message[0][:115] if message else type(error).__name__
    ax.set_title("Orbit context", loc="left")
    ax.text(0.5, 0.5, f"Orbit context unavailable\n{reason}", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _draw_spectrum(ax, window, info, *, mirror_focus=False):
    values, timing, failure = window
    if failure is None:
        dt = timing["dt_s"]
        freqs, _, psd = _one_sided_spectra(values, dt)
        if not np.isfinite(psd).all():
            failure = ("psd_failed", "nonfinite_power")
    title = "FFT power spectral density (whole interval)"
    ax.set_title(title, loc="left", fontsize="medium", pad=24 if failure is None else None)
    if failure is not None:
        reason = failure[1].replace("_", " ")
        ax.text(0.5, 0.5, f"FFT unavailable\n{reason.capitalize()}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    magnitude_psd, parallel_psd = psd[1:, :2].T
    transverse_psd = psd[1:, 2] + psd[1:, 3]
    curves = []
    for power, color, width, label in (
        (magnitude_psd, _MIRROR_COLOR, 1.6 if mirror_focus else 1.0, r"$S_{|B|}$"),
        (parallel_psd, "#202329", 1.0, r"$S_\parallel$"),
        (transverse_psd, _PERP_1_COLOR, 1.0 if mirror_focus else 1.6, r"$S_\perp=S_{\perp 1}+S_{\perp 2}$"),
    ):
        line, = ax.plot(freqs[1:], np.where(power > 0, power, np.nan), color=color, lw=width, label=label)
        curves.append(line)

    resolved_min = info.get("resolved_min_Hz", np.nan)
    usable_max = info.get("usable_max_Hz", np.nan)
    if pd.isna(resolved_min):
        resolved_min = 3 / timing["duration_s"]
    if pd.isna(usable_max):
        usable_max = 1 / (4 * dt)

    regions, markers = [], []
    for mode, name, band_color, peak_color in (
        ("mirror", "mirror", _MIRROR_COLOR, _MIRROR_COLOR),
        ("icw", "ICW", _PERP_1_COLOR, _PEAK_COLOR),
    ):
        low = info.get(f"{mode}_fmin_Hz", np.nan)
        high = info.get(f"{mode}_fmax_Hz", np.nan)
        if np.isfinite(low) and np.isfinite(high) and low < high:
            ax.axvspan(low, high, color=band_color, alpha=0.10, zorder=0)
            regions.append((low, high, f"{name} band", band_color))
        peak = info.get(f"{mode}_f_peak_Hz", np.nan)
        if np.isfinite(peak):
            ax.axvline(peak, ls=":", lw=1.3, color=peak_color)
            markers.append((peak, f"{name} peak", peak_color))

    for low, high, label in (
        (freqs[1], resolved_min, "< 3 cycles"),
        (usable_max, freqs[-1], "< 4 samples/cycle"),
    ):
        if low < high:
            ax.axvspan(low, high, facecolor="#f2f3f5", edgecolor="#b8bec8", linewidth=0,
                       hatch="//", alpha=0.65, zorder=0.1)
            regions.append((low, high, label, "#565f6d"))

    reference = info.get("icw_reference_fc_Hz", np.nan)
    if freqs[1] <= reference <= freqs[-1]:
        ax.axvline(reference, ls="--", lw=1, color=_REFERENCE_COLOR)
        markers.append((reference, r"$f_c$", _REFERENCE_COLOR))
    ax.set(xscale="log", xlim=(freqs[1], freqs[-1]), xlabel="Frequency (Hz)",
           ylabel=r"PSD (nT$^2$/Hz)")
    if np.any(psd[1:] > 0):
        ax.set_yscale("log")
    ax.grid(which="major", alpha=0.18)
    ax.legend(handles=curves, loc="lower right", bbox_to_anchor=(1, 1.01), borderaxespad=0,
              frameon=False, fontsize="small", ncols=3, columnspacing=0.9)

    # Pack labels from left to right in log-frequency coordinates. Mirror
    # labels get first choice of space when the available width is small.
    fmin, fmax = ax.get_xlim()
    log_min = np.log(fmin)
    log_span = np.log(fmax) - log_min
    axis_width_pt = 72 * ax.get_position().width * ax.figure.get_figwidth()
    fontsize = 8
    text_transform = ax.get_xaxis_transform()
    filled_rows = ([], [])
    vertical_regions = []
    for low, high, label, color in regions:
        low, high = max(low, fmin), min(high, fmax)
        if low >= high:
            continue
        text_width = 0.6 * fontsize * len(label) / axis_width_pt
        rotate = text_width > np.log(high / low) / log_span
        if rotate:
            text_width = (fontsize + 3) / axis_width_pt
        if text_width >= 0.96:
            continue
        center = (0.5 * (np.log(low) + np.log(high)) - log_min) / log_span
        half_width = text_width / 2 + 0.02
        center = np.clip(center, half_width, 1 - half_width)
        left, right = center - half_width, center + half_width
        for y, occupied in zip((0.98, 0.88), filled_rows):
            if any(right > a and left < b for a, b in vertical_regions):
                continue
            if rotate and any(right > a and left < b for a, b in filled_rows[1]):
                continue
            if all(right <= a or left >= b for a, b in occupied):
                ax.text(np.exp(log_min + center * log_span), y, label, color=color,
                        ha="center", va="top", fontsize=fontsize, rotation=90 if rotate else 0,
                        clip_on=True, transform=text_transform)
                occupied.append((left, right))
                if rotate:
                    vertical_regions.append((left, right))
                break

    occupied = vertical_regions.copy()
    text_width = (fontsize + 3) / axis_width_pt
    for frequency, label, color in markers:
        if not fmin <= frequency <= fmax:
            continue
        center = (np.log(frequency) - log_min) / log_span
        for left in (center + 0.012, center - 0.012 - text_width):
            right = left + text_width
            if left < 0 or right > 1 or any(right > a and left < b for a, b in occupied):
                continue
            ax.text(np.exp(log_min + left * log_span), 0.77, label, color=color,
                    ha="left", va="top", rotation=90, fontsize=fontsize, clip_on=True,
                    transform=text_transform)
            occupied.append((left, right))
            break


def plot_wave_timeseries(df, *, start_time=None, end_time=None):
    """Return a Matplotlib figure for an optional half-open interval in UTC."""
    data = _select_interval(df, start_time, end_time)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10.5, 5.4), layout="constrained")
    dated = data.loc[data["TIME"].notna()]
    times = dated["TIME"]
    references = _fac_references(dated)
    switches = _fac_switches(references)
    _draw_timeseries(dict(zip(_RAW_ROWS, axes)), dated, references)
    _mark_fac_switches(axes[-1], times, references, switches)
    axes[0].set_title(_event_title(times), loc="left")
    _label_time_axes(axes, times)
    return fig


def _focused_panels(focus, label):
    if focus == "mirror" or (focus == "auto" and label == "mirror_like"):
        return _MIRROR_WAVE_PANELS, "parallel"
    if focus == "auto" and label == "mixed":
        return (*DEFAULT_WAVE_PANELS, "mirror_amplitude"), "transverse"
    return DEFAULT_WAVE_PANELS, "transverse"


def plot_wave_panel(
    df, *, start_time=None, end_time=None, panels=None, show_wavelet=True, figsize=None,
    diagnostic_window_s=120.0, diagnostic_step_s=20.0, orbit_frame="CASSINI_KSM",
    layout="wide", hodogram_interval=None, focus="auto", _classification=None, _include_orbit=False,
    **classifier_options,
):
    """Plot named panels with a shared UTC column and independent detail plots.

    focus="auto" uses the mirror panels for mirror-only candidates and the ICW
    panels otherwise, with mirror amplitude added for mixed candidates. Use
    focus="icw" or "mirror" to choose a layout explicitly. The mirror focus
    shows parallel wavelet power. An explicit panels list overrides the rows.
    Wide defaults put FFT, hodogram and orbit in that order on the right.
    Explicit panels preserve their order within each column. layout="stacked"
    places every panel in one column. Axes labels identify the rows.
    show_wavelet=False selects the original raw and FFT rows by default, or
    removes the wavelet row from explicit panels.

    The hodogram uses unfiltered FAC perturbations and defaults to a centered
    40 s excerpt. Pass hodogram_interval=(start, stop) for a different half-open
    UTC excerpt, (None, None) for the full interval, or "peak_rms" for six
    peak-frequency cycles (at least 40 s) around the highest transverse RMS
    window. An unavailable peak uses the centered 40 s excerpt. The path
    breaks at gaps and reference switches. Unknown references appear as points.
    Supply half-open UTC bounds to select data. Extra keyword options are
    forwarded to the classifier for whole-event and sliding diagnostics.
    Dipole references obtain missing positions from local SPICE kernels.
    """
    if focus not in ("auto", "icw", "mirror"):
        raise ValueError("focus must be 'auto', 'icw', or 'mirror'")
    if df.empty:
        fig = plt.figure(figsize=figsize or (10.5, 1.6))
        _draw_classification_header(fig, "Cassini MAG", {"label": "no_data"})
        return fig

    data = _select_interval(df, start_time, end_time)
    data = _with_dipole_position(
        data, icw_fc_reference=classifier_options.get("icw_fc_reference", DEFAULT_ICW_FC_REFERENCE),
        equatorial_field_nT=classifier_options.get("equatorial_field_nT"),
    )
    # The notebook quick check supplies its whole-event result for these same samples.
    info = _classification
    if info is None:
        options = {**classifier_options, "return_details": True}
        _, _, _, info = check_ICW_mirror(data, **options)
    defaults, wavelet_component = _focused_panels(focus, info["label"])
    if panels is None:
        names = defaults if show_wavelet else (*_RAW_ROWS, "fft")
        if _include_orbit and "orbit" not in names:
            names = (*names, "orbit")
    else:
        names = (panels,) if isinstance(panels, str) else tuple(panels)
    unknown = set(names).difference(_ROW_HEIGHTS)
    if unknown:
        raise ValueError(f"Unknown panels {sorted(unknown)}. Choose from {list(_ROW_HEIGHTS)}")
    if not show_wavelet:
        names = tuple(name for name in names if name != "wavelet")
    if not names or len(names) != len(set(names)):
        raise ValueError("Select one or more distinct panel names")
    if layout not in ("wide", "stacked"):
        raise ValueError("layout must be 'wide' or 'stacked'")

    timeline_names = [name for name in names if name in _TIMELINE_ROWS]
    peak_hodogram = ("hodogram" in names and isinstance(hodogram_interval, str)
                     and hodogram_interval == "peak_rms")
    time_names = tuple(name for name in names if name not in _DETAIL_ROWS)
    detail_names = tuple(name for name in names if name in _DETAIL_ROWS)
    wide = bool(layout == "wide" and time_names and detail_names)
    if wide and panels is None:
        detail_names = tuple(name for name in _DETAIL_ROWS if name in detail_names)
        names = (*time_names, *detail_names)
    groups = (time_names, detail_names) if wide else (names,)
    tallest = max(sum(_ROW_HEIGHTS[name] for name in group) for group in groups)
    if figsize is None:
        figsize = (15.5 if wide else 10.5, max(3.2, 1.75 + tallest * 1.18))
    fig = plt.figure(figsize=figsize, layout="constrained")
    outer = fig.add_gridspec(1, len(groups), width_ratios=(1.95, 1) if wide else (1,))
    slots = {}
    for column, group in enumerate(groups):
        heights = [_ROW_HEIGHTS[name] for name in group]
        if column == 1:
            scale = sum(_ROW_HEIGHTS[name] for name in time_names) / sum(heights)
            heights = [height * scale for height in heights]
        widths = (1, 0.028) if "wavelet" in group else (1,)
        grid = outer[0, column].subgridspec(len(heights), len(widths), height_ratios=heights,
                                          width_ratios=widths)
        for row, name in enumerate(group):
            slots[name] = grid[row, 0]
            if name == "wavelet":
                slots["wavelet_colorbar"] = grid[row, 1]

    dated = data.loc[data["TIME"].notna()]
    times = dated["TIME"]
    references = _fac_references(dated)
    switches = _fac_switches(references)
    title = _event_title(times)
    time_axes = {}
    axes = {}
    for name in names:
        share_time = name in time_names
        anchor = next(iter(time_axes.values()), None) if share_time else None
        axes[name] = fig.add_subplot(slots[name], sharex=anchor, label=name)
        if share_time:
            time_axes[name] = axes[name]

    raw_axes = {name: axes[name] for name in _RAW_ROWS if name in axes}
    if raw_axes:
        _draw_timeseries(raw_axes, dated, references)
    if {"wavelet", "fft"}.intersection(axes):
        window = _prepare_wave_window(data, min_points=4, min_duration_s=0)
    if timeline_names or peak_hodogram:
        diagnostics = icw_window_diagnostics(
            data, window_s=diagnostic_window_s, step_s=diagnostic_step_s, **classifier_options,
        )
        if timeline_names:
            options = {**classifier_options, "return_details": True, "min_duration_s": diagnostic_window_s}
            settings = signature(check_ICW_mirror).bind(data, **options)
            settings.apply_defaults()
            note = f"{diagnostic_window_s:g} s diagnostic windows · {diagnostic_step_s:g} s step"
            for index, name in enumerate(timeline_names):
                _draw_timeline(axes[name], name, diagnostics, info, settings.arguments,
                               note=note if index == 0 else None)
        if peak_hodogram:
            hodogram_interval = _peak_rms_hodogram_interval(diagnostics)
    if "wavelet" in axes:
        mesh = _draw_wavelet(axes["wavelet"], data, window, info, component=wavelet_component)
        if mesh is not None:
            color_ax = fig.add_subplot(slots["wavelet_colorbar"], label="wavelet_colorbar")
            fig.colorbar(mesh, cax=color_ax, label=r"Wavelet power (nT$^2$)", extend="min")
    if "hodogram" in axes:
        selected = _hodogram_selection(times, hodogram_interval)
        excerpt = dated.iloc[selected]
        _draw_hodogram(axes["hodogram"], excerpt, references[selected], _cadence_breaks(times)[selected],
                       is_icw=info["is_ICW"])
        if "transverse" in axes and selected.any() and not selected.all():
            sample_times = excerpt["TIME"]
            axes["transverse"].axvspan(sample_times.iloc[0], sample_times.iloc[-1],
                                      color=_HODOGRAM_SPAN_COLOR, alpha=0.14, zorder=0,
                                      gid="hodogram_interval")
    if "fft" in axes:
        _draw_spectrum(axes["fft"], window, info, mirror_focus=wavelet_component == "parallel")
    if "orbit" in axes:
        _draw_orbit(axes["orbit"], times, orbit_frame)
        if wide and detail_names[-1] == "orbit":
            axes["orbit"].set_anchor("S")
    if time_axes:
        mark_axis = time_axes.get("transverse", next(iter(time_axes.values())))
        _mark_fac_switches(mark_axis, times, references, switches)
        _label_time_axes(list(time_axes.values()), times)
    _draw_classification_header(fig, title, info)
    return fig


@dataclass
class EventCheck:
    """Processed MAG samples, classifier results, and time-series and panel figures."""

    df: pd.DataFrame
    is_icw: bool
    is_mirror: bool
    label: str
    details: dict
    figures: tuple[Figure, ...]


def _print_event_summary(df, label, details, options):
    def display(key, denominator=1):
        value = details.get(key, np.nan)
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = value / denominator
            return "unavailable" if np.isnan(value) else f"{value:.6g}"
        return str(value)

    def print_gates(gates, mode, heading):
        failures = details[f"{mode}_fail_reasons"]
        for gate, title, key, limit in gates:
            verdict = ("fail" if gate in failures else "pass") if details[f"{mode}_eligible"] else "not evaluated"
            print(f"  {title} {display(key)} · requires {limit} · {verdict}")
        print(f"{heading} fail reasons {', '.join(failures) if failures else 'none'}")

    steps = df["TIME"].sort_values().diff().dt.total_seconds()
    cadence = steps[steps.gt(0)].median()
    gaps = int(steps.gt(1.05 * cadence).sum())
    resolution = df.attrs["resolution"]
    reference = details.get("icw_reference", options["icw_fc_reference"])
    status = details["icw_status"]
    eligible = details["icw_eligible"]
    print(f"Samples {len(df)} · resolution {resolution} · gaps {gaps}")
    print(f"ICW reference {reference} · B_eq {display('B_eq_nT')} nT · f_c {display('icw_reference_fc_Hz')} Hz · "
          f"L_dipole_mean {display('L_dipole_mean')} · B_meas_over_B_dip {display('B_meas_over_B_dip')}")
    print(f"ICW band {display('icw_fmin_Hz')} to {display('icw_fmax_Hz')} Hz · "
          f"usable {display('resolved_min_Hz')} to {display('usable_max_Hz')} Hz · "
          f"peak {display('icw_f_peak_Hz')} Hz")
    print(f"Label {label} · ICW {'eligible' if eligible else 'unobservable'} ({status})")

    settings = signature(check_ICW_mirror).bind(df, **options)
    settings.apply_defaults()
    limits = settings.arguments
    gates = (
        ("compressibility", "C_B", "icw_C_B", f"≤ {limits['icw_cb_max']:g}"),
        ("peak_prominence", "Peak prominence", "icw_peak_prominence", f"≥ {limits['icw_peak_prominence_min']:g}"),
        ("svd_status", "SVD status", "icw_svd_status", "estimated"),
        ("svd_ellipticity", "SVD ellipticity", "icw_svd_ellipticity", f"≥ {limits['icw_svd_ellipticity_min']:g}"),
        ("svd_planarity", "SVD planarity", "icw_svd_planarity", f"≥ {limits['icw_planarity_min']:g}"),
        ("polarization_degree", "Polarization degree", "icw_polarization_degree", f"≥ {limits['icw_polarization_min']:g}"),
        ("wave_normal_angle", "Wave-normal angle (°)", "icw_wave_normal_deg", f"≤ {limits['icw_wave_normal_max_deg']:g}"),
        ("rotation_sense", "Rotation sense", "icw_rotation_sense",
         "left" if limits["icw_require_left_handed"] else "any (gate disabled)"),
    )
    print_gates(gates, "icw", "ICW")
    if status in ("lower_edge_below_resolution", "too_few_band_bins", "too_few_background_bins"):
        print("Hint · Lengthen the interval to resolve the ICW band and its background at this f_c.")
    elif status == "upper_edge_above_usable_band":
        if resolution == "sec" or np.isclose(cadence, 1):
            print('Hint · The upper band edge exceeds the 1 s limit. Try resolution="full".')
        else:
            print("Hint · The upper band edge exceeds the usable limit at this cadence. Use faster sampling.")
    elif status in ("insufficient_duration", "too_few_points"):
        print("Hint · Choose a longer interval with continuous MAG coverage.")
    elif status == "outside_dipole_mapping_range":
        print(f'Hint · Mean L exceeds the dipole mapping limit of {limits["icw_max_L"]:g}. '
              'Use icw_fc_reference="local" for a local-field check.')
    mirror = "candidate" if details["is_mirror"] else "no candidate" if details["mirror_eligible"] else "unobservable"
    print(f"Mirror-like {mirror} ({details['mirror_status']})")
    local_fc = details.get("fc_Hz_H2Oplus", np.nan)
    print(f"Mirror band {display('mirror_fmin_Hz')} to {display('mirror_fmax_Hz')} Hz · "
          f"{display('mirror_fmin_Hz', local_fc)} to {display('mirror_fmax_Hz', local_fc)} × local f_c · "
          f"peak {display('mirror_f_peak_Hz')} Hz")
    mirror_gates = (
        ("compressibility", "C_B", "mirror_C_B", f"≥ {limits['mirror_cb_min']:g}"),
        ("absB_amplitude", "|B| RMS / B0", "mirror_fractional_rms_absB",
         f"≥ {limits['mirror_fractional_rms_absB_min']:g}"),
        ("peak_prominence", "Peak prominence", "mirror_peak_prominence", f"≥ {limits['mirror_peak_prominence_min']:g}"),
    )
    print_gates(mirror_gates, "mirror", "Mirror")


def quick_event_check(
    start, stop, *, padding="10min", resolution="auto", icw_fc_reference=DEFAULT_ICW_FC_REFERENCE,
    panels=None, hodogram_interval="peak_rms", focus="auto", show=True, **classifier_options,
):
    """Check one half-open UTC event and return its data, classification and two figures.

    Naive timestamps use UTC. Padding is a timedelta or a duration string.
    FAC processing trims ten minutes from each read edge. The default panel
    includes the orbit and follows the focus options of plot_wave_panel. The
    hodogram, when present, defaults to the strongest window. `show=False`
    leaves display to the caller. Empty reads return a no-data panel only.
    """
    if focus not in ("auto", "icw", "mirror"):
        raise ValueError("focus must be 'auto', 'icw', or 'mirror'")
    bounds = _parse_time_range(start, stop)
    if bounds is None or bounds[0] == bounds[1]:
        raise ValueError("Provide valid UTC timestamps with start earlier than stop")
    start, stop = bounds
    padding = pd.Timedelta(padding)
    if pd.isna(padding) or padding < pd.Timedelta(0):
        raise ValueError("padding must be a nonnegative duration")

    df = read_and_process_data(start - padding, stop + padding, resolution=resolution)
    if not df.empty:
        times = pd.to_datetime(df["TIME"], utc=True).dt.tz_localize(None)
        df = df.loc[times.ge(start) & times.lt(stop)]
    df = df.copy()
    df.attrs.setdefault("resolution", resolution)
    df.attrs["event_bounds_utc"] = bounds
    print(f"{start} to {stop} UTC")
    if df.empty:
        print("No processed MAG samples in this event interval.")
        fig = plot_wave_panel(df, focus=focus)
        if show:
            plt.show()
        return EventCheck(df, False, False, "no_data", {"reason": "no_data"}, (fig,))

    df = _with_dipole_position(
        df, icw_fc_reference=icw_fc_reference,
        equatorial_field_nT=classifier_options.get("equatorial_field_nT"),
    )
    options = {**classifier_options, "icw_fc_reference": icw_fc_reference,
               "min_duration_s": (stop - start).total_seconds(), "return_details": True}
    is_icw, is_mirror, label, details = check_ICW_mirror(df, **options)
    _print_event_summary(df, label, details, options)
    figures = (
        plot_wave_timeseries(df, start_time=start, end_time=stop),
        plot_wave_panel(
            df, start_time=start, end_time=stop,
            panels=panels, hodogram_interval=hodogram_interval, focus=focus,
            _classification=details, _include_orbit=True, **options,
        ),
    )
    if show:
        plt.show()
    return EventCheck(df, is_icw, is_mirror, label, details, figures)
