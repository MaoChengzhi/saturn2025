"""Configurable event panels for field-aligned Cassini MAG data."""

from dataclasses import dataclass
from datetime import timezone
from inspect import signature
import re
from textwrap import fill

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.textpath import TextPath
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd
from scipy.signal import detrend

from data_process import _parse_time_range
from do_wavelet import wavelet
from missions import get_mission
from wave_classification import (
    DEFAULT_ICW_FC_REFERENCE, _DIPOLE_REFERENCES, _cyclotron_freq_hz, _with_dipole_position,
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
_LIMIT_COLOR = "#6b7480"
_WAVELET_LABEL_SIZE = 7.5
_WAVELET_LABEL_GAP = 10.0  # points from the wavelet edge to its reference labels
_RAW_ROWS = ("field", "parallel", "transverse")
_DETAIL_ROWS = ("fft", "hodogram", "orbit")
# The wide layout puts these side by side below the single column of other rows.
_SIDE_ROWS = ("hodogram", "orbit")

# Change the order of this tuple to change the default panel order.
DEFAULT_WAVE_PANELS = (
    *_RAW_ROWS, "wavelet", "fft", "amplitude", "peak_frequency", "polarization", "wave_normal",
    "compressibility", "hodogram",
)
_MIRROR_WAVE_PANELS = (*_RAW_ROWS, "wavelet", "fft", "compressibility", "mirror_amplitude", "orbit")
_ROW_HEIGHTS = {
    "field": 1.0, "parallel": 1.0, "transverse": 1.0, "wavelet": 2.3,
    "amplitude": 0.9, "peak_frequency": 0.9, "polarization": 1.0, "wave_normal": 0.9,
    "compressibility": 0.9, "mirror_amplitude": 0.9,
    "fft": 3.8, "hodogram": 2.9, "orbit": 2.9,
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


def _legend_right(ax, handles=None, labels=None):
    """Put a panel legend in the shared right gutter, beside the axes and clear of the data."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.012, 0.5), borderaxespad=0,
                     frameon=False, fontsize="small", handlelength=1.5)


def _reference_fc_label(info):
    """Label for the ICW reference gyrofrequency, naming how its field was obtained."""
    kind = {"dipole_mapped": "dipole-mapped", "dipole": "dipole", "local": "local",
            "equatorial": "given B_eq"}.get(info.get("icw_reference"))
    return rf"$f_c$ ({kind})" if kind else r"$f_c$"


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
        _legend_right(field)
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
        _legend_right(transverse)
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


def _event_title(times, instrument="Cassini MAG"):
    first, last = (
        time.tz_localize(None).isoformat(sep=" ", timespec="milliseconds").rstrip("0").rstrip(".")
        for time in (times.iloc[0], times.iloc[-1])
    )
    return f"{instrument}   {first} to {last} UTC"


def _label_time_axes(axes, times):
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    bottom = axes[-1]
    if len(times) > 1 and times.iloc[0] < times.iloc[-1]:
        bottom.set_xlim(times.iloc[0], times.iloc[-1])
    locator = mdates.AutoDateLocator(tz=timezone.utc, minticks=5, maxticks=10, interval_multiples=False)
    # Allow 2- and 3-minute and 20-second steps so narrow panels keep several ticks.
    locator.intervald[mdates.MINUTELY] = [1, 2, 3, 5, 10, 15, 30]
    locator.intervald[mdates.SECONDLY] = [1, 2, 5, 10, 15, 20, 30]
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
    outline = [withStroke(linewidth=2.0, foreground="#222222")]
    # Labels sit in the right gutter, joined to their lines; _spread_wavelet_labels separates them.
    label_coords = offset_copy(ax.transAxes, fig=ax.figure, x=_WAVELET_LABEL_GAP, units="points")
    for key, label in _wavelet_references(info, component):
        frequency = info.get(key, np.nan)
        if fmin <= frequency <= fmax:
            ax.axhline(frequency, color="white", ls="--", lw=0.8, zorder=3,
                       path_effects=outline, gid="wavelet_reference")
            ax.annotate(label, xy=(1, frequency), xycoords=ax.get_yaxis_transform(),
                        xytext=(1, np.log(frequency / fmin) / np.log(fmax / fmin)), textcoords=label_coords,
                        ha="left", va="center", color="#222222", fontsize=_WAVELET_LABEL_SIZE,
                        annotation_clip=False, gid="wavelet_reference_label",
                        arrowprops={"arrowstyle": "-", "color": _LIMIT_COLOR, "linewidth": 0.7,
                                    "shrinkA": 1.5, "shrinkB": 0})
    return mesh


def _wavelet_references(info, component):
    """Return (frequency key, label) pairs for the wavelet reference lines."""
    def ratio(key, reference):
        value = info.get(key, np.nan) / reference
        return f"{value:.2g}" if np.isfinite(value) else "?"

    if component == "parallel":
        local = info.get("fc_Hz_local", np.nan)
        raised = info.get("mirror_fmin_Hz") != info.get("mirror_nominal_fmin_Hz")
        low = "Mirror low (3-cycle limit)" if raised else rf"Mirror low ({ratio('mirror_fmin_Hz', local)} local $f_c$)"
        return (("mirror_fmin_Hz", low),
                ("mirror_fmax_Hz", rf"Mirror high ({ratio('mirror_fmax_Hz', local)} local $f_c$)"))
    reference = info.get("icw_reference_fc_Hz", np.nan)
    return (("icw_fmin_Hz", rf"ICW low ({ratio('icw_fmin_Hz', reference)} $f_c$)"),
            ("icw_reference_fc_Hz", _reference_fc_label(info)),
            ("icw_fmax_Hz", rf"ICW high ({ratio('icw_fmax_Hz', reference)} $f_c$)"))


def _wavelet_labels(ax):
    return [text for text in ax.texts if text.get_gid() == "wavelet_reference_label"]


def _wavelet_label_room(labels):
    """Return the gutter width in points that the wavelet labels need before the colorbar."""
    if not labels:
        return 0.0
    widths = [TextPath((0, 0), label.get_text(), size=_WAVELET_LABEL_SIZE).get_extents().width
              for label in labels]
    return _WAVELET_LABEL_GAP + max(widths) + 2.0


def _spread_wavelet_labels(ax, labels):
    """Space the wavelet labels at least one text line apart, in frequency order.

    Overlapping labels are centered on their lines' mean height and kept beside
    the axes. Call after the figure layout is final.
    """
    if not labels:
        return
    fmin, fmax = ax.get_ylim()
    desired = np.array([np.log(label.xy[1] / fmin) / np.log(fmax / fmin) for label in labels])
    height_pt = 72 * ax.get_position().height * ax.figure.get_figheight()
    gap = (_WAVELET_LABEL_SIZE + 2.5) / height_pt

    def start(cluster):
        return np.mean(cluster) - gap * (len(cluster) - 1) / 2

    order = np.argsort(desired)
    clusters = []
    for value in desired[order]:
        clusters.append([value])
        while len(clusters) > 1 and start(clusters[-1]) - (start(clusters[-2]) + gap * (len(clusters[-2]) - 1)) < gap:
            clusters[-2:] = [clusters[-2] + clusters[-1]]
    positions = np.concatenate([start(cluster) + gap * np.arange(len(cluster)) for cluster in clusters])
    positions[-1] = min(positions[-1], 1.0)
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - gap)
    positions[0] = max(positions[0], 0.0)
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)
    for index, position in zip(order, positions):
        labels[index].xyann = (1, position)


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
    # Short statuses fit beside the title in the narrow column; the header gives the reasons.
    if candidate:
        status = f"{mode_name} candidate"
    elif info[f"{mode}_eligible"]:
        status = f"no {mode_name} candidate"
    else:
        status = f"{mode_name} not applicable"
    ax.set_title(f"{title}\n{note}" if note else title, loc="left", fontsize="medium")
    # With a two-line title, keep the status on the first line so it clears the note.
    ax.set_title(f"{status}\n" if note else status, loc="right", fontsize="small")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)

    # Gate lines are named in the legend rather than labelled over the data.
    gate_handles = []
    for key, label, multiplier, color in _TIMELINE_GATES.get(name, ()):
        value = limits[key] * multiplier
        if np.isfinite(value):
            gate_handles.append(ax.axhline(value, ls="--", lw=0.8, color=color, alpha=0.65, zorder=1,
                                           label=label.format(value=value)))
    gate_values = [handle.get_ydata()[0] for handle in gate_handles]

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
        # Expected frequency: the whole-interval ICW reference gyrofrequency, as on the FFT panel.
        reference = info.get("icw_reference_fc_Hz", np.nan)
        if np.isfinite(reference):
            ax.axhline(reference, ls="--", lw=1, color=_REFERENCE_COLOR, zorder=1.5,
                       label=_reference_fc_label(info))

    if name in ("polarization", "compressibility"):
        ax.set(ylim=(-0.02, 1.04), yticks=[0, 0.5, 1])
    elif name == "wave_normal":
        ax.set(ylim=(0, 90), yticks=[0, 30, 60, 90])
    elif name == "mirror_amplitude" and gate_values:
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.12 * max(gate_values)))
    elif name in ("amplitude", "mirror_amplitude"):
        ax.set_ylim(bottom=0)

    if not has_values:
        message = "No complete diagnostic window" if diagnostics.empty else "No supported estimates"
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
        if not gate_values:
            ax.set_yticks([])
    # Single-series panels name only their gates; the y label names the series.
    handles, labels = ax.get_legend_handles_labels()
    entries = [(handle, label) for handle, label in zip(handles, labels) if handle not in gate_handles]
    if not (len(lines) > 1 or name == "peak_frequency"):
        entries = []
    entries += [(handle, handle.get_label()) for handle in gate_handles]
    if entries:
        _legend_right(ax, *map(list, zip(*entries)))


def _draw_orbit(ax, times, frame, mission=None):
    try:
        from spiceypy.utils.exceptions import SpiceyError
        draw_wave_orbit_context = get_mission(mission).hook("orbit_drawer")
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


def _ion_tex(ion):
    """Mathtext for an ion label: SO2+ gives SO₂⁺ and O+/S++ gives O⁺/S⁺⁺."""
    parts = []
    for part in str(ion).split("/"):
        body, charge = re.fullmatch(r"(.*?)(\+*)", part).groups()
        body = re.sub(r"(\d+)", r"_{\1}", body)
        parts.append(rf"\mathrm{{{body}}}" + (f"^{{{charge}}}" if charge else ""))
    return "/".join(parts)


def _draw_spectrum(ax, window, info, *, mirror_focus=False, fc_lines=()):
    """Draw the FFT panel and return its vertical marks for _place_spectrum_labels.

    fc_lines is a sequence of (ion, frequency_Hz) gyrofrequency marks. Bands and
    unresolved frequencies go in the legend below the axes; the marks are
    labelled above the axes once the figure layout is known.
    """
    values, timing, failure = window
    if failure is None:
        dt = timing["dt_s"]
        freqs, _, psd = _one_sided_spectra(values, dt)
        if not np.isfinite(psd).all():
            failure = ("psd_failed", "nonfinite_power")
    ax.set_title("FFT power spectral density (whole interval)", loc="left", fontsize="medium")
    if failure is not None:
        reason = failure[1].replace("_", " ")
        ax.text(0.5, 0.5, f"FFT unavailable\n{reason.capitalize()}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return []

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
    fmin, fmax = freqs[1], freqs[-1]

    marks = []

    def mark(frequency, label, color, style, width, priority):
        # Lower priority values are labelled first and sit closest to the axes.
        if np.isfinite(frequency) and fmin <= frequency <= fmax:
            ax.axvline(frequency, color=color, ls=style, lw=width, zorder=1.5)
            marks.append({"frequency": frequency, "label": label, "color": color,
                          "style": style, "width": width, "priority": priority})

    ion = info.get("ion_label")
    ion_text = rf" ${_ion_tex(ion)}$" if ion else ""
    bands = []
    for mode, name, band_color, peak_color in (
        ("icw", "ICW", _PERP_1_COLOR, _PEAK_COLOR),
        ("mirror", "Mirror", _MIRROR_COLOR, _MIRROR_COLOR),
    ):
        low = info.get(f"{mode}_fmin_Hz", np.nan)
        high = info.get(f"{mode}_fmax_Hz", np.nan)
        if np.isfinite(low) and np.isfinite(high) and low < high:
            ax.axvspan(low, high, color=band_color, alpha=0.10, zorder=0)
            band_name = f"{name}{ion_text}" if mode == "icw" else name
            bands.append(Patch(facecolor=band_color, alpha=0.25,
                               label=f"{band_name} band {low:.3g}–{high:.3g} Hz"))
        peak = info.get(f"{mode}_f_peak_Hz", np.nan)
        mark(peak, f"{name} peak {peak:.3g} Hz", peak_color, ":", 1.3, 0 if mode == "icw" else 1)

    hatch_style = {"facecolor": "#f2f3f5", "edgecolor": "#b8bec8", "hatch": "//"}
    unresolved = False
    for low, high in ((fmin, resolved_min), (usable_max, fmax)):
        if low < high:
            ax.axvspan(low, high, linewidth=0, alpha=0.65, zorder=0.1, **hatch_style)
            unresolved = True
    if unresolved:
        bands.append(Patch(**hatch_style, label="Unresolved: < 3 cycles or < 4 samples/cycle"))

    reference = info.get("icw_reference_fc_Hz", np.nan)
    mark(reference, _reference_fc_label(info), _REFERENCE_COLOR, "--", 1.0, 2)
    gyro_style = {"color": _INACTIVE_COLOR, "ls": (0, (1, 2)), "lw": 0.9}
    for line_ion, frequency in fc_lines:
        if not np.isclose(frequency, reference):
            mark(frequency, rf"${_ion_tex(line_ion)}$", gyro_style["color"], gyro_style["ls"],
                 gyro_style["lw"], 4)
    if any(item["priority"] == 4 for item in marks):
        bands.append(Line2D([], [], **gyro_style, label="Local gyrofrequency of each ion"))
    for frequency, label in ((resolved_min, "3-cycle limit"), (usable_max, "4 samples/cycle"),
                             (fmax, "Nyquist")):
        mark(frequency, label, _LIMIT_COLOR, "-", 0.8, 3)

    ax.set(xscale="log", xlim=(fmin, fmax), xlabel="Frequency (Hz)", ylabel=r"PSD (nT$^2$/Hz)")
    if np.any(psd[1:] > 0):
        ax.set_yscale("log")
    ax.grid(which="major", alpha=0.18)

    # Legend columns fill top to bottom, so interleave to read curves on row 1, bands on row 2.
    entries = [*curves, *bands]
    rows = -(-len(entries) // 3)
    order = sorted(range(len(entries)), key=lambda index: (index % 3, index // 3))
    ax.legend(handles=[entries[index] for index in order], ncols=3 if rows > 1 else len(entries),
              loc="upper center", bbox_to_anchor=(0.5, 0), frameon=False, fontsize="small",
              bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, y=-36, units="points"),
              columnspacing=1.4, handlelength=1.8)
    return marks


def _place_spectrum_labels(ax, marks, *, fontsize=8):
    """Extend the FFT marks above the axes and label each at its tip.

    Labels sit in tiers above the axes, to the right of their line where they
    fit. Working from the right, a mark takes the lowest tier where its label
    overlaps no other label, every line under its label is shorter, and its own
    line passes under no label. Call after the figure layout is final.
    """
    if not marks:
        return
    fmin, fmax = ax.get_xlim()
    width_pt = 72 * ax.get_position().width * ax.figure.get_figwidth()
    log_span = np.log(fmax / fmin)
    base, step, gap, pad, clearance, max_tier = 6.0, fontsize + 3.0, 2.5, 4.0, 1.5, 10
    labels = [{**item, "x": np.log(item["frequency"] / fmin) / log_span * width_pt,
               "text_width": TextPath((0, 0), item["label"], size=fontsize).get_extents().width}
              for item in marks]

    def span(label, on_right):
        x, text_width = label["x"], label["text_width"]
        return (x + gap, x + gap + text_width) if on_right else (x - gap - text_width, x - gap)

    def under(bounds, x):
        return bounds[0] - clearance <= x <= bounds[1] + clearance

    # Face right unless that covers another line while the left side is clear, or it leaves the axes.
    for label in labels:
        right_span, left_span = span(label, True), span(label, False)
        others = [other["x"] for other in labels if other is not label]
        fits_right, fits_left = right_span[1] <= width_pt, left_span[0] >= 0
        crowded_right = any(under(right_span, x) for x in others)
        clear_left = fits_left and not any(under(left_span, x) for x in others)
        label["on_right"] = fits_right and not (crowded_right and clear_left) or not fits_left
        label["span"] = span(label, label["on_right"])
        label["tier"] = 0

    # Raise labels until every line under a label is shorter and labels in a tier do not overlap.
    # Of two overlapping labels the left one moves up; at the same frequency, the less important one.
    for index, label in enumerate(labels):
        label["order"] = (label["x"], -label["priority"], index)
    for _ in range(10 * len(labels)):
        changed = False
        for line in labels:
            for label in labels:
                if label is line or label["tier"] >= max_tier:
                    continue
                if under(label["span"], line["x"]) and label["tier"] <= line["tier"]:
                    label["tier"], changed = line["tier"] + 1, True
                elif (label["tier"] == line["tier"] and label["order"] < line["order"]
                      and label["span"][1] + pad > line["span"][0] and label["span"][0] - pad < line["span"][1]):
                    label["tier"], changed = label["tier"] + 1, True
        if not changed:
            break

    for label in labels:
        height = base + label["tier"] * step
        anchor = {"xy": (label["frequency"], 1), "xycoords": ax.get_xaxis_transform(),
                  "textcoords": "offset points", "annotation_clip": False}
        ax.annotate("", xytext=(0, height), arrowprops={
            "arrowstyle": "-", "shrinkA": 0, "shrinkB": 0, "color": label["color"],
            "linestyle": label["style"], "linewidth": label["width"]}, **anchor)
        ax.annotate(label["label"], xytext=(gap if label["on_right"] else -gap, height),
                    ha="left" if label["on_right"] else "right", va="center", fontsize=fontsize,
                    color=label["color"], **anchor)
    top = base + max(label["tier"] for label in labels) * step + fontsize
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize="medium", pad=top + 4)


def plot_wave_timeseries(df, *, start_time=None, end_time=None, mission=None):
    """Return a Matplotlib figure for an optional half-open interval in UTC."""
    mission = get_mission(mission)
    data = _select_interval(df, start_time, end_time)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10.5, 5.4), layout="constrained")
    dated = data.loc[data["TIME"].notna()]
    times = dated["TIME"]
    references = _fac_references(dated)
    switches = _fac_switches(references)
    _draw_timeseries(dict(zip(_RAW_ROWS, axes)), dated, references)
    _mark_fac_switches(axes[-1], times, references, switches)
    axes[0].set_title(_event_title(times, f"{mission.name} MAG"), loc="left")
    _label_time_axes(axes, times)
    return fig


def _mission_classifier_options(mission, options, ion=None):
    """Fill the mission's ion and field reference into classifier options unless given."""
    options = dict(options)
    if ion is not None or "ion_mass_amu" not in options:
        ion = ion or mission.default_ion
        options["ion_mass_amu"] = mission.ion_mass(ion)
        options["ion_label"] = ion
    options.setdefault("icw_fc_reference", mission.icw_fc_reference)
    if (mission.position_lookup is None and options["icw_fc_reference"] in _DIPOLE_REFERENCES
            and options.get("equatorial_field_nT") is None):
        raise ValueError(f"{mission.name} has no dipole position lookup. Use "
                         'icw_fc_reference="local" or pass equatorial_field_nT.')
    return options


def _focused_panels(focus, label):
    if focus == "mirror" or (focus == "auto" and label == "mirror_like"):
        return _MIRROR_WAVE_PANELS, "parallel"
    if focus == "auto" and label == "mixed":
        return (*DEFAULT_WAVE_PANELS, "mirror_amplitude"), "transverse"
    return DEFAULT_WAVE_PANELS, "transverse"


def plot_wave_panel(
    df, *, start_time=None, end_time=None, panels=None, show_wavelet=True, figsize=None,
    diagnostic_window_s=120.0, diagnostic_step_s=20.0, orbit_frame=None,
    layout="wide", hodogram_interval=None, focus="auto", _classification=None, _include_orbit=False,
    mission=None, **classifier_options,
):
    """Plot named panels with a shared UTC column and independent detail plots.

    focus="auto" uses the mirror panels for mirror-only candidates and the ICW
    panels otherwise, with mirror amplitude added for mixed candidates. Use
    focus="icw" or "mirror" to choose a layout explicitly. The mirror focus
    shows parallel wavelet power. An explicit panels list overrides the rows.
    The wide layout puts every row in one column, with the hodogram and orbit
    side by side in a bottom row. Explicit panels keep their order within the
    column and the bottom row. layout="stacked" also stacks the hodogram and
    orbit in the column. Axes labels identify the rows.
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
    mission ("cassini" by default, or "juno") sets the title, the orbit panel
    and its default frame (orbit_frame=None), and the ion gyrofrequencies
    marked on the FFT panel.
    """
    if focus not in ("auto", "icw", "mirror"):
        raise ValueError("focus must be 'auto', 'icw', or 'mirror'")
    mission = get_mission(mission)
    instrument = f"{mission.name} MAG"
    orbit_frame = orbit_frame or mission.orbit_frame
    classifier_options = _mission_classifier_options(mission, classifier_options)
    if df.empty:
        fig = plt.figure(figsize=figsize or (10.5, 1.6))
        _draw_classification_header(fig, instrument, {"label": "no_data"})
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
    main_names = tuple(name for name in names if name not in _SIDE_ROWS)
    side_names = tuple(name for name in names if name in _SIDE_ROWS)
    wide = bool(layout == "wide" and main_names and side_names)
    if wide and panels is None:
        side_names = tuple(name for name in _SIDE_ROWS if name in side_names)
        names = (*main_names, *side_names)
    # One column of rows; the wide layout puts the hodogram and orbit side by side underneath.
    column = main_names if wide else names
    column_height = sum(_ROW_HEIGHTS[name] for name in column)
    side_height = max((_ROW_HEIGHTS[name] for name in side_names), default=0) if wide else 0
    if figsize is None:
        figsize = (14.0 if wide else 10.5, max(3.2, 1.75 + (column_height + side_height) * 1.18))
    fig = plt.figure(figsize=figsize, layout="constrained")
    outer = fig.add_gridspec(2 if wide else 1, 1,
                             height_ratios=(column_height, side_height) if wide else (1,))
    # Legends and the wavelet colorbar share the right gutter beside this column.
    grid = outer[0, 0].subgridspec(len(column), 1, height_ratios=[_ROW_HEIGHTS[name] for name in column])
    slots = {name: grid[row, 0] for row, name in enumerate(column)}
    if wide:
        side_grid = outer[1, 0].subgridspec(1, len(side_names))
        for index, name in enumerate(side_names):
            slots[name] = side_grid[0, index]

    dated = data.loc[data["TIME"].notna()]
    times = dated["TIME"]
    references = _fac_references(dated)
    switches = _fac_switches(references)
    title = _event_title(times, instrument)
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
    wavelet_labels = []
    if "wavelet" in axes:
        mesh = _draw_wavelet(axes["wavelet"], data, window, info, component=wavelet_component)
        if mesh is not None:
            # The colorbar sits right of the reference labels in the gutter.
            wavelet_labels = _wavelet_labels(axes["wavelet"])
            bar_coords = offset_copy(axes["wavelet"].transAxes, fig=fig,
                                     x=_wavelet_label_room(wavelet_labels), units="points")
            color_ax = axes["wavelet"].inset_axes([1.012, 0, 0.016, 1], transform=bar_coords,
                                                  label="wavelet_colorbar")
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
    spectrum_marks = []
    if "fft" in axes:
        mean_field = data["BTOTAL"].mean()
        fc_lines = [(ion, _cyclotron_freq_hz(mean_field, mission.ions[ion]))
                    for ion in mission.fc_line_ions]
        spectrum_marks = _draw_spectrum(axes["fft"], window, info,
                                        mirror_focus=wavelet_component == "parallel", fc_lines=fc_lines)
    if "orbit" in axes:
        _draw_orbit(axes["orbit"], times, orbit_frame, mission)
    if time_axes:
        mark_axis = time_axes.get("transverse", next(iter(time_axes.values())))
        _mark_fac_switches(mark_axis, times, references, switches)
        # Label the bottom of each unbroken run of time rows, e.g. the wavelet above the FFT.
        run = []
        for name in (*column, None):
            if name in time_axes:
                run.append(axes[name])
            elif run:
                _label_time_axes(run, times)
                run = []
    _draw_classification_header(fig, title, info)
    if spectrum_marks or wavelet_labels:
        # Measure the final axes before packing their labels.
        fig.get_layout_engine().execute(fig)
        if spectrum_marks:
            _place_spectrum_labels(axes["fft"], spectrum_marks)
        if wavelet_labels:
            if spectrum_marks:
                fig.get_layout_engine().execute(fig)  # The FFT title pad changes the row heights.
            _spread_wavelet_labels(axes["wavelet"], wavelet_labels)
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
    local_fc = details.get("fc_Hz_local", details.get("fc_Hz_H2Oplus", np.nan))
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
    start, stop, *, padding="10min", resolution=None, icw_fc_reference=None,
    panels=None, hodogram_interval="peak_rms", focus="auto", show=True,
    mission=None, ion=None, reader_options=None, **classifier_options,
):
    """Check one half-open UTC event and return its data, classification and two figures.

    Naive timestamps use UTC. Padding is a timedelta or a duration string.
    FAC processing trims ten minutes from each read edge. The default panel
    includes the orbit and follows the focus options of plot_wave_panel. The
    hodogram, when present, defaults to the strongest window. `show=False`
    leaves display to the caller. Empty reads return a no-data panel only.

    mission is "cassini" (default) or "juno". ion selects the ICW ion from the
    mission's list (default H2O+ for Cassini, S+ for Juno); resolution and
    icw_fc_reference default to the mission's settings ("auto"; Cassini
    "dipole_mapped", Juno "local"). reader_options go to the mission's data
    reader, e.g. {"background_s": 300} for a shorter FAC background.
    """
    if focus not in ("auto", "icw", "mirror"):
        raise ValueError("focus must be 'auto', 'icw', or 'mirror'")
    mission = get_mission(mission)
    resolution = resolution or mission.default_resolution
    if resolution not in mission.resolutions:
        raise ValueError(f"resolution for {mission.name} must be one of {mission.resolutions}")
    if icw_fc_reference is not None:
        classifier_options["icw_fc_reference"] = icw_fc_reference
    classifier_options = _mission_classifier_options(mission, classifier_options, ion)
    icw_fc_reference = classifier_options.pop("icw_fc_reference")
    bounds = _parse_time_range(start, stop)
    if bounds is None or bounds[0] == bounds[1]:
        raise ValueError("Provide valid UTC timestamps with start earlier than stop")
    start, stop = bounds
    padding = pd.Timedelta(padding)
    if pd.isna(padding) or padding < pd.Timedelta(0):
        raise ValueError("padding must be a nonnegative duration")

    reader = mission.hook("reader")
    reader_options = dict(reader_options or {})
    if "ion_mass_amu" in signature(reader).parameters:
        reader_options.setdefault("ion_mass_amu", classifier_options["ion_mass_amu"])
    df = reader(start - padding, stop + padding, resolution=resolution, **reader_options)
    if not df.empty:
        times = pd.to_datetime(df["TIME"], utc=True).dt.tz_localize(None)
        df = df.loc[times.ge(start) & times.lt(stop)]
    df = df.copy()
    df.attrs.setdefault("resolution", resolution)
    df.attrs["event_bounds_utc"] = bounds
    print(f"{start} to {stop} UTC")
    if df.empty:
        print("No processed MAG samples in this event interval.")
        fig = plot_wave_panel(df, focus=focus, mission=mission)
        if show:
            plt.show()
        return EventCheck(df, False, False, "no_data", {"reason": "no_data"}, (fig,))

    df = _with_dipole_position(
        df, icw_fc_reference=icw_fc_reference,
        equatorial_field_nT=classifier_options.get("equatorial_field_nT"),
    )
    options = {**classifier_options, "icw_fc_reference": icw_fc_reference,
               "min_duration_s": (stop - start).total_seconds() - mission.duration_slack_s,
               "return_details": True}
    is_icw, is_mirror, label, details = check_ICW_mirror(df, **options)
    _print_event_summary(df, label, details, options)
    figures = (
        plot_wave_timeseries(df, start_time=start, end_time=stop, mission=mission),
        plot_wave_panel(
            df, start_time=start, end_time=stop,
            panels=panels, hodogram_interval=hodogram_interval, focus=focus,
            _classification=details, _include_orbit=True, mission=mission, **options,
        ),
    )
    if show:
        plt.show()
    return EventCheck(df, is_icw, is_mirror, label, details, figures)
