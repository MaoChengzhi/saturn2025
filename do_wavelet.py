"""Continuous wavelet analysis and the standalone wavelet plot.

The Fourier-domain transform and significance formulas follow Torrence and
Compo (1998), using the same normalization as the repository's original
Predybaylo Python port. Wavelet power is in squared input units.
"""

# Wavelet software originally provided by C. Torrence and G. P. Compo.
# Copyright (C) 1995-2004, Christopher Torrence and Gilbert P. Compo.
# Original Python port by Evgeniya Predybaylo (2014).
# The original software may be used, copied, or redistributed if it is not
# sold and this notice is reproduced. It is provided without warranties.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft, signal
from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy.stats import chi2


def _wavelet_parameters(mother, param):
    name = mother.upper() if mother != -1 else "MORLET"
    if name == "MORLET":
        k0 = 6.0 if param in (None, -1) else float(param)
        if not np.isfinite(k0) or k0 <= 0:
            raise ValueError("Morlet frequency must be positive and finite")
        factor = 4 * np.pi / (k0 + np.sqrt(2 + k0**2))
        empirical = (0.776, 2.32, 0.60) if k0 == 6 else None
        return name, k0, factor, factor / np.sqrt(2), 2, empirical
    if name not in ("PAUL", "DOG"):
        raise ValueError("mother must be MORLET, PAUL, or DOG")
    order = (4 if name == "PAUL" else 2) if param in (None, -1) else param
    if not np.isfinite(order) or order < 1 or order != int(order):
        raise ValueError("Paul and DOG orders must be positive integers")
    order = int(order)
    if name == "PAUL":
        factor = 4 * np.pi / (2 * order + 1)
        empirical = (1.132, 1.17, 1.5) if order == 4 else None
        return name, order, factor, factor * np.sqrt(2), 2, empirical
    factor = 2 * np.pi * np.sqrt(2 / (2 * order + 1))
    empirical = {2: (3.541, 1.43, 1.4), 6: (1.966, 1.37, 0.97)}.get(order)
    return name, order, factor, factor / np.sqrt(2), 1, empirical


def _daughter(name, param, k, scale, dt):
    sk = scale * k
    norm = np.sqrt(2 * np.pi * scale / dt)
    if name == "MORLET":
        return norm * np.pi**(-0.25) * np.exp(-0.5 * (sk - param)**2) * (k > 0)
    if name == "PAUL":
        positive = np.maximum(sk, 0)
        norm *= 2.0**param / np.sqrt(param * gamma(2 * param))
        return norm * positive**param * np.exp(-positive)
    norm *= -(1j**param) / np.sqrt(gamma(param + 0.5))
    return norm * sk**param * np.exp(-0.5 * sk**2)


def wavelet(Y, dt, pad=1, dj=-1, s0=-1, J1=-1, mother="MORLET", param=-1, *, freq=None):
    """Return coefficients, Fourier periods, scales, and cone-of-influence periods.

    Input samples occupy the last axis of Y and must already be on a uniform
    grid. Coefficients have shape (n_scales, *Y.shape) for any input parity.
    Frequencies are in cycles per unit of dt and retain their supplied order.
    Periods above coi[t] are affected by the finite record at sample t.
    """
    values = np.asarray(Y, dtype=float)
    if values.ndim == 0 or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError("Wavelet input needs at least two finite samples per series")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if pad not in (0, 1):
        raise ValueError("pad must be 0 or 1")

    name, parameter, factor, coi_factor, _, _ = _wavelet_parameters(mother, param)
    count = values.shape[-1]
    if freq is None:
        s0 = 2 * dt if s0 == -1 else s0
        dj = 0.25 if dj == -1 else dj
        if not np.isfinite([s0, dj]).all() or s0 <= 0 or dj <= 0:
            raise ValueError("s0 and dj must be positive and finite")
        if J1 == -1:
            J1 = max(0, np.floor(np.log2(count * dt / s0) / dj))
        if not np.isfinite(J1) or J1 < 0 or J1 != int(J1):
            raise ValueError("J1 must be a nonnegative integer")
        scales = s0 * 2.0**(np.arange(int(J1) + 1) * dj)
        periods = scales * factor
    else:
        frequencies = np.atleast_1d(np.asarray(freq, dtype=float))
        if (frequencies.ndim != 1 or not frequencies.size
                or not np.isfinite(frequencies).all()
                or np.any((frequencies <= 0) | (frequencies > 0.5 / dt))):
            raise ValueError("freq must contain positive frequencies up to Nyquist")
        periods = 1 / frequencies
        scales = periods / factor

    # Retain the original Torrence–Compo padding and positive Nyquist convention.
    length = 2 ** (int(np.log2(count) + 0.4999) + 1) if pad else count
    k = 2 * np.pi * fft.fftfreq(length, d=dt)
    if length % 2 == 0:
        k[length // 2] = np.pi / dt
    spectrum = fft.fft(values - values.mean(axis=-1, keepdims=True), n=length, axis=-1)
    coefficients = np.empty((len(scales), *values.shape), dtype=complex)
    for row, scale in enumerate(scales):
        kernel = _daughter(name, parameter, k, scale, dt)
        coefficients[row] = fft.ifft(spectrum * kernel, axis=-1)[..., :count]

    samples = np.arange(count)
    coi = coi_factor * dt * np.minimum(samples, count - 1 - samples)
    return coefficients, periods, scales, coi


def wave_signif(Y, dt, scale, sigtest=0, lag1=0.0, siglvl=0.95,
                dof=None, mother="MORLET", param=None, gws=None):
    """Torrence–Compo local, time-averaged or scale-averaged power threshold."""
    _, _, factor, _, base_dof, empirical = _wavelet_parameters(mother, param)
    scales = np.asarray(scale, dtype=float)
    if (scales.ndim != 1 or not scales.size or not np.isfinite(scales).all()
            or np.any(scales <= 0) or not np.isfinite(dt) or dt <= 0):
        raise ValueError("Scale and dt must be positive and finite")
    if not -1 < lag1 < 1 or not 0 < siglvl < 1:
        raise ValueError("Require abs(lag1) < 1 and 0 < siglvl < 1")
    values = np.ravel(Y)
    variance = values[0] if values.size == 1 else np.var(values)
    if gws is None:
        phase = 2 * np.pi * dt / (factor * scales)
        background = variance * (1 - lag1**2) / (1 - 2 * lag1 * np.cos(phase) + lag1**2)
    else:
        background = np.broadcast_to(np.asarray(gws, dtype=float), scales.shape)

    if sigtest == 0:  # Equation (18).
        return background * chi2.ppf(siglvl, base_dof) / base_dof
    if empirical is None:
        raise ValueError("Averaged significance requires a calibrated mother/parameter pair")
    reconstruction, decorrelation, scale_width = empirical
    if sigtest == 1:  # Equation (23). dof is a number of time samples.
        average_count = np.maximum(1, base_dof if dof is None else dof)
        effective_dof = base_dof * np.sqrt(1 + (average_count * dt / (decorrelation * scales))**2)
        return background * chi2.ppf(siglvl, effective_dof) / effective_dof
    if sigtest != 2 or dof is None or len(dof) != 2:
        raise ValueError("sigtest 2 requires a two-element scale interval in dof")
    lower, upper = dof
    selected = (scales >= lower) & (scales <= upper)
    if lower <= 0 or lower >= upper or scales.size < 2 or not selected.any():
        raise ValueError("No wavelet scales in the requested scale interval")
    spacing = np.diff(np.log2(scales))
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0]):
        raise ValueError("Scale-averaged significance requires increasing logarithmic scales")
    inverse_sum = 1 / np.sum(1 / scales[selected])
    count = np.count_nonzero(selected)
    effective_dof = (base_dof * count * inverse_sum / np.sqrt(lower * upper)
                     * np.sqrt(1 + (count * spacing[0] / scale_width)**2))
    # Equations (25)–(28), with the scale interval supplied by the caller.
    band_background = np.sum(background[selected] / scales[selected])
    return (spacing[0] * dt / reconstruction * band_background
            * chi2.ppf(siglvl, effective_dof) / effective_dof)


def _uniform_time(time, count, dt):
    if time is None:
        return np.arange(count) * dt
    times = np.asarray(time)
    if times.shape != (count,):
        raise ValueError("time must have one value per sample")
    if np.issubdtype(times.dtype, np.number):
        offsets = times - times[0]
    else:
        times = pd.DatetimeIndex(pd.to_datetime(time, utc=True))
        offsets = (times - times[0]).total_seconds().to_numpy()
    spacing = np.diff(offsets)
    if (not np.isfinite(offsets).all() or np.any(spacing <= 0)
            or np.any(np.abs(spacing - dt) > 0.05 * dt)
            or np.any(np.abs(offsets - dt * np.arange(count)) > 0.1 * dt)):
        raise ValueError("time must be a continuous uniform grid matching dt")
    return times


def _gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-0.5 * ((x - center) / width)**2)


def _global_peaks(periods, power, threshold, refine):
    peaks, _ = signal.find_peaks(power)
    peaks = peaks[power[peaks] > threshold[peaks]]
    if peaks.size == 0:
        return [], np.array([]), np.array([])
    _, _, left, right = signal.peak_widths(power, peaks, rel_height=1.0)
    indices = np.arange(len(periods))
    starts, ends = (np.interp(edge, indices, periods) for edge in (left, right))
    centers = periods[peaks].tolist()
    if refine:
        for i, (lo, hi) in enumerate(zip(left.astype(int), np.ceil(right).astype(int))):
            if hi - lo < 3:
                continue
            width = ends[i] - starts[i]
            try:
                fit, _ = curve_fit(
                    _gaussian, periods[lo:hi + 1], power[lo:hi + 1],
                    p0=(power[peaks[i]], centers[i], width / 4),
                    bounds=([0, starts[i], width * 1e-6], [np.inf, ends[i], width]),
                )
                centers[i] = float(fit[1])
            except (ValueError, RuntimeError):
                # Keep the sampled peak when the local fit is poorly conditioned.
                continue
    return centers, starts, ends


def do_wavelet(sst, dt, time=None, unit="Year", fig_name=None, num=50,
               title="Signal", ylabel="Variation", cmap="Purples",
               show_peaks=False, fill_nan=True, idx=None, fontsize=15, noplot=True,
               *, refine_peaks=False, lag1=0.0, siglvl=0.95, **wavelet_options):
    """Analyze one uniform series and optionally plot or save the result.

    noplot=True returns (period, global_power, peaks, starts, ends). With
    noplot=False it returns (peaks, starts, ends). Peak periods are on the
    native wavelet grid unless refine_peaks=True requests a local Gaussian fit.
    Missing values can be interpolated on a valid sample grid without mutating
    the input. Timestamp gaps are rejected when time is supplied.
    """
    values = np.array(sst, dtype=float, copy=True)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("do_wavelet needs a one-dimensional series with at least two samples")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    times = _uniform_time(time, len(values), dt)
    finite = np.isfinite(values)
    if not finite.all():
        if not fill_nan or np.count_nonzero(finite) < 2:
            raise ValueError("Too few finite values for wavelet analysis")
        values[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), values[finite])

    coefficients, period, scale, coi = wavelet(values, dt, **wavelet_options)
    order = np.argsort(period)
    period, scale = period[order], scale[order]
    power = np.abs(coefficients[order])**2
    global_power = power.mean(axis=1)
    noise = {key: wavelet_options[key] for key in ("mother", "param") if key in wavelet_options}
    variance = np.var(values, ddof=1)
    local = wave_signif(variance, dt, scale, lag1=lag1, siglvl=siglvl, **noise)
    average_count = len(values) - scale / dt
    global_threshold = wave_signif(
        variance, dt, scale, sigtest=1, lag1=lag1, siglvl=siglvl, dof=average_count, **noise,
    )
    centers, starts, ends = _global_peaks(period, global_power, global_threshold, refine_peaks)
    if fig_name is not None or not noplot:
        fig, axes = plt.subplot_mosaic(
            [["series", "series"], ["wavelet", "global"]],
            num=num, clear=True, figsize=(12, 8), layout="constrained",
            width_ratios=(3, 1),
        )
        series, local_ax, global_ax = (axes[k] for k in ("series", "wavelet", "global"))
        series.plot(times, values - values.mean(), color="black", linewidth=1)
        series.set(title=title, ylabel=ylabel)
        if idx is not None:
            series.axvline(times[idx], color="black", linestyle="--")
        mesh = local_ax.pcolormesh(times, period, power, shading="auto", cmap=cmap)
        fig.colorbar(mesh, ax=local_ax, label="Wavelet power")
        if len(period) > 1 and np.any(power > local[:, None]) and np.any(power < local[:, None]):
            local_ax.contour(times, period, power - local[:, None], levels=[0], colors="black", linewidths=0.7)
        local_ax.fill_between(times, np.clip(coi, period[0], period[-1]), period[-1],
                              color="gray", alpha=0.25)
        local_ax.set(title="Wavelet power", ylabel=f"Period ({unit})" if unit else "Period")
        global_ax.plot(global_power, period, color="black")
        global_ax.plot(global_threshold, period, color="black", linestyle="--")
        global_ax.set(title="Global spectrum", xlabel="Power")
        for ax in (local_ax, global_ax):
            ax.set_yscale("log", base=2)
            if period[0] < period[-1]:
                ax.set_ylim(period[0], period[-1])
        if show_peaks:
            for center in centers:
                global_ax.annotate(f"{center:.3g}", (np.interp(center, period, global_power), center),
                                   fontsize=fontsize)
        for ax in (series, local_ax):
            ax.set_xlabel(f"Time ({unit})" if unit else "Time")
            ax.set_xlim(times[0], times[-1])
        if fig_name is not None:
            path = Path(fig_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        if noplot or fig_name is not None:
            plt.close(fig)
        else:
            plt.show()
    if noplot:
        return period, global_power, centers, starts, ends
    return centers, starts, ends


if __name__ == "__main__":
    do_wavelet(np.loadtxt(Path(__file__).with_name("sst_nino3.dat")), 0.25, noplot=False)
