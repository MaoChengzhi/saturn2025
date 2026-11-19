"""Plots for the Cassini location tables produced by the SPICE workflow."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from cassini_spice.locations import SATURN_RADIUS_KM, read_saturn_positions


__all__ = [
    "plot_cassini_trajectory",
    "plot_trajectory_2d",
    "plot_trajectory_3d",
    "plot_cassini_overview",
    "plot_cassini_speed",
    "plot_radial_distance",
    "plot_reference_speeds",
    "plot_l_mlt_coverage",
]

_BODY_COLORS = {
    "Cassini": "#23354D",
    "Enceladus": "#147A77",
    "Mimas": "#B75926",
    "Tethys": "#76518E",
}

_CARTESIAN_COLUMNS = {
    "J2000": ("x_Rs", "y_Rs", "z_Rs"),
    "CASSINI_KSM": ("x_ksm_Rs", "y_ksm_Rs", "z_ksm_Rs"),
    "ENIS": ("x_enis_RE", "y_enis_RE", "z_enis_RE"),
}
_FRAME_ALIASES = {"KSM": "CASSINI_KSM", "CUSTOM_ENIS_OFFICIAL_DEFINITION": "ENIS"}


def _trajectory_coordinates(positions, frame):
    if positions.empty:
        raise ValueError("No positions to plot")
    saved_frame = positions.attrs.get("frame", "").upper()
    saved_frame = _FRAME_ALIASES.get(saved_frame, saved_frame)
    frame = frame.upper() if frame else saved_frame
    frame = _FRAME_ALIASES.get(frame, frame)
    if saved_frame and saved_frame != frame:
        raise ValueError(f"Positions are in {saved_frame}, but {frame} was requested")
    if not frame:
        frame = next(
            (name for name, columns in _CARTESIAN_COLUMNS.items()
             if all(column in positions for column in columns)),
            "",
        )
    if frame not in _CARTESIAN_COLUMNS:
        raise ValueError("Cartesian trajectories require J2000, CASSINI_KSM, or ENIS positions")

    columns = _CARTESIAN_COLUMNS[frame]
    if frame == "CASSINI_KSM" and columns[0] not in positions:
        # read_saturn_positions uses generic columns and records its frame in attrs.
        columns = _CARTESIAN_COLUMNS["J2000"]
    units = "R_E" if frame == "ENIS" else "R_S"
    return positions.loc[:, list(columns)].to_numpy(), frame, units


def _plot_axes(ax, figsize):
    return ax if ax is not None else plt.subplots(figsize=figsize, layout="constrained")[1]


def _format_time_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_cassini_trajectory(
    start_time, end_time, *, frame="CASSINI_KSM", plane="xy", step_seconds=60,
    show_moon_tracks=True, ax=None,
):
    """Plot Cassini and the three moons in one coordinate plane and return (figure, axes).

    The filled markers show simultaneous end positions. The open marker is
    Cassini's start. Dashed curves show the moons' motion over the selected interval.
    """
    plane = plane.lower()
    if plane not in ("xy", "xz", "yz"):
        raise ValueError("plane must be 'xy', 'xz' or 'yz'")
    positions = read_saturn_positions(start_time, end_time, frame=frame, step_seconds=step_seconds)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7), layout="constrained")
    horizontal, vertical = (f"{axis}_Rs" for axis in plane)

    ax.add_patch(Circle((0, 0), 1, color="#D1AD77", alpha=0.6, label="Saturn reference radius"))
    for body, color in _BODY_COLORS.items():
        track = positions.loc[positions["body"] == body]
        x, y = track[horizontal], track[vertical]
        is_cassini = body == "Cassini"
        if is_cassini or show_moon_tracks:
            ax.plot(x, y, "-" if is_cassini else "--", color=color, lw=2 if is_cassini else 1.4)
        ax.scatter(x.iloc[-1], y.iloc[-1], color=color, s=38, label=body, zorder=4)
        if is_cassini:
            ax.scatter(
                x.iloc[0], y.iloc[0], facecolors="none", edgecolors=color, s=48,
                label="Cassini start", zorder=4,
            )

    start, end = positions["utc"].min(), positions["utc"].max()
    frame = positions.attrs["frame"]
    ax.set(
        title=f"Saturn-centered trajectories in {frame}\n"
              f"{start:%Y-%m-%d %H:%M:%S} to {end:%Y-%m-%d %H:%M:%S} UTC",
        xlabel=f"{plane[0].upper()} ($R_S$)", ylabel=f"{plane[1].upper()} ($R_S$)",
        aspect="equal",
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=9)
    return ax.figure, ax


def plot_trajectory_2d(positions, *, frame=None, plane="xy", ax=None):
    """Plot a Cartesian projection, with ENIS positions centered on Enceladus."""
    plane = plane.lower()
    if plane not in ("xy", "xz", "yz"):
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    if "body" in positions:
        positions = positions.loc[positions["body"] == "Cassini"]
    xyz, frame, units = _trajectory_coordinates(positions, frame)
    x, y = (xyz[:, "xyz".index(axis)] for axis in plane)
    ax = _plot_axes(ax, (8, 6))
    ax.plot(x, y, color="tab:blue", label="Cassini")
    ax.scatter(0, 0, color="gold", edgecolor="black",
               label="Enceladus" if frame == "ENIS" else "Saturn", zorder=3)
    ax.scatter(x[0], y[0], facecolors="none", edgecolors="tab:blue", label="Start", zorder=4)
    ax.scatter(x[-1], y[-1], color="tab:blue", label="End", zorder=4)
    ax.set(
        title=f"Cassini trajectory in {frame}",
        xlabel=f"{plane[0]} (${units}$)", ylabel=f"{plane[1]} (${units}$)",
        aspect="equal",
    )
    ax.legend(loc="best")
    return ax.figure, ax


def plot_trajectory_3d(positions, *, frame=None, interactive=False, color_by_time=False):
    """Plot three Cartesian components. Interactive tracks can be colored by UTC."""
    if "body" in positions:
        positions = positions.loc[positions["body"] == "Cassini"]
    xyz, frame, units = _trajectory_coordinates(positions, frame)
    if color_by_time and not interactive:
        raise ValueError("Set interactive=True to color the 3D track by time")
    if interactive:
        import plotly.graph_objects as go

        line = {"width": 3, "color": "royalblue"}
        if color_by_time:
            times = pd.DatetimeIndex(positions["utc"]).as_unit("ns")
            seconds = times.asi8 / 1e9
            indices = np.unique(np.linspace(0, len(times) - 1, min(len(times), 3), dtype=int))
            line.update(
                color=seconds,
                colorscale="Viridis",
                colorbar={
                    "title": "UTC",
                    "tickvals": seconds[indices],
                    "ticktext": times[indices].strftime("%Y-%m-%d %H:%M"),
                },
            )
        fig = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", line=line))
        fig.update_layout(
            title=f"Cassini trajectory in {frame}",
            scene={
                "xaxis_title": f"x ({units})",
                "yaxis_title": f"y ({units})",
                "zaxis_title": f"z ({units})",
                "aspectmode": "data",
            },
            template="plotly_white",
        )
        return fig

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": "3d"}, layout="constrained")
    ax.plot(*xyz.T, color="tab:blue")
    ax.set(
        title=f"Cassini trajectory in {frame}",
        xlabel=f"x (${units}$)", ylabel=f"y (${units}$)", zlabel=f"z (${units}$)",
    )
    span = np.ptp(xyz, axis=0)
    if span.max() > 0:
        ax.set_box_aspect(np.maximum(span, 0.01 * span.max()))
    return fig, ax


def plot_cassini_speed(positions, *, x="utc", show_radius=False, ax=None):
    """Plot speed versus time or radial distance, with an optional radius trace."""
    if x not in ("utc", "r_Rs"):
        raise ValueError("x must be 'utc' or 'r_Rs'")
    if show_radius and x != "utc":
        raise ValueError("The secondary radius trace requires x='utc'")
    ax = _plot_axes(ax, (8, 5))
    ax.plot(positions[x], positions["speed_km_s"], color="tab:purple", label="Speed")
    ax.set(xlabel="UTC" if x == "utc" else r"Radial distance ($R_S$)",
           ylabel="Speed (km/s)", title="Cassini speed")
    if x == "utc":
        _format_time_axis(ax)
    if show_radius:
        radius_ax = ax.twinx()
        radius_ax.plot(positions["utc"], positions["r_Rs"], color="tab:orange", ls="--")
        radius_ax.set_ylabel(r"Radial distance ($R_S$)", color="tab:orange")
        ax.yaxis.label.set_color("tab:purple")
    return ax.figure, ax


def plot_cassini_overview(positions, *, frame=None):
    """Plot the XY trajectory beside time series of speed and radial distance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    plot_trajectory_2d(positions, frame=frame, ax=axes[0])
    plot_cassini_speed(positions, show_radius=True, ax=axes[1])
    return fig, axes


def plot_radial_distance(positions, *, histogram=False, log_scale=False, ax=None):
    """Plot the supplied radial distances in their native table units."""
    if "r_Rs" in positions:
        radius, units = positions["r_Rs"], r"$R_S$"
    else:
        radius, units = positions["R_km"], "km"
    ax = _plot_axes(ax, (8, 5))
    label = f"Radial distance ({units})"
    if histogram:
        ax.hist(radius.dropna(), bins="auto")
        ax.set(xlabel=label, ylabel="Count", title="Cassini radial distance distribution")
        if log_scale:
            ax.set_xscale("log")
    else:
        ax.plot(positions["utc"], radius)
        ax.set(xlabel="UTC", ylabel=label, title="Cassini radial distance")
        _format_time_axis(ax)
        if log_scale:
            ax.set_yscale("log")
    return ax.figure, ax


def plot_reference_speeds(positions=None, *, interactive=False):
    """Plot the notebook's ideal corotation, Keplerian, and pickup speeds."""
    radius_rs = np.linspace(1, 10, 1000)
    radius_km = radius_rs * SATURN_RADIUS_KM
    corotation = (2 * np.pi / (10.7 * 3600)) * radius_km
    keplerian = np.sqrt((6.67430e-20 * 5.683e26) / radius_km)
    curves = {
        "Plasma corotation": corotation,
        "Neutral Keplerian motion": keplerian,
        "Corotation − Keplerian": corotation - keplerian,
    }
    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure([
            go.Scatter(x=radius_rs, y=values, mode="lines", name=label)
            for label, values in curves.items()
        ])
        if positions is not None:
            fig.add_scatter(x=positions["r_Rs"], y=positions["speed_km_s"],
                            mode="markers", name="Cassini speed")
        fig.update_layout(
            title="Reference speeds around Saturn",
            xaxis_title="Radial distance (R_S)", yaxis_title="Speed (km/s)",
            template="plotly_white",
        )
        fig.update_xaxes(range=[1, 10])
        return fig

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for label, values in curves.items():
        ax.plot(radius_rs, values, label=label)
    if positions is not None:
        ax.scatter(positions["r_Rs"], positions["speed_km_s"], s=8, alpha=0.5, label="Cassini speed")
    ax.set(xlabel=r"Radial distance ($R_S$)", ylabel="Speed (km/s)",
           title="Reference speeds around Saturn", xlim=(1, 10))
    ax.legend()
    ax.grid(alpha=0.2)
    return fig, ax


def plot_l_mlt_coverage(positions, *, stride=1, limit=8, ax=None):
    """Plot approximate L-shell versus MLT, with midnight at +X and dawn at +Y."""
    if stride < 1 or limit <= 0:
        raise ValueError("stride must be at least 1 and limit must be positive")
    data = positions.iloc[::stride]
    angle = 2 * np.pi * data["MLT_hr"] / 24
    ax = _plot_axes(ax, (7, 7))
    ax.plot(data["L_approx"] * np.cos(angle), data["L_approx"] * np.sin(angle))
    for radius, label in ((4.0, "Enceladus"), (6.2, "Dione")):
        ax.add_patch(Circle((0, 0), radius, fill=False, linestyle="--", edgecolor="0.4"))
        if radius < limit:
            ax.text(radius, 0, label, va="bottom")
    mlt_extent = limit * 0.94
    for x, y, hour, ha, va in (
        (mlt_extent, 0, "00", "left", "center"),
        (0, mlt_extent, "06", "center", "bottom"),
        (-mlt_extent, 0, "12", "right", "center"),
        (0, -mlt_extent, "18", "center", "top"),
    ):
        ax.text(x, y, hour, ha=ha, va=va)
    ax.set(
        xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
        xlabel="L, 00 MLT direction", ylabel="L, 06 MLT direction",
        title="Cassini coverage in approximate L–MLT coordinates",
    )
    return ax.figure, ax
