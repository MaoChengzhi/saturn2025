"""Juno and Galilean-moon positions from the local Juno SPICE kernels, and the orbit panel.

Kernels (downloaded by jupiter/juno_download.py) live in
data/raw/spice_kernels/spice-juno/: juno_rec_orbit.bsp (Juno), jup380s.bsp and
jup388s.bsp (moons), de440s.bsp (planets), naif0012.tls, pck00010.tpc and
juno_v12.tf (JUNO_JSS and other frames). They are loaded once per session.

Frames: IAU_JUPITER is Jupiter-centred and rotates with System III, like the
PC frame of the MAG files. JUNO_JSS (Jupiter-De-Spun-Sun) does not rotate with
Jupiter: z along the spin axis, x toward the Sun's meridian, y toward dusk.
"""

from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import spiceypy as spice
from matplotlib.patches import Circle

JUPITER_RADIUS_KM = 71_492.0
SPICE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "spice_kernels" / "spice-juno"
KERNELS = ("lsk/naif0012.tls", "pck/pck00010.tpc", "fk/juno_v12.tf", "spk/de440s.bsp",
           "spk/jup380s.bsp", "spk/jup388s.bsp", "spk/juno_rec_orbit.bsp")
MOONS = ("IO", "EUROPA", "GANYMEDE", "CALLISTO")
_BODY_COLORS = {"JUNO": "#1c71ae", "IO": "#c9a227", "EUROPA": "#8c6d46",
                "GANYMEDE": "#6f6f6f", "CALLISTO": "#4f5d75"}
_LOCK = Lock()
_loaded = False


def load_kernels():
    """Load the seven Juno kernels once; raise FileNotFoundError if any is missing."""
    global _loaded
    with _LOCK:
        if _loaded:
            return
        paths = [SPICE_DIR / kernel for kernel in KERNELS]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Juno SPICE kernels: {missing}. "
                                    "Run jupiter/juno_download.py step 2.")
        for path in paths:
            spice.furnsh(str(path))
        _loaded = True


def _ephemeris_times(times):
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if times.hasnans:
        raise ValueError("times contain missing values")
    load_kernels()
    utc = times.tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return np.array([spice.str2et(value) for value in utc])


def positions(times, body="JUNO", frame="JUNO_JSS", center="JUPITER"):
    """Return an (n, 3) array of body positions in km relative to center, in frame."""
    et = _ephemeris_times(times)
    with _LOCK:
        xyz, _ = spice.spkpos(body, et, frame, "NONE", center)
    return np.asarray(xyz, dtype=float).reshape(-1, 3)


def juno_position_at(times, frame="IAU_JUPITER", center="JUPITER", step_s=60):
    """Juno position in km at many sample times, interpolated from a step_s grid."""
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if len(times) == 0:
        return np.empty((0, 3))
    step = pd.Timedelta(seconds=step_s)
    grid = pd.date_range(times.min().floor(step) - step, times.max().ceil(step) + step, freq=step)
    grid_xyz = positions(grid, "JUNO", frame, center)
    offsets = (times - grid[0]).total_seconds()
    grid_offsets = (grid - grid[0]).total_seconds()
    return np.column_stack([np.interp(offsets, grid_offsets, grid_xyz[:, axis]) for axis in range(3)])


def body_table(start_time, end_time, *, frame="JUNO_JSS", step_seconds=60, bodies=("JUNO", *MOONS)):
    """Positions of Juno and the moons on a UTC grid, in Jupiter radii (long table)."""
    start, end = (pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None
                  else pd.Timestamp(value).tz_convert("UTC") for value in (start_time, end_time))
    grid = pd.date_range(start, end, freq=f"{step_seconds}s")
    if len(grid) == 0 or grid[-1] < end:
        grid = grid.append(pd.DatetimeIndex([end]))
    rows = []
    for body in bodies:
        xyz = positions(grid, body, frame) / JUPITER_RADIUS_KM
        rows.append(pd.DataFrame({"utc": grid, "body": body, "x_RJ": xyz[:, 0],
                                  "y_RJ": xyz[:, 1], "z_RJ": xyz[:, 2]}))
    table = pd.concat(rows, ignore_index=True)
    table.attrs["frame"] = frame
    return table


def draw_wave_orbit_context(ax, start_time, end_time, *, frame="JUNO_JSS"):
    """Draw Juno's event track, a +/-6 h context track, and the Galilean moons on ``ax``.

    Jupiter-centred XY projection in Jupiter radii (71,492 km). Moon orbits are
    circles at each moon's distance at the event midpoint; filled dots are the
    moon positions at the midpoint.
    """
    start, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
    midpoint = start + (end - start) / 2
    context = body_table(midpoint - pd.Timedelta(hours=6), midpoint + pd.Timedelta(hours=6),
                         frame=frame, step_seconds=300, bodies=("JUNO",))
    track = body_table(start, end, frame=frame, step_seconds=30, bodies=("JUNO",))
    at_mid = body_table(midpoint, midpoint, frame=frame, bodies=("JUNO", *MOONS))

    ax.add_patch(Circle((0, 0), 1, color="#D1AD77", alpha=0.7, label="Jupiter"))
    for row in at_mid.itertuples(index=False):
        if row.body == "JUNO":
            continue
        radius = np.hypot(row.x_RJ, row.y_RJ)
        ax.add_patch(Circle((0, 0), radius, fill=False, ls=":", lw=0.8,
                            color=_BODY_COLORS[row.body], alpha=0.8))
        ax.scatter(row.x_RJ, row.y_RJ, color=_BODY_COLORS[row.body], s=30, zorder=4,
                   label=row.body.title())
    ax.plot(context["x_RJ"], context["y_RJ"], ls="--", lw=1, color=_BODY_COLORS["JUNO"],
            alpha=0.5, label="Juno ±6 h")
    ax.plot(track["x_RJ"], track["y_RJ"], lw=2.2, color=_BODY_COLORS["JUNO"],
            label="Juno event track", zorder=3)
    juno = at_mid.loc[at_mid["body"].eq("JUNO")].iloc[0]
    ax.scatter(juno.x_RJ, juno.y_RJ, marker="D", facecolors="none",
               edgecolors=_BODY_COLORS["JUNO"], s=40, zorder=5, label="Juno midpoint")

    reach = max(np.abs(context[["x_RJ", "y_RJ"]].to_numpy()).max(), 7.0) * 1.08
    ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal",
           xlabel=r"X ($R_J$)", ylabel=r"Y ($R_J$)")
    r_mid = np.sqrt(juno.x_RJ**2 + juno.y_RJ**2 + juno.z_RJ**2)
    epoch = midpoint.tz_localize(None) if midpoint.tzinfo else midpoint
    ax.set_title(f"Orbit context · {frame} XY · r = {r_mid:.2f} $R_J$\n"
                 f"Positions at {epoch.isoformat(sep=' ')} UTC", loc="left", fontsize="medium")
    ax.grid(alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize="small")
    return ax
