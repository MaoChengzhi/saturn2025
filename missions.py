"""Mission settings shared by the quick event check: Cassini at Saturn, Juno at Jupiter.

The wave analysis (FAC decomposition, spectra, ICW and mirror-like screening,
panels) is common code. A Mission names what differs between spacecraft: the
data reader, the position lookup for dipole references, the orbit panel, the
planet, and the ions whose gyrofrequencies matter. Hooks are "module:function"
strings imported on first use, so loading one mission never imports the other.
"""

from dataclasses import dataclass, field
from importlib import import_module


@dataclass(frozen=True)
class Mission:
    name: str                      # spacecraft, used in figure titles
    planet: str
    planet_radius_km: float
    ions: dict                     # ion label -> mass-to-charge ratio (amu per elementary charge)
    default_ion: str               # ion used for the ICW band and the auto cadence rule
    icw_fc_reference: str          # default ICW field reference
    resolutions: tuple             # values accepted by the reader
    default_resolution: str
    reader: str                    # read_and_process(start, stop, resolution=..., ion_mass_amu=...)
    orbit_drawer: str              # draw(ax, start, stop, frame=...)
    orbit_frame: str
    position_lookup: str | None = None   # adds r and latitude columns for dipole references
    fc_line_ions: tuple = field(default=())  # ions marked on the FFT panel

    def hook(self, name):
        path = getattr(self, name)
        if path is None:
            return None
        module, function = path.split(":")
        return getattr(import_module(module), function)

    def ion_mass(self, ion=None):
        ion = ion or self.default_ion
        if ion not in self.ions:
            raise ValueError(f"Unknown ion {ion!r} for {self.name}. Choose from {list(self.ions)}")
        return float(self.ions[ion])


CASSINI = Mission(
    name="Cassini",
    planet="Saturn",
    planet_radius_km=60_268.0,
    ions={"H+": 1, "H2+": 2, "H3+": 3, "O+": 16, "OH+": 17, "H2O+": 18, "H3O+": 19,
          "O2+": 32},
    default_ion="H2O+",
    icw_fc_reference="dipole_mapped",
    resolutions=("auto", "sec", "full"),
    default_resolution="auto",
    reader="data_process:read_and_process_data",
    orbit_drawer="cassini_spice.plots:draw_wave_orbit_context",
    orbit_frame="CASSINI_KSM",
    position_lookup="cassini_spice.locations:add_dipole_position",
)

# Io-genic ions and other species used in Juno wave studies
# (Cao et al. 2025, GRL; Chakraborty et al. 2025, GRL). O+ and S++ share m/q = 16.
JUNO = Mission(
    name="Juno",
    planet="Jupiter",
    planet_radius_km=71_492.0,
    ions={"H+": 1, "He+": 4, "O++": 8, "O+/S++": 16, "S+": 32, "SO+": 48, "SO2+": 64},
    default_ion="S+",
    # Jupiter's dipole is tilted about 10 degrees from the spin axis, so the
    # spin-aligned dipole mapping used at Saturn does not apply. Use the local field.
    icw_fc_reference="local",
    resolutions=("auto", "sec", "full", "flyby"),
    default_resolution="auto",
    reader="jupiter.juno_data:read_and_process_data",
    orbit_drawer="jupiter.juno_spice:draw_wave_orbit_context",
    orbit_frame="JUNO_JSS",
    position_lookup=None,
    fc_line_ions=("He+", "O++", "O+/S++", "S+", "SO+", "SO2+"),
)

MISSIONS = {"cassini": CASSINI, "juno": JUNO}


def get_mission(mission=None):
    """Return a Mission from an instance, a name ("cassini", "juno"), or None (Cassini)."""
    if mission is None:
        return CASSINI
    if isinstance(mission, Mission):
        return mission
    try:
        return MISSIONS[str(mission).lower()]
    except KeyError as error:
        raise ValueError(f"Unknown mission {mission!r}. Choose from {list(MISSIONS)}") from error
