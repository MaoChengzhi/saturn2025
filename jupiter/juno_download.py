"""Download Juno MAG (FGM) data from NASA PDS PPI (UCLA) and Juno SPICE kernels from NAIF.

Every download function defaults to ``dry_run=True``: it lists what would be
downloaded and the total size, and writes nothing. Pass ``dry_run=False`` to
download.

Downloads are incremental. A file already on disk is kept when the server
listing shows the same name, a matching size, and no newer modification time,
so repeating a call months later fetches only files that are new or were
reprocessed on the server. Files are written as ``.part`` and renamed only when
complete, so an interrupted run can simply be repeated.

Notebook use, from the saturn2025 folder:

    from jupiter import juno_download as jd
    jd.download_juno_starter()               # dry run: steps 0, 1, 2 and 4
    jd.download_juno_native_all()            # dry run: step 6, every native day ("all")
    jd.download_juno_native_all(years=[2016, 2017])   # dry run: two years only
    jd.update_juno()                         # dry run: everything new since last time
    jd.update_juno(dry_run=False)            # download what is new

Destination folders (parallel to the Cassini folders):

    data/raw/PDS/juno/docs/            step 0  archive documents and file index
    data/raw/PDS/juno/mag_pc_flyby/    step 1  native-resolution moon-flyby files
    data/raw/spice_kernels/spice-juno/ step 2  SPICE kernels for positions
    data/raw/PDS/juno/mag_pc_1s/       step 4  1-second MAG, whole mission
    data/raw/PDS/juno/mag_pc_native/   step 5  native MAG for selected days
                                       step 6  native MAG, whole mission (~382 GiB)

All MAG files are in the planetocentric (PC) frame: Jupiter-centred, body-fixed
(System III). The PC folders hold only Jupiter-orbit data, starting 2016-07-07,
two days after orbit insertion. Each ``.sts`` table is saved with its PDS3
``.lbl`` label.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import threading
import time

import requests
from tqdm.auto import tqdm


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
JUNO_PDS_DIR = DATA_DIR / "raw" / "PDS" / "juno"
JUNO_SPICE_DIR = DATA_DIR / "raw" / "spice_kernels" / "spice-juno"
MAG_DIRS = {
    "r1s": JUNO_PDS_DIR / "mag_pc_1s",
    "native": JUNO_PDS_DIR / "mag_pc_native",
}
FLYBY_DIR = JUNO_PDS_DIR / "mag_pc_flyby"
DOCS_DIR = JUNO_PDS_DIR / "docs"

PDS_ROOT = "https://pds-ppi.igpp.ucla.edu/data/JNO-J-3-FGM-CAL-V1.0/"
PC_ROOT = PDS_ROOT + "DATA/JUPITER/PC/"
NAIF = "https://naif.jpl.nasa.gov/pub/naif/"

# Kernels for spacecraft and moon positions. Attitude (CK), events (EK) and
# shape (DSK) kernels are not needed: the calibrated MAG vectors are already
# rotated into the PC frame. NAIF extends juno_rec_orbit.bsp under the same
# name, so an update re-downloads it when its size changes.
SPICE_KERNELS = {
    "spk/juno_rec_orbit.bsp": NAIF + "JUNO/kernels/spk/juno_rec_orbit.bsp",  # Juno, whole mission
    "spk/jup380s.bsp": NAIF + "JUNO/kernels/spk/jup380s.bsp",                # Galilean moons
    "spk/jup388s.bsp": NAIF + "JUNO/kernels/spk/jup388s.bsp",                # Galilean moons, newer
    "spk/de440s.bsp": NAIF + "generic_kernels/spk/planets/de440s.bsp",       # Sun and planets
    "lsk/naif0012.tls": NAIF + "JUNO/kernels/lsk/naif0012.tls",              # leap seconds
    "pck/pck00010.tpc": NAIF + "JUNO/kernels/pck/pck00010.tpc",              # Jupiter size and spin
    "fk/juno_v12.tf": NAIF + "JUNO/kernels/fk/juno_v12.tf",                  # JUNO_JSS and other frames
}

# Moon flybys known when this file was written. download_juno_flybys() also
# finds moon-flyby products added to the archive later.
FLYBYS = {
    "ganymede_2021-06-07": (34, "2021158", "ga"),
    "europa_2022-09-29": (45, "2022272", "eu"),
    "io_2023-12-30": (57, "2023364", "io"),
    "io_2024-02-03": (58, "2024034", "io"),
}
MOON_TAGS = ("io", "eu", "ga")

_FGM_NAME = re.compile(
    r"fgm_jno_l3_(?P<day>\d{7})pc(?:_(?P<tag>io|eu|ga|pj\d+))?"
    r"(?:_(?P<cad>r1s|r60s))?_v(?P<ver>\d\d)\.sts"
)
# One file row of an Apache listing: name, modification time (UTC), rounded size.
_LISTING_ROW = re.compile(
    r'href="([^"?/][^"]*)">[^<]*</a>\s+(\d{4}-\d\d-\d\d \d\d:\d\d)\s+([\d.]+[KMG]?)\s'
)
_SIZE_UNITS = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30}
_LISTING_TTL_S = 3600
_listing_cache = {}
_listing_info = {}  # url -> (server modification time, rounded size text)
_thread_state = threading.local()


def _session():
    """One HTTP session per thread."""
    if not hasattr(_thread_state, "session"):
        session = requests.Session()
        # The PDS server gzips text files on request and then omits Content-Length.
        # Asking for uncompressed bytes keeps the size checks exact.
        session.headers["Accept-Encoding"] = "identity"
        _thread_state.session = session
    return _thread_state.session


def _listing_text(url):
    cached = _listing_cache.get(url)
    if cached and time.time() - cached[0] < _LISTING_TTL_S:
        return cached[1]
    response = _session().get(url, timeout=120)
    response.raise_for_status()
    _listing_cache[url] = (time.time(), response.text)
    return response.text


def _links(url):
    """File and folder names in an Apache directory listing."""
    names = re.findall(r'href="([^"?/][^"]*)"', _listing_text(url))
    return [name for name in dict.fromkeys(names) if not name.startswith(("http", "/"))]


def _listing(url):
    """{file name: (modification time as epoch seconds, rounded size text)} for a listing."""
    rows = {}
    for name, stamp, size in _LISTING_ROW.findall(_listing_text(url)):
        modified = datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        rows[name] = (modified.timestamp(), size)
    return rows


def _remote_size(url):
    response = _session().head(url, timeout=120, allow_redirects=True)
    response.raise_for_status()
    return int(response.headers["Content-Length"])


def _size_matches(local_bytes, size_text):
    """Whether an exact byte count agrees with a rounded listing size such as '85M' or '5.1M'."""
    number, unit = re.fullmatch(r"([\d.]+)([KMG]?)", size_text).groups()
    if not unit:
        return local_bytes == int(number)
    step = 0.1 if "." in number else 1.0
    return abs(local_bytes / _SIZE_UNITS[unit] - float(number)) <= step


def _is_current(dest, url):
    """True when ``dest`` exists and the listing shows no newer or different file."""
    info = _listing_info.get(url)
    if info is None or not dest.is_file():
        return False
    modified, size_text = info
    stat = dest.stat()
    return stat.st_mtime >= modified and _size_matches(stat.st_size, size_text)


def _fetch(url, dest, size):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with _session().get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
    if partial.stat().st_size != size:
        raise IOError(f"Incomplete download ({partial.stat().st_size} of {size} bytes): {url}")
    partial.replace(dest)
    return size


def _run(jobs, label, dry_run, by_year=False, verify=False, workers=4):
    """Report and optionally download a list of (url, destination) pairs.

    Files that the listing shows as unchanged are not checked again on the
    server. ``verify=True`` asks the server for the exact size of every file.
    ``workers`` files are downloaded at the same time (default 4); size checks
    use 8 parallel requests.
    """
    jobs = [(url, Path(dest)) for url, dest in jobs]
    current = [] if verify else [(url, dest) for url, dest in jobs if _is_current(dest, url)]
    current_set = set(current)
    candidates = [job for job in jobs if job not in current_set]
    with ThreadPoolExecutor(8) as pool:
        sizes = list(tqdm(pool.map(_remote_size, [url for url, _ in candidates]),
                          total=len(candidates), desc=f"{label}: checking", leave=False))
    todo = []
    for (url, dest), size in zip(candidates, sizes):
        if dest.is_file() and dest.stat().st_size == size:
            current.append((url, dest))
        else:
            todo.append((url, dest, size))

    have = sum(dest.stat().st_size for _, dest in current)
    need = sum(size for _, _, size in todo)
    print(f"{label}: {len(jobs)} files, {(have + need) / 2**30:.3f} GiB in total")
    print(f"  on disk and current: {len(current)} files, {have / 2**30:.3f} GiB")
    print(f"  new or changed:      {len(todo)} files, {need:,} bytes = {need / 2**30:.3f} GiB")
    if by_year and todo:
        years = {}
        for _, dest, size in todo:
            year = re.search(r"_l3_(\d{4})", dest.name)[1]
            years[year] = years.get(year, 0) + size
        print("  to download, by year:")
        for year, size in sorted(years.items()):
            print(f"    {year}: {size / 2**30:8.2f} GiB")
    summary = {"files": len(jobs), "current": len(current), "new": len(todo),
               "bytes_new": need, "downloaded": 0}
    if dry_run:
        print("  dry run, nothing written. Pass dry_run=False to download.")
        return summary
    if not todo:
        return summary

    target = todo[0][1].parent
    while not target.exists():
        target = target.parent
    free = shutil.disk_usage(target).free
    margin = 20 * 2**30  # keep at least 20 GiB free for the system and notebooks
    if need > free - margin:
        raise OSError(f"Not enough disk space: need {need / 2**30:.1f} GiB, "
                      f"free {free / 2**30:.1f} GiB (keeping a 20 GiB margin).")

    with ThreadPoolExecutor(workers) as pool, \
            tqdm(total=need, unit="B", unit_scale=True, desc=label) as bar:
        futures = [pool.submit(_fetch, url, dest, size) for url, dest, size in todo]
        for future in as_completed(futures):
            bar.update(future.result())
            summary["downloaded"] += 1
    print(f"  {summary['downloaded']} files downloaded")
    return summary


def list_perijove_folders():
    """Orbit folders on the server, e.g. [0, 1, ..., 83]. New orbits appear automatically."""
    return sorted(int(name[5:-1]) for name in _links(PC_ROOT) if re.fullmatch(r"PERI-\d+/", name))


def list_juno_fgm(cadence="r1s", perijoves=None, days=None, tags=("day",)):
    """Return relative paths (``PERI-NN/<file>.sts``) of selected PC-frame MAG files.

    cadence : "r1s" (1 sample per second) or "native" (full resolution, 8-64 Hz)
    perijoves : orbit numbers to search; None searches every orbit folder
    days : "YYYYDDD" strings to keep; None keeps every day
    tags : "day" for the plain daily files; "io", "eu", "ga" for moon-flyby
        products; "pj" for the per-orbit copies of the perijove day

    For each day and tag only the highest version (v01, v02, ...) is kept.
    """
    if cadence not in MAG_DIRS:
        raise ValueError(f"cadence must be one of {sorted(MAG_DIRS)}")
    days = set(days or ())
    best = {}
    for peri in perijoves if perijoves is not None else list_perijove_folders():
        folder = f"PERI-{peri:02d}/"
        rows = _listing(PC_ROOT + folder)
        for name in rows:
            match = _FGM_NAME.fullmatch(name)
            if not match or (match["cad"] or "native") != cadence:
                continue
            tag = match["tag"] or "day"
            if tag.rstrip("0123456789") not in tags or (days and match["day"] not in days):
                continue
            key, version = (match["day"], tag), int(match["ver"])
            if key not in best or version > best[key][0]:
                best[key] = (version, folder + name)
        for name, info in rows.items():
            _listing_info[PC_ROOT + folder + name] = info
    return sorted(path for _, path in best.values())


def _mag_jobs(paths, out_dir):
    jobs = []
    for path in paths:
        for rel in (path, path[:-4] + ".lbl"):
            jobs.append((PC_ROOT + rel, Path(out_dir) / Path(rel).name))
    return jobs


def _report_superseded(paths, out_dir):
    """List local files replaced on the server by a newer version (v01 -> v02). Nothing is deleted."""
    wanted = {Path(path).name for path in paths}
    keys = {}
    for name in wanted:
        match = _FGM_NAME.fullmatch(name)
        keys[(match["day"], match["tag"], match["cad"])] = int(match["ver"])
    older = []
    for local in sorted(Path(out_dir).glob("fgm_jno_l3_*.sts")):
        match = _FGM_NAME.fullmatch(local.name)
        newest = keys.get((match["day"], match["tag"], match["cad"])) if match else None
        if newest is not None and int(match["ver"]) < newest:
            older.append(local)
    if older:
        print(f"  {len(older)} older versions on disk now replaced by a newer version "
              f"(kept; delete them and their .lbl files if you do not need them):")
        for local in older[:20]:
            print(f"    {local.name}")
        if len(older) > 20:
            print(f"    ... and {len(older) - 20} more")
    return older


def download_juno_docs(dry_run=True):
    """Step 0: archive read-me, errata, catalog, instrument documents and file index (~10 MB)."""
    jobs = [(PDS_ROOT + name, DOCS_DIR / name) for name in
            ("AAREADME.TXT", "ERRATA.TXT", "VOLDESC.CAT", "INDEX/INDEX.TAB", "INDEX/INDEX.LBL")]
    for folder in ("CATALOG/", "DOCUMENT/"):
        jobs += [(PDS_ROOT + folder + name, DOCS_DIR / folder / name)
                 for name in _links(PDS_ROOT + folder) if not name.endswith("/")]
    return _run(jobs, "Step 0 docs", dry_run)


def download_juno_flybys(dry_run=True, flybys=None):
    """Step 1: native-resolution moon-flyby files with labels (~170 MB for the first four).

    flybys : names from ``FLYBYS``; None finds every moon-flyby product in the
        archive, including ones added after this file was written.
    These compact products repeat part of the native daily file for that day.
    """
    if flybys is None:
        paths = list_juno_fgm("native", tags=MOON_TAGS)
    else:
        paths = []
        for name in flybys:
            peri, day, tag = FLYBYS[name]
            found = list_juno_fgm("native", perijoves=[peri], days=[day], tags=(tag,))
            if not found:
                raise FileNotFoundError(f"No {tag} product for {day} in PERI-{peri:02d}")
            paths += found
    result = _run(_mag_jobs(paths, FLYBY_DIR), "Step 1 flybys", dry_run)
    _report_superseded(paths, FLYBY_DIR)
    return result


def download_juno_spice(dry_run=True):
    """Step 2: trajectory, moon, planet, time and frame kernels (~265 MB).

    Kernels have no listing check, so each one is size-checked on the server;
    a kernel extended under the same name is downloaded again.
    """
    jobs = [(url, JUNO_SPICE_DIR / rel) for rel, url in SPICE_KERNELS.items()]
    return _run(jobs, "Step 2 SPICE", dry_run)


def _select_years(paths, years):
    """Keep paths whose file date falls in ``years``; "all" or None keeps everything."""
    if years is None or (isinstance(years, str) and years.lower() == "all"):
        return paths
    if isinstance(years, (int, str)):
        years = [years]
    keep = {str(int(year)) for year in years}
    available = sorted({re.search(r"_l3_(\d{4})", path)[1] for path in paths})
    missing = sorted(keep - set(available))
    if missing:
        print(f"  no files for year(s) {missing}; available: {available[0]}-{available[-1]}")
    return [path for path in paths if re.search(r"_l3_(\d{4})", path)[1] in keep]


def download_juno_mag(cadence="r1s", perijoves=None, days=None, years="all", dry_run=True,
                      out_dir=None, verify=False, workers=4):
    """Steps 4 and 5: plain daily MAG files at one cadence, with labels.

    perijoves, days : optional orbit numbers or "YYYYDDD" days to keep
    years : "all" (default) for every year in the archive, or e.g. 2017 or [2016, 2017]
    workers : files downloaded at the same time (default 4)

    The whole mission is about 36 GiB at 1 s and about 382 GiB at native
    resolution (see also ``download_juno_native_all``).
    """
    paths = _select_years(list_juno_fgm(cadence, perijoves=perijoves, days=days), years)
    out_dir = Path(out_dir or MAG_DIRS[cadence])
    label = f"MAG {cadence}" + ("" if years in (None, "all") else f", years {years}")
    result = _run(_mag_jobs(paths, out_dir), label, dry_run,
                  by_year=cadence == "native", verify=verify, workers=workers)
    _report_superseded(paths, out_dir)
    return result


def download_juno_native_all(years="all", dry_run=True, verify=False, workers=4):
    """Step 6: full-resolution (native) daily MAG since Jupiter orbit insertion.

    years : "all" (default) downloads every year available on the server,
        including years added in later PDS releases. Give one year (2017) or a
        list ([2016, 2017]) to download in parts.

    About 382 GiB for July 2016 to May 2026. Repeat the call later to fetch
    only new or reprocessed days. Refuses to start if the disk would keep less
    than 20 GiB free.
    """
    return download_juno_mag("native", years=years, dry_run=dry_run, verify=verify,
                             workers=workers)


def download_juno_starter(dry_run=True):
    """Steps 0, 1, 2 and 4 in order: docs, flybys, SPICE, then 1 s MAG for the whole mission."""
    return {
        "docs": download_juno_docs(dry_run),
        "flybys": download_juno_flybys(dry_run),
        "spice": download_juno_spice(dry_run),
        "mag_1s": download_juno_mag("r1s", dry_run=dry_run),
    }


def update_juno(dry_run=True, native=True, verify=False):
    """Bring every Juno folder up to date: docs, flybys, SPICE, 1 s MAG and native MAG.

    Run every few months after PDS releases new orbits. Only new or changed
    files are downloaded. ``native=False`` skips the full-resolution data.
    ``verify=True`` checks the exact size of every file on the server (slower).
    """
    result = {
        "docs": download_juno_docs(dry_run),
        "flybys": download_juno_flybys(dry_run),
        "spice": download_juno_spice(dry_run),
        "mag_1s": download_juno_mag("r1s", dry_run=dry_run, verify=verify),
    }
    if native:
        result["mag_native"] = download_juno_native_all(dry_run=dry_run, verify=verify)
    return result
