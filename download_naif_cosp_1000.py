#!/usr/bin/env python3
"""
Download everything under a NAIF Apache-style directory listing, recursively.
No command-line args needed: just run `python download_naif_cosp_1000.py`.

Edit BASE_URL and OUT_DIR below if you want to change source or destination.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ====== EDIT THESE TWO LINES IF YOU WANT ======
BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/pds/data/co-s_j_e_v-spice-6-v1.0/cosp_1000/"
OUT_DIR = "data/spice/"   # can be absolute: r"C:\data\cosp_1000" or "/data/cosp_1000"
# =============================================


# Optional knobs (safe defaults)
WORKERS = 6
TIMEOUT_LISTING = 60
TIMEOUT_FILE = 120
CHUNK_SIZE = 1024 * 1024  # 1 MB
POLITE_SLEEP = 0.0        # set to 0.1~0.3 if you want to be extra gentle
OVERWRITE = False         # True to re-download even if file exists


@dataclass(frozen=True)
class Item:
    url: str
    is_dir: bool


def normalize_url(u: str) -> str:
    """Remove query/fragment and keep a clean URL."""
    p = urlparse(u)
    return urlunparse(p._replace(query="", fragment=""))


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "naif-mirror/1.0 (+requests; respectful downloader)",
            "Accept": "text/html,application/octet-stream,*/*",
        }
    )
    return s


def list_directory(dir_url: str, base_url: str) -> List[Item]:
    """Parse an Apache-style 'Index of' directory and return child entries."""
    dir_url = normalize_url(dir_url)
    if not dir_url.endswith("/"):
        dir_url += "/"

    s = make_session()
    r = s.get(dir_url, timeout=TIMEOUT_LISTING)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    items: List[Item] = []

    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue

        text = (a.get_text() or "").strip().lower()
        if "parent directory" in text:
            continue
        if href.startswith("#"):
            continue

        child = normalize_url(urljoin(dir_url, href))

        # Stay inside the base URL
        if not child.startswith(base_url):
            continue

        is_dir = href.endswith("/") or child.endswith("/")
        items.append(Item(url=child, is_dir=is_dir))

    return items


def url_to_local_path(file_url: str, base_url: str, out_dir: Path) -> Path:
    """Map URL under base_url to a local path under out_dir (preserve tree)."""
    rel = file_url[len(base_url):].lstrip("/")
    return out_dir / rel


def crawl_all_files(base_url: str) -> List[str]:
    """Crawl entire directory tree under base_url; return list of file URLs."""
    base_url = normalize_url(base_url)
    if not base_url.endswith("/"):
        base_url += "/"

    seen_dirs: Set[str] = set()
    queue: List[str] = [base_url]
    files: List[str] = []

    while queue:
        d = normalize_url(queue.pop())
        if not d.endswith("/"):
            d += "/"
        if d in seen_dirs:
            continue
        seen_dirs.add(d)

        try:
            children = list_directory(d, base_url)
        except Exception as e:
            print(f"[WARN] Failed to list {d}: {e}", file=sys.stderr)
            continue

        for it in children:
            if it.is_dir:
                if it.url not in seen_dirs:
                    queue.append(it.url)
            else:
                files.append(it.url)

    return files


def download_file(file_url: str, base_url: str, out_dir: Path) -> Tuple[str, bool, str]:
    """Download one file with resume support (.part)."""
    file_url = normalize_url(file_url)

    local_path = url_to_local_path(file_url, base_url, out_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and not OVERWRITE:
        return (file_url, True, "skip")

    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    existing = tmp_path.stat().st_size if tmp_path.exists() else 0

    headers = {}
    mode = "wb"
    if existing > 0 and not OVERWRITE:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    if POLITE_SLEEP > 0:
        time.sleep(POLITE_SLEEP)

    s = make_session()
    try:
        with s.get(file_url, stream=True, headers=headers, timeout=TIMEOUT_FILE) as r:
            # If server ignores Range and returns full file, restart cleanly
            if r.status_code == 200 and "Range" in headers:
                existing = 0
                mode = "wb"

            # If Range is not satisfiable, we already have it complete
            if r.status_code == 416:
                tmp_path.replace(local_path)
                return (file_url, True, "done")

            r.raise_for_status()

            with open(tmp_path, mode) as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

        tmp_path.replace(local_path)
        return (file_url, True, "ok")
    except Exception as e:
        return (file_url, False, str(e))


def main() -> int:
    base_url = normalize_url(BASE_URL)
    if not base_url.endswith("/"):
        base_url += "/"

    out_dir = Path(OUT_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Base URL: {base_url}")
    print(f"[INFO] Output folder: {out_dir}")

    print("[INFO] Crawling directory tree...")
    files = crawl_all_files(base_url)
    print(f"[INFO] Found {len(files)} files.")

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as ex:
        futures = [ex.submit(download_file, u, base_url, out_dir) for u in files]
        with tqdm(total=len(futures), unit="file") as pbar:
            for fut in as_completed(futures):
                url, ok, msg = fut.result()
                if not ok:
                    failures += 1
                    print(f"\n[ERR] {url}\n      {msg}", file=sys.stderr)
                pbar.update(1)

    if failures:
        print(f"[DONE] Finished with {failures} failures. Re-run to retry.", file=sys.stderr)
        return 2

    print("[DONE] All files downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

