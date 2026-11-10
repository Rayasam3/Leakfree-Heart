"""Download the datasets used in the project.

    python scripts/download_data.py            # UCI Heart Disease (4 sites)  -> data/raw/uci/
    python scripts/download_data.py --kaggle   # also try the Kaggle copy      -> data/raw/kaggle_heart.csv

UCI Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease  (CC BY 4.0)
Creators: Janosi A., Steinbrunn W., Pfisterer M., Detrano R.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakfree_heart.data import UCI_EXPECTED_ROWS, UCI_FILES  # noqa: E402

BASES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/",
    "https://archive.ics.uci.edu/static/public/45/",
]
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
EXTRA_FILES = ["heart-disease.names"]


def _get(url: str, tries: int = 3, timeout: int = 60) -> bytes | None:
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "leakfree-heart/1.0"})
            if r.status_code == 200 and r.content:
                return r.content
            print(f"   HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            print(f"   attempt {i + 1} failed: {e}")
        time.sleep(1.5)
    return None


def download_uci(out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for site, fname in UCI_FILES.items():
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[skip] {fname} already present")
            continue
        data = None
        for base in BASES[:1]:
            print(f"[get ] {base}{fname}")
            data = _get(base + fname)
            if data:
                break
        if data is None:
            ok = False
            continue
        dest.write_bytes(data)
    if not ok:
        print("[try ] zip archive fallback")
        z = _get(ZIP_URL, timeout=120)
        if z:
            with zipfile.ZipFile(io.BytesIO(z)) as zf:
                for name in zf.namelist():
                    base = Path(name).name
                    if base in UCI_FILES.values() or base in EXTRA_FILES:
                        (out_dir / base).write_bytes(zf.read(name))
            ok = all((out_dir / f).exists() for f in UCI_FILES.values())
    for f in EXTRA_FILES:
        if not (out_dir / f).exists():
            d = _get(BASES[0] + f, tries=1)
            if d:
                (out_dir / f).write_bytes(d)
    return ok


def verify_uci(out_dir: Path) -> bool:
    good = True
    for site, fname in UCI_FILES.items():
        p = out_dir / fname
        if not p.exists():
            print(f"[FAIL] missing {fname}")
            good = False
            continue
        n = sum(1 for line in p.read_text(errors="ignore").splitlines() if line.strip())
        flag = "ok" if n == UCI_EXPECTED_ROWS[site] else "UNEXPECTED"
        print(f"[{flag:^10}] {fname}: {n} rows (expected {UCI_EXPECTED_ROWS[site]})")
        good &= n == UCI_EXPECTED_ROWS[site]
    return good


def download_kaggle(raw_dir: Path) -> bool:
    dest = raw_dir / "kaggle_heart.csv"
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return True
    try:
        import kagglehub  # type: ignore
    except ImportError:
        print("[info] kagglehub not installed: pip install kagglehub  (optional, only for the leakage audit)")
        return False
    try:
        path = Path(kagglehub.dataset_download("johnsmith88/heart-disease-dataset"))
        src = next(path.glob("*.csv"))
        dest.write_bytes(src.read_bytes())
        print(f"[ok  ] saved {dest}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Kaggle download failed: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", action="store_true", help="also fetch the Kaggle copy via kagglehub")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    ok = download_uci(raw / "uci")
    ok = verify_uci(raw / "uci") and ok
    if args.kaggle:
        download_kaggle(raw)
    if not ok:
        print("\nUCI download incomplete. Manual option: download 'heart+disease.zip' from\n"
              "  https://archive.ics.uci.edu/dataset/45/heart+disease\n"
              f"and extract the four processed.*.data files into {raw / 'uci'}")
        return 1
    print("\nDone. UCI files are in", raw / "uci")
    return 0


if __name__ == "__main__":
    sys.exit(main())
