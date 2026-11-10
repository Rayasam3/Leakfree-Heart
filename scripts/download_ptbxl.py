"""Download the PTB-XL ECG dataset (100 Hz version) from PhysioNet.

    python scripts/download_ptbxl.py                  # full 100 Hz data (about 0.7 GB, 21,799 ECGs)
    python scripts/download_ptbxl.py --limit 500      # only the first 500 ECGs (quick test)
    python scripts/download_ptbxl.py --dest D:\\data\\ptbxl

PTB-XL 1.0.3, PhysioNet, licence CC BY 4.0.
Wagner P. et al., "PTB-XL, a large publicly available electrocardiography dataset", Scientific Data 7:154 (2020).
https://physionet.org/content/ptb-xl/1.0.3/

The download is resumable: files that already exist are skipped, so you can simply run it again after an interruption.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
META_FILES = ["ptbxl_database.csv", "scp_statements.csv"]
UA = {"User-Agent": "leakfree-heart-ptbxl/1.0"}


def fetch(url: str, dest: Path, tries: int = 4, timeout: int = 60) -> bool:
    """Download one file; skip if present. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers=UA)
            if r.status_code == 200 and r.content:
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                return True
            if r.status_code in (403, 404):
                return False
        except requests.RequestException:
            pass
        time.sleep(1.0 + i)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "data" / "raw" / "ptbxl"))
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--limit", type=int, default=0, help="download only the first N ECGs (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[ptbxl] destination: {dest}")
    for f in META_FILES:
        print(f"[get ] {f}")
        if not fetch(base + f, dest / f):
            print(f"\nCould not download {f} from {base}")
            print("Manual route: open https://physionet.org/content/ptb-xl/1.0.3/ , download the zip,")
            print("and copy ptbxl_database.csv, scp_statements.csv and the records100 folder into:", dest)
            return 1

    meta = pd.read_csv(dest / "ptbxl_database.csv", index_col="ecg_id")
    names = meta["filename_lr"].tolist()           # e.g. records100/00000/00001_lr
    if args.limit:
        names = names[: args.limit]
    todo = [(base + n + ext, dest / (n + ext)) for n in names for ext in (".hea", ".dat")]
    todo = [(u, d) for u, d in todo if not (d.exists() and d.stat().st_size > 0)]
    print(f"[ptbxl] {len(names)} ECGs wanted, {len(todo)} files still to download")

    failed, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, u, d): (u, d) for u, d in todo}
        for fut in as_completed(futs):
            done += 1
            if not fut.result():
                failed.append(futs[fut][0])
            if done % 2000 == 0 or done == len(todo):
                print(f"   {done}/{len(todo)} files  ({time.time() - t0:.0f}s, {len(failed)} failed)", flush=True)

    if failed:
        print(f"\n{len(failed)} files failed (first: {failed[0]}). Run the script again to retry them.")
        return 1 if len(failed) > 0.005 * max(1, len(todo)) else 0
    print("\n[ptbxl] done. Next:  python scripts/run_ecg.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
