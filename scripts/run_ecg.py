"""Run the ECG (multimodal) study on PTB-XL.

    python scripts/download_ptbxl.py                      # once: download the data
    python scripts/run_ecg.py                             # all stages
    python scripts/run_ecg.py --only audit                # quick data audit only (no training)
    python scripts/run_ecg.py --only train evaluate
    python scripts/run_ecg.py --demo                      # smoke test on a tiny SYNTHETIC dataset (not real results)
    python scripts/run_ecg.py --quick                     # small real-data run (3,000 ECGs, 3 epochs)

Results: results_ecg/tables/*.csv, results_ecg/figures/*.png
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakfree_heart.ecg.study import STAGES, make_ecg_ctx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(STAGES))
    ap.add_argument("--skip", nargs="*", default=[], choices=list(STAGES))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    try:
        ctx = make_ecg_ctx(a.config, quick=a.quick, demo=a.demo, device=a.device)
    except FileNotFoundError as e:
        print(f"\nPTB-XL data not found: {e}\nRun first:  python scripts/download_ptbxl.py")
        return 1
    todo = [s for s in (a.only or list(STAGES)) if s not in a.skip]
    failed, t0 = [], time.time()
    for s in todo:
        ctx.say(f"\n===== {s} =====")
        try:
            STAGES[s](ctx)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failed.append(s)
    ctx.say(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Failed stages: {', '.join(failed) or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
