"""Run the experiments that produce every table and figure of the paper.

    python scripts/run_experiments.py                       # everything
    python scripts/run_experiments.py --only leakage benchmark
    python scripts/run_experiments.py --quick               # tiny smoke-test run (minutes)

Stages (names): leakage benchmark external trust xai faith features missing fairness report
Results: results/tables/*.csv|json, results/figures/*.png, results/RESULTS.md
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakfree_heart.experiments import benchmark, faith, leakage, robustness, trust, xai  # noqa: E402
from leakfree_heart.experiments.common import make_ctx  # noqa: E402
from leakfree_heart.experiments.report import write_report  # noqa: E402

STAGES = {
    "leakage": leakage.run,
    "benchmark": benchmark.run,
    "external": benchmark.run_external,
    "trust": trust.run,
    "xai": xai.run,
    "faith": faith.run,
    "features": robustness.run_feature_reduction,
    "missing": robustness.run_missing,
    "fairness": robustness.run_fairness,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(STAGES) + ["report"], help="run only these stages")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(STAGES) + ["report"])
    ap.add_argument("--quick", action="store_true", help="small/fast settings for a smoke test")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    ctx = make_ctx(args.config, quick=args.quick)
    todo = [s for s in (args.only or list(STAGES) + ["report"]) if s not in args.skip]
    failed = []
    t0 = time.time()
    for s in todo:
        if s == "report":
            continue
        ctx.say(f"\n===== {s} =====")
        try:
            STAGES[s](ctx)
        except FileNotFoundError as e:
            ctx.say(f"[{s}] SKIPPED: {e}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failed.append(s)
            ctx.say(f"[{s}] FAILED: {type(e).__name__}: {e}")
    if "report" in todo:
        write_report(ctx)
    ctx.say(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Failed stages: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
