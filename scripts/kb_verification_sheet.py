"""Step 1: write knowledge_base/kb_verification.csv (one row per seed entry).
Open it in Excel. For each row read the cited source, compare with `text`, then set
   verified = 1 (text is an accurate paraphrase of the source) or 0 (wrong or unsupported; fix the text in seed_kb.json)
and write the page/section you checked in `checked_in`.
Step 2: python scripts/kb_verification_sheet.py --apply   (copies verified=1 into seed_kb.json and rebuilds nothing else)
Then run:  python -m leakfree_heart build-kb"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "knowledge_base" / "seed" / "seed_kb.json"
CSV = ROOT / "knowledge_base" / "kb_verification.csv"

d = json.loads(SEED.read_text(encoding="utf-8"))
if "--apply" in sys.argv:
    t = pd.read_csv(CSV).set_index("id")
    n = 0
    for e in d["entries"]:
        if e["id"] in t.index and int(t.loc[e["id"], "verified"] or 0) == 1:
            e["verified"] = True
            n += 1
    SEED.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"marked {n}/{len(d['entries'])} entries verified. Now run: python -m leakfree_heart build-kb")
else:
    rows = [dict(id=e["id"], source=e.get("source", ""), year=e.get("year", ""), features=",".join(e.get("features", [])),
                 text=e["text"], verified=0, checked_in="", note="") for e in d["entries"]]
    pd.DataFrame(rows).to_csv(CSV, index=False, encoding="utf-8-sig")
    print(f"wrote {CSV} ({len(rows)} rows). Fill 'verified' (1/0) and 'checked_in', then run with --apply")
