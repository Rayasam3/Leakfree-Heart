"""Download the public web pages listed (fetch: true) in knowledge_base/sources.yaml and store plain text in
knowledge_base/raw_text/<id>.txt.   Guideline PDFs (manual: true) must be downloaded by hand into knowledge_base/raw/.

    python scripts/download_guidelines.py
Then:  python scripts/build_kb.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"
UA = {"User-Agent": "Mozilla/5.0 (research prototype; leakfree-heart)"}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [x.get_text(" ", strip=True) for x in main.find_all(["h1", "h2", "h3", "p", "li"])]
    return "\n".join(l for l in lines if len(l.split()) >= 4)


def main() -> int:
    meta = yaml.safe_load((KB / "sources.yaml").read_text(encoding="utf-8"))["sources"]
    (KB / "raw_text").mkdir(parents=True, exist_ok=True)
    (KB / "raw").mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for s in meta:
        if s.get("manual"):
            have = (KB / "raw" / f"{s['id']}.pdf").exists()
            print(f"[{'have' if have else 'TODO'}] manual PDF  {s['id']}.pdf  <- {s['title']}")
            continue
        if not s.get("fetch"):
            continue
        out = KB / "raw_text" / f"{s['id']}.txt"
        if out.exists() and out.stat().st_size > 500:
            print(f"[skip] {s['id']}")
            continue
        try:
            r = requests.get(s["url"], headers=UA, timeout=45)
            r.raise_for_status()
            text = html_to_text(r.text)
            if len(text.split()) < 80:
                raise ValueError("page text too short (blocked or script-rendered)")
            out.write_text(text, encoding="utf-8")
            print(f"[ ok ] {s['id']}  ({len(text.split())} words)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {s['id']}: {e}")
            fail += 1
        time.sleep(1.0)
    print(f"\nFetched {ok}, failed {fail}. Failed pages can be saved manually as knowledge_base/raw_text/<id>.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
