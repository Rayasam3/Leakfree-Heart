"""Build the RAG knowledge base index.

    python scripts/build_kb.py                # seed entries + every text/PDF found in knowledge_base/
    python scripts/build_kb.py --seed-only    # only the bundled (unverified) seed entries

Steps: PDFs dropped into knowledge_base/raw/ are converted to knowledge_base/raw_text/<name>.txt, then everything is
chunked, tagged with the clinical features it discusses, and saved to knowledge_base/index/chunks.json.
Add a metadata entry for each document in knowledge_base/sources.yaml (id = file name without extension).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakfree_heart.config import load_config, path  # noqa: E402
from leakfree_heart.rag.kb import build_chunks, save_chunks  # noqa: E402


def pdf_to_text(pdf: Path) -> str:
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    return "\n".join((p.extract_text() or "") for p in r.pages)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    kb = path(cfg, "knowledge_base")
    (kb / "raw").mkdir(exist_ok=True)
    (kb / "raw_text").mkdir(exist_ok=True)
    if not args.seed_only:
        for pdf in sorted((kb / "raw").glob("*.pdf")):
            out = kb / "raw_text" / (pdf.stem + ".txt")
            if not out.exists():
                print(f"[pdf ] extracting {pdf.name}")
                out.write_text(pdf_to_text(pdf), encoding="utf-8")
    chunks = build_chunks(kb, cfg["rag"]["chunk_words"], cfg["rag"]["chunk_overlap"], include_documents=not args.seed_only)
    save_chunks(chunks, kb / "index" / "chunks.json")
    n_seed = sum(c.kind in ("guideline", "dataset_doc", "public_health", "project_observation") and c.id.startswith("KB-") for c in chunks)
    unverified = sum(not c.verified for c in chunks if c.id.startswith("KB-"))
    print(f"[kb  ] {len(chunks)} chunks ({n_seed} seed entries, {len(chunks) - n_seed} from documents) -> {kb / 'index' / 'chunks.json'}")
    if unverified:
        print(f"[warn] {unverified} seed entries are marked verified=false. Check them against the cited sources "
              f"before publishing (see knowledge_base/seed/seed_kb.json).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
