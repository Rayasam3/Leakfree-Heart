"""Knowledge-base construction: seed entries + chunked full-text documents (guideline PDFs / web pages)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..data import FEATURES

FEATURE_KEYWORDS = {
    "age": ["age-related", "older adults", "aged "],
    "sex": ["women", "men ", "sex differences", "female", "male"],
    "cp": ["chest pain", "angina"],
    "trestbps": ["blood pressure", "hypertension", "systolic"],
    "chol": ["cholesterol", "ldl", "lipid", "statin"],
    "fbs": ["glucose", "diabetes", "glycemi", "glycaemi"],
    "restecg": ["electrocardiogra", "ecg", "ekg", "left ventricular hypertrophy"],
    "thalach": ["heart rate", "chronotropic", "maximal heart"],
    "exang": ["exercise-induced", "exercise testing", "treadmill", "stress test"],
    "oldpeak": ["st depression", "st-segment depression", "st segment depression", "st-segment"],
    "slope": ["st slope", "upsloping", "downsloping"],
    "ca": ["coronary angiograph", "stenosis", "multivessel", "vessel disease", "fluoroscop"],
    "thal": ["thallium", "perfusion", "scintigraph", "spect"],
    "prevention": ["lifestyle", "physical activity", "diet", "smoking cessation", "weight loss", "prevention"],
}


@dataclass
class Chunk:
    id: str
    text: str
    title: str = ""
    source: str = ""
    year: int | None = None
    features: list[str] = field(default_factory=list)
    kind: str = "document"
    verified: bool = False
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_seed(seed_path: Path) -> list[Chunk]:
    data = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    return [Chunk(id=e["id"], text=e["text"], title=e.get("title", ""), source=e.get("source", ""),
                  year=e.get("year"), features=e.get("features", []), kind=e.get("kind", "guideline"),
                  verified=bool(e.get("verified", False)), url=e.get("url", "")) for e in data["entries"]]


def tag_features(text: str) -> list[str]:
    low = text.lower()
    return [f for f, kws in FEATURE_KEYWORDS.items() if any(k in low for k in kws)]


def chunk_words(text: str, size: int = 140, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


# ---- cleaning of scraped web text (MedlinePlus, NHLBI, WHO ...): remove navigation / link-list boilerplate
_BOILER = re.compile(
    r"(also in spanish|medical encyclopedia|medlineplus health topic|clinicaltrials\.gov|journal articles|"
    r"see more articles|related health topics|fact ?sheet|national library of medicine|texas heart institute|"
    r"american college of|american heart association\)|health topic \(|article:|learn more|read more|"
    r"view the fact sheet|for parents|last updated|skip to|sign up|subscribe|cookie|privacy policy|"
    r"copyright|all rights reserved|return to top|share this|print this|\(spanish\))", re.I)


_GLOSSARY = re.compile(r"\b[\w-]+ \([A-Za-z]{2,}(?:-[A-Za-z]{2,})*\): .*?\. (?=[a-z,\-]|\. )")


def clean_web_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = _GLOSSARY.sub("", text)                       # inline tooltip glossary (NHLBI): "artery (ARR-tuh-ree): A blood vessel ..."
    text = re.sub(r"See, Play and Learn|Prevention and Risk Factors|< Back To Home", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"(?:\s*\.){2,}", ".", text)
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    keep = []
    for sn in sents:
        sn = sn.strip()
        w = sn.split()
        if len(w) < 7 or len(w) > 80 or sn.endswith("?"):   # too short, too long, or a question heading
            continue
        if _BOILER.search(sn):
            continue
        caps = sum(1 for x in w if x[:1].isupper())
        if caps / len(w) > 0.45:          # title lists / link lists
            continue
        if sn.count("(") >= 3:
            continue
        keep.append(sn)
    return " ".join(keep)


def chunk_sentences(text: str, min_words: int = 40, max_words: int = 110) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    out, cur, n = [], [], 0
    for sn in sents:
        k = len(sn.split())
        if cur and n + k > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sn)
        n += k
    if cur:
        out.append(" ".join(cur))
    return [c for c in out if len(c.split()) >= min_words]


def load_document_chunks(kb_dir: Path, size: int, overlap: int) -> list[Chunk]:
    """Chunk every extracted text file in knowledge_base/raw_text/ (created by scripts/download_guidelines.py
    or build_kb.py from PDFs in knowledge_base/raw/). Metadata comes from knowledge_base/sources.yaml.
    Scraped text is cleaned first and cut on sentence boundaries."""
    meta: dict[str, dict] = {}
    sy = Path(kb_dir) / "sources.yaml"
    if sy.exists():
        for s in yaml.safe_load(sy.read_text(encoding="utf-8")).get("sources", []):
            meta[s["id"]] = s
    out: list[Chunk] = []
    for f in sorted((Path(kb_dir) / "raw_text").glob("*.txt")):
        sid = f.stem
        m = meta.get(sid, {})
        text = clean_web_text(f.read_text(encoding="utf-8", errors="ignore"))
        for i, ch in enumerate(chunk_sentences(text)):
            feats = [x for x in tag_features(ch) if x in FEATURES or x == "prevention"]
            out.append(Chunk(id=f"{sid}-{i:03d}", text=ch, title=m.get("title", sid),
                             source=m.get("citation", m.get("title", sid)), year=m.get("year"),
                             features=feats or ["general"], kind=m.get("kind", "document"),
                             verified=False, url=m.get("url", "")))
    return out


def build_chunks(kb_dir: Path, size: int = 140, overlap: int = 30, include_documents: bool = True) -> list[Chunk]:
    chunks = load_seed(Path(kb_dir) / "seed" / "seed_kb.json")
    if include_documents:
        chunks += load_document_chunks(kb_dir, size, overlap)
    return chunks


def save_chunks(chunks: list[Chunk], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([c.to_dict() for c in chunks], indent=1), encoding="utf-8")


def load_chunks(kb_dir: Path, rebuild: bool = False, **kw) -> list[Chunk]:
    idx = Path(kb_dir) / "index" / "chunks.json"
    if idx.exists() and not rebuild:
        return [Chunk(**d) for d in json.loads(idx.read_text(encoding="utf-8"))]
    return build_chunks(kb_dir, **kw)
