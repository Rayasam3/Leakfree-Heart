"""Explanation generation: prompts, LLM back-ends (Ollama / OpenAI-compatible / deterministic mock) and the
verify-and-repair loop.

Conditions compared in the faithfulness study
    template         deterministic text from SHAP only (no LLM, no evidence)
    template_rag     deterministic text from SHAP + retrieved evidence (faithful by construction)
    llm_norag        LLM sees patient + SHAP, no evidence
    llm_rag          LLM sees patient + SHAP + retrieved evidence, must cite it
    llm_rag_verified llm_rag + claim-level verifier: failing claims are sent back to the LLM for repair
                     (max_repairs rounds) and finally pruned if still unsupported
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import requests

from ..clinical_text import value_phrase
from ..data import FEATURE_INFO
from .faithfulness import Verification, Verifier
from .kb import Chunk

SYSTEM_RULES = """You are a careful clinical decision-support writer for a research prototype.
You explain WHY a heart-disease risk model produced its output. You never diagnose and never give medication names or doses.
Return ONLY one JSON object with this exact schema:
{"summary": str,
 "claims": [{"feature": str, "direction": "increases" | "decreases", "statement": str, "citations": [str]}],
 "recommendations": [{"text": str, "citations": [str]}]}
Rules:
1. Write exactly one claim per listed driver and use its feature code unchanged.
2. direction must match the sign of its SHAP value: positive = increases the predicted risk, negative = decreases it.
3. Quote the patient's value exactly as given. Never introduce any other number unless it appears in the evidence.
4. If evidence passages are provided: every claim and recommendation must cite at least one passage id from the list,
   and may only state things supported by the cited passage. Never invent ids. If no evidence is provided leave citations empty.
5. Statements are plain language, at most 40 words. Recommendations are lifestyle or follow-up advice
   (at most 3), and always suggest discussing results with a clinician."""


# ------------------------------------------------------------------- prompts
def build_user_prompt(patient: dict, prob: float, top: list[tuple[str, float]], evidence: list[Chunk] | None) -> str:
    lines = [f"Model output: estimated probability of heart disease = {prob * 100:.0f}%.", "",
             "Top drivers (SHAP, in percentage points of probability; positive raises risk):"]
    for f, s in top:
        lines.append(f"- {f}: {value_phrase(f, patient.get(f))}; SHAP {s * 100:+.1f}")
    lines.append("")
    if evidence:
        lines.append("Evidence passages (cite by id):")
        for c in evidence:
            lines.append(f"[{c.id}] ({c.source}) {c.text}")
    else:
        lines.append("No evidence passages are provided.")
    return "\n".join(lines)


def _cites(v) -> list:
    if v is None:
        return []
    if isinstance(v, (str, int)):
        v = [v]
    return [str(x) for x in v if isinstance(x, (str, int))] if isinstance(v, list) else []


def normalize_expl(d) -> dict | None:
    """Coerce small-LLM output to the expected schema (strings instead of objects, scalar citations, ...)."""
    if not isinstance(d, dict):
        return None
    out = {"summary": d.get("summary") if isinstance(d.get("summary"), str) else ""}
    claims = d.get("claims") if isinstance(d.get("claims"), list) else []
    out["claims"] = []
    for c in claims:
        if isinstance(c, str):
            c = {"statement": c}
        if isinstance(c, dict):
            c = dict(c)
            c["citations"] = _cites(c.get("citations"))
            c["statement"] = c["statement"] if isinstance(c.get("statement"), str) else str(c.get("statement", ""))
            out["claims"].append(c)
    recs = d.get("recommendations") if isinstance(d.get("recommendations"), list) else []
    out["recommendations"] = []
    for r in recs:
        if isinstance(r, str):
            r = {"text": r}
        if isinstance(r, dict):
            r = dict(r)
            r["citations"] = _cites(r.get("citations"))
            r["text"] = r["text"] if isinstance(r.get("text"), str) else str(r.get("text", ""))
            out["recommendations"].append(r)
    for k, v in d.items():
        out.setdefault(k, v)
    return out


def parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return normalize_expl(json.loads(text))
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return normalize_expl(json.loads(m.group(0)))
        except Exception:  # noqa: BLE001
            return None
    return None


# ------------------------------------------------------------------ back-ends
class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.2, timeout: int = 240):
        self.model, self.host, self.temperature, self.timeout = model, host.rstrip("/"), temperature, timeout
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        try:
            r = requests.get(self.host + "/api/tags", timeout=3)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return r.status_code == 200 and any(self.model.split(":")[0] in n for n in names)
        except Exception:  # noqa: BLE001
            return False

    def generate(self, messages: list[dict], payload: dict | None = None) -> str:
        # num_predict caps runaway generation (small models in JSON mode can loop on whitespace until the
        # context is full, which looks like a hang on a 4 GB GPU). One retry on timeout.
        body = {"model": self.model, "messages": messages, "stream": False, "format": "json", "keep_alive": "30m",
                "options": {"temperature": self.temperature, "num_ctx": 4096, "num_predict": 900}}
        last = None
        for _ in range(2):
            try:
                r = requests.post(self.host + "/api/chat", timeout=self.timeout, json=body)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except requests.exceptions.ReadTimeout as ex:
                last = ex
        raise last


class OpenAICompatBackend:
    def __init__(self, base_url: str, model: str, api_key: str = "", temperature: float = 0.2, timeout: int = 120):
        self.base, self.model, self.key, self.temperature, self.timeout = base_url.rstrip("/"), model, api_key, temperature, timeout
        self.name = f"openai:{model}"

    def available(self) -> bool:
        return bool(self.base and self.model)

    def generate(self, messages: list[dict], payload: dict | None = None) -> str:
        h = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = requests.post(self.base + "/chat/completions", headers=h, timeout=self.timeout, json={
            "model": self.model, "messages": messages, "temperature": self.temperature,
            "response_format": {"type": "json_object"}})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def first_sentence(text: str, max_words: int = 40) -> str:
    """First sentence; if it is too long, cut at the last clause boundary before max_words."""
    s = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    w = s.split()
    if len(w) <= max_words:
        return s if s.endswith((".", "!", "?")) else s + "."
    cut = " ".join(w[:max_words])
    k = max(cut.rfind(","), cut.rfind(";"))
    if k > len(cut) * 0.5:
        cut = cut[:k]
    return cut.rstrip(",; ") + "."


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class MockBackend:
    """Deterministic, offline generator that builds a *faithful* explanation straight from the structured payload.
    It is the 'template' baseline and the source of clean explanations for fault-injection tests."""
    name = "template"

    def available(self) -> bool:
        return True

    def generate(self, messages: list[dict], payload: dict | None = None) -> str:
        assert payload is not None, "MockBackend needs the structured payload"
        return json.dumps(template_explanation(payload["patient"], payload["prob"], payload["top"], payload.get("evidence")))


def template_explanation(patient: dict, prob: float, top: list[tuple[str, float]], evidence: list[Chunk] | None) -> dict:
    claims = []
    for f, s in top:
        up = s > 0
        stmt = f"{cap(value_phrase(f, patient.get(f)))} {'raises' if up else 'lowers'} the model's estimated risk."
        cites = []
        if evidence:
            match = next((c for c in evidence if f in c.features and "prevention" not in c.features), None) or \
                next((c for c in evidence if f in c.features), None)
            if match:
                stmt += " " + first_sentence(match.text)
                cites = [match.id]
        claims.append({"feature": f, "direction": "increases" if up else "decreases", "statement": stmt, "citations": cites})
    recs = []
    if evidence:
        for c in [c for c in evidence if "prevention" in c.features][:2]:
            recs.append({"text": first_sentence(c.text), "citations": [c.id]})
    recs.append({"text": "Discuss this result with a clinician; it is a research estimate, not a diagnosis.", "citations": []})
    level = "higher" if prob >= 0.5 else "lower"
    return {"summary": f"The model estimates a {level} probability of heart disease ({prob * 100:.0f}%).",
            "claims": claims, "recommendations": recs}


def get_backend(llm_cfg: dict):
    kind = llm_cfg.get("backend", "auto")
    if kind in ("auto", "ollama"):
        b = OllamaBackend(llm_cfg["model"], llm_cfg["host"], llm_cfg["temperature"])
        if b.available():
            return b
        if kind == "ollama":
            raise RuntimeError(f"Ollama model {llm_cfg['model']} not reachable at {llm_cfg['host']}")
    if kind == "openai" or (kind == "auto" and llm_cfg.get("openai_base_url")):
        import os
        return OpenAICompatBackend(llm_cfg["openai_base_url"], llm_cfg["openai_model"],
                                   os.environ.get("OPENAI_API_KEY", ""), llm_cfg["temperature"])
    return MockBackend()


# ------------------------------------------------------------ orchestration
@dataclass
class GeneratedExplanation:
    condition: str
    expl: dict | None
    raw: str
    verification: Verification
    n_llm_calls: int = 0
    n_repairs: int = 0
    n_pruned: int = 0
    backend: str = ""
    evidence_ids: list[str] = field(default_factory=list)


def _feedback(v: Verification) -> str:
    probs = [f"- claim about '{c.feature}': " + "; ".join(c.reasons(v.rag)) for c in v.failing()]
    return ("Your previous answer failed automatic verification:\n" + "\n".join(probs) +
            "\nReturn the complete corrected JSON object. Fix only these problems, keep everything else.")


def prune(expl: dict, v: Verification) -> tuple[dict, int]:
    bad = {id(c): c for c in v.failing()}
    keep, dropped = [], 0
    fail_keys = {(c.feature, c.direction, tuple(c.citations)) for c in bad.values()}
    for cl in expl.get("claims", []):
        key = (str(cl.get("feature", "")), str(cl.get("direction", "")), tuple(str(x) for x in (cl.get("citations") or [])))
        if key in fail_keys:
            dropped += 1
        else:
            keep.append(cl)
    out = dict(expl)
    out["claims"] = keep
    return out, dropped


def generate_explanation(condition: str, backend, verifier: Verifier, patient: dict, prob: float,
                         top: list[tuple[str, float]], attr: dict[str, float], evidence: list[Chunk] | None,
                         max_repairs: int = 2) -> GeneratedExplanation:
    rag = condition in ("template_rag", "llm_rag", "llm_rag_verified")
    ev = evidence if rag else None
    ev_ids = [c.id for c in ev] if ev else []
    payload = {"patient": patient, "prob": prob, "top": top, "evidence": ev}
    be = MockBackend() if condition.startswith("template") else backend
    messages = [{"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": build_user_prompt(patient, prob, top, ev)}]
    raw = be.generate(messages, payload)
    calls = 0 if condition.startswith("template") else 1
    expl = parse_json(raw)
    ver = verifier.verify(expl, attr, patient, set(ev_ids), rag=rag)
    repairs, pruned = 0, 0
    if condition == "llm_rag_verified":
        while ver.valid_json and ver.failing() and repairs < max_repairs:
            messages = messages + [{"role": "assistant", "content": raw}, {"role": "user", "content": _feedback(ver)}]
            raw = be.generate(messages, payload)
            calls += 1
            repairs += 1
            expl = parse_json(raw)
            ver = verifier.verify(expl, attr, patient, set(ev_ids), rag=rag)
        if ver.valid_json and ver.failing():
            expl, pruned = prune(expl, ver)
            ver = verifier.verify(expl, attr, patient, set(ev_ids), rag=rag)
    return GeneratedExplanation(condition, expl, raw, ver, calls, repairs, pruned, getattr(be, "name", ""), ev_ids)


def render_text(expl: dict | None) -> str:
    if not expl:
        return ""
    expl = normalize_expl(expl) or {}
    parts = [expl.get("summary", "")]
    for c in expl.get("claims", []):
        cite = " " + "".join(f"[{x}]" for x in c.get("citations", [])) if c.get("citations") else ""
        parts.append(f"- {c.get('statement', '')}{cite}")
    for r in expl.get("recommendations", []):
        cite = " " + "".join(f"[{x}]" for x in r.get("citations", [])) if r.get("citations") else ""
        parts.append(f"* {r.get('text', '')}{cite}")
    return "\n".join(p for p in parts if p)
