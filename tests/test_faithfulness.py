import numpy as np
import pytest

from leakfree_heart.config import load_config, path
from leakfree_heart.data import FEATURES
from leakfree_heart.rag import Retriever, build_chunks
from leakfree_heart.rag.faithfulness import Verifier
from leakfree_heart.rag.generator import MockBackend, generate_explanation, parse_json, template_explanation
from leakfree_heart.experiments.faith import EXPECTED_CHECK, FAULTS, inject_fault

CFG = load_config()
KB = path(CFG, "knowledge_base")


@pytest.fixture(scope="module")
def kit():
    chunks = build_chunks(KB, include_documents=False)
    return chunks, Retriever(chunks, backend="tfidf"), Verifier(chunks)


PATIENT = dict(age=62.0, sex=1.0, cp=3.0, trestbps=145.0, chol=268.0, fbs=1.0, restecg=1.0, thalach=112.0,
               exang=1.0, oldpeak=2.8, slope=1.0, ca=2.0, thal=3.0)
ATTR = dict(zip(FEATURES, [-0.01, 0.04, 0.08, 0.01, 0.01, 0.0, -0.01, 0.04, 0.07, 0.07, 0.04, 0.18, 0.07]))


def _setup(kit):
    chunks, retr, ver = kit
    top = sorted(ATTR.items(), key=lambda kv: -abs(kv[1]))[:5]
    topk = [f for f, _ in top]
    ev = retr.retrieve_for_explanation(PATIENT, topk, 2)
    return top, topk, ev, ver


def test_clean_template_passes_every_check(kit):
    top, topk, ev, ver = _setup(kit)
    g = generate_explanation("template_rag", MockBackend(), ver, PATIENT, 0.9, top, ATTR, ev)
    m = g.verification.metrics()
    assert m["fully_verified_rate"] == 1.0 and m["recall_at_k"] == 1.0 and m["efs"] == 1.0


@pytest.mark.parametrize("fault", FAULTS)
def test_every_injected_fault_is_caught(kit, fault):
    top, topk, ev, ver = _setup(kit)
    clean = template_explanation(PATIENT, 0.9, top, ev)
    rng = np.random.default_rng(0)
    caught = tried = 0
    for _ in range(8):
        bad, idx = inject_fault(clean, fault, topk, PATIENT, {c.id: c for c in ev}, rng)
        if bad is None:
            continue
        tried += 1
        v = ver.verify(bad, ATTR, PATIENT, {c.id for c in ev}, rag=True)
        caught += not v.checks[idx].passed
    assert tried > 0 and caught == tried


def test_unparseable_output_is_scored_as_invalid(kit):
    top, topk, ev, ver = _setup(kit)
    v = ver.verify(parse_json("I think the patient is at risk."), ATTR, PATIENT, set(), rag=False)
    assert not v.valid_json and v.metrics()["efs"] == 0.0


def test_no_rag_citations_are_fabricated_by_definition(kit):
    top, topk, ev, ver = _setup(kit)
    e = template_explanation(PATIENT, 0.9, top, None)
    e["claims"][0]["citations"] = ["KB-018"]
    v = ver.verify(e, ATTR, PATIENT, set(), rag=False)
    assert v.checks[0].fabricated_ids == ["KB-018"]
