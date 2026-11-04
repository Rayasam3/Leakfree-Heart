from leakfree_heart.rag.generator import parse_json, render_text


def test_malformed_llm_output_is_normalised():
    raw = '{"summary": "s", "claims": ["plain string", {"statement": "x", "citations": "KB-1"}], "recommendations": ["see a doctor"]}'
    e = parse_json(raw)
    assert e["claims"][0]["statement"] == "plain string"
    assert e["claims"][1]["citations"] == ["KB-1"]
    assert "see a doctor" in render_text(e)
