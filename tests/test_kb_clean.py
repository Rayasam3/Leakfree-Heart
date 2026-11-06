from leakfree_heart.rag.kb import chunk_sentences, clean_web_text

MESSY = ("Coronary artery disease happens when the arteries that supply blood to the heart muscle become hardened and narrowed. "
         "Also in Spanish Hypertension - overview (Medical Encyclopedia) Also in Spanish High Blood Pressure Facts (Centers for Disease Control and Prevention). "
         "Regular physical activity and a heart-healthy diet can lower the risk of coronary artery disease over time. "
         "Systolic Blood Pressure Intervention Trial (SPRINT) Study (National Heart, Lung, and Blood Institute) ClinicalTrials.gov: Hypertension. "
         "Quitting smoking is one of the most effective steps a person can take to protect the heart and blood vessels from damage.")


def test_boilerplate_removed():
    c = clean_web_text(MESSY)
    assert "Also in Spanish" not in c and "ClinicalTrials" not in c and "Encyclopedia" not in c
    assert "physical activity" in c and "Quitting smoking" in c
    ch = chunk_sentences(c, min_words=10)
    assert ch and all("Spanish" not in x for x in ch)


def test_shap_contribution_number_is_grounded():
    from leakfree_heart.rag import faithfulness as F
    import inspect
    assert "_is_contrib" in inspect.getsource(F)
