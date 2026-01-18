from eiaf.audit.artifacts import ExplanationBundle

def test_bundle_to_dict():
    b = ExplanationBundle(
        schema_version="1.0",
        model={"name":"m","type":"t"},
        data_summary={"n_rows":10,"n_cols":2,"missing_by_feature":{"a":0,"b":1}},
        global_explanations={"permutation_importance":{"a":0.2}},
        local_explanations={"sample_index":0,"what_if_contributions":{"a":0.1}},
        reason_codes=[{"code":"RC-01","message":"x","feature":"a","direction":"up","strength":0.1}],
        stability=None,
        model_card_markdown="# card"
    )
    d = b.to_dict()
    assert d["schema_version"] == "1.0"
    assert "global_explanations" in d
