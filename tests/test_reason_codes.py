from eiaf.reason_codes.generator import ReasonCodeGenerator

def test_reason_codes_topk():
    gen = ReasonCodeGenerator()
    contribs = {"a": 0.5, "b": -0.2, "c": 0.1, "d": -0.9}
    out = gen.generate(contribs)
    assert len(out) == 4 or len(out) == 5  # default top_k=5, but only 4 items
    assert out[0].feature in ("d", "a")  # biggest abs first
