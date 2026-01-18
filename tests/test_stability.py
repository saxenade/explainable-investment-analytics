import numpy as np
from eiaf.metrics.stability import compute_psi

def test_compute_psi_shapes():
    train = np.random.RandomState(0).normal(size=(100, 3))
    score = np.random.RandomState(1).normal(size=(120, 3))
    psi = compute_psi(train, score, ["a", "b", "c"], bins=8)
    assert set(psi.keys()) == {"a", "b", "c"}
    assert all(isinstance(v, float) for v in psi.values())
