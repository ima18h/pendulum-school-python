from exp_decay import ExponentialDecay
import pytest


def test_ExponentialDecay():
    t = 1 #?
    u_t = 3.2
    a = 0.4
    exp_decay = ExponentialDecay(a)
    expected = -1.28
    tol = 1e-14
    assert abs(exp_decay(t, u_t) - expected) < tol

def test_is_a_negative_raise_ValueError():
    with pytest.raises(ValueError):
        ExponentialDecay(-5)