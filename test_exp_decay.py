from exp_decay import ExponentialDecay
from scipy.integrate import solve_ivp
import numpy as np
import pytest

# Oppgave b)
def test_ExponentialDecay():
    t = 1 # Kan velges fritt, endrer ikke svaret til den deriverte
    u_t = 3.2
    a = 0.4
    exp_decay = ExponentialDecay(a)
    expected = -1.28
    tol = 1e-14
    assert abs(exp_decay(t, u_t) - expected) < tol
    assert exp_decay(0, 0.728) == exp_decay(2, 0.728)

def test_is_a_negative_raise_ValueError():
    with pytest.raises(ValueError):
        ExponentialDecay(-5)



# Oppgave 1d)
# TO DO: fix last test for 1d)
@pytest.mark.parametrize(
    "a, u0, T", [(2, 3, 4), (0.2, 6, 10), (1, 5, 20)]
)
def test_exp_decay_solve_timesteps(a, u0, T):
    dt = 1 
    t = np.linspace(0, T, int(T/dt)+1)
    u = ExponentialDecay(a).solve(u0, T, dt)
    tol = 1e-14
    assert np.all(abs(u[0]-t) < tol)

@pytest.mark.parametrize(
    "a, u0, T", [(2, 3, 4), (0.2, 6, 10), (1, 5, 20)]
)
def test_exp_decay_solve_solutions(a, u0, T):
    dt = 1 
    u = ExponentialDecay(a).solve(u0, T, dt)
    expected = u0*np.exp(-a*u[0])
    tol = 1e-14
    print(expected[:5], u[1][:5])
    # fun  = lambda t, u: -a*u
    # expected = solve_ivp(fun, [0, T], [u0], t_eval = np.linspace(0, T, T//dt))
    assert np.all(abs(u[1]-expected) < tol) # f"max diff: {np.max(abs(u[1]-expected.y[0]))}\nmin diff: {np.min(abs(u[1]-expected.y[0]))}\n{len((abs(u[1]-expected.y[0]) < tol) == False)}"
    
    