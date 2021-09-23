import pytest
import numpy as np
from double_pendulum import DoublePendulum

# Oppgave 3b)
@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0),
        (0, 0.5, 3.386187037),
        (0.5, 0, -7.678514423),
        (0.5, 0.5, -4.703164534),
    ]
)
def test_domega1_dt(theta1, theta2, expected):
    dp = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = dp(t, y)
    assert np.isclose(dtheta1_dt, 0.25)
    assert np.isclose(domega1_dt, expected)


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0.0),
        (0, 0.5, -7.704787325),
        (0.5, 0, 6.768494455),
        (0.5, 0.5, 0.0),
    ],
)
def test_domega2_dt(theta1, theta2, expected):
    dp = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = dp(t, y)
    assert np.isclose(dtheta2_dt, 0.15)
    assert np.isclose(domega2_dt, expected)


# Oppgave 3f)
# Need more work for these tests? What to tests? 
# Is it even possible to test solve since it doesn't return anything!
@pytest.mark.parametrize(
    "theta, omega, T, dt", 
    [
        (np.pi/2, 0.15, 5, 1), 
        (np.pi/3, 0.5, 10, 0.02), 
        (np.pi/6, 1, 20, 2)
    ]
)
def test_DoublePendulum_solve_timesteps(y0, T, dt):
    t = np.linspace(0, T, int(T/dt)+1)
    y = DoublePendulum().solve(y0, T, dt)
    tol = 1e-14
    assert np.all(abs(y[0]-t) < tol) 

def test_properties_raise_AssertionError_before_solve():
    double_pendulum = DoublePendulum()
    with pytest.raises(AssertionError):
        double_pendulum.t, double_pendulum.theta1, double_pendulum.theta2, 
        double_pendulum.omega1, double_pendulum.omega2 

def test_kinetic_and_potential_is_constant():
    pass
