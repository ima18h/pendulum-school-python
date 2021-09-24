import pytest
import numpy as np
from double_pendulum import DoublePendulum, ODEsNotSolve

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
TOL = 1e-14

def test_DoublePendulum_at_rest():
    y = (0, 0, 0, 0)
    expected_theta1_deriv = 0
    expected_theta2_deriv = 0
    expected_omega1_deriv = 0
    expected_omega2_deriv = 0
    double_pend = DoublePendulum()
    theta1_deriv, omega1_deriv, theta2_deriv, omega2_deriv = double_pend(0, y)
    assert abs(theta1_deriv-expected_theta1_deriv) < TOL
    assert abs(theta2_deriv-expected_theta2_deriv) < TOL
    assert abs(omega1_deriv-expected_omega1_deriv) < TOL
    assert abs(omega2_deriv-expected_omega2_deriv) < TOL

def test_calling_private_attributes_raises_ODEsNotSolveError():
    double_pend = DoublePendulum()
    with pytest.raises(ODEsNotSolve):
        double_pend.t, double_pend.theta1, double_pend.theta2, 
        double_pend.omega1, double_pend.omega2
    
    double_pend.solve([np.pi/6, 0.15, np.pi/3, 0.15], 10, 1)
    double_pend.t, double_pend.theta1, double_pend.theta2, double_pend.omega1, 
    double_pend.omega2

def test_initial_condition_zero_gives_arrays_zero():
    T = 10
    dt = 1
    double_pend = DoublePendulum()
    double_pend.solve((0, 0, 0, 0), T, dt)

    expected_theta1 = np.zeros_like(double_pend.theta1)
    expected_theta2 = np.zeros_like(double_pend.theta2)
    expected_omega1 = np.zeros_like(double_pend.omega1)
    expected_omega2 = np.zeros_like(double_pend.omega2)

    assert np.all(double_pend.theta1 == expected_theta1)
    assert np.all(double_pend.theta2 == expected_theta2)
    assert np.all(double_pend.omega1 == expected_omega1)
    assert np.all(double_pend.omega2 == expected_omega2)
    assert np.all(double_pend.t >= 0)
    assert double_pend.t[0] == 0
    assert double_pend.t[-1] == T
