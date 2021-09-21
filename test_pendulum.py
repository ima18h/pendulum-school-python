import numpy as np
from pendulum import Pendulum
import pytest

TOL = 1e-14

# Oppgave 2b)
def test_Pendulum_call():
    y = (np.pi/6, 0.15) 
    L = 2.7
    expected_theta_deriv = 0.15
    expected_omega_deriv = -1.816666666666667
    pendulum = Pendulum(L=L)
    theta_deriv, omega_deriv = pendulum(0, y)
    assert abs(theta_deriv-expected_theta_deriv) < TOL
    assert abs(omega_deriv-expected_omega_deriv) < TOL

def test_Pendulum_at_rest():
    y = (0, 0)
    expected_theta_deriv = 0
    expected_omega_deriv = 0
    pendulum = Pendulum()
    theta_deriv, omega_deriv = pendulum(0, y)
    assert abs(theta_deriv-expected_theta_deriv) < TOL
    assert abs(omega_deriv-expected_omega_deriv) < TOL

# Oppgave 2e)
def test_calling_private_attributes_raises_AssertionError():
    pendulum = Pendulum()
    with pytest.raises(AssertionError):
        pendulum.t, pendulum.theta, pendulum.omega
    
    pendulum.solve()
    pendulum.t, pendulum.theta, pendulum.omega

def test_initial_condition_zero_gives_arrays_zero():
    T = 10
    dt = 1
    pendulum = Pendulum()
    pendulum.solve((0, 0), T, dt)

    expected_theta = np.zeros_like(pendulum.theta)
    expected_omega = np.zeros_like(pendulum.omega)
    expected_t_values = [i * dt for i in range(T+dt)]

    assert np.all(pendulum.theta == expected_theta)
    assert np.all(pendulum.omega == expected_omega)
    assert np.all(abs(pendulum.t - expected_t_values) - TOL)