import numpy as np
from pendulum import Pendulum

TOL = 1e-14

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


