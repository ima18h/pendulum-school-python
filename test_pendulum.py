import numpy as np
from pendulum import Pendulum

def test_Pendulum():
    t = 0  
    y = (np.pi/6, 0.15) 
    L = 2.7
    tol = 1e-12
    expected_dtheta = 0.15
    expected_domega = -0.03320283660974
    pendulum_object = Pendulum(L=L)
    dtheta_dt, domega_dt = pendulum_object(t, y)
    print(domega_dt, expected_domega)
    assert abs(dtheta_dt-expected_dtheta) < tol
    assert abs(domega_dt-expected_domega) < tol
