import numpy as np


class DoublePendulum:
    def __init__(self, L1=1, L2=1, g=9.81):
        self._L1 = L1
        self._L2 = L2
        self._g = g

    def __call__(self, t, y):
        theta1, omega1, theta2, omega2 = y
        dtheta1 = omega1
        dtheta2 = omega2
        delta_t = theta2 - theta1
        domega1 = ((self._L1 * omega1 ** 2 * np.sin(delta_t) * np.cos(delta_t))
                   + (self._g * np.sin(theta2) * np.cos(delta_t))
                   + (self._L2 * omega2 ** 2 * np.sin(delta_t))
                   - (2 * self._g * np.sin(theta1))) / ((2 * self._L1)
                                                        - (self._L1 * np.cos(delta_t) ** 2))

        domega2 = (-(self._L2 * omega2 ** 2 * np.sin(delta_t) * np.cos(delta_t))
                   + (2 * self._g * np.sin(theta1) * np.cos(delta_t))
                   - (2 * self._L1 * omega1 ** 2 * np.sin(delta_t))
                   - (2 * self._g * np.sin(theta2))) / ((2 * self._L2)
                                                        - (self._L2 * np.cos(delta_t) ** 2))
        return dtheta1, domega1, dtheta2, domega2
