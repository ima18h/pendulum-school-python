import numpy as np
from scipy.integrate import solve_ivp


class DoublePendulum:
    def __init__(self, L1=1, L2=1, g=9.81):
        self._L1 = L1
        self._L2 = L2
        self._g = g
        self._t = None
        self._theta1, self._omega1, self._theta2, self._omega2 = None, None, None, None

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

    def solve(self, y0, T, dt, angle="rad"):
        """ Beregner løsninger av ODE-systemet for 0 <= t <= T """
        if angle == "deg":
            angle = np.radians("deg")

        sol = solve_ivp(self, (0, T), y0, max_step=dt)
        self._t = sol.t
        self._theta1, self._omega1, self._theta2, self._omega2 = sol.y[0], sol.y[1], sol.y[2], sol.y[3]

    @property
    def t(self):
        return self._t

    @property
    def theta1(self):
        return self._theta1

    @property
    def theta2(self):
        return self._theta2

    @property
    def x1(self):
        return self._L1 * np.sin(self.theta1)

    @property
    def y1(self):
        return -self._L1 * np.cos(self.theta1)

    @property
    def x2(self):
        return self.x1 + self._L2 * np.sin(self.theta2)

    @property
    def y2(self):
        return self.y1 - self._L2 * np.cos(self.theta2)

    @property
    def potential(self):
        # masses are 1
        p1 = self._g * (self.y1 + self._L1)
        p2 = self._g * (self.y2 + self._L2 + self._L1)
        return p1 + p2

    @property
    def vx1(self):
        return np.gradient(self.x1)

    @property
    def vy1(self):
        return np.gradient(self.y1)

    @property
    def vx2(self):
        return np.gradient(self.x2)

    @property
    def vy2(self):
        return np.gradient(self.y2)

    @property
    def kinetic(self):
        # masses are 1
        k1 = 1/2 * (self.vx1**2 + self.vy1**2)
        k2 = 1/2 * (self.vx2**2 + self.vy2**2)
        return k1 + k2

