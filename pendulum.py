import numpy as np
from scipy.integrate import solve_ivp

# Oppgave 2a)
class Pendulum: 
    def __init__(self, L=1, M=1, g=9.81):
        """
        Tar inn parameterne L, M og g. Om noe annet ikke er gitt så er 
        L = 1 m, M = 1 kg og g = 9.81 m/s^2 som standardverdi.
        """
        self.__t = None
        self._theta, self._omega = None, None
        self.__L = L
        self.__M = M
        self.__g = g
        self.__solved = False

    @property
    def t(self):
        if self.__solved:
            return self.__t
        else:
            raise AttributeError("not solved yet")

    @property
    def theta(self):
        if self.__solved:
            return self._theta
        else:
            raise AttributeError("not solved yet")

    @property
    def omega(self):
        if self.__solved:
            return self._omega
        else:
            raise AttributeError("not solved yet")

    def __call__(self, t, y):
        """
        Tar inn parameterne t og y som brukes i to funksjoner og
        returnerer de deriverte (h.s. av ODE-systemene).
        """
        theta, omega = y
        theta_deriv = omega
        omega_deriv = -self.__g/self.__L*np.sin(theta)
        return theta_deriv, omega_deriv

    # Oppgave 2c)
    def solve(self, y0, T, dt, angle="rad"):
        """ Beregner løsninger av ODE-systemet for 0 <= t <= T """
        if angle == "deg": 
            angle = np.radians("deg")

        sol = solve_ivp(self, (0, T), y0, max_step=dt)
        self.__t = sol.t
        self._theta, self._omega = sol.y[0], sol.y[1]
        self.__solved = True


# just for basic test, need better test case
pend = Pendulum()
pend.solve((0, 0), 1, 0.1)
tt = pend.t
