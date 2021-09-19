import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

    @property
    def x(self):
        return self.__L * np.sin(self.theta)

    @property
    def y(self):
        return -(self.__L * np.cos(self.theta))

    @property
    def potential(self):
        return self.__M * self.__g * (self.y + self.__L)

    @property
    def vx(self):
        return np.gradient(self.x)

    @property
    def vy(self):
        return np.gradient(self.y)

    # TODO: det er noe som feiler formlene.
    # kinetic er alt for lav, eller så er potential alt for høy.
    # virker som periodene stemmer, men størrelsen på kinetic og potential er rar.
    @property
    def kinetic(self):
        return (1/2) * self.__M * (self.vx**2 + self.vy**2)

    def __call__(self, t, y):
        """
        Tar inn parameterne t og y som brukes i to funksjoner og
        returnerer de deriverte (h.s. av ODE-systemene).
        """
        theta, omega = y
        theta_deriv = omega
        omega_deriv = -self.__g / self.__L*np.sin(theta)
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
# TODO: add titles and such to plots
pend = Pendulum()
pend.solve((np.pi/2, 1), 4, 0.1)

plt.plot(pend.t, pend.theta)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(pend.t, pend.kinetic)
ax2.plot(pend.t, pend.potential, 'tab:orange')
ax3.plot(pend.t, pend.theta, 'tab:green')
ax4.plot(pend.t, pend.y, 'tab:red')
plt.show()
print(pend.potential+pend.kinetic)
