import numpy as np
from numpy.lib.function_base import kaiser
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Oppgave 2a)
class Pendulum:
    def __init__(self, L=1, M=1, g=9.81):
        """
        Tar inn parameterne L, M og g. Om noe annet ikke er gitt så er 
        L = 1 m, M = 1 kg og g = 9.81 m/s^2 som standardverdi.
        """
        self.L = L
        self.M = M
        self.g = g
        self._solved = False

    @property
    def t(self):
        if self._solved:
            return self._t
        else:
            raise AssertionError(
                "No solution found. Did you remember to call solve?"
            )

    @property
    def theta(self):
        if self._solved:
            return self._theta
        else:
            raise AssertionError(
                "No solution found. Did you remember to call solve?"
            )

    @property
    def omega(self):
        if self._solved:
            return self._omega
        else:
            raise AssertionError(
                "No solution found. Did you remember to call solve?"
            )

    @property
    def x(self):
        return self.L * np.sin(self.theta)

    @property
    def y(self):
        return -(self.L * np.cos(self.theta))

    @property
    def potential(self):
        return self.M * self.g * (self.y + self.L)

    @property
    def vx(self):
        return np.gradient(self.x, self.t)

    @property
    def vy(self):
        return np.gradient(self.y, self.t)

    # TO DO: det er noe som feiler formlene.
    # kinetic er alt for lav, eller så er potential alt for høy.
    # virker som periodene stemmer, men størrelsen på kinetic og potential er rar.
    @property
    def kinetic(self):
        return (1 / 2) * self.M * (self.vx ** 2 + self.vy ** 2)

    def __call__(self, t, y):
        """
        Tar inn parameterne t og y som brukes i to funksjoner og
        returnerer de deriverte (h.s. av ODE-systemene).
        """
        theta, omega = y
        theta_deriv = omega
        omega_deriv = -self.g / self.L * np.sin(theta)
        return theta_deriv, omega_deriv

    # Oppgave 2c)
    def solve(self, y0, T, dt, angle="rad"):
        """ Beregner løsninger av ODE-systemet for 0 <= t <= T """
        if angle == "deg":
            y0[0] = np.radians(y0[0])

        sol = solve_ivp(self, (0, T), y0, t_eval=np.linspace(0, T, int(T/dt)+1))
        self._t = sol.t
        self._theta, self._omega = sol.y[0], sol.y[1]
        self._solved = True


class DampenedPendulum(Pendulum):
    def __init__(self, B, L=1, M=1, g=9.81):
        super().__init__(L, M, g)
        self._B = B

    def __call__(self, t, y):
        theta, omega = y
        theta_deriv = omega
        omega_deriv = (
            (-self.g / self.L) * np.sin(theta)
        ) - (self._B / self.M) * omega
        return theta_deriv, omega_deriv


if __name__ == '__main__': 
    pend = Pendulum()
    pend.solve((np.pi / 3, 0), 10, 0.001)

    plt.plot(pend.t, pend.kinetic + pend.potential, label="kinetic + potential")
    plt.plot(pend.t, pend.potential, label="potential energy")
    plt.plot(pend.t, pend.kinetic, label="kinetic energy")
    plt.plot(pend.t, pend.theta, label=r"$\theta$-values")
    plt.legend()
    plt.title("Graphs of energy conservation")
    plt.show()
