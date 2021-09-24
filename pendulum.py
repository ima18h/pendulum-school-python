import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Oppgave 2a)
class Pendulum:
    """
    Klasse Pendulum som består av en masse, M, og en masseløs tråd, L, som 
    beregner hvordan en pendulum oppfører seg når den svinger fritt fra et 
    festepunkt og bare blir påvirka av tyngdekraften g."""
    def __init__(self, L=1, M=1, g=9.81):
        """
        Tar inn parameterne L, M og g. Om noe annet ikke er gitt så er 
        L = 1 m, M = 1 kg og g = 9.81 m/s^2 som standardverdi.
        """
        self.L = L
        self.M = M
        self.g = g
        self._solved = False

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
        """
        Beregner løsninger av ODE-systemet for 0 <= t <= T ved å bruke scipys
        innebygde metoden solve_ivp. Solve tar også inn en vinkel som enten gis
        i radianer eller grader. Har satt til radianer som standard, og dersom
        vinkelen blir gitt som grader, gjøres den om til radianer.
        """
        if angle == "deg":
            y0[0] = np.radians(y0[0])

        sol = solve_ivp(self, (0, T), y0, max_step=dt)
        self._t = sol.t
        self._theta, self._omega = sol.y[0], sol.y[1]
        self._solved = True

# Oppgave 2d)
    @property
    def t(self):
        """
        Lager t som en privat attributt, og kaster en AssertionError om man 
        prøver å hente den før metoden solve blir kalt på.
        """
        if self._solved:
            return self._t
        else:
            raise AssertionError(
                "No solution found. Did you remember to call solve?"
            )

    @property
    def theta(self):
        """
        Lagrer theta som en privat attributt og kaster en AssertionError om man 
        prøver å hente den før metoden solve blir kalt på.
        """
        if self._solved:
            return self._theta
        else:
            raise AssertionError(
                "No solution found. Did you remember to call solve?"
            )

    @property
    def omega(self):
        """
        Lagrer omega som en privat attributt og kaster en AssertionError om man 
        prøver å hente den før metoden solve blir kalt på.
        """
        if self._solved:
            return self._omega
        else:
            raise AssertionError(
                "No solution found. Did you remember to call solve?"
            )

# Oppgave 2f)
    @property
    def x(self):
        """
        Returnerer en array av horisontale verdier i kartetiske koordinater.
        Plasseres i midten av koordinatsystemet i festepunktet til pendelen.
        """
        return self.L * np.sin(self.theta)

    @property
    def y(self):
        """
        Returnerer en array av vertikale verdier i kartetiske koordinater.
        Plasseres i midten av koordinatsystemet i festepunktet til pendelen."
        """
        return -(self.L * np.cos(self.theta))

# Oppgave 2g)
    @property
    def potential(self):
        """Beregner potensiell energi."""
        return self.M * self.g * (self.y + self.L)

    @property
    def vx(self):
        """Farten, v_x, til x-verdiene i pendulumen."""
        return np.gradient(self.x, self.t)

    @property
    def vy(self):
        """Farten, v_y, til y-verdiene i pendulumen."""
        return np.gradient(self.y, self.t)

    @property
    def kinetic(self):
        """Beregner kinetisk energi."""
        return (1 / 2) * self.M * (self.vx ** 2 + self.vy ** 2)


class DampenedPendulum(Pendulum):
    """
    Lik metode som Pendulum, men her avtar svingningene mye kraftigere fordi vi
    ikke bevarer energien som vi har gjort i Pendulum.
    """
    def __init__(self, B, L=1, M=1, g=9.81):
        """
        Arver paramterne L, M og G paramaterne fra Pendulum. 
        I tillegg lagrer den paramteren B som en privat attributt.
        """
        super().__init__(L, M, g)
        self._B = B

    def __call__(self, t, y):
        """
        Beregner og returnerer den deriverte av omega og deriverte av theta
        """
        theta, omega = y
        theta_deriv = omega
        omega_deriv = (
            (-self.g / self.L) * np.sin(theta)
        ) - (self._B / self.M) * omega
        return theta_deriv, omega_deriv


# Oppgave 2h) og 2i) 
if __name__ == '__main__': 
    """Tester Pendulum og DampenedPendulum klassene med plots."""
    pend = Pendulum()
    #pend = DampenedPendulum(0.25)
    pend.solve((3 * np.pi / 7, 0), 10, 0.01)

    plt.plot(pend.t, pend.kinetic + pend.potential, label="Total energy")
    plt.plot(pend.t, pend.potential, label="Potential energy")
    plt.plot(pend.t, pend.kinetic, label="Kinetic energy")
    plt.plot(pend.t, pend.theta, label=r"$\theta$-values")
    plt.legend()
    plt.title("Graphs of energy conservation")
    plt.show()
