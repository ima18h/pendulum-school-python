import numpy as np
from scipy.integrate import solve_ivp

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

    def __call__(self, t, y):
        """
        Tar inn parameterne t og y som brukes i to funksjoner og
        returnerer de deriverte (h.s. av ODE-systemene).
        """
        theta, omega = y
        theta_deriv = omega
        omega_deriv = -self.g/self.L*np.sin(theta)
        return theta_deriv, omega_deriv

    # Oppgave 2c)
    def solve(self, y0, T, dt, angle="rad"):
        """ Beregner løsninger av ODE-systemet for 0 <= t <= T """
        if angle == "deg": 
            angle = np.radians("deg")

        solution = solve_ivp(
            Pendulum(self.L, self.M, self.g), [0, T], [y0],
            t_eval=np.linspace(0, T, T // dt)
        )
        self.solution_t, self.solution_y = solution.t, solution.y[0]
        