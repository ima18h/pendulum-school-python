import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class ExponentialDecay: 
    def __init__(self, a): 
        self.a = a
        if a < 0: 
            raise ValueError("The constant a cannot be negative")

    def __call__(self, t, u):   
        """Definerer funksjonen f(t, u)"""
        du_dt = -self.a*u
        return du_dt            # Returnerer den deriverte du/dt = -a*u

# Oppgave 1c)
    def solve(self, u0, T, dt): 
        """ Beregner løsninger for 0 <= t <= T """
        solution = solve_ivp(
            ExponentialDecay(self.a), 
            [0, T], 
            [u0], 
            t_eval = np.linspace(0, T, T//dt)
        )
        return solution.t, solution.y[0]

# Tester solve metoden
a = 0.05
u0 = 8
T = 100
dt = 1
decay_model = ExponentialDecay(a)
t, u = decay_model.solve(u0, T, dt)

plt.plot(t, u)
plt.show()
