import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class ExponentialDecay: 
    def __init__(self, a): 
        """Tar inn en konstant a som kun kan være positiv."""
        self.a = a
        if a < 0: 
            raise ValueError("The constant a cannot be negative")

    def __call__(self, t, u):  
        """
        Tar inn parameterne t og u i en funksjon f(t, u) og
        returnerer den deriverte (h.s. av ODE-systemet).
        """
        du_dt = -self.a*u
        return du_dt            

# Oppgave 1c)
    def solve(self, u0, T, dt):
        """ Beregner løsninger av ODE-systemet for 0 <= t <= T """
        solution = solve_ivp(
            ExponentialDecay(self.a), [0, T], [u0],
            t_eval=np.linspace(0, T, T // dt)
        )
        return solution.t, solution.y[0]


# Tester solve metoden
a = 0.05
u0_list = [2, 4, 8, 12]
T = 100
dt = 1
for u0 in u0_list:
    decay_model = ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)
    plt.plot(t, u, label=fr"$u_0$ = {u0}")

plt.xlabel("x-akse")
plt.ylabel("y-akse")
plt.title(r"Løsning av ODE-systemet $ \frac{du}{dt} = -a \cdot u$")
plt.legend()
plt.show()
