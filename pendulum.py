import numpy as np

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
        dtheta_dt = omega
        domega_dt = -self.g/self.L*np.sin(theta)
        return dtheta_dt, domega_dt
