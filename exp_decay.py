import numpy as np
import matplotlib.pyplot as plt


class ExponentialDecay: 
    def __init__(self, a): 
        self.a = a
        if a < 0: 
            raise ValueError("The constant a cannot be negative")

    def __call__(self, t, u):   # Makes the function f(t, u)
        self.t = t
        self.u = u
        du_dt = -self.a*u
        return du_dt     # Returns the derivative du/dt = -a*u
