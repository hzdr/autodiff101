import jax
import jax.numpy as np


def mysin(x):
    return x - x**3/6 + x**5/120 - x**7/5040
