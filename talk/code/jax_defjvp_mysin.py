import jax
import jax.numpy as np

@jax.custom_jvp
def mysin(x):
    return x - x**3/6 + x**5/120 - x**7/5040
