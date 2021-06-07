# https://github.com/google/jax/blob/master/jax/_src/lax/lax.py

def sin(x: Array) -> Array:
  r"""Elementwise sine: :math:`\mathrm{sin}(x)`."""
  return sin_p.bind(x)

def mul(x: Array, y: Array) -> Array:
  r"""Elementwise multiplication: :math:`x \times y`."""
  return mul_p.bind(x, y)

sin_p = standard_unop(_float | _complex, 'sin')
ad.defjvp(sin_p, lambda g, x: mul(g, cos(x)))
