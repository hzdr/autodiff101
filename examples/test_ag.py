"""autograd examples (https://github.com/hips/autograd).

Define custom derivatives via VJPs (reverse mode).

The code examples are not useful production code since those derivatives are
already implemented. We use them to learn and to show how the library operates
inside.
"""

from autograd import grad, elementwise_grad, jacobian
from autograd.extend import primitive, defvjp
import autograd.numpy as anp

import numpy as np

rand = np.random.rand


@primitive
def pow2(x):
    return x**2


def pow2_vjp(ans, x):
    # correct for x scalar or (n,), use this in production code
    return lambda v: v * 2*x


def pow2_vjp_with_jac(ans, x):
    """VJP where we really build the intermediate Jacobian. Not
    necessary in practice, only for demonstration.
    """
    # jacobian() works for scalar and 1d array input, diag() doesn't
    if x.shape == ():
        return lambda v: v * 2*x
    else:
        ##jac = jacobian(lambda x: anp.power(x,2))(x)
        jac = anp.diag(2*x)
        return lambda v: anp.dot(v, jac)


@primitive
def mysin(x):
    return anp.sin(x)


def mysin_vjp(ans, x):
    return lambda v: v * anp.cos(x)


@primitive
def mysum(x):
    return anp.sum(x)


def mysum_vjp(ans, x):
    """
    See autograd/numpy/numpy_vjps.py -> grad_np_sum() for how they do it. The
    returned v's shape must be corrected sometimes. Here we explicitly write
    the VJP using jacobian(anp.sum)(x) == anp.ones_like(x) which is always
    correct. See test_jax.py:mysum_jvp() for more comments. Note that in
    contrast to JAX, here v is *always* scalar, a fact that we can't explain
    ATM. As JAX is autograd 2.0, we consider this an autograd quirk and leave
    it be.
    """
    return lambda v: anp.dot(v, anp.ones_like(x))


def func(x):
    return anp.sum(anp.power(anp.sin(x),2))


def func_with_vjp(x):
    return mysum(pow2(mysin(x)))


def test():
    # scalar derivative
    # df/dx : R -> R
    assert anp.allclose(grad(anp.sin)(1.234), anp.cos(1.234))

    x = rand(10)*5 - 5
    assert anp.allclose(jacobian(anp.sin)(x), anp.diag(anp.cos(x)))

    # elementwise_grad(f) : R^n -> R^n (of f: R^n -> R^n), returns the column sum of
    # the Jacobian
    assert anp.allclose(elementwise_grad(anp.sin)(x), anp.cos(x))
    assert anp.allclose(jacobian(anp.sin)(x).sum(axis=0), anp.cos(x))


    defvjp(mysin, mysin_vjp)
    defvjp(mysum, mysum_vjp)
    for p2_jvp in [pow2_vjp, pow2_vjp_with_jac]:
        defvjp(pow2, p2_jvp)

        assert anp.allclose([func(xi)          for xi in x],
                            [func_with_vjp(xi) for xi in x])

        assert anp.allclose(func(x),
                            func_with_vjp(x))

        assert anp.allclose([grad(func)(xi)          for xi in x],
                            [grad(func_with_vjp)(xi) for xi in x])

        assert anp.allclose(elementwise_grad(func)(x),
                            elementwise_grad(func_with_vjp)(x))

        assert anp.allclose(grad(func)(x),
                            grad(func_with_vjp)(x))


if __name__ == '__main__':
    test()
