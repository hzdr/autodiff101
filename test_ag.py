from autograd import grad, elementwise_grad, jacobian
from autograd.extend import primitive, defvjp
import autograd.numpy as wnp
import numpy as np

rand = np.random.rand

# scalar derivative
# df/dx : R -> R
assert wnp.allclose(grad(wnp.sin)(1.234), wnp.cos(1.234))

# The Jacobian of a vectorized numpy function is diagonal and square $J \in
# R^{n\times n}$ with
# $J_{ii} = df_i/dx_i$ where $f_i \equiv f \forall i$
# $J : R^n -> R^n$
x = rand(10)*5 - 5
assert wnp.allclose(jacobian(wnp.sin)(x), wnp.diag(wnp.cos(x)))

# `elementwise_grad` df/dx : R^n -> R^n (vectorized), returns the column sum of
# the Jacobian
assert wnp.allclose(elementwise_grad(wnp.sin)(x), wnp.cos(x))
assert wnp.allclose(jacobian(wnp.sin)(x).sum(axis=0), wnp.cos(x))


@primitive
def pow2(x):
    return x**2


@primitive
def mysin(x):
    return wnp.sin(x)


@primitive
def mysum(x):
    return wnp.sum(x)


def pow2_vjp(ans, x):
    # correct for x scalar or (n,), use this in production code
    return lambda g: g * 2*x


def pow2_vjp_with_jac(ans, x):
    """ VJP where we really build the intermediate Jacobian. Not
    necessary in practice, only for demonstration.
    """
    # called from grad(func)(x) with x scalar, here we get passed np.array(x)
    # which has shape ()
    if x.shape == ():
        return lambda g: g * 2*x
    # called from elementwise_grad(func)(x) with x 1d array
    else:
        # The same:
        #   jac = np.diag(2*x)
        jac = jacobian(lambda x: wnp.power(x,2))(x)
        return lambda g: wnp.dot(g, jac)


def mysin_vjp(ans, x):
    return lambda g: g * wnp.cos(x)


# Works in JAX but not here in the elementwise_grad() test. See
# autograd/numpy/numpy_vjps.py -> grad_np_sum() for ow they do it. The returned
# g's shape must be corrected sometimes.
##def mysum_vjp(ans, x):
##    return lambda g: g


def func(x):
    return wnp.sum(wnp.power(wnp.sin(x),2))


def func_with_vjp(x):
    return wnp.sum(pow2(mysin(x)))


defvjp(mysin, mysin_vjp)
for p_jvp in [pow2_vjp, pow2_vjp_with_jac]:
    defvjp(pow2, p_jvp)

    assert wnp.allclose([func(xi)          for xi in x],
                        [func_with_vjp(xi) for xi in x])

    assert wnp.allclose(func(x),
                        func_with_vjp(x))

    assert wnp.allclose([grad(func)(xi)          for xi in x],
                        [grad(func_with_vjp)(xi) for xi in x])

    assert wnp.allclose(elementwise_grad(func)(x),
                        elementwise_grad(func_with_vjp)(x))
