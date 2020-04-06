from autograd import grad, elementwise_grad, jacobian
from autograd.extend import primitive, defvjp
import autograd.numpy as np

# XXX pytorch now has a jax-like API
# https://github.com/pytorch/pytorch/commit/1f4a4aaf643b70ebcb40f388ae5226a41ca57d9b


# scalar derivative
# df/dx : R -> R
dfdx = grad(np.sin)
assert np.allclose(grad(np.sin)(1.234), np.cos(1.234))
assert np.allclose(dfdx(1.234), np.cos(1.234))

# The Jacobian of a vectorized numpy function is diagonal and square $J \in
# R^{n\times n}$ with
# $J_{ii} = df_i/dx_i$ where $f_i \equiv f \forall i$
# $J : R^n -> R^n$
x = np.random.rand(10)*5 - 5
assert np.allclose(jacobian(np.sin)(x), np.diag(np.cos(x)))

# `elementwise_grad` df/dx : R^n -> R^n (vectorized), returns the column sum of
# the Jacobian
assert np.allclose(elementwise_grad(np.sin)(x), np.cos(x))
assert np.allclose(jacobian(np.sin)(x).sum(axis=0), np.cos(x))


@primitive
def pow2(x):
    return x**2


@primitive
def mysin(x):
    return np.sin(x)


@primitive
def mysum(x):
    return np.sum(x)


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
        jac = jacobian(lambda x: np.power(x,2))(x)
        return lambda g: np.dot(g, jac)


def mysin_vjp(ans, x):
    return lambda g: g * np.cos(x)


# Works in JAX but not here in the elementwise_grad() test. See
# autograd/numpy/numpy_vjps.py -> grad_np_sum() for ow they do it. The returned
# g's shape must be corrected sometimes.
##def mysum_vjp(ans, x):
##    return lambda g: g


def func(x):
    return np.sum(np.power(np.sin(x),2))


def func_with_vjp(x):
    return np.sum(pow2(mysin(x)))


defvjp(mysin, mysin_vjp)
for p_jvp in [pow2_vjp, pow2_vjp_with_jac]:
    defvjp(pow2, p_jvp)

    assert np.allclose([func(xi)          for xi in x],
                       [func_with_vjp(xi) for xi in x])

    assert np.allclose(func(x),
                       func_with_vjp(x))

    assert np.allclose([grad(func)(xi)          for xi in x],
                       [grad(func_with_vjp)(xi) for xi in x])

    assert np.allclose(elementwise_grad(func)(x),
                       elementwise_grad(func_with_vjp)(x))
