from jax import grad, vmap, jacobian
import jax
import jax.numpy as np
import numpy as onp

# emulate elementwise_grad from autograd
def elementwise_grad(func):
    return vmap(grad(func))

dfdx = grad(np.sin)
assert np.allclose(grad(np.sin)(1.234), np.cos(1.234))
assert np.allclose(dfdx(1.234), np.cos(1.234))

x = onp.random.rand(10)*5 - 5
assert np.allclose(jacobian(np.sin)(x), np.diag(np.cos(x)))

assert np.allclose(elementwise_grad(np.sin)(x), np.cos(x))
assert np.allclose(jacobian(np.sin)(x).sum(axis=0), np.cos(x))


@jax.custom_transforms
def pow2(x):
    return x**2


def pow2_vjp(x):
    # correct for x scalar or (n,), use this in production code
    return pow2(x), lambda g: (2*g*x,)


@jax.custom_transforms
def mysin(x):
    return np.sin(x)

def pow2_vjp_with_jac(x):
    """ VJP where we really build the intermediate Jacobian. Not
    necessary in practice, only for demonstration.
    """
    # called from elementwise_grad(func)(x) with x 1d array
    if x.shape != ():
        # The same:
        #    jac = jacobian(lambda x: np.power(x,2))(x)
        jac = np.diag(2*x)
        return pow2(x), lambda g: (np.dot(g, jac),)
    # called from grad(func)(x) with x scalar
    else:
        return pow2(x), lambda g: (2*g*x,)

def mysin_vjp(x):
    return np.sin(x), lambda g: (g * np.cos(x),)


def func(x):
    return np.sum(np.power(np.sin(x),2))

def func_with_vjp(x):
    return np.sum(pow2(mysin(x)))


for p_jvp in [pow2_vjp, pow2_vjp_with_jac]:
    jax.defvjp_all(pow2, p_jvp)
    jax.defvjp_all(mysin, mysin_vjp)


    assert np.allclose([func(xi)          for xi in x],
                       [func_with_vjp(xi) for xi in x])

    assert np.allclose(func(x),
                       func_with_vjp(x))

    assert np.allclose([grad(func)(xi)          for xi in x],
                       [grad(func_with_vjp)(xi) for xi in x])

    assert np.allclose(elementwise_grad(func)(x),
                       elementwise_grad(func_with_vjp)(x))
