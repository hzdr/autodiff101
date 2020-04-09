from jax import grad, vmap, jacobian
import jax
import jax.numpy as wnp
import numpy as np

rand = np.random.rand


# emulate elementwise_grad from autograd
def elementwise_grad(func):
    return vmap(grad(func))

assert wnp.allclose(grad(wnp.sin)(1.234), wnp.cos(1.234))

x = np.random.rand(10)*5 - 5
assert wnp.allclose(jacobian(wnp.sin)(x), wnp.diag(wnp.cos(x)))

assert wnp.allclose(elementwise_grad(wnp.sin)(x), wnp.cos(x))
assert wnp.allclose(jacobian(wnp.sin)(x).sum(axis=0), wnp.cos(x))


@jax.custom_transforms
def pow2(x):
    return x**2


def pow2_vjp(x):
    # correct for x scalar or (n,), use this in production code
    return pow2(x), lambda g: (g * 2*x,)


@jax.custom_transforms
def mysin(x):
    return wnp.sin(x)

@jax.custom_transforms
def mysum(x):
    return wnp.sum(x)

def pow2_vjp_with_jac(x):
    """ VJP where we really build the intermediate Jacobian. Not
    necessary in practice, only for demonstration.
    """
    # called from elementwise_grad(func)(x) with x 1d array
    if x.shape != ():
        # The same:
        #   jac = np.diag(2*x)
        jac = jacobian(lambda x: wnp.power(x,2))(x)
        return pow2(x), lambda g: (wnp.dot(g, jac),)
    # called from grad(func)(x) with x scalar
    else:
        return pow2(x), lambda g: (g * 2*x,)

def mysin_vjp(x):
    return wnp.sin(x), lambda g: (g * wnp.cos(x),)

def mysum_vjp(x):
    return wnp.sum(x), lambda g: (g,)

def func(x):
    return wnp.sum(wnp.power(wnp.sin(x),2))

def func_with_vjp(x):
    return mysum(pow2(mysin(x)))


jax.defvjp_all(mysin, mysin_vjp)
jax.defvjp_all(mysum, mysum_vjp)
for p_jvp in [pow2_vjp, pow2_vjp_with_jac]:
    jax.defvjp_all(pow2, p_jvp)


    assert wnp.allclose([func(xi)          for xi in x],
                        [func_with_vjp(xi) for xi in x])

    assert wnp.allclose(func(x),
                        func_with_vjp(x))

    assert wnp.allclose([grad(func)(xi)          for xi in x],
                        [grad(func_with_vjp)(xi) for xi in x])

    assert wnp.allclose(elementwise_grad(func)(x),
                        elementwise_grad(func_with_vjp)(x))
