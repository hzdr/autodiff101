"""When we call
    >>> grad(func)(x)

all x and v in each custom_jvp function have x's shape, which makes sense and
is not always the case in autograd.

    >>> # scalar
    >>> x=wnp.array(1.234)
    >>> # array
    >>> x=wnp.array(np.random.rand(3))
"""

from jax import grad, vmap, jacobian
import jax
import jax.numpy as wnp
import numpy as np

rand = np.random.rand


# Emulate elementwise_grad() from autograd.
def elementwise_grad(func):
    return vmap(grad(func))


@jax.custom_jvp
def pow2(x):
    return x**2


# We call defjvp() below in the tests for pow2()'s two JVP implementations.

##@pow2.defjvp
def pow2_jvp(primals, tangents):
    x, = primals
    v, = tangents
    return pow2(x), 2*x * v


##@pow2.defjvp
def pow2_jvp_with_jac(primals, tangents):
    """JVP where we really build the intermediate Jacobian. Not
    necessary in practice, only for demonstration.
    """
    x, = primals
    v, = tangents
    # called from grad(func)(x) with x scalar, here we get passed np.array(x)
    # with x.shape == ()
    if x.shape == ():
        return pow2(x), 2*x * v
    # called from grad(func)(x) with x.shape == (n,)
    else:
        # The same:
        #   jac = np.diag(2*x)
        jac = jacobian(lambda x: wnp.power(x,2))(x)
        return pow2(x), wnp.dot(jac, v)


@jax.custom_jvp
def mysin(x):
    return wnp.sin(x)


@mysin.defjvp
def mysin_jvp(primals, tangents):
    x, = primals
    v, = tangents
    return wnp.sin(x), wnp.cos(x) * v


@jax.custom_jvp
def mysum(x):
    return wnp.sum(x)


@mysum.defjvp
def mysum_jvp(primals, tangents):
    """
    jacobian(wnp.sum)(x) == wnp.ones_like(x), i.e. 1st row of J b/c sum: R^n ->
    R, so dot(jac, v) == sum(v). However, note that when v is scalar, e.g.
    wnp.array(1.234), dot() does NOT perform a sum, but only multipiles
    (scalar-vector product). Oddly enough, in this case returning either a
    scalar, e.g. one of
        sum(v)
        v
    or a vector, one of
        dot(wnp.ones_like(x), v)
        wnp.ones_like(x) * v
    works.
    """
    x, = primals
    v, = tangents
    # v scalar or array
    return wnp.sum(x), wnp.sum(v)
##    return wnp.sum(x), wnp.dot(wnp.ones_like(x), v)
    # v scalar
##    return wnp.sum(x), wnp.ones_like(x) * v
##    return wnp.sum(x), v


def func(x):
    return wnp.sum(wnp.power(wnp.sin(x),2))


def func_with_jvp(x):
    return mysum(pow2(mysin(x)))


def test():
    assert wnp.allclose(grad(wnp.sin)(1.234), wnp.cos(1.234))

    x = np.random.rand(10)*5 - 5
    assert wnp.allclose(jacobian(wnp.sin)(x), wnp.diag(wnp.cos(x)))
    assert wnp.allclose(jacobian(wnp.sin)(x).sum(axis=0), wnp.cos(x))
    assert wnp.allclose(elementwise_grad(wnp.sin)(x), wnp.cos(x))
    assert (jacobian(wnp.sum)(x) == wnp.ones_like(x)).all()

    for p2_jvp in [pow2_jvp, pow2_jvp_with_jac]:
        pow2.defjvp(p2_jvp)
        assert wnp.allclose([func(xi)          for xi in x],
                            [func_with_jvp(xi) for xi in x])

        assert wnp.allclose(func(x),
                            func_with_jvp(x))

        assert wnp.allclose([grad(func)(xi)          for xi in x],
                            [grad(func_with_jvp)(xi) for xi in x])

        assert wnp.allclose(elementwise_grad(func)(x),
                            elementwise_grad(func_with_jvp)(x))

        assert wnp.allclose(grad(func)(x),
                            grad(func_with_jvp)(x))


if __name__ == '__main__':
    test()
