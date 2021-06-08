"""jax examples (https://github.com/google/jax).

Define custom derivatives via JVPs (forward mode).

The code examples are not useful production code since most derivatives that we
implement are already the correct default anyway (e.g. jax.grad(jax.numpy.sin)
-> jax.numpy.cos). We use them to learn and to show how the library operates
inside.

When we call

    >>> grad(func)(x)

all x and v in each custom_jvp function have x's shape, which makes sense and
is not always the case in autograd.

    >>> # scalar
    >>> x=wnp.array(1.234)
    >>> # array
    >>> x=wnp.array(rand(3))
"""

from jax import grad, vmap, jacobian, random
import jax
import jax.numpy as wnp


def elementwise_grad(func):
    """Emulate elementwise_grad() from autograd."""
    return vmap(grad(func))


@jax.custom_jvp
def pow2(x):
    return wnp.power(x, 2.0)


# pow2(), mysin(): We call defjvp() below in tests to check several JVP
# implementations. That's why we skip the decorators here.


##@pow2.defjvp
def pow2_jvp(primals, tangents):
    (x,) = primals
    (v,) = tangents
    return pow2(x), 2 * x * v


##@pow2.defjvp
def pow2_jvp_with_jac(primals, tangents):
    """JVP where we really build the intermediate Jacobian. Not
    necessary in practice, only for demonstration.
    """
    (x,) = primals
    (v,) = tangents
    # jacobian() works for scalar and 1d array input, diag() doesn't
    if x.shape == ():
        return pow2(x), 2 * x * v
    else:
        ##jac = jacobian(lambda x: wnp.power(x,2))(x)
        jac = wnp.diag(2 * x)
        return pow2(x), wnp.dot(jac, v)


@jax.custom_jvp
def mysin(x):
    """Fake use case for custom sin(): We pretend that we have a super fast
    approximation of sin(x): Some terms of Taylor around x=0. Actually, this is
    much slower than np.sin() :-D
    """
    ##return wnp.sin(x)
    return x - x ** 3 / 6 + x ** 5 / 120 - x ** 7 / 5040 + x ** 9 / 362880


def mycos(x):
    """Here is a real use case for implementing a custom_jvp, in this case for
    mysin():

        "Approximate the derivative, not differentiate the approximation."

    The analytic (and thus the AD) deriv of mysin() is

        1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320

    But that's not accurate enough! By ADing the approximate sine, we get an
    approximate cosine which is worse. We need the x**10 term as well. With that
    |mycos(x) - cos(x)| is slightly better than |mysin(x) - sin(x)|, w/o it
    slightly worse. Both grow beyond 1e-8 outside of ~ [-1,1].
    """
    return (
        1
        - x ** 2 / 2
        + x ** 4 / 24
        - x ** 6 / 720
        + x ** 8 / 40320
        - x ** 10 / 3628800
    )


##@mysin.defjvp
def mysin_jvp(primals, tangents):
    (x,) = primals
    (v,) = tangents
    ##return wnp.sin(x), wnp.cos(x) * v
    return mysin(x), mycos(x) * v


##@mysin.defjvp
def mysin_jvp_with_jac(primals, tangents):
    (x,) = primals
    (v,) = tangents
    # jacobian() works for scalar and 1d array input, diag() doesn't
    if x.shape == ():
        return mysin(x), mycos(x) * v
    else:
        # The same, using exact results:
        #   jac = jacobian(wnp.sin)(x)
        #   jac = wnp.diag(wnp.cos(x))
        # but:
        #   jac = jacobian(mysin)(x)
        # doesn't work b/c we can't use a function to calculate its own deriv
        # (jacobian() would call the JVP which are about to define right here).
        jac = wnp.diag(mycos(x))
        return mysin(x), wnp.dot(jac, v)


@jax.custom_jvp
def mysum(x):
    return wnp.sum(x)


@mysum.defjvp
def mysum_jvp(primals, tangents):
    """
    jac = jacobian(wnp.sum)(x) == wnp.ones_like(x), i.e. 1st row of J b/c sum:
    R^n -> R, so dot(jac, v) == sum(v). However, note that when v is scalar,
    e.g. wnp.array(1.234), dot() does NOT perform a sum, but only multiplies
    (scalar-vector product). Oddly enough, in this case returning either a
    scalar, e.g. one of
        sum(v)
        v
    or a vector, one of
        dot(wnp.ones_like(x), v)
        wnp.ones_like(x) * v
    works.
    """
    (x,) = primals
    (v,) = tangents
    # v scalar or array
    return wnp.sum(x), wnp.sum(v)
    ##return wnp.sum(x), wnp.dot(wnp.ones_like(x), v)
    # v scalar only
    ##return wnp.sum(x), wnp.ones_like(x) * v
    ##return wnp.sum(x), v


def func(x):
    return wnp.sum(wnp.power(wnp.sin(x), 2))


def func_with_jvp(x):
    return mysum(pow2(mysin(x)))


def test():
    assert wnp.allclose(grad(wnp.sin)(1.234), wnp.cos(1.234))

    # Keep slightly tighter than -pi/2 .. pi/2 to keep mysin() and mycos()
    # errors below 1e-8, else tune allclose() default thresholds.
    x = random.uniform(key=random.PRNGKey(123), shape=(10,)) * 2 - 1

    assert wnp.allclose(jacobian(wnp.sin)(x), wnp.diag(wnp.cos(x)))
    assert wnp.allclose(jacobian(wnp.sin)(x).sum(axis=0), wnp.cos(x))
    assert wnp.allclose(elementwise_grad(wnp.sin)(x), wnp.cos(x))
    assert (jacobian(wnp.sum)(x) == wnp.ones_like(x)).all()

    for p2_jvp, s_jvp in [
        (pow2_jvp, mysin_jvp),
        (pow2_jvp_with_jac, mysin_jvp_with_jac),
    ]:
        pow2.defjvp(p2_jvp)
        mysin.defjvp(s_jvp)
        assert wnp.allclose(
            [func(xi) for xi in x], [func_with_jvp(xi) for xi in x]
        )

        assert wnp.allclose(func(x), func_with_jvp(x))

        assert wnp.allclose(
            [grad(func)(xi) for xi in x], [grad(func_with_jvp)(xi) for xi in x]
        )

        assert wnp.allclose(
            elementwise_grad(func)(x), elementwise_grad(func_with_jvp)(x)
        )

        assert wnp.allclose(grad(func)(x), grad(func_with_jvp)(x))


if __name__ == "__main__":
    test()
