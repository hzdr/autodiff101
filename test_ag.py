from autograd import grad, elementwise_grad, jacobian
from autograd.extend import primitive, defvjp
import autograd.numpy as wnp
import numpy as np

rand = np.random.rand


@primitive
def pow2(x):
    return x**2


def pow2_vjp(ans, x):
    # correct for x scalar or (n,), use this in production code
    return lambda v: v * 2*x


def pow2_vjp_with_jac(ans, x):
    """ VJP where we really build the intermediate Jacobian. Not
    necessary in practice, only for demonstration.
    """
    # called from grad(func)(x) with x scalar, here we get passed np.array(x)
    # with x.shape == ()
    if x.shape == ():
        return lambda v: v * 2*x
    # called from grad(func)(x) and elementwise_grad(func)(x) with x.shape ==
    # (n,)
    else:
        # The same:
        #   jac = np.diag(2*x)
        jac = jacobian(lambda x: wnp.power(x,2))(x)
        return lambda v: wnp.dot(v, jac)


@primitive
def mysin(x):
    return wnp.sin(x)


def mysin_vjp(ans, x):
    return lambda v: v * wnp.cos(x)


@primitive
def mysum(x):
    return wnp.sum(x)


def mysum_vjp(ans, x):
    """
    See autograd/numpy/numpy_vjps.py -> grad_np_sum() for how they do it. The
    returned v's shape must be corrected sometimes. Here we explicitly write
    out the JVP using jacobian(wnp.sum)(x) == wnp.ones_like(x) which is always
    correct. See test_jax.py:mysum_jvp() for more comments. Note that in
    contrast to JAX, here v is *always* scalar, a fact that we can't explain
    ATM. As JAX is autograd 2.0, we consider this an autograd quirk and leave
    it be.
    """
    return lambda v: wnp.dot(v, wnp.ones_like(x))


def func(x):
    return wnp.sum(wnp.power(wnp.sin(x),2))


def func_with_vjp(x):
    return mysum(pow2(mysin(x)))


def test():
    # scalar derivative
    # df/dx : R -> R
    assert wnp.allclose(grad(wnp.sin)(1.234), wnp.cos(1.234))

    x = rand(10)*5 - 5
    assert wnp.allclose(jacobian(wnp.sin)(x), wnp.diag(wnp.cos(x)))

    # elementwise_grad(f) : R^n -> R^n (of f: R^n -> R^n), returns the column sum of
    # the Jacobian
    assert wnp.allclose(elementwise_grad(wnp.sin)(x), wnp.cos(x))
    assert wnp.allclose(jacobian(wnp.sin)(x).sum(axis=0), wnp.cos(x))


    defvjp(mysin, mysin_vjp)
    defvjp(mysum, mysum_vjp)
    for p2_jvp in [pow2_vjp, pow2_vjp_with_jac]:
        defvjp(pow2, p2_jvp)

        assert wnp.allclose([func(xi)          for xi in x],
                            [func_with_vjp(xi) for xi in x])

        assert wnp.allclose(func(x),
                            func_with_vjp(x))

        assert wnp.allclose([grad(func)(xi)          for xi in x],
                            [grad(func_with_vjp)(xi) for xi in x])

        assert wnp.allclose(elementwise_grad(func)(x),
                            elementwise_grad(func_with_vjp)(x))

        assert wnp.allclose(grad(func)(x),
                            grad(func_with_vjp)(x))


if __name__ == '__main__':
    test()
