"""
pytorch has a "functional" grad API [1,2] as of v1.5

    torch.autograd.functional

in addition to

    # like jax.nn and jax.experimental.stax
    torch.nn.functional

However, unlike jax, torch.autograd.functional's functions don't return
functions. One needs to supply the function to differentiate along with the
input at which grad(func) shall be evaluated.

    # like jax.grad(func)(x)
    #
    # default vector v in VJP is v=None -> v=1 -> return grad(func)(x)
    torch.autograd.functional.vjp(func, x) -> Tensor

    torch.autograd.functional.hessian(func, x) -> Tensor

whereas

    jax.grad(func) -> grad_func
    jax.grad(func)(x) -> grad_func(x) -> DeviceArray

    jax.hessian(func) -> hess_func
    jax.hessian(func)(x) -> hess_func(x) -> DeviceArray


resources
    https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#autograd
    https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
    https://pytorch.org/docs/stable/autograd.html
    https://pytorch.org/docs/stable/notes/autograd.html


[1] https://github.com/pytorch/pytorch/commit/1f4a4aaf643b70ebcb40f388ae5226a41ca57d9b
[2] https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api
"""

import torch
import numpy as np

rand = torch.rand

### gradient of scalar result c w.r.t. x, evaluated at x
### step by step, see grad_fn
##x = torch.rand(3, requires_grad=True)
####c = x.sin().pow(2.0).sum()
##a = torch.sin(x)
##print(a)
##b = torch.pow(a, 2.0)
##print(b)
##c = torch.sum(b)
##print(c)
### same as torch.autograd.grad(c,x)
##c.backward()
##print(x.grad)
##
##
### VJP: extract one row of J
##x = torch.rand(3, requires_grad=True)
##v = torch.tensor([1.0,0,0])
##b = x.sin().pow(2.0)
##b.backward(v)
##print(x.grad)


func_plain_torch = lambda x: torch.sin(x).pow(2.0).sum()


def copy(x, requires_grad=False):
    _x = x.clone().detach()
    if not requires_grad:
        assert not _x.requires_grad
    else:
        _x.requires_grad = requires_grad
    return _x


# -----------------------------------------------------------------------------
# poor man's jax-like API
# -----------------------------------------------------------------------------


def _wrap_input(func):
    def wrapper(_x):
        if isinstance(_x, torch.Tensor):
            x = _x
        else:
            x = torch.Tensor(np.atleast_1d(_x))
        x.requires_grad = True
        if x.grad is not None:
            x.grad.zero_()
        return func(x)

    return wrapper


# only to make scalar args work
@_wrap_input
def cos(x):
    return torch.cos(x)


@_wrap_input
def func(x):
    return func_plain_torch(x)


def grad(func):
    @_wrap_input
    def _gradfunc(x):
        out = func(x)
        out.backward(torch.ones_like(out))
        # x.grad is a Tensor of x.shape which holds the derivatives of func
        # w.r.t each x[i,j,k,...] evaluated at x, got it?
        return x.grad

    return _gradfunc


elementwise_grad = grad


def test():
    # Check that grad() works
    assert torch.allclose(grad(torch.sin)(1.234), cos(1.234))
    x = rand(10) * 5 - 5
    assert torch.allclose(elementwise_grad(torch.sin)(x), torch.cos(x))
    assert grad(func)(x).shape == x.shape

    # Show 4 different pytorch grad APIs
    x1 = rand(3, requires_grad=True)

    # 1
    c1 = func_plain_torch(x1)
    c1.backward()
    g1 = x1.grad

    # 2
    x2 = copy(x1, requires_grad=True)
    c2 = func_plain_torch(x2)
    torch.autograd.backward(c2)
    g2 = x2.grad
    assert (g1 == g2).all()

    # 3
    x2 = copy(x1, requires_grad=True)
    c2 = func_plain_torch(x2)
    g2 = torch.autograd.grad(c2, x2)[0]
    assert (g1 == g2).all()

    # 4
    x2 = copy(x1)
    g2 = torch.autograd.functional.vjp(func_plain_torch, x2)[1]
    assert (g1 == g2).all()

    # jax-like functional API defined here
    x2 = copy(x1)
    g2 = grad(func)(x2)
    assert (g1 == g2).all()


if __name__ == "__main__":
    test()
