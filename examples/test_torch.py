"""
pytorch has a "functional" grad API [1,2] as of v1.5

    torch.autograd.functional

in addition to

    # jax.nn and jax.experimental.stax
    torch.nn.functional

However, unlike jax, torch.autograd.functional's functions don't return
functions. Also one has to supply the function to differentiate *and* the
input.

    torch.autograd.functional.hessian(func, x) -> Tensor

which is the same API we find with torch.autograd.grad():

    x=torch.rand(3)
    x.requires_grad_()
    a=x.sin().sum()
    torch.autograd.grad(a, x) -> x.grad
    # same as
    #   a.backward()
    #   x.grad

whereas

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


#-----------------------------------------------------------------------------
# poor man's jax-like functional API
#-----------------------------------------------------------------------------

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


@_wrap_input
def cos(x):
    return torch.cos(x)


@_wrap_input
def func(x):
    return torch.sin(x).pow(2.0).sum()


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
    assert torch.allclose(grad(torch.sin)(1.234), cos(1.234))
    x = rand(10)*5 - 5
    assert torch.allclose(elementwise_grad(torch.sin)(x), torch.cos(x))
    assert grad(func)(x).shape == x.shape

    # Different grad APIs
    x1 = rand(3, requires_grad=True)
    # This is how one copies an array in pytorch
    x2 = x1.clone().detach()
    x2.requires_grad = True
    c1 = func(x1)
    # same as torch.autograd.backward(c1)
    c1.backward()
    g1 = x1.grad
    c2 = func(x2)
    g2 = torch.autograd.grad(c2, x2)[0]
    assert (g1==g2).all()

if __name__ == '__main__':
    test()
