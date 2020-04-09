# pytorch now has a jax-like functional grad API, i.e. jacobian(), hessian(),
# jvp() etc, but no grad() for grad(func)??? AFAIK torch.autograd.grad()
# operates on tensors
#   torch.autograd.functional
# https://github.com/pytorch/pytorch/commit/1f4a4aaf643b70ebcb40f388ae5226a41ca57d9b
#
# Not in 1.4 :-/, but on GH already in 1.5 release

### like jax
##torch.autograd.functional

### jax.nn
### jax.jax.experimental.stax
##torch.nn.functional

# AG resources
#   https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#autograd
#   https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
#   https://pytorch.org/docs/stable/autograd.html
#   https://pytorch.org/docs/stable/notes/autograd.html

import torch
wnp = torch
import numpy as np

rand = torch.rand

# gradient of scalar result c w.r.t. x, evaluated at x
x = torch.rand(3, requires_grad=True)
c = torch.sin(x).pow(2.0).sum()
c.backward()
print(x.grad)

# VJP: extract one row of J
x = torch.rand(3, requires_grad=True)
v = torch.tensor([1.0,0,0])
b = torch.sin(x).pow(2.0)
b.backward(v)
print(x.grad)

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
    return wnp.cos(x)

@_wrap_input
def func(x):
    return wnp.sin(x).pow(2.0).sum()


def grad(func):
    @_wrap_input
    def _gradfunc(x):
        out = func(x)
        out.backward(wnp.ones_like(out))
        # x.grad is a Tensor of x.shape which holds the derivatives of func
        # w.r.t each x[i,j,k,...] evaluated at x .... great API design, tho
        return x.grad
    return _gradfunc

elementwise_grad = grad


assert wnp.allclose(grad(wnp.sin)(1.234), cos(1.234))
x = rand(10)*5 - 5
assert wnp.allclose(elementwise_grad(wnp.sin)(x), wnp.cos(x))
