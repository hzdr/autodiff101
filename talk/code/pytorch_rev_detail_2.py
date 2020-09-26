>>> b = torch.pow(a, 2.0)
>>> v = torch.tensor([1.0, 0.0, 0.0])
>>> b.backward(v)
>>> x.grad
tensor([0.9619, 0.0000, 0.0000])
