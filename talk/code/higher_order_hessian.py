>>> hessian(f)(x)
DeviceArray([[-5.1981254,  0.       ,  0.       ],
             [ 0.       , 11.531276 ,  0.       ],
             [ 0.       ,  0.       , 12.378209 ]],
             dtype=float32)
>>> jacfwd(jacrev(f))(x)
>>> jacfwd(grad(f))(x)

>>> f=lambda x: torch.sum(torch.pow(torch.sin(x),2))
>>> torch.autograd.functional.hessian(f, torch.tensor(x))
