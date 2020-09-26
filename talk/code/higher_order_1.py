>>> from jax import grad, hessian, jacfwd, jacrev
>>> f=lambda x: np.sum(np.power(np.sin(x),2))
>>> grad(grad(f))(23.0)
DeviceArray(-5.4309087, dtype=float32)
