@mysin.defjvp
def mysin_jvp_with_jac(primals, tangents):
    x, = primals
    v, = tangents
    ##jac = jax.jacobian(np.sin)(x)
    jac = np.diag(np.cos(x))
    return mysin(x), np.dot(jac, v)
