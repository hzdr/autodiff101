@mysin.defjvp
def mysin_jvp(primals, tangents):
    x, = primals
    v, = tangents
    return mysin(x), np.cos(x) * v
