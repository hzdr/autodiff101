>>> x=rand(3)
>>> grad(grad(f))(x)
TypeError: Gradient only defined for scalar-output functions.
    Output had shape: (3,).
