>>> f=lambda x: np.sum(np.power(np.sin(x), 2))
>>> x=rand(3)
>>> jax.make_jaxpr(f)(x)
{ lambda  ; a.
  let b = sin a
      c = integer_pow[ y=2 ] b
      d = reduce_sum[ axes=(0,) ] c
  in (d,) }
>>> jax.make_jaxpr(grad(f))(x)
{ lambda  ; a.
  let b = sin a
      c = cos a
      d = integer_pow[ y=2 ] b
      e = mul 2.0 b
      _ = reduce_sum[ axes=(0,) ] d
      f = broadcast_in_dim[ broadcast_dimensions=(  )
                            shape=(3,) ] 1.0
      g = mul f e
      h = mul g c
  in (h,) }
