# Autodiff 101

An introduction to Automatic Differentiation.


`theory/`: AD background theory, introducing the concept of forward and
reverse mode plus Jacobian-vector / vector-Jacobian products. To go deeper,
make sure to check out the excellent [JAX autodiff
cookbook][jax_autodiff_cookbook].

`talk/`: Talk version of those notes. The talk was given at the @hzdr
<http://helmholtz.ai> local unit's Machine Learning journal club and at a @hzdr
and @casus workshop on physics-informed neural networks, both organized by Nico
Hoffmann (@nih23).

`examples/`: AD examples using [autograd], [jax] and [pytorch]. The examples
focus mostly on how to define custom derivatives in jax (and autograd). This
has helped to understand how Jacobian-vector products actually work. More
examples to come!



# PDFs

Download CI-built PDF files:

* from the Actions tab after each CI run (on `push`): All workflows > Build
  and release PDF > click latest Workflow run > Artifacts > download `.zip`
  file
* from the Releases page directly for each `git tag`


[autograd]: https://github.com/HIPS/autograd
[jax]: https://github.com/google/jax
[pytorch]: https://github.com/pytorch/pytorch
[jax_autodiff_cookbook]: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
