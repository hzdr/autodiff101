[![GitHub release (latest by date)](https://img.shields.io/github/v/release/hzdr/autodiff101?label=latest%20release&style=flat-square)][releases]
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/hzdr/autodiff101/Build%20and%20release%20PDF?label=Build%20and%20release%20PDF&style=flat-square)][actions]

# An introduction to Automatic Differentiation

`theory/`: AD background theory, introducing the concept of forward and reverse
mode plus Jacobian-vector / vector-Jacobian products. To go deeper, make sure
to check out the excellent [JAX autodiff cookbook][jax_autodiff_cookbook] as
well as @mattjj's [talk on autograd][mattjj_talk].

`talk/`: Talk version of those notes. The talk was given at the @hzdr
<http://helmholtz.ai> local unit's Machine Learning journal club and at a @hzdr
and @casus workshop on physics-informed neural networks, both organized by Nico
Hoffmann (@nih23).

`examples/`: AD examples using [autograd], [jax] and [pytorch]. The examples
focus mostly on how to define custom derivatives in jax (and autograd). This
has helped to understand how Jacobian-vector products actually work. More
examples to come!

Download `talk` and `theory` PDF files from the [Releases page][releases] or
the latest [CI run][actions]. You can also click the badges above.


[autograd]: https://github.com/HIPS/autograd
[jax]: https://github.com/google/jax
[pytorch]: https://github.com/pytorch/pytorch
[jax_autodiff_cookbook]: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
[mattjj_talk]: http://videolectures.net/deeplearning2017_johnson_automatic_differentiation
[releases]: https://github.com/hzdr/autodiff101/releases/latest
[actions]: https://github.com/hzdr/autodiff101/actions
