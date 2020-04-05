---
header-includes:
- |
  ```{=latex}
     \usepackage{bm}
     \newcommand{\ve}[1]{\ensuremath{\bm{\mathit{#1}}}}
     \newcommand{\ma}[1]{\ensuremath{\bm{\mathbf{#1}}}}
     \newcommand{\dd}{\text{d}}
     \newcommand{\ra}{\rightarrow}
  ```
---

Methods for diff:

* work out derivs by hand and code them
* call symbolic engines (Mathematica, Maple, Maxima, sympy) in your code
  * special sub-type: manually construct computation graph by framework
    mini-language (TensorFlow, Theano) (e.g. `tf.while`, `tf.cond`), the
    framework then does symbolic derivs and thus creates a new computation
    graph for derivs which is used in SGD
* numerical derivs by FD (use at least central diffs, `numdifftools`), good for
  checking AD gradients, scales as $\mathcal O(n)$ for $\nabla f(\ve x), \ve x\in\mathbb
  R^n$, slow when $n\sim 10^6$
* AD: directly provides numerical values for gradients at some point $\ve x$
  (e.g. $\nabla f|_{\ve x}$) with machine precision, *not* symbolic expressions
  of derivatives, uses code tracing and operator overloading, can be applied to
  "any" code using arbitrary control flows


Typical in ML: a multivariate function $f(\ve x): \mathbb R^n\ra \mathbb R$
that is composed of a number of vector-valued functions $\ve a(\ve z), \ve
b(\ve z): \mathbb R^m\ra \mathbb R^n$ and a final contraction $c(\ve z): \mathbb R^n\ra \mathbb R$.

$$f(\ve x) = c(\ve b(\ve a(\ve x)))$$


```py
    # y: scalar, x: 1d array
    y = c(b(a(x)))
    y = sum(pow2(sin(x)))
```


what                            | scipy [*]                     | numdifftools   | jax               | autogtad
-|-|-|-|-
$\dd f(x)/\dd x$                | `M.gradient`                  | `Derivative`   | `grad`            | `grad`
$\nabla f(\ve x)$               | `O.approx_fprime`             | `Gradient`     | `vmap(grad(.))`   | `elementwise_grad`
$\dd\ve f/\dd\ve x$   | `N.approx_derivative`         | `Jacobian`     | `jacobian`        | `jacobian`

Table: [*] `M=scipy.misc`, `O=scipy.optimize`, `N=optimize._numdiff`


Refs:

* <https://insidehpc.com/2017/12/deep-learning-automatic-differentiation-theano-pytorch/>
* <https://arxiv.org/abs/1502.05767>
* <http://www.autodiff.org>
