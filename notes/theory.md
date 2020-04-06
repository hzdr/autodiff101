---
header-includes:
- |
  ```{=latex}
     \usepackage{bm}
     \usepackage{easylist}

     \newcommand{\ve}[1]{\ensuremath{\bm{\mathit{#1}}}}
     \newcommand{\ma}[1]{\ensuremath{\bm{\mathbf{#1}}}}
     \newcommand{\dd}{\text{d}}
     \newcommand{\ra}{\rightarrow}
     \newcommand{\la}{\leftarrow}
     \newcommand{\pd}[2]{\dfrac{\partial #1}{\partial #2}}
     \newcommand{\td}[2]{\dfrac{\dd #1}{\dd #2}}
     \newcommand{\red}[1]{{\color{red}{#1}}}
     \newcommand{\green}[1]{{\color{green}{#1}}}
     \newcommand{\blue}[1]{{\color{blue}{#1}}}
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
* AD
  * directly provides numerical values for gradients at some point $\ve x$
    (e.g. $\nabla f|_{\ve x}$) with machine precision, *not* symbolic
    expressions of derivatives
  * uses operator overloading (or other methods) to carry out derivs
  * can be applied to "any" code using arbitrary control flows
  * in reverse mode uses code tracing ("taping") by one forward eval to
    construct computation graph
  * no "code swell" (complex expressions that grow which each application of
    $\partial/\partial x_i$ as in symbolic engines) which need an additional
    simplification step (Theano does this)


First consider a scalar function of scalar inputs $f:\mathbb R\ra\mathbb R$
$$f(x) = c(b(a(x)))$$

```py
    f = lambda x: c(b(a(x)))
    f = lambda x: sum(pow2(sin(x)))
```

which gives the straight graph $x\ra a\ra b\ra c = f$. Forward = $x \ra c$ (inputs
to outputs). Reverse = $x \la c$ (outputs to inputs).
The derivative using the chain rule
    \begin{align*}
        \td{f}{x}
            &= \td{c}{b}\,\td{b}{a}\,\td{a}{x}\\
            &= \td{c}{b}\,\left(\td{b}{a}\,\td{a}{x}\right)\quad\text{forward}\\
            &= \left(\td{c}{b}\,\td{b}{a}\right)\,\td{a}{x}\quad\text{reverse}\\
    \end{align*}
Now we consider a multivariate vector function $\ve f: \mathbb R^n \ra \mathbb R^m$.
The Jacobian is

$$\ve f(\ve x): \mathbb R^n \ra \mathbb R^m\quad \ma J =
\begin{pmatrix}
\pd{f_1}{x_1} & \cdots & \pd{f_1}{x_n}  \\
\vdots        & \ddots & \vdots         \\
\pd{f_m}{x_1} & \cdots & \pd{f_m}{x_n}
\end{pmatrix}
\in\mathbb R^{m\times n}
$$
One edge case is
$$\ve f(x): \mathbb R^1 \ra \mathbb R^m\quad \ma J =
\begin{pmatrix}
\pd{f_1}{x_1}  \\
\vdots         \\
\pd{f_m}{x_1}
\end{pmatrix}
\in\mathbb R^{m\times 1}
$$
where we need only the first column $\ma J[:,1]$.
In the other edge case
$$f(\ve x): \mathbb R^n \ra \mathbb R^1\quad \ma J =
\begin{pmatrix}
\pd{f_1}{x_1} & \cdots & \pd{f_1}{x_n}
\end{pmatrix}
\in\mathbb R^{1\times n}
\equiv \nabla f\equiv \pd{f}{\ve x}
$$
we need the first row $\ma J[1,:]$ only.
Again we have
$$\ve f(\ve x) = \ve c(\ve b(\ve a(\ve x)))$$
and can use the chain rule
$$\ma J=\pd{\ve f}{\ve x} = \pd{\ve c}{\ve b}\,\pd{\ve b}{\ve a}\,\pd{\ve a}{\ve x}$$
which gives a series of matrix multiplications of individual Jacobians $\pd{\ve
a}{\ve x}$, $\pd{\ve b}{\ve a}$, $\pd{\ve c}{\ve b}$.

Forward: extract $j$-th column $\ma J[:,j]$ by multiplying from the right with
a one-hot vector $\partial\ve x/\partial x_j = \ve e_j = [0,0,\cdots,1,\cdots,
0]$ to select the $x_j$ w.r.t. to which we want to calculate the derivative.
Apply \blue{Jacobian vector products (JVPs)} as we go.
Apply all possible $\ve e_j$ to build up full $\ma J$, one column at a time.
For the edge case $\ve f(x): \mathbb R^1 \ra \mathbb R^m$ we are finished in one
forward pass.
    \begin{align*}
    \pd{\ve f}{\ve x}\,\red{\pd{\ve x}{x_j}}
        &= \pd{\ve c}{\ve b}\,\pd{\ve b}{\ve a}\,\pd{\ve a}{\ve x}\,\red{\pd{\ve x}{x_j}}\\
        &= \pd{\ve c}{\ve b}\,\pd{\ve b}{\ve a}\,\pd{\ve a}{\ve x}\,\red{\ve e_j}\\
        &= \pd{\ve c}{\ve b}\,\pd{\ve b}{\ve a}\,\blue{\pd{\ve a}{\ve x}\,\ve e_j} \\
        &= \pd{\ve c}{\ve b}\,\pd{\ve b}{\ve a}\,\blue{\ve e_j'} \\
        &= \pd{\ve c}{\ve b}\,\ve e_j'' \\
        &= \ma J[:,j] \\
    \end{align*}
Reverse: First, do a forward pass to trace execution and build up the graph
$\ve x\ra \ve a\ra \ve b\ra \ve c = \ve f$.
Extract $i$-th row $\ma J[i,:]$ by multiplying from the left with a
one-hot vector $\ve e_i$. Apply \blue{vector Jacobian products (VJPs)} as we go.
Apply all possible $\ve e_i$ to build up full $\ma J$, one row at a time.
For the edge case $f(\ve x): \mathbb R^n \ra \mathbb R^1$ we
are done in one backward pass.
    \begin{align*}
    \red{\pd{\ve x}{x_i}}\,\pd{\ve f}{\ve x}
        &= \red{\pd{\ve x}{x_i}}\,\pd{\ve c}{\ve b}\,\pd{\ve b}{\ve a}\,\pd{\ve a}{\ve x}\\
        &= \red{\ve e_i}\,\pd{\ve c}{\ve b}\,\pd{\ve b}{\ve a}\,\pd{\ve a}{\ve x}\\
        &= \blue{\ve e_i\,\pd{\ve c}{\ve b}}\,\pd{\ve b}{\ve a}\,\pd{\ve a}{\ve x}\\
        &= \ldots\\
        &= \ma J[i,:]
    \end{align*}
Note that this the same as right-multiplying the transpose of $\ma J$
$$\ve e_i\,\ma J = \ma J^\top\,\ve e_i$$
i.e. use JVPs on $\ma J^\top$.

Typical in ML: a multivariate function $f(\ve x): \mathbb R^n\ra \mathbb R$
where $n$ is "large" (as in $10^5$) and
that is composed of a number of vector-valued functions $\ve a(\ve z), \ve
b(\ve z): \mathbb R^p\ra \mathbb R^q$ and a final contraction $c(\ve z): \mathbb R^r\ra \mathbb R$.
For example, $f$ is the loss function in NN training and $\ve x$ are the
network's parameters.

$$f(\ve x) = c(\ve b(\ve a(\ve x)))$$



what                  | scipy [*]                     | numdifftools   | jax               | autograd
-|-|-|-|-
$\dd f(x)/\dd x$      | `M.gradient`                  | `Derivative`   | `grad`            | `grad`
$\nabla f(\ve x)$     | `O.approx_fprime`             | `Gradient`     | `vmap(grad(.))`   | `elementwise_grad`
$\dd\ve f/\dd\ve x$   | `N.approx_derivative`         | `Jacobian`     | `jacobian`        | `jacobian`

Table: [*] `M=scipy.misc`, `O=scipy.optimize`, `N=optimize._numdiff`


# Syntax

JAX (and mostly also autograd)

math                        |   code
-|-
$\td{f}{x}$                 | `grad(f)`
$\pd{f(\ve x)}{x_i}$        | `grad(f, argnums=i)`
$\nabla f = \left[\pd{f}{x_0}\cdots\pd{f}{x_{n-1}}\right]$ | `jnp.array(grad(f, argnums=tuple(range(n)))`
fake $\nabla f$ (e.g. `f=np.sin`)| `vmap(grad(f))`
$\ma J$                     | `jacobian(f)`, `jacfwd(f)`, `jacrev(f)`

# JVPs (forward) and VJPs (reverse)

* autograd
  * they define a VJP for "each" numpy function in
    [`numpy_vjps.py`](https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py)
    using `defvjp`, also some JVPs in
    [`numpy_jvps.py`](https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_jvps.py),
    but autograd is mostly only reverse mode
  * numpy API definition
    [`numpy_wrapper.py`](https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_wrapper.py)
* JAX
  * numpy primitives defined in
    [`jax.lax`](https://github.com/google/jax/blob/master/jax/lax/lax.py) using
    only `defjvp`, i.e. only forward
  * numpy API def
    [`jax.numpy.lax_numpy`](https://github.com/google/jax/blob/master/jax/numpy/lax_numpy.py)
  * they don't define explicit VJPs (for reverse mode), instead they use the
    forward trace, which has to be done anyway, and then run that backwards,
    transposing the JVP operations to get a reverse operation

Refs:

* <https://insidehpc.com/2017/12/deep-learning-automatic-differentiation-theano-pytorch/>
* <https://arxiv.org/abs/1502.05767>
* <http://www.autodiff.org>
* Talk by Matthew Johnson (former HIPS, now Goolge, JAX) about <https://github.com/HIPS/autograd>: <http://videolectures.net/deeplearning2017_johnson_automatic_differentiation>
