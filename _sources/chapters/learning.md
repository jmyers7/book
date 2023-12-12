---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

**THIS CHAPTER IS CURRENTLY UNDER CONSTRUCTION!!!**

(learning)=
# Learning

In the [last chapter](prob-models), we studied general probabilistic models and described several specific and important examples. These descriptions included careful identifications of the parameters of the models, but the question was left open concerning exactly _how_ these parameters are chosen in practice. To cut straight to the chase:

> The goal is to _learn_ the parameters of a model based on an observed dataset.

The actual implementation of a concrete learning procedure is called a _learning algorithm_ by machine learning researchers and engineers, and they will refer to _training_ or _fitting_ a model. Statisticians refer to learning as _parameter estimation_. But no matter what you call them, the values of the parameters that these learning procedures seek are very often solutions to some sort of optimization problem. Intuitively, we want to choose parameters to minimize the "distance" between the model probability distribution and the empirical probability distribution of the dataset:

```{image} ../img/prob-distance.svg
:width: 75%
:align: center
```
&nbsp;

How one precisely defines and measures "distance" (or "discrepancy") is essentially a matter of choosing an objective function to minimize. Some learning algorithms we will study below are actually posed as maximization problems, but these may be reframed as minimization problems via the usual trick of replacing the objective function with its negative.

So, our first goal in this chapter is to describe objective functions for parameter learning. In some form or fashion, all these objectives will involve the data and model probability functions described in {numref}`Chapter %s <prob-models>`, though these functions will be called _likelihood functions_ in this chapter. Thus, all the learning algorithms in this book are _likelihood based_. For some simple models, the solutions to these optimization problems may be obtained in closed form; for others, the gradient-based optimization algorithms that we studied in {numref}`Chapter %s <optim>` are required to obtain approximate solutions.

Our focus in this chapter is using likelihood-based learning algorithms in a framework inspired by machine learning practice; in the chapters that follow, we will turn toward theoretical and statistical properties of likelihood-based parameter estimators in a more traditional statistics-based context.









## Likelihood-based learning objectives

To help motivate likelihood-based learning objectives, let's begin with a simple example. Suppose that we flip a coin $m\geq 1$ times and let $x^{(i)}$ be the number of heads obtained on the $i$-th toss; thus, $x^{(i)}$ is an observed value of a random variable

$$
X \sim \Ber(\theta).
$$

This is a very simple example of a probabilistic graphical model whose underlying graph consists of only two nodes, one for the parameter $\theta$ and one for the (observed) random variable $X$:

```{image} ../img/bern-pgm.svg
:width: 17%
:align: center
```
&nbsp;

Our observations together form a dataset of size $m$:

$$
x^{(1)},\ldots,x^{(m)} \in \{0,1\}.
$$

Based on this dataset, our goal is to _learn_ an optimal value for $\theta$ that minimizes the discrepancy between the model distribution and the empirical distribution of the dataset. To do this, it will be convenient to introduce the sum

$$
\Sigma x = x^{(1)} + \cdots + x^{(m)}
$$ (sum-dep-eqn)

which counts the total number of heads seen during the $m$ flips of the coin. To make this concrete, suppose that $m=10$ and $\Sigma x=7$, so that we see seven heads over ten flips. Then, intuition suggests that $\theta=0.7$ would be a "more optimal" estimate for the parameter then, say, $\theta=0.1$. Indeed, if $\theta=0.1$, we would expect it highly unlikely to observe seven heads over ten flips when there is only a one-in-ten chance of seeing a head on a single flip.

We may confirm our hunch by actually computing probabilities. Assuming, as always, that the observations in the dataset are independent, we have

$$
p\big(x^{(1)},\ldots,x^{(m)};\theta\big) = \prod_{i=1}^m \theta^{x^{(i)}}(1-\theta)^{1-x^{(i)}} = \theta^x (1-\theta)^{m-\Sigma x}.
$$ (likelihood-bern-eqn)

Notice that the value of the joint mass function depends only on the sum {eq}`sum-dep-eqn`. If this sum is $\Sigma x=7$ and we have $m=10$ and $\theta=0.1$, then

$$
p\big(x^{(1)},\ldots,x^{(m)};\theta=0.1\big) = 0.1^{7} (1-0.1)^{10-7} = 7.29 \times 10^{-8}.
$$

On the other hand, when $\theta=0.7$, we have

$$
p\big(x^{(1)},\ldots,x^{(m)};\theta=0.7\big) = 0.7^{7} (1-0.7)^{10-7} \approx 2.22 \times 10^{-3}.
$$

Thus, it is five orders of magnitude more likely to observe a dataset with $x=7$ for $\theta=0.7$ compared to $\theta=0.1$. In fact, when $\Sigma x=7$ and $m=10$, the value $\theta = 0.7$ is a global maximizer of {eq}`likelihood-bern-eqn` as a function of $\theta$, which may be verified by inspecting the graph:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

import torch
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
from itertools import product
import warnings
plt.style.use('../aux-files/custom_style_light.mplstyle')
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
warnings.filterwarnings("ignore")
blue = '#486AFB'
magenta = '#FD46FC'

def likelihood(theta, x, m):
    return (theta ** x) * ((1 - theta) ** (m - x))

m = 10
x = 7
grid = np.linspace(0, 1)
plt.plot(grid, likelihood(grid, x, m))
plt.axvline(x=0.7, color=magenta, linestyle='--')
plt.xlabel('$\\theta$')
plt.ylabel('likelihood')
plt.gcf().set_size_inches(w=5, h=3)
plt.tight_layout()
```

```{margin}
**Warning**: Note that the likelihood function $\mathcal{L}(\theta)$ is **not** a probability density function over $\theta$!
```

Note the label along the vertical axis; when the dataset is held fixed, the values of the joint mass function {eq}`likelihood-bern-eqn` as a function of the parameter $\theta$ are referred to as _likelihoods_. This function is called the _data likelihood function_ and is denoted

$$
\mathcal{L}\big(\theta;x^{(1)},\ldots,x^{(m)}\big) = p\big(x^{(1)},\ldots,x^{(m)};\theta\big).
$$

When the dependence of the likelihood function on the dataset does not need to be explicitly indicated, we shall often simply write $\mathcal{L}(\theta)$.

Thus, we see that the parameter $\theta = 0.7$ is a solution to the optimization problem that consists of maximizing the likelihood function $\mathcal{L}(\theta)$. This is a simple example of _maximum likelihood estimation_, or _MLE_.

We see from {eq}`likelihood-bern-eqn` that the data likelihood function is a product of probabilities. Thus, if $m$ is very large, the values of $\mathcal{L}(\theta)$ will be very small. For example, in the case that $m=100$ and $\Sigma x=70$ (which are still quite small values), we get the following plot:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

m = 100
x = 70

plt.plot(grid, likelihood(grid, x, m))
plt.axvline(x=0.7, color=magenta, linestyle='--')
plt.xlabel('$\\theta$')
plt.ylabel('likelihood')
plt.gcf().set_size_inches(w=5, h=3)
plt.tight_layout()
```

This often leads to difficulties when implementing MLE in computer algorithms due to numerical round-off. The machine is liable to round very small numbers to $0$. For this reason (and others), we often work with the (base $e$) logarithm of the data likelihood function, denoted by

$$
\ell\big(\theta; x^{(1)},\ldots,x^{(m)}\big) = \log{\mathcal{L}\big(\theta; x^{(1)},\ldots,x^{(m)}\big)}.
$$

This is called the _data log-likelihood function_. As with the data likelihood function, if the dataset does not need to be explicitly mentioned, we will often write $\ell(\theta)$.

MLE is the optimization problem with the data likelihood function $\mathcal{L}(\theta)$ as the objective function. But it is not hard to prove (see the suggested problems for this section) that the maximizers of the data likelihood function $\mathcal{L}(\theta)$ are the _same_ as the maximizers of the data log-likelihood function $\ell(\theta)$. For our Bernoulli model with $m=100$ and $\Sigma x=70$, a visual comparison of $\mathcal{L}(\theta)$ and $\ell(\theta)$ is given in:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

def log_likelihood(theta, x, m):
    return x * np.log(theta) + (m - x) * np.log(1 - theta)

grid = np.linspace(0, 1)
_, axes = plt.subplots(ncols=2, figsize=(10, 3))

axes[0].plot(grid, likelihood(grid, x, m))
axes[1].plot(grid, log_likelihood(grid, x, m))
axes[0].axvline(x=0.7, color=magenta, linestyle='--')
axes[1].axvline(x=0.7, color=magenta, linestyle='--')
axes[0].set_xlabel('$\\theta$')
axes[0].set_ylabel('likelihood')
axes[1].set_xlabel('$\\theta$')
axes[1].set_ylabel('log-likelihood')
plt.tight_layout()
```

Notice that the values of $\ell(\theta)$ are on a much more manageable scale compared to the values of $\mathcal{L}(\theta)$, and that the two functions have the same global maximizer at $\theta=0.7$.


```{margin}

The acronym _MLE_ often serves double duty: It stands for the procedure of _maximum likelihood estimation_, but it also sometimes stands for the results of this procedure, called _maximum likelihood estimates_.
```

Using the data log-likelihood function as the objective, we may easily compute the MLE in closed form for our Bernoulli model:

```{prf:theorem} MLE for the Bernoulli model

Consider the Bernoulli model described above and suppose that $0 < \Sigma x < m$. The (unique) global maximizer $\theta^\star$ of the data log-likelihood function $\ell(\theta)$ over $\theta \in (0,1)$ is given by $\theta^\star = \Sigma x/m$. Thus, $\theta^\star=\Sigma x/m$ is the maximum likelihood estimate.
```

To begin the simple proof, first note that

$$
\ell(\theta) = \Sigma x \log{\theta} + (m-\Sigma x) \log{(1-\theta)}
$$ (data-log-like-bern-eqn)

from {eq}`likelihood-bern-eqn`.  As you well know, the maximizers of $\ell(\theta)$ over $(0,1)$ must occur at points where $\ell'(\theta)=0$. But

$$
\ell'(\theta) = \frac{\Sigma x}{\theta} - \frac{m-\Sigma x}{1-\theta},
$$

and a little algebra yields the solution $\theta = \Sigma x/m$ to the equation $\ell'(\theta)=0$. To confirm that $\theta = \Sigma x/m$ is a global maximizer over $(0,1)$, note that the second derivatives of both $\log{\theta}$ and $\log{(1-\theta)}$ are always negative, and hence $\ell''(\theta)<0$ as well since $\Sigma x$ and $m-\Sigma x$ are positive (this is a manifestation of [concavity](https://en.wikipedia.org/wiki/Concave_function)). Thus, $\theta^\star = \Sigma x/m$ must be the (unique) global maximizer of $\ell(\theta)$.

Note that the data likelihood function

$$
\mathcal{L}\big(\theta; x^{(1)},\ldots,x^{(m)}\big) = p\big(x^{(1)},\ldots,x^{(m)};\theta\big)
$$

is exactly the _data joint probability function_, in the language of {numref}`Chapter %s <prob-models>`. The latter is the product

$$
p\big(x^{(1)},\ldots,x^{(m)};\theta\big) = \prod_{i=1}^m p\big(x^{(i)};\theta\big)
$$

where

$$
p(x;\theta) = \theta^x (1-\theta)^{1-x}
$$

is the _model probability function_. As a function of $\theta$ with $x$ held fixed, we call

$$
\mathcal{L}(\theta; x) \stackrel{\text{def}}{=} p(\theta; x)
$$

the _model likelihood function_ and

$$
\ell(\theta;x) \stackrel{\text{def}}{=} \log{\mathcal{L}(\theta;x)}
$$

the _model log-likelihood function_. When the data point $x$ does not need to be mentioned explicitly, we will write $\mathcal{L}(\theta)$ and $\ell(\theta)$ in place of $\mathcal{L}(\theta;x)$ and $\ell(\theta;x)$. Note that this clashes with our usage of $\mathcal{L}(\theta)$ and $\ell(\theta)$ to represent the _data_ likelihood and log-likelihood functions when the dataset is not made explicit. You will need to rely on context to clarify which of the two types of likelihood functions (data or model) is meant when we write $\mathcal{L}(\theta)$ or $\ell(\theta)$.

It will be convenient to describe an optimization problem involving the _model_ likelihood function that is equivalent to MLE. Here, _equivalence_ means that the two optimization problems have the same solutions. This new (but equivalent!) optimization problem is appealing in part because it directly uses the empirical probability distribution of the dataset and thus more closely aligns us with the intuitive scheme described in the introduction to this chapter, that the goal of parameter learning is to minimize the "distance" (or "discrepancy") between the model distribution and the empirical distribution. This optimization problem is also useful because it opens the door for the _stochastic gradient descent algorithm_ from {numref}`Chapter %s <optim>` when closed form solutions are not available.

To describe the new optimization problem, let's consider again our Bernoulli model. Let $\hat{p}(x)$ be the empirical mass function of the dataset

$$
x^{(1)},\ldots,x^{(m)} \in \{0,1\}.
$$

Thus, in general we have

$$
\hat{p}(x) = \frac{\text{number of data points $x^{(i)}$ that match $x$}}{m}
$$

for all $x\in \bbr$, but for our particular Bernoulli model, this simplifies to

$$
\hat{p}(0) = \frac{m-\Sigma x}{m} \quad \text{and} \quad \hat{p}(1) = \frac{\Sigma x}{m},
$$

where $\Sigma x=x^{(1)} + \cdots + x^{(m)}$. Letting $\widehat{X}$ be a Bernoulli random variable with $\hat{p}(x)$ as its mass function, we consider the stochastic objective function

$$
J(\theta) \stackrel{\text{def}}{=} E \big[ \ell\big(\theta; \widehat{X} \big) \big],
$$

where $\ell(\theta;x)$ is the model log-likelihood function. Note that

$$
J(\theta) = \ell(\theta;1) \hat{p}(1)+ \ell(\theta; 0 ) \hat{p}(0) = \frac{1}{m} \left[ \Sigma x \log{\theta}  + (m-\Sigma x)\log{(1-\theta)} \right],
$$

and so by comparison with {eq}`data-log-like-bern-eqn` we see that the stochastic objective function $J(\theta)$ differs from the data log-likelihood function $\ell\big( \theta; x^{(1)},\ldots,x^{(m)}\big)$ only by a constant factor of $1/m$.