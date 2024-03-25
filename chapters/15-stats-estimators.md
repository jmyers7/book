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

(stats-estimators)=
# Statistics and general parameter estimation








## Statistics

```{prf:definition}
:label: statistic-def

Let $\bX$ be a $k$-dimensional random vector. A _($d$-dimensional) statistic_ is a random vector of the form

$$
T = r(\bX),
$$

where $r:\bbr^k \to \bbr^d$ is a vector-valued function. An observed value $t$ of $T$ is called an _observed statistic_ or _empirical statistic_.
```

If we conceptualize the random vector $\bX$ as (theoretical) data, then a statistic is simply a function of the data. Crucially important examples of statistics include those defined as follows:

```{prf:definition}
:label: sample-mean-var-def

Let $\bX = (X_1,\ldots,X_m)$ be an $m$-dimensional random vector. The _sample mean_ is defined to be the statistic

$$
\overline{X} \def \frac{1}{m}(X_1+\cdots+X_m),
$$

while the _sample variance_ is defined to be the statistic

$$
S^2 \def \frac{1}{m-1} \sum_{i=1}^m(X_i - \overline{X})^2.
$$

The corresponding empirical statistics are the _empirical mean_ and _empirical variance_ defined as

$$
\overline{x} \def \frac{1}{m}(x_1+\cdots+x_m) \quad \text{and} \quad s^2 = \frac{1}{m-1} \sum_{i=1}^m(x_i - \overline{x})^2.
$$

```

Very often, the component random variables $X_1,\ldots,X_m$ of the random vector $\bX$ in the definition are assumed to form a random sample, i.e., an IID sequence of random variables. The dimension $m$ is then referred to as the _sample size_. In principle, then, the sample size $m$ can be _any_ positive integer, and so it is often convenient to write $\overline{X}_m$ for the sample mean, explicitly displaying the sample size. This gives us an entire _infinite sequence_ of sample means:

$$
\overline{X}_1,\overline{X}_2,\ldots,\overline{X}_m, \ldots.
$$ (seq-means-eqn)

Since statistics are random vectors, they have their own probability distributions. These are given special names:

```{prf:definition}
:label: samp-dist-def

The probability distribution of a statistic $T$ is called the _sampling distribution_ of $T$.
```

The sampling distributions for sample means $\overline{X}_m$ are particularly important, and one of the main goals of {numref}`Chapter %s <asymptotic>` is to study the limiting behavior (or _asymptotic behavior_) of the sampling distributions in the sequence {eq}`seq-means-eqn` as $m\to \infty$.

In general, however, computing the sampling distributions is difficult. But if we actually have _observed_ data $x_1,x_2,\ldots,x_m$, then (as you will explore in the programming assignment) there is a resampling method known as _bootstrapping_ that yields  approximations to sampling distributions. An example is given by the histogram (with KDE) on the right-hand side of the following figure, where a histogram (with KDE) of the empirical distribution of an observed dataset is given on the left-hand side:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib_inline.backend_inline
import matplotlib.colors as clr
import warnings
plt.style.use('../aux-files/custom_style_light.mplstyle')
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
warnings.filterwarnings("ignore")
blue = '#486AFB'
magenta = '#FD46FC'

np.random.seed(42)
X = sp.stats.gamma(a=5)

sample_size = 100
resample_size = 1000
random_sample = X.rvs(size=sample_size)
replicate_means = []
num_resamples = 10000

for _ in range(num_resamples):
    sample = np.random.choice(a=random_sample, size=resample_size, replace=True)
    replicate_means.append(sample.mean())

_, axes = plt.subplots(ncols=2, figsize=(10, 3))

sns.histplot(x=random_sample, ec='black', stat='density', kde=True, ax=axes[0])
sns.histplot(x=replicate_means, ec='black', stat='density', kde=True, ax=axes[1])
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('density')
axes[0].set_title('observed data')
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('density')
axes[1].set_title('bootstrap sampling distribution of sample mean')
plt.tight_layout()
```

Observe that the sampling distribution on the right-hand side appears to be well approximated by a normal distribution. This is actually a manifestation of the asymptotic behavior of sample means that we alluded to above; indeed, as we will see in {numref}`Chapter %s <asymptotic>`, the Central Limit Theorem tells us that the sequence {eq}`seq-means-eqn` of sample means converges (in distribution) to a normal distribution as $m\to \infty$, provided that the random variables are IID. This is true even though the observed data are definitely _not_ normally distributed. Moreover, the mean of the sampling distribution is approximately $4.924$, while the mean of the observed data is approximately $4.928$. The fact that these means are nearly equal is a consequence of another theorem in {numref}`Chapter %s <asymptotic>` called the Law of Large Numbers. These asymptotic results provide the foundation for the large-sample _confidence intervals_ that we will construct in {numref}`Chapter %s <CIs>`.

Let's consider the sample mean a little closer:

```{prf:theorem} Properties of the sample mean
:label: prop-sample-mean-thm

Let $X_1,\ldots,X_m$ be an IID random sample from a distribution with mean $\mu$ and standard deviation $\sigma$.

1. The expectation of the sample mean $\overline{X}$ is $\mu$.
2. The variance of the sample mean $\overline{X}$ is $\sigma^2/m$, and hence its standard deviation is $\sigma/\sqrt{m}$.
3. If the $X_i$'s are normally distributed, then so too is the sample mean $\overline{X}$.
```

You will prove this theorem as







## General parametric models

Abstracting away all the intricate particularities of the fully-observed probabilistic graphical models in {numref}`Chapter %s <prob-models>` reveals that they all are examples of the following type of general structure:

```{prf:definition}
:label: gen-para-model-def

Let $\bX$ be a $k$-dimensional random vector and let $\Omega$ be a (nonempty) subset of a Euclidean space $\bbr^d$. A _parametric probabilistic model_ (or simply a _parametric model_) for $\bX$ is a specification of a dependence of the probability distribution of $\bX$ on values $\btheta \in \Omega$. In other words, a _parametric model_ is simply a family

$$
\calP_0 = \{P_\btheta : \btheta \in \Omega\}
$$ (para-model-eqn)

of probability distributions on $\bbr^k$ such that $\bX \sim P_\btheta$. In this context, the set $\Omega$ is called the _parameter space_, each $\btheta \in \Omega$ is called a ($d$-dimensional) _parameter_, and the vector $\bX$ is called the _data_.
```

Very often, we shall specify a parametric model {eq}`para-model-eqn` by listing the density functions of the probability measures $P_\btheta$, provided that these exist. In other words, we will write

$$
\calP_0 = \{ p(\bx; \btheta): \btheta \in \Omega\}.
$$

The simplest examples of parametric models are the univariate models introduced and studied in {numref}`Chapter %s <examples>`. Indeed, if we have $X \sim \Ber(\theta)$, then

$$
\mathcal{P}_0 = \{ p(x;\theta) : \theta \in [0,1] \}, \quad p(x;\theta) = \theta^x (1-\theta)^{1-x}, \ x\in \{0,1\},
$$

is a parametric model with $1$-dimensional parameter space $\Omega = [0,1]$. Similarly, if we have $X\sim \calN(\mu,\sigma^2)$, then

$$
\mathcal{P}_0 = \{ p(x;\btheta) : \btheta = (\mu,\sigma^2) \in \bbr \times (0,\infty) \}, \quad p(x;\btheta) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp \left[ -\frac{1}{2\sigma^2} (x-\mu)^2\right],
$$

is a parametric model with $2$-dimensional parameter space $\Omega = \bbr \times (0,\infty)$.

As we mentioned above, all the fully-observed probabilistic graphical models studied in {numref}`Chapter %s <prob-models>` are examples of this general type of parametric model. For example, a linear regression model with $n$-dimensional predictor vector $\bX$ and response variable $Y$ defines a parametric model

$$
\calP_0 = \{ p(\bx, y; \btheta) : \btheta = (\beta_0,\bbeta,\sigma^2) \in \bbr \times \bbr^n \times (0,\infty)\}
$$

with $(n+2)$-dimensional parameter space $\Omega = \bbr \times \bbr^n \times (0,\infty)$ and where

$$
p(\bx, y; \btheta) = p(y \mid \bx; \btheta) p(\bx) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp \left[ -\frac{1}{2\sigma^2} (y-\beta_0 - \bx^\intercal \bbeta)^2\right] p(\bx).
$$

In this latter example, notice that the $(n+1)$-dimensional random vector $(\bX,Y)$ plays the roll of the vector $\bX$ in {prf:ref}`para-model-def` (so that $k=n+1$).

The "plated" versions of the fully-observed PGMs in {numref}`Chapter %s <prob-models>` also define parametric models in the sense of {prf:ref}`para-model-def`. For example, a "plated" version of our linear regression model from above would define the parametric model

$$
\calP_0 = \{ p(\bx_1,\ldots,\bx_m,y_1,\ldots,y_m; \btheta) : \btheta = (\beta_0,\bbeta,\sigma^2) \in \Omega\}
$$

where $\Omega$ is the same parameter space as above and where

\begin{align*}
p(\bx_1,\ldots,\bx_m,y_1,\ldots,y_m; \btheta) &= p(y_1,\ldots,y_m \mid \bx_1,\ldots,\bx_m) p(\bx_1,\ldots,\bx_m) \\
&= \left\{\prod_{i=1}^m \frac{1}{\sqrt{2\pi \sigma^2}}\exp \left[ -\frac{1}{2\sigma^2} (y_i-\beta_0 - \bx_i^\intercal \bbeta)^2\right]\right\} p(\bx_1,\ldots,\bx_m).
\end{align*}

In this case, the $(mn + m)$-dimensional random vector $(\bX_1,\ldots,\bX_m,y_1,\ldots,y_m)$ plays the roll of the vector $\bX$ in {prf:ref}`para-model-def` (so that $k=mn+m$).















## Parameter estimators

```{prf:definition}
:label: para-model-def

Let $\calP_0$ be a parametric model for a $k$-dimensional random vector $\bX$ with $d$-dimensional parameter space $\Omega$. A _parameter estimator_ (or simply an _estimator_) is a statistic

$$
\hatbtheta = \delta(\bX),
$$

where $\delta: \bbr^k \to \bbr^d$ is a vector-valued function. An observed value of $\hatbtheta$ is called a _point estimate_.
```

There will be much abuse of terminology and notation regarding parameter estimators; it seems wise, then, to formally issue the following:

```{warning}

1. Following our previously established convention of representing random objects with capital letters, we _should_ write a parameter estimator as $\widehat{\boldsymbol \Theta}$, where $\boldsymbol\Theta$ is a capital theta. However, this is notationally awkward, so we will not do this.
2. Though technically the parameter estimator is the random vector $\hatbtheta$, we will use the word _estimator_ to also refer to the function $\delta$. Moreover, we will often use the notations $\hatbtheta$ and $\delta$ interchangeably.
3. To complicate things even more, we will sometimes write $\hatbtheta$ to refer to a point estimate.
```

Thus, the single piece of notation $\hatbtheta$ might stand for one of three things: Either the random vector $\delta(\bX)$, the function $\delta$, or an observed value of the random vector $\delta(\bX)$. You will need to rely on context to determine which of these three objects is meant when you encounter the symbol $\hatbtheta$.


