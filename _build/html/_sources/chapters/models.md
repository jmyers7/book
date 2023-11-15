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

(prob-models)=
# Probabilistic models

**THIS CHAPTER IS CURRENTLY UNDER CONSTRUCTION!!!**

## A first look at probabilistic models

## Graphs and factorizations

At the most basic level, a _directed graph_ $G$ consists of two sets $V$ and $E$ of _vertices_ and _edges_. The vertices are visualized as nodes, and the edges are arrows that run between the nodes. For example, consider the following graph $G$:

```{image} ../img/graph-01.svg
:width: 70%
:align: center
```
&nbsp;

This graph consists of five vertices and five edges:

$$
V = \{v_1,v_2,v_3,v_4,v_5\}, \quad E = \{e_1,e_2,e_3,e_4,e_5\}.
$$

Notice that this graph is _acyclic_, which means that beginning at any given node, there is no (directed) path along the edges that returns to the original node. Thus, our graph $G$ is an example of a _directed acyclic graph_, or _DAG_ for short.

If there is a directed edge $v_i \to v_j$ from a node $v_i$ to a node $v_j$, then $v_i$ is said to be a _parent_ of $v_j$ and $v_j$ is called a _child_ of $v_i$. For example, in our graph $G$ above, the set of parents of $v_4$ is $\{v_1,v_2\}$, while the set of children of $v_4$ is $\{v_5\}$.

More generally, if there exists a directed path from $v_i$ to $v_j$ of any length, say

$$
v_i \to \cdots \to v_j,
$$

then $v_j$ is called a _descendant_ of $v_i$. Thus, $v_j$ is a child of $v_i$ if there exists a directed path of length $1$. In our graph above, the node $v_5$ is a descendant of $v_2$ which is not a child.


The vertex set of a graph can be any set whatsoever; in particular, we can take the vertex set of a DAG to be a set of random variables $V = \{X_1,\ldots,X_n\}$. In our running example from the previous section, we might imagine that our graph has vertex set consisting of five random variables:

```{image} ../img/graph-02.svg
:width: 70%
:align: center
```
&nbsp;

I have omitted the edge labels for clarity, which I will continue to do in what follows.


```{prf:definition}

Let $V = \{X_1,\ldots,X_n\}$ be a collection of random variables, $G$ a graph with vertex set $V$, and $p(x_1,\ldots,x_n)$ the joint probability function. We shall say that $G$ _represents_ the joint probability distribution, or that $p$ _factors over $G$_, if

$$
p(x_1,\ldots,x_n) = \prod_{i=1}^n p(x_i | \text{parents of $x_i$}).
$$

```

Note that I am intentionally confusing an observed value $x_i$ of a random variable $X_i$ with the random variable itself, so that the "parents of $x_i$" actually makes sense.

In our running example, the graph $G$ represents the joint probability distribution provided that

$$
p(x_1,x_2,x_3,x_4,x_5) = p(x_1|x_2)p(x_2)p(x_3|x_2)p(x_4|x_1,x_2)p(x_5|x_4).
$$

Notice that the random variable $X_2$ has no parents in $G$, so that the marginal probability function $p(x_2)$ serves in place of a conditional distribution.


## Deterministic nodes, parameters, and plate notation




## Linear regression models

````{prf:definition}

A _linear regression model_ is a probabilistic graphical model whose underlying graph is of the form

```{image} ../img/lin-reg-00.svg
:width: 50%
:align: center
```
&nbsp;

where $\mathbf{x} \in \mathbb{R}^n$. The model has the following parameters:

* A parameter vector $\boldsymbol\beta \in \mathbb{R}^{n}$.

* A real parameter $\beta_0\in \mathbb{R}$.

* A positive real parameter $\sigma^2 \in \mathbb{R}$.

The link function at $Y$ is given by

$$
\mu = \mathbf{x}^T \boldsymbol\beta + \beta_0, \quad \text{where} \quad Y ; \mathbf{x}, \boldsymbol\beta,\beta_0,\sigma^2 \sim \mathcal{N}\big(\mu,\sigma^2\big).
$$
````

The components of the vector $\mathbf{x}$ are referred to as _predictors_, _regressors_, _explanatory variables_, or _independent variables_, while the random variable $Y$ is called the _response variable_ or the _dependent variable_. In the case that $n=1$, the model is called a _simple linear regression model_; otherwise, it is called a _multiple linear regression model_.

````{prf:theorem} Plate notation for linear regression models

A linear regression model in plate notation is given by

```{image} ../img/lin-reg-01.svg
:width: 50%
:align: center
```
&nbsp;

The joint probability function factors as

$$
p(y_1,\ldots,y_m; \mathbf{x}_1,\ldots,\mathbf{x}_m, \boldsymbol\beta,\beta_0,\sigma^2) = \frac{1}{\big(2\pi \sigma^2\big)^{m/2}} \exp \left[ - \frac{1}{2\sigma^2} \sum_{i=1}^m \big( y_i - \mu_i\big)^2 \right]
$$

where $\mu_i = \mathbf{x}_i^T \boldsymbol\beta + \beta_0$.
````

The proof of the factorization is a simple computation:

\begin{align*}
p(y_1,\ldots,y_m; \mathbf{x}_1,\ldots,\mathbf{x}_m, \boldsymbol\beta,\beta_0,\sigma^2) &= \prod_{i=1}^m p\big( y_i; \mathbf{x}_i, \boldsymbol\beta,\beta_0,\sigma^2\big) \\
&= \prod_{i=1}^m \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left[ -\frac{1}{2\sigma^2}\big(y_i - \mu_i\big)^2\right] \\
&= \frac{1}{\big(2\pi \sigma^2\big)^{m/2}} \exp \left[ - \frac{1}{2\sigma^2} \sum_{i=1}^m \big( y_i - \mu_i\big)^2 \right].
\end{align*}

Note that

$$
E\big(Y_i\big) = \mu_{i} = \beta_0 + \beta_1 x_{i,1} + \cdots + \beta_n x_{i,n},
$$

and so a linear regression model assumes (among other things) that the means of the response variables are linearly related to the regressors through the function

$$
\mu = \beta_0 + \beta_1z_1 + \cdots + \beta_n z_n \quad (\mu, z_1,\ldots,z_n\in \mathbb{R}).
$$ (lin-reg-line-eqn)

The parameter $\beta_0$ is often called the _intercept coefficient_, while the other $\beta_j$'s (for $j>0$) are called _slope coefficients_ since they are exactly the (infinitesimal) slopes:

$$
\frac{\partial \mu}{\partial z_j} = \beta_j.
$$

The line {eq}`lin-reg-line-eqn` is often called the _regression line_ of the model, and it is often displayed in a scatter plot with observed data. For example, suppose we consider the Ames housing data from the <a href="https://github.com/jmyers7/stats-book-materials/tree/main/programming-assignments">third programming assignment</a>. In this dataset, we have two columns of observations on _price_ (measured in thousands of US dollars) and _area_ (measured in square feet):

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 70%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import scipy as sp
from itertools import product
import statsmodels.formula.api as smf
import warnings
plt.style.use('../aux-files/custom_style_light.mplstyle')
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
warnings.filterwarnings("ignore")
magenta = '#FD46FC'

url = 'https://raw.githubusercontent.com/jmyers7/stats-book-materials/main/data/data-3-1.csv'
df = pd.read_csv(url, usecols=['area', 'price'])
df
```

We might believe that the price observations $y_1,\ldots,y_{2{,}930}$ come from an IID random sample $Y_1,\ldots,Y_{2{,}930}$ that may be modeled with the area observations $x_1,\ldots,x_{2{,}930}$ through a simple linear regression model. As we will see below (in {numref}`mle-lin-reg-sec`), it is possible to choose an "optimal" value for the parameter vector $\beta = (\beta_0,\beta_1)$ leading to a "best fit" regression line, also called a _least squares line_. Using these values of $\beta_0$ and $\beta_1$, we may plot this regression line along with the data in a scatter plot:


```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 70%

lr = smf.ols(formula='price ~ area', data=df).fit()
beta_0, beta_1 = lr.params

min_data = df['area'].min()
max_data = df['area'].max()
grid = np.linspace(min_data, max_data)
df.plot(kind='scatter', x='area', y='price', alpha=0.15)
plt.plot(grid, beta_0 + beta_1 * grid, color=magenta)
plt.show()
```

In general, the random variables

$$
\epsilon_{i} \stackrel{\text{def}}{=} Y_i - \mu_i
$$

in a linear regression model are called the _error terms_. Note then that

$$
Y_i = \beta_0 + \beta_1 x_{i,1} + \cdots + \beta_n x_{i,n} + \epsilon_i
$$

and $\epsilon_i \sim \mathcal{N}\big( 0,\sigma^2\big)$ for each $i=1,\ldots,m$. Thus, in a linear regression model, all error terms share the same variance; this assumption is called _homoscedasticity_. If we have observed values $y_{1},\ldots,y_{m}$, then the differences

$$
y_i - \hat{y}_i
$$ (resid-eqn)

are observed values of the error terms, where

$$
\hat{y}_i \stackrel{\text{def}}{=} \mu_{i} = \beta_0 + \beta_1x_{i,1} + \cdots + \beta_n x_{i,n}
$$

are the _predicted values_ of the $y_i$'s. The differences {eq}`resid-eqn` are called the _residuals_.

Based on the scatter plot above, it is apparent that the homoscedasticity assumption is violated in the Ames dataset. This is made even more apparent by plotting the residuals against the predictor variable:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 70%

resid = lr.resid
plt.scatter(x=df['area'], y=resid, alpha=0.20)
plt.xlabel('area')
plt.ylabel('residuals')
plt.show()
```

Indeed, the distributions of the residuals appear to widen as the area variable increases.

As with the parameter vector $\beta$, it is also possible to estimate an "optimal" value of the variance $\sigma^2$ in the linear regression model for the Ames dataset. Given these parameters, we may then generate new datasets by sampling from the normal distributions

$$
\mathcal{N}\big(\mu_i, \sigma^2\big)
$$

for each $i=1,2,\ldots,2{,}930$. It is interesting to produce scatter plots of a few generated datasets and compare their shape to the real dataset above:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 100%

np.random.seed(42)
sigma = np.sqrt(lr.scale)
y_hat = lr.predict()

_, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharex=True, sharey=True)
for i, j in product(range(2), range(2)):
    y_gen = sp.stats.norm(loc=y_hat, scale=sigma).rvs(2930)
    df_gen = pd.DataFrame({'area': df['area'], 'price': y_gen})
    df_gen.plot(kind='scatter', x='area', y='price', alpha=0.15, ax=axes[i, j])
    axes[i, j].plot(grid, beta_0 + beta_1 * grid, color=magenta)
axes[0, 0].set_title(f'generated dataset 1')
axes[0, 1].set_title(f'generated dataset 2')
axes[1, 0].set_title(f'generated dataset 3')
axes[1, 1].set_title(f'generated dataset 4')
plt.tight_layout()
```

The lines in these plots are copies of the original "best fit" regression line. For comparison, let's plot contours of KDEs for the true dataset and the fourth generated one:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 100%

df['indicator'] = 'true data PDF'
df_gen['indicator'] = 'generated data PDF'
df_combined = pd.concat(objs=[df, df_gen], axis=0)

g = sns.kdeplot(data=df_combined, x='area', y='price', hue='indicator', levels=6)
g.get_legend().set_title(None)
sns.move_legend(obj=g, loc='upper left')
plt.xlim(250, 3000)
plt.ylim(-50, 450)
plt.tight_layout()
```

For smaller values of area, we see that the distributions of the true prices are narrower compared to the generated prices, while for larger values of area, the distributions of the true prices are wider.

The residuals for the four generated datasets look as follows:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 100%

np.random.seed(42)

_, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharex=True, sharey=True)
for i, j in product(range(2), range(2)):
    y_gen = sp.stats.norm(loc=y_hat, scale=sigma).rvs(2930)
    df_gen = pd.DataFrame({'area': df['area'], 'price': y_gen})
    y_hat = lr.predict(df_gen)
    resid = y_gen - y_hat
    axes[i, j].scatter(x=df['area'], y=resid, alpha=0.20)
    axes[i, j].set_xlabel('area')
    axes[i, j].set_ylabel('residuals')
axes[0, 0].set_title(f'generated dataset 1')
axes[0, 1].set_title(f'generated dataset 2')
axes[1, 0].set_title(f'generated dataset 3')
axes[1, 1].set_title(f'generated dataset 4')
plt.tight_layout()
```












## Logistic regression models

````{prf:definition}

A _logistic regression model_ is a probabilistic graphical model whose underlying graph is of the form

```{image} ../img/log-reg-00.svg
:width: 50%
:align: center
```
&nbsp;

where $\mathbf{x}\in \mathbb{R}^n$. The model has the following parameters:

* A parameter vector $\boldsymbol\beta \in \mathbb{R}^n$.

* A real parameter $\beta_0\in \mathbb{R}$.

The link function at $Y$ is given by

$$
\phi = \sigma(\mathbf{x}^T\boldsymbol\beta + \beta_0), \quad \text{where} \quad Y; \mathbf{x},\boldsymbol\beta,\beta_0 \sim \mathcal{B}er(\phi).
$$
````

````{prf:theorem} Plate notation for logistic regression models

A logistic regression model in plate notation is given by

```{image} ../img/log-reg-01.svg
:width: 50%
:align: center
```
&nbsp;

The joint probability function factors as

$$
p(y_1,\ldots,y_m; \mathbf{x}_1,\ldots,\mathbf{x}_m, \boldsymbol\beta,\beta_0) = \prod_{i=1}^m \phi_i^{y_i}(1-\phi_i)^{1-y_i}
$$

where $\phi_i = \sigma(\mathbf{x}_i^T\boldsymbol\beta + \beta_0)$.
````



## Neural network models

````{prf:definition}

A _neural network model_ is a probabilistic graphical model whose underlying graph is of the form

```{image} ../img/nn-00.svg
:width: 50%
:align: center
```
&nbsp;

where $\mathbf{x} \in \mathbb{R}^n$ and $\mathbf{z}\in \mathbb{R}^k$. The model has the following parameters:

* A parameter matrix $\boldsymbol\alpha \in \mathbb{R}^{n\times k}$.

* A parameter vector $\boldsymbol\alpha_0 \in \mathbb{R}^k$

* A parameter vector $\boldsymbol\beta \in \mathbb{R}^{k}$.

* A real parameter $\beta_0 \in \mathbb{R}$.

The link function at $\mathbf{z}$ is given by

$$
\mathbf{z} = \sigma(\mathbf{x}^T\boldsymbol\alpha + \boldsymbol\alpha_0),
$$

while the link function at $Y$ is given by

$$
\phi = \sigma(\mathbf{z}^T\boldsymbol\beta + \beta_0), \quad \text{where} \quad Y ; \mathbf{z}, \boldsymbol\beta,\beta_0 \sim \mathcal{B}er\big(\phi\big).
$$
````

````{prf:theorem} Plate notation for neural network models

A neural network model in plate notation is given by

```{image} ../img/nn-01.svg
:width: 50%
:align: center
```
&nbsp;

The joint probability function factors as

$$
p(y_1,\ldots,y_m; \mathbf{z}_1,\ldots,\mathbf{z}_m,\boldsymbol\beta,\beta_0) = \prod_{i=1}^m \phi_i^{y_i}(1-\phi_i)^{1-y_i}
$$

where $\phi_i = \sigma(\mathbf{z}_i^T\boldsymbol\beta + \beta_0)$.
````





## Gaussian mixture models