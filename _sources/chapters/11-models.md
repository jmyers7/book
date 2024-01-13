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

(prob-models)=
# Probabilistic graphical models









## Probabilistic graphical models

Let's begin in the simple case with two deterministic vectors $\bx\in \bbr^n$ and $\by \in \bbr^m$. By saying that there is a _deterministic flow of influence_ from $\bx$ to $\by$, we shall mean simply that there is a function

$$
g: \bbr^n \to \bbr^m, \quad \by = g(\bx),
$$

called a _link function_. It will be convenient to depict this situation graphically by representing the variables $\bx$ and $\by$ as nodes in a <a href="https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)">graph</a> and the link function $g$ as an arrow between them:

```{image} ../img/det-link.svg
:width: 30%
:align: center
```
&nbsp;

Very often, the label $g$ on the link function will be omitted.

It could be the case that flow of influence is parametrized. For example, $g$ might be a linear transformation that is represented by a matrix $\mathcal{A} \in \bbr^{m\times n}$, with the entries in the matrix serving as parameters for the flow. We would represent this situation as

```{image} ../img/det-link-2.svg
:width: 30%
:align: center
```
&nbsp;

where the parameter matrix is represented by an un-circled node.

For a more complex example, consider the following graph:

```{image} ../img/det-link-3.svg
:width: 30%
:align: center
```
&nbsp;

This might represent a link function of the form

$$
\bz = \mathcal{A} \bx + \mathcal{B} \by, \quad \bx \in \bbr^{n\times 1}, \ \by\in \bbr^{k\times 1}, \ \bz \in \bbr^{m\times 1},
$$

which is parametrized by matrices $\mathcal{A} \in \bbr^{m\times n}$ and $\mathcal{B} \in \bbr^{m\times k}$.

The vectors in our discussion might be random, rather than deterministic, say $\bX$ and $\bY$. In this case, a _stochastic flow of influence_ from $\bX$ to $\bY$ would be visualized just as before:


```{image} ../img/random-link.svg
:width: 30%
:align: center
```
&nbsp;

This flow is represented mathematically via a _link function_ $\btheta = g(\bx)$ where $\bx$ is an observed value of $\bX$ and $\btheta$ is a parameter that uniquely determines the probability distribution of $\bY$. So, in this case, an observed value $\bx$ does _not_ determine a particular observed value $\by$ of $Y$, but rather an entire probability distribution over the $\by$'s. This probability distribution is conditioned on $\bX$, so the link function is often specified by giving the functional form of the conditional probability function $p(\by | \bx)$. Notice that only observed values $\bx$ of $\bX$ are used to determine the distribution of $\bY$ through the link---the distribution of $\bX$ itself plays no role.

These stochastic flows might be parametrized. For example, suppose $\bY$ is $1$-dimensional, equal to a random variable $Y$, while $\bX\in \mathbb{R}^{1\times n}$ is an $n$-dimensional random row vector. Then, a particular example of a stochastic flow of influence is given by the graph


```{image} ../img/lin-reg-0.svg
:width: 45%
:align: center
```
&nbsp;

The parameters consist of a real number $\beta_0 \in \bbr$, a column vector $\bbeta \in \bbr^{n\times 1}$, and a positive number $\sigma^2 >0$. A complete description of the link function at $Y$ is given by

$$
\mu \stackrel{\text{def}}{=} \beta_0 + \bx \bbeta, \quad \text{where} \quad Y \mid \bX; \ \beta_0, \bbeta,\sigma^2 \sim \mathcal{N}(\mu, \sigma^2).
$$

In fact, this is exactly a _linear regression model_, which we will see again in {numref}`lin-reg-sec` below, as well as in {numref}`Chapters %s <learning>` and {numref}`%s <lin-reg>`.


We shall take a flow of influence of the form

```{image} ../img/mixed-1.svg
:width: 30%
:align: center
```
&nbsp;

from a deterministic vector $\bx$ to a stochastic one $\bY$ to mean that there is a link function $\btheta = g(\bx)$ where $\btheta$ is a parameter that uniquely determines the distribution of $\bY$. Such a link function is often specified by giving the functional form of the parametrized probability function $p(\by; \bx)$.

A flow of influence of the form

```{image} ../img/mixed-2.svg
:width: 30%
:align: center
```
&nbsp;

from a random vector $\bX$ to a deterministic vector $\by$ means that there is a link function of the form $\by = g(\bx)$, so that observed values of $\bX$ uniquely determine values of $\by$.

The probabilistic graphical models that we will study in this chapter are meant to model real-world datasets. These datasets will often be conceptualized as observations of random or deterministic vectors, and these vectors are then integrated into a graphical model. These vectors are called _observed_ or _visible_, while all others are called _latent_ or _hidden_. To visually represent observed vectors in the graph structure, their nodes will be shaded; the nodes associated with _hidden_ vectors are left unshaded. For example, if we draw

```{image} ../img/shaded.svg
:width: 30%
:align: center
```
&nbsp;

then we mean that $\bX$ is observed while $\by$ is hidden.

It is important to note that for the simple types of models we consider in this chapter, the datasets consist of observations across _all_ observed nodes in the model. For example, let's suppose that we have a graphical structure of the form

```{image} ../img/unplated.svg
:width: 50%
:align: center
```
&nbsp;

with two observed random vectors $\bY$ and $\bZ$ and one hidden. Then, by saying that $\bY$ and $\bZ$ are observed, we mean that we have in possession a pair $(\by, \bz) $ consisting of observed values of $\bY$ and $\bZ$.

We may integrate IID random samples into our graphical framework as follows. Suppose that instead of a single copy of the graph above, we have a collection of graphs

```{image} ../img/unplated-02.svg
:width: 50%
:align: center
```
&nbsp;

one for each $i=1,\ldots,m$, where the random vector $\bX$ and the parameters $\balpha$ and $\bbeta$ are assumed to be _shared_ across all $i$. In the case that $m=3$ (for example), we may assemble all these graphs together into a single large graph

```{image} ../img/unplated-03.svg
:width: 65%
:align: center
```
&nbsp;

which explicitly shows that $\bX$, $\balpha$, and $\bbeta$ are shared across all $i$. Clearly, drawing these types of graphs becomes unwieldy for large $m$, so analysts have invented a method for depicting repetition in graphs by drawing a rectangle around the portion that is supposed to be duplicated:

```{image} ../img/plated-01.svg
:width: 55%
:align: center
```
&nbsp;

This is called _plate notation_, where the rectangle is called the _plate_. The visible nodes in the plate are assumed to be grouped as pairs $(\bY^{(i)},\bZ^{(i)})$, and altogether they form an IID random sample

$$
(\bY^{(1)},\bZ^{(1)}),\ldots,(\bY^{(m)},\bZ^{(m)}).
$$

We now have everything that we need to define _probabilistic graphical models_ in general. After the definition, the remaining sections in this chapter are devoted to the study of particular examples of such models.

```{prf:definition}
:label: pgm-def

A _probabilistic graphical model_ (_PGM_) consists of the following:

1. A set of vectors, some random and some deterministic, and some marked as observed and all others as hidden.

2. A graphical structure depicting the vectors as nodes and flows of influence as arrows between the nodes. If any of these flows are parametrized, then the graphical structure also has (un-circled) nodes for the parameters.

3. Mathematical descriptions of the flows as (possibly parametrized) link functions.
```























(lin-reg-sec)=
## Linear regression models

The type of PGM defined in this section is one of the simplest, but also one of the most important. Its goal is to model an observed dataset

$$
(\bx^{(1)}, y^{(1)}), (\bx^{(2)},y^{(2)}),\ldots, (\bx^{(m)},y^{(m)}) \in \bbr^{1\times n} \times \bbr
$$

where we believe that

```{math}
:label: approx-linear-eqn

y^{(i)} \approx \beta_0 + \bx^{(i)}\bbeta
```

for some parameters $\beta_0 \in \bbr$ and $\bbeta \in \bbr^{n\times 1}$. For example, let's consider the Ames housing dataset from the <a href="https://github.com/jmyers7/stats-book-materials/tree/main/programming-assignments">third programming assignment</a> and {numref}`Chapter %s <random-vectors>`; it consists of $m=2{,}930$ bivariate observations

$$
(x^{(1)}, y^{(1)}), (x^{(2)},y^{(2)}),\ldots, (x^{(m)},y^{(m)}) \in \bbr^2
$$

where $x^{(i)}$ and $y^{(i)}$ are the size (in square feet) and selling price (in thousands of US dollars) of the $i$-th house in the dataset. A scatter plot of the dataset looks like

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib as mpl
import matplotlib_inline.backend_inline
import seaborn as sns
import scipy as sp
from itertools import product
import warnings
plt.style.use('../aux-files/custom_style_light.mplstyle')
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
warnings.filterwarnings('ignore')
blue = '#486AFB'
magenta = '#FD46FC'

# linear regression example begins below

# import linear regression model from scikit-learn
from sklearn.linear_model import LinearRegression

# import data
url = 'https://raw.githubusercontent.com/jmyers7/stats-book-materials/main/data/data-3-1.csv'
df = pd.read_csv(url, usecols=['area', 'price'])

# pull out the 'area' column and 'price column from the data and convert them to numpy arrays
X = df['area'].to_numpy().reshape(-1, 1)
y = df['price'].to_numpy()

# instantiate a linear regression model
model = LinearRegression()

# train the model
model.fit(X=X, y=y)

# get the learned parameters
beta, beta_0 = model.coef_, model.intercept_

# build a grid for the regression line
grid = np.linspace(X.min(), X.max())

# plot the regression line
plt.plot(grid, beta * grid + beta_0, color=magenta)

# plot the data
plt.scatter(x=X, y=y, alpha=0.15)

plt.xlabel('area')
plt.ylabel('price')
plt.gcf().set_size_inches(w=5, h=3)
plt.tight_layout()
```

The positively-sloped line is used to visualize the approximate linear relationship {eq}`approx-linear-eqn`. This is a so-called _least squares line_ or _regression line_; we will learn how to compute them in {numref}`Chapter %s <learning>`.

But for now, let's define our first PGM:

````{prf:definition}
:label: linear-reg-def

A _linear regression model_ is a probabilistic graphical model whose underlying graph is of the form

```{image} ../img/lin-reg-00.svg
:width: 50%
:align: center
```
&nbsp;

where $\bX\in \bbr^{1\times n}$. The model has the following parameters:

* A real parameter $\beta_0\in \mathbb{R}$.

* A parameter vector $\bbeta \in \mathbb{R}^{n\times 1}$.

* A positive real parameter $\sigma^2>0$.

The link function at $Y$ is given by

$$
Y \mid \bX; \ \beta_0,\bbeta,\sigma^2 \sim \mathcal{N}\big(\mu,\sigma^2\big), \quad \text{where} \quad \mu = \beta_0 + \bx \bbeta.
$$
````

Before we introduce important terminology associated with linear regression models and look at an example, we need to discuss two probability density functions that will play a crucial role in the [next chapter](learning). The first is just the conditional density function of $Y$ given $\bX$:

```{prf:definition}
:label: linear-reg-pf-def

The _model probability function for a linear regression model_ is the conditional probability density function

$$
p\big(y \mid \bx ; \ \beta_0, \bbeta, \sigma^2\big).
$$

On its support consisting of all $y\in \bbr$ and $\bx \in \bbr^{1\times n}$, it is given by the formula

$$
p\big(y \mid \bx ; \ \beta_0, \bbeta, \sigma^2\big) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left(- \frac{1}{2\sigma^2} ( y - \mu)^2 \right),
$$

where $\mu = \beta_0 + \bx \bbeta$.
```

The second important probability density function is obtained from the plated version of a linear regression model:

```{image} ../img/lin-reg-00-plated.svg
:width: 50%
:align: center
```
&nbsp;

Observations of the visible nodes correspond to an observed dataset. Then:

```{prf:definition}
:label: lin-reg-data-pf-def

Given an observed dataset

$$
(\bx^{(1)}, y^{(1)}), (\bx^{(2)},y^{(2)}),\ldots, (\bx^{(m)},y^{(m)}) \in \bbr^{1\times n} \times \bbr,
$$

the _data probability function for a linear regression model_ is the conditional probability density function

$$
p\big(y^{(1)},\ldots,y^{(m)} \mid \bx^{(1)},\ldots,\bx^{(m)}; \ \beta_0, \bbeta,\sigma^2 \big) = \prod_{i=1}^m p\big(y^{(i)} \mid \bx^{(i)} ; \ \beta_0, \bbeta, \sigma^2\big).
$$ (data-pf-eqn)
```

Note that the data probability function appears to be _defined_ as a product of model probability functions. However, using independence of the random sample

$$
(\bX^{(1)},Y^{(1)}),\ldots,(\bX^{(m)}, Y^{(m)}),
$$

one may actually _prove_ that the left-hand side of {eq}`data-pf-eqn` is equal to the product on the right-hand side; see the [suggested problems](https://github.com/jmyers7/stats-book-materials/blob/main/suggested-problems/10-2-suggested-problems.md#problem-1-solution) for this section.

The components of the vector $\bX$ are referred to as _predictors_, _regressors_, _explanatory variables_, or _independent variables_, while the random variable $Y$ is called the _response variable_ or the _dependent variable_. In the case that $n=1$, the model is called a _simple linear regression model_; otherwise, it is called a _multiple linear regression model_.

Note that

$$
E\big(Y \mid \bX = \bx \big) = \mu = \beta_0 + \bx \bbeta,
$$

and so a linear regression model assumes (among other things) that the conditional mean of the response variable is linearly related to the regressors through the link function

$$
\mu = \beta_0 + \bx \bbeta.
$$ (lin-reg-line-eqn)

The parameter $\beta_0$ is often called the _intercept_ or _bias term_, while the other $\beta_j$'s (for $j>0$) are called _weights_ or _slope coefficients_ since they are exactly the (infinitesimal) slopes:

$$
\frac{\partial \mu}{\partial x_j} = \beta_j.
$$

The random variable

$$
\dev \stackrel{\text{def}}{=} Y - \beta_0 - \bX\bbeta
$$

in a linear regression model is called the _error term_; note then that

$$
Y = \beta_0 + \bX\bbeta + \dev \quad \text{and} \quad \dev \sim \mathcal{N}(0, \sigma^2).
$$ (random-lin-rel-eqn)

This is the manifestation in terms of random vectors and variables of the approximate linear relationship {eq}`approx-linear-eqn` described at the beginning of this section.

Suppose we are given an observed dataset

$$
(\bx^{(1)}, y^{(1)}), (\bx^{(2)},y^{(2)}),\ldots, (\bx^{(m)},y^{(m)}) \in \bbr^{1\times n} \times \bbr.
$$

If for each $i=1,\ldots,m$, we define the _predicted values_

$$
\hat{y}^{(i)} = \beta_0 + \bx^{(i)}\bbeta
$$

and the _residuals_

$$
\dev^{(i)} = y^{(i)} - \hat{y}^{(i)},
$$

then from {eq}`random-lin-rel-eqn` we get

$$
y^{(i)} = \beta_0 + \bx^{(i)} \bbeta + \dev^{(i)}.
$$

This shows that the residuals $\dev^{(i)}$ are observations of the error term $\dev \sim \mathcal{N}(0,\sigma^2)$. Thus, in a linear regression model, all residuals from a dataset are assumed to be modeled by a normal distribution with mean $0$ and a _fixed_ variance; the fixed-variance assumption is sometimes called _homoscedasticity_.

In {numref}`Chapter %s <learning>`, we will learn how to train a linear regression model on a dataset to obtain optimal values of the parameters $\beta_0$ and $\bbeta$. Using these training methods, we obtained values for the parameters $\beta_0$ and $\bbeta = \beta_1$ for the Ames housing dataset mentioned at the beginning of this section. The positively-sloped line in the scatter plot was the line traced out by the link function $\mu = \beta_0 + \beta_1 x $. The predicted values $\hat{y}^{(i)}$ lie along this line, and the magnitude of the residual $\dev^{(i)}$ may be visualized as the vertical distance from the true data point $y^{(i)}$ to this line. We may plot the residuals $\dev^{(i)}$ against the predictor variables $x^{(i)}$ to get:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

# get the predictions
y_hat = model.predict(X=X)

# get the residuals
resid = y - y_hat

# plot the residuals vs. area
plt.scatter(x=X, y=resid, alpha=0.20)

plt.xlabel('area')
plt.ylabel('residuals')
plt.gcf().set_size_inches(w=5, h=3)
plt.tight_layout()
```

It is evident from this plot that the homoscedasticity assumption is violated since the distributions of the residuals appear to widen as the area variable increases.

As with the parameters $\beta_0$ and $\bbeta$, it is also possible to learn an optimal value of the variance $\sigma^2$. As another method of model checking, given all the learned parameters $\beta_0$, $\beta_1$, and $\sigma^2$ for the Ames dataset, we may generate a new dataset by sampling from the normal distributions 

$$
\mathcal{N}\big(\hat{y}^{(i)}, \sigma^2\big)
$$

for each $i=1,2,\ldots,m$. A scatter plot of one simulated dataset is:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

# import statsmodels
import statsmodels.formula.api as smf

# instantiate and train a linear regression model from statsmodels
model = smf.ols(formula='price ~ area', data=df).fit()

# get the learned standard deviation
sigma = np.sqrt(model.scale)

# generate the dataset
np.random.seed(42)
y_gen = sp.stats.norm(loc=y_hat, scale=sigma).rvs(2930)
df_gen = pd.DataFrame({'area': df['area'], 'price': y_gen})

# plot the dataset
df_gen.plot(kind='scatter', x='area', y='price', alpha=0.15)

# plot the original regression line
plt.plot(grid, beta_0 + beta * grid, color=magenta)

plt.gcf().set_size_inches(w=5, h=3)
plt.tight_layout()
```

To compare this simulated dataset against the real one, let's compare KDEs:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

df['indicator'] = 'true data PDF'
df_gen['indicator'] = 'simulated data PDF'
df_combined = pd.concat(objs=[df, df_gen], axis=0)

g = sns.kdeplot(data=df_combined, x='area', y='price', hue='indicator', levels=6)
g.get_legend().set_title(None)
sns.move_legend(obj=g, loc='upper left')
plt.xlim(250, 3000)
plt.ylim(-50, 450)
plt.gcf().set_size_inches(w=5, h=4)
plt.tight_layout()
```

For smaller values of area, the distribution of the true prices is narrower compared to the simulated prices, while for larger values of area, the distribution of the true prices is wider.











(log-reg-sec)=
## Logistic regression models

The types of models studied in this section are closely related to the linear regression models in the previous, but here the goal is to model a dataset of the form

$$
(\bx^{(1)}, y^{(1)}), (\bx^{(2)},y^{(2)}),\ldots, (\bx^{(m)},y^{(m)}) \in \bbr^{1\times n} \times \{0,1\}.
$$

Such datasets arise naturally in _binary classification problems_, where we aim to determine which of two classes a given object lies in based on predictor features. The true class of the $i$-th object is indicated by the value of $y^{(i)}$, while the vector $\bx^{(i)}$ consists of the predictor features.

As a running example through this and the next section, consider the following scatter plot:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

# import the data
url = 'https://raw.githubusercontent.com/jmyers7/stats-book-materials/main/data/ch10-book-data-01.csv'
df = pd.read_csv(url)

# plot the data
g = sns.scatterplot(data=df, x='x_1', y='x_2', hue='y')

# change the default seaborn legend
g.legend_.set_title(None)
new_labels = ['class 0', 'class 1']
for t, l in zip(g.legend_.texts, new_labels):
    t.set_text(l)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.gcf().set_size_inches(w=5, h=3)
plt.tight_layout()
```

The points represent the $2$-dimensional predictors

$$
\bx^{(i)} = \begin{bmatrix} x^{(i)}_1 & x^{(i)}_2 \end{bmatrix},
$$

while the color indicates the class $y^{(i)} \in \{0,1\}$. Our goal in this section is to capture the evident pattern in the data using a _logistic regression model_.

To define these models, we first need to discuss the important _sigmoid function_, defined as

$$
\sigma: \bbr \to (0,1), \quad \sigma(x) = \frac{1}{1+e^{-x}}.
$$

Its graph is:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

import torch
import torch.nn.functional as F

grid = torch.linspace(start=-10, end=10, steps=300)
plt.plot(grid, F.sigmoid(grid))
plt.gcf().set_size_inches(w=5, h=3)
plt.xlabel('$x$')
plt.ylabel('$\sigma(x)$')
plt.tight_layout()
```

Since the outputs of the sigmoid function land in the open interval $(0,1)$, we may use it to convert _any_ real number into a _probability_. Indeed, this is precisely its role in a _logistic regression model_:


````{prf:definition}
:label: log-reg-def

A _logistic regression model_ is a probabilistic graphical model whose underlying graph is of the form

```{image} ../img/log-reg-00.svg
:width: 50%
:align: center
```
&nbsp;

where $\bX\in \bbr^{1\times n}$. The model has the following parameters:

* A real parameter $\beta_0\in \mathbb{R}$.

* A parameter vector $\bbeta \in \mathbb{R}^{n\times 1}$.

The link function at $Y$ is given by

$$
Y \mid \bX; \ \beta_0,\bbeta \sim \mathcal{B}er(\phi), \quad \text{where} \quad \phi = \sigma(\beta_0 + \bx\bbeta),
$$

and where $\sigma$ is the sigmoid function.
````

Notice that the link function $\phi = \sigma(\beta_0 + \bx\bbeta)$ in a logistic regression model is precisely the affine link function $\mu = \beta_0 + \bx\bbeta$ of a linear regression model composed with the sigmoid function.

The two probability functions that we will use to train logistic regression models in the [next chapter](learning) are given as follows. For the first:

```{prf:definition}
:label: log-reg-pf-def

The _model probability function for a logistic regression model_ is the conditional probability density function

$$
p\big(y \mid \bx ; \ \beta_0, \bbeta\big).
$$

On its support consisting of all $y\in \{0,1\}$ and $\bx \in \bbr^{1\times n}$, it is given by the formula

$$
p\big(y \mid \bx ; \ \beta_0, \bbeta\big) = \phi^y (1-\phi)^{1-y}
$$

where $\phi = \sigma(\beta_0 + \bx \bbeta)$.
```

As with linear regression models, the second probability function is obtained from the plated version of a logistic regression model:

```{image} ../img/log-reg-00-plated.svg
:width: 50%
:align: center
```
&nbsp;

Then:

```{prf:definition}
:label: logreg-data-pf-def

Given a dataset

$$
(\bx^{(1)}, y^{(1)}), (\bx^{(2)},y^{(2)}),\ldots, (\bx^{(m)},y^{(m)}) \in \bbr^{1\times n} \times \{0,1\},
$$

the _data probability function for a logistic regression model_ is the conditional probability density function

$$
p\big(y^{(1)},\ldots,y^{(m)} \mid \bx^{(1)},\ldots,\bx^{(m)}; \ \beta_0, \bbeta\big) = \prod_{i=1}^m p\big(y^{(i)} \mid \bx^{(i)} ; \ \beta_0, \bbeta\big).
$$ (log-reg-data-pf-eqn)
```

As in {prf:ref}`lin-reg-data-pf-def`, one may _prove_ that the data probability function of a logistic regression model is given by the product of model probability functions in {eq}`log-reg-data-pf-eqn`.

Let's return to our toy dataset introduced at the beginning of the section. To aid with training, it is often helpful to _standardize_ the predictor features

$$
\bx^{(1)},\ldots,\bx^{(m)} \in \bbr^{1\times n}.
$$

This means that we compute the (empirical) mean $\bar{x}_j$ and standard deviation $s_j$ of each sequence

$$
x_j^{(1)},\ldots,x_j^{(m)} \in \bbr
$$

of components, and then replace each $x_j^{(i)}$ with

$$
\frac{x_j^{(i)} - \bar{x}_j}{s_j}.
$$

It is convenient to visualize this process in terms of the so-called _design matrix_

$$
\mathcal{X} = \begin{bmatrix} \leftarrow & \bx^{(1)} & \rightarrow \\ \vdots & \vdots & \vdots \\ \leftarrow & \bx^{(m)} & \rightarrow \end{bmatrix} = \begin{bmatrix} x_1^{(1)} & \cdots & x_n^{(1)} \\
\vdots & \ddots & \vdots \\
x_1^{(m)} & \cdots & x_n^{(m)}
\end{bmatrix}.
$$

Then the empirical means $\bar{x}_j$ and standard deviations $s_j$ are precisely the means and standard deviations of the columns.

If we standardize our toy dataset, we get the following:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

# import scaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# convert the data to numpy arrays
X = df[['x_1', 'x_2']].to_numpy()
y = df['y'].to_numpy()

# scale the input data
ss = StandardScaler()
X = ss.fit_transform(X=X)

# replaced the columns of the dataframe with the transformed data
df['x_1'] = X[:, 0]
df['x_2'] = X[:, 1]

# plot the scaled data
g = sns.scatterplot(data=df, x='x_1', y='x_2', hue='y')

# change the default seaborn legend
g.legend_.set_title(None)
new_labels = ['class 0', 'class 1']
for t, l in zip(g.legend_.texts, new_labels):
    t.set_text(l)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.gcf().set_size_inches(w=5, h=3)
plt.tight_layout()
```

Notice that the values of the two features $x_1$ and $x_2$ now lie in comparable ranges, while the overall _shape_ of the dataset has not changed.

Along with linear regression models, in the [next chapter](learning) we will see how to learn optimal values of the parameters $\beta_0$ and $\bbeta$ from data. With these parameters in hand, one way to check how well a logistic regression model captures the data is to draw a contour plot of the function $\phi = \sigma( \beta_0 + \bx \bbeta)$. This contour plot appears on the left in the following:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

# import logistic regression model from scikit-learn
from sklearn.linear_model import LogisticRegression

# instantiate a logistic regression model
model = LogisticRegression()

# train the model
model.fit(X=X, y=y)

# get the grid for the contour plot
resolution = 1000
x_1 = (-2, 2)
x_2 = (-3.5, 3.5)
x1_grid, x2_grid = torch.meshgrid(torch.linspace(*x_1, resolution), torch.linspace(*x_2, resolution))
grid = torch.column_stack((x1_grid.reshape((resolution ** 2, -1)), x2_grid.reshape((resolution ** 2, -1))))

# define colormaps for the contour plots
desat_blue = '#7F93FF'
desat_magenta = '#FF7CFE'
diverging_cmap = clr.LinearSegmentedColormap.from_list(name='diverging', colors=[desat_blue, 'white', desat_magenta], N=10)
binary_cmap = clr.LinearSegmentedColormap.from_list(name='binary', colors=[desat_blue, desat_magenta], N=2)

_, axes = plt.subplots(ncols=3, figsize=(10, 3), width_ratios=[10, 10, 1])

# generate the contour plots
z = model.predict_proba(grid)[:, 1]
z = z.reshape(resolution, resolution)
axes[0].contourf(x1_grid, x2_grid, z, cmap=diverging_cmap, levels=diverging_cmap.N)
z = model.predict(grid)
z = z.reshape(resolution, resolution)
axes[1].contourf(x1_grid, x2_grid, z, cmap=binary_cmap, levels=binary_cmap.N)

# create the colorbar
plt.colorbar(mpl.cm.ScalarMappable(cmap=diverging_cmap), cax=axes[2], orientation='vertical')

# plot the data
for axis in axes[:-1]:
    sns.scatterplot(data=df, x='x_1', y='x_2', hue='y', ax=axis, legend=False)
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')

plt.tight_layout()
```

To interpret this plot, remember that $\phi = \sigma( \beta_0 + \bx \bbeta)$ is the probability parameter for the class indicator variable $Y \sim \Ber(\phi)$, so we should interpret $\phi$ as the probability that the point $\bx$ is in class $1$ (corresponding to $y=1$). In the right-hand plot, we have "thresholded" the probability $\phi$ at $0.5$, creating a _predictor function_

$$
f:\bbr^{1\times 2} \to \{0,1\}, \quad f(\bx) = \begin{cases}
0 & : \sigma(\beta_0 + \bx\bbeta) < 0.5, \\
1 & : \sigma(\beta_0 + \bx\bbeta) \geq 0.5. \\
\end{cases}
$$

The _decision boundary_ is exactly the curve in $\bbr^2$ consisting of those $\bx$ for which the predictor $f$ is "flipping a coin," i.e., it consists of those points $\bx$ such that

$$
\sigma(\beta_0 + \bx \bbeta) = 0.5,
$$

which is equivalent to

$$
\beta_0 + \bx \bbeta = 0.
$$

Notice that this defines a _linear_ decision boundary that separates $\bbr^2$ into two unbounded half planes based on whether

$$
\beta_0 + \bx \bbeta > 0 \quad \text{or} \quad \beta_0 + \bx \bbeta < 0.
$$

Those vectors $\bx$ satisfying the first inequality would be predicted to belong to class $1$, while those satisfying the latter inequality would be predicted to belong to class $0$. As is evident from the plots, our logistic regression model is doing its best to accurately classify as many data points as possible, but our model is handicapped by the fact it will _always_ produce a linear decision boundary.












(nn-sec)=
## Neural network models

The desire to obtain _nonlinear_ decision boundaries is (in part) the motivation for the probabilistic graphical models studied in this section, called _neural networks_. While there are many (_many!_) different types of neural network architectures in current use, the particular type that we shall begin our study with are _fully-connected, feedforward neural networks with one hidden layer_.

Essentially, these types of neural networks are logistic regression models with a hidden deterministic node $\bz$ sandwiched between the predictor features $\bX$ and the response variable $Y$. The link from $\bz$ to $Y$ goes through the same sigmoid function used in the definition of logistic regression models, but the link from $\bX$ to $\bz$ goes through a function called the _rectified linear unit_ (_ReLU_), defined as

$$
\rho: \bbr \to [0,\infty), \quad \rho(x) = \max\{0, x\}.
$$

The ReLU function is piecewise linear, with a graph of the form:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

relu_grid = torch.linspace(start=-2, end=2, steps=200)
plt.plot(relu_grid, F.relu(relu_grid))
plt.gcf().set_size_inches(w=5, h=3)
plt.xlabel('$x$')
plt.ylabel('$\\rho(x)$')
plt.tight_layout()
```

We may apply the ReLU function to vectors $\bx\in \bbr^{1\times n}$ by "vectorization" (in pythonic language), which just means that we apply it componentwise:

$$
\rho(\bx) \def \begin{bmatrix} \rho(x_1) & \cdots & \rho(x_n) \end{bmatrix}.
$$

Using these pieces, we now state the official definition in the case that the neural network has one hidden layer; later, we shall indicate how one obtains "deeper" neural networks by adding additional hidden layers.


````{prf:definition}
:label: neural-net-def

A _(fully-connected, feedforward) neural network with one hidden layer_ is a probabilistic graphical model whose underlying graph is of the form

```{image} ../img/nn-00.svg
:width: 50%
:align: center
```
&nbsp;

where $\bX\in \bbr^{1\times n}$ and $\bz \in \bbr^{1\times k}$. The model has the following parameters:

* A parameter matrix $\mathcal{W} \in \mathbb{R}^{n\times k}$.

* A parameter vector $\bb \in \mathbb{R}^{1\times k}$.

* A parameter vector $\bw \in \mathbb{R}^{k\times 1}$.

* A real parameter $b \in \mathbb{R}$.

The link function at $\mathbf{z}$ is given by

$$
\mathbf{z} = \rho(\mathbf{x}\mathcal{W} + \bb),
$$

while the link function at $Y$ is given by

$$
Y ;\  \mathbf{z}, \bw, b \sim \mathcal{B}er\big(\phi\big), \quad \text{where} \quad \phi = \sigma(\bz\bw + b).
$$

Here, $\rho$ is the ReLU function and $\sigma$ is the sigmoid function.
````

The name "neural network" comes from a loose analogy with networks of biological neurons in the human brain. For this reason, sometimes neural networks just defined are called _artificial neural networks_ (*ANN*s).

Following the pattern begun with linear and logistic regression models, we first want to give the probability functions that we will use in the [next chapter](learning) to train neural network models. The first one is:

```{prf:definition}
:label: neural-net-pf-def

The _model probability function for a neural network model_ is the conditional probability density function

$$
p\big(y \mid \bx ; \ \mathcal{W}, \bb, \bw, b \big).
$$

On its support consisting of all $y\in \{0,1\}$ and $\bx \in \bbr^{1\times n}$, it is given by the formula

$$
p\big(y \mid \bx ; \ \mathcal{W}, \bb, \bw, b \big) = \phi^y (1-\phi)^{1-y}
$$

where $\phi = \sigma(\bz \bw + b)$ and $\bz = \sigma(\bx \mathcal{W} + \bb)$.
```

The second probability function is obtained from the plated version of a neural network:

```{image} ../img/nn-00-plated.svg
:width: 50%
:align: center
```
&nbsp;

Then:

```{prf:definition}
:label: neural-net-data-pf-def

Given a dataset

$$
(\bx^{(1)}, y^{(1)}), (\bx^{(2)},y^{(2)}),\ldots, (\bx^{(m)},y^{(m)}) \in \bbr^{1\times n} \times \{0,1\},
$$

the _data probability function for a neural network model_ is the conditional probability density function

$$
p\big(y^{(1)},\ldots,y^{(m)} \mid \bx^{(1)},\ldots,\bx^{(m)}; \ \mathcal{W}, \bb, \bw, b\big) = \prod_{i=1}^m p\big(y^{(i)} \mid \bx^{(i)} ; \ \mathcal{W}, \bb, \bw, b\big).
$$
```

Very often, one sees the underlying graph of a neural network displayed in terms of the components of the vectors (with the parameters omitted). For example, in the case that $\bX$ is $3$-dimensional and $\bz$ is $4$-dimensional, we might see the graph of the neural network drawn as

```{image} ../img/nn-neuron.svg
:width: 60%
:align: center
```
&nbsp;

In this format, the nodes are often called _(artificial) neurons_ or _units_. The visible neurons $X_1,X_2,X_3$ are said to comprise the _input layer_ of the network, the hidden neurons $z_1,z_2,z_3,z_4$ make up a _hidden layer_, and the single visible neuron $Y$ makes up the _output layer_. The network is called _fully-connected_ because there is a link function at a given neuron _from_ every neuron in the previous layer and _to_ every neuron in the subsequent layer; it is called a _feedfoward_ network because the link functions only go in one direction, with no feedback links. The link function at $z_j$ is of the form

$$
z_j = \rho(\bx \bw_j + b_j),
$$

where

$$
\mathcal{W} = \begin{bmatrix} \uparrow & \uparrow & \uparrow & \uparrow \\ \bw_1 & \bw_2 & \bw_3 & \bw_4 \\
\downarrow & \downarrow & \downarrow & \downarrow \end{bmatrix} \quad \text{and} \quad \bb = \begin{bmatrix} b_{1} & b_2 & b_3 & b_{4} \end{bmatrix}.
$$

The link function at $Y$ is given by the same formula as before using the sigmoid function. In the literature, the ReLU function $\rho$ and the sigmoid function $\sigma$ are often called _activation functions_ of the network. The parameters $\mathcal{W}$ and $\bw$ are called _weights_, while the parameters $\bb$ and $b$ are called _biases_.

From our networks with just one hidden layer, it is easy to imagine how we might obtain "deeper" networks by adding additional hidden layers; for example, a network with two hidden layers might look like this:

```{image} ../img/nn-neuron-02.svg
:width: 80%
:align: center
```
&nbsp;

If we collapse the neurons into vectors and bring in the parameters, this network would be drawn as

```{image} ../img/nn-02.svg
:width: 65%
:align: center
```
&nbsp;

There are now _two_ weight matrices $\mathcal{W}^{[1]}$ and $\mathcal{W}^{[2]}$, along with _two_ bias vectors $\bb^{[1]}$ and $\bb^{[2]}$. The link functions at $\bz^{[1]}$ and $\bz^{[2]}$ are given by

$$
\bz^{[i]} = \rho\big(\bz^{[i-1]} \mathcal{W}^{[i]} + \bb^{[i]} \big) \quad \text{for $i=1,2$,}
$$

where we set $\bz^{[0]} = \bx$. The link function at $Y$ is the same as it was before:

$$
Y; \ \bz^{[2]}, \bw, b \sim \Ber(\phi), \quad \text{where} \quad \phi = \sigma \big( \bz^{[2]} \bw + b\big).
$$

The _depth_ of a neural network is defined to be one less than the total number of layers. The "one less" convention is due to the fact that only the hidden and output layers are associated with trainable parameters. Equivalently, the _depth_ is the number of "layers" of link functions (with trainable parameters). The _widths_ of a network are defined to be the dimensions of the hidden vectors.

Let's return to our toy dataset from the [previous section](log-reg-sec), but with an extra four "blobs" of data just to make things interesting:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

url = 'https://raw.githubusercontent.com/jmyers7/stats-book-materials/main/data/ch10-book-data-03.csv'
df = pd.read_csv(url)

# convert the data to numpy arrays
X = df[['x_1', 'x_2']].to_numpy()
y = df['y'].to_numpy()

# scale the input data
ss = StandardScaler()
X = ss.fit_transform(X=X)

# replaced the columns of the dataframe with the transformed data
df['x_1'] = X[:, 0]
df['x_2'] = X[:, 1]

# convert the data to torch tensors
X = torch.tensor(data=X, dtype=torch.float32)
y = torch.tensor(data=y, dtype=torch.float32)

# plot the data
g = sns.scatterplot(data=df, x='x_1', y='x_2', hue='y')

# change the default seaborn legend
g.legend_.set_title(None)
new_labels = ['class 0', 'class 1']
for t, k2 in zip(g.legend_.texts, new_labels):
    t.set_text(k2)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.gcf().set_size_inches(w=5, h=3)
plt.tight_layout()
```

Trained on the original dataset (without the "blobs"), we saw that a logistic regression model produces a _linear_ decision boundary and thus misclassifies a nontrivial number of data points. In comparison, not only will a neural network produce a nonlinear decision boundary dividing the data in the original dataset, it will also correctly classify the data in the four new "blobs." Indeed, using the techniques in the [next chapter](learning), we trained a neural network on the new dataset with _three_ hidden layers of widths $8$, $8$, and $4$. Then, a contour plot of the function

$$
\phi = \sigma\big(\bz^{[3]}\bw + b\big)
$$

appears on the left-hand side of the following figure, while the "thresholded" version (at $0.5$) appears on the right-hand side displaying the (nonlinear!) decision boundaries:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

import torch.nn as nn

# define the neural network model architecture
torch.manual_seed(42)
k1 = 8 # width of first hidden layer
k2 = 8 # width of second hidden layer
k3 = 4 # width of third hidden layer

class NeuralNetwork(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()

        # three hidden layers...
        self.hidden1_linear = nn.Linear(in_features=input_dimension, out_features=k1)
        self.hidden1_act = nn.ReLU()
        self.hidden2_linear = nn.Linear(in_features=k1, out_features=k2)
        self.hidden2_act = nn.ReLU()
        self.hidden3_linear = nn.Linear(in_features=k2, out_features=k3)
        self.hidden3_act = nn.ReLU()
        
        # ...and one output layer
        self.output_linear = nn.Linear(in_features=k3, out_features=1)
        self.output_act = nn.Sigmoid()

    def forward(self, X):
        X = self.hidden1_act(self.hidden1_linear(X))
        hidden_output_1 = X
        X = self.hidden2_act(self.hidden2_linear(X))
        hidden_output_2 = X
        X = self.hidden3_act(self.hidden3_linear(X))
        hidden_output_3 = X
        X = self.output_act(self.output_linear(X))

        return X, hidden_output_1, hidden_output_2, hidden_output_3
    
model = NeuralNetwork(input_dimension=2)

# define the loss function and optimizer
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=5e-1)

# train the model
num_epochs = 4000
for _ in range(num_epochs):
    optimizer.zero_grad()
    y_hat = model(X)[0]
    loss = loss_fn(y_hat.squeeze(), y)
    loss.backward()
    optimizer.step()

_, axes = plt.subplots(ncols=3, figsize=(10, 4), width_ratios=[10, 10, 1])

# get the grid for the contour plot
resolution = 1000
x1_grid = torch.linspace(-1.75, 1.75, resolution)
x2_grid = torch.linspace(-1.5, 1.5, resolution)
x1_grid, x2_grid = torch.meshgrid(x1_grid, x2_grid)
grid = torch.column_stack((x1_grid.reshape((resolution ** 2, -1)), x2_grid.reshape((resolution ** 2, -1))))

# generate the contour plots
grid_outputs = model(grid)
z = grid_outputs[0].detach()
z = z.reshape(resolution, resolution)
axes[0].contourf(x1_grid, x2_grid, z, cmap=diverging_cmap, levels=diverging_cmap.N)
z = grid_outputs[0] >= 0.5
z = z.reshape(resolution, resolution)
axes[1].contourf(x1_grid, x2_grid, z, cmap=binary_cmap, levels=binary_cmap.N)

# create the colorbar
plt.colorbar(mpl.cm.ScalarMappable(cmap=diverging_cmap), cax=axes[2], orientation='vertical')

# plot the data
for axis in axes[:-1]:
    sns.scatterplot(data=df, x='x_1', y='x_2', hue='y', ax=axis, legend=False)
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')

plt.tight_layout()
```

Notice that the band of white dividing the original dataset (representing values $\phi \approx 0.5$) in the left-hand plot is much narrower compared to the same plot for the logistic regression model. This indicates that the neural network is making much more confident predictions up to its decision boundary (displayed in the right-hand plot) compared to the logistic regression model.


```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

fig = plt.figure(constrained_layout=True, figsize=(14, 14))
subfigs = fig.subfigures(ncols=2, nrows=3, hspace=0.03, height_ratios=[2, 2, 1], width_ratios=[18, 1])

light_cmap = clr.LinearSegmentedColormap.from_list(name='light', colors=['white', magenta], N=16)

# hidden layer 1, with 8 neurons
subfig = subfigs[0, 0]
subfig.suptitle(f'neurons in hidden layer 1')
axes = subfig.subplots(nrows=2, ncols=4, sharex=True, sharey=True)
    
for j, axis in enumerate(axes.flatten()):
    z = grid_outputs[1][:, j].detach().numpy()
    z = z.reshape(resolution, resolution)
    contour = axis.contourf(x1_grid, x2_grid, z, cmap=light_cmap, levels=light_cmap.N, vmin=0, vmax=3)
    
    sns.scatterplot(data=df, x='x_1', y='x_2', hue='y', ax=axis, legend=False, zorder=3)
    axis.set_title(f'neuron {j + 1}')
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')

# hidden layer 2, with 4 neurons
subfig = subfigs[1, 0]
subfig.suptitle(f'neurons in hidden layer 2')
axes = subfig.subplots(nrows=2, ncols=4, sharex=True, sharey=True)

for j, axis in enumerate(axes.flatten()):
    z = grid_outputs[2][:, j].detach().numpy()
    z = z.reshape(resolution, resolution)
    axis.contourf(x1_grid, x2_grid, z, cmap=light_cmap, levels=light_cmap.N, vmin=0, vamx=3)

    sns.scatterplot(data=df, x='x_1', y='x_2', hue='y', ax=axis, legend=False, zorder=3)
    axis.set_title(f'neuron {j + 1}')
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')

subfig = subfigs[2, 0]
subfig.suptitle('neurons in hidden layer 3')
axes = subfig.subplots(nrows=1, ncols=4, sharex=True, sharey=True)

for j, axis in enumerate(axes.flatten()):
    z = grid_outputs[3][:, j].detach().numpy()
    z = z.reshape(resolution, resolution)
    axis.contourf(x1_grid, x2_grid, z, cmap=light_cmap, levels=light_cmap.N, vmin=0, vamx=10)

    sns.scatterplot(data=df, x='x_1', y='x_2', hue='y', ax=axis, legend=False, zorder=3)
    axis.set_title(f'neuron {j + 1}')
    axis.set_xlabel('$x_1$')
    axis.set_ylabel('$x_2$')

# plot the colorbars
for subfig in subfigs[:, 1]:
    axis = subfig.subplots()
    cbar = subfig.colorbar(mpl.cm.ScalarMappable(cmap=light_cmap), cax=axis, orientation='vertical')
    cbar.set_ticklabels([round(3 / 5 * k, 1) for k in range(5)] + ['>5.0'])
```












(gmm-sec)=
## Gaussian mixture models

````{prf:definition}
:label: gmm-def

A _(two-component) Gaussian model_ is a probabilistic graphical model whose underlying graph is of the form

```{image} ../img/gmm.svg
:width: 50%
:align: center
```
&nbsp;

The model has the following parameters:

* A real parameter $\phi \in [0,1]$.

* Two real parameters $\mu_0,\mu_1\in \bbr$.

* Two positive real parameters $\sigma_0^2,\sigma_1^2 >0$.

We have $Z \sim \mathcal{B}er(\phi)$, and the link functions at $X$ are given by

$$
\mu = \mu_0(1-z) + \mu_1z, \quad \sigma^2 = \sigma_0^2(1-z) + \sigma_1^2z
$$

and

$$
X \mid Z; \ \mu_0,\sigma_0^2,\mu_1, \sigma^2_1 \sim \mathcal{N}(\mu,\sigma^2).
$$
````

```{prf:definition}
:label: gmm-pf-def

1. The _model joint probability function for a Gaussian mixture model_ is the joint probability density function

    $$
    p\big(x, z ; \ \mu_0,\mu_1, \sigma_0^2, \sigma_1^2, \phi \big).
    $$

    On its support consisting of all $x\in \bbr$ and $z \in \{0,1\}$, it is given by the formula

    $$
    p\big(x, z ; \ \mu_0,\mu_1, \sigma_0^2, \sigma_1^2, \phi \big) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left( - \frac{1}{2\sigma^2} (x-\mu)^2 \right) \phi^z (1-\phi)^{1-z}
    $$

    where $\mu = \mu_0(1-z) + \mu_1z$ and $\sigma^2 = \sigma_0^2(1-z) + \sigma_1^2z$.

2. Given a dataset

    $$
    (x^{(1)}, z^{(1)}), (x^{(2)},z^{(2)}),\ldots, (x^{(m)},z^{(m)}) \in \bbr \times \{0,1\},
    $$

    the _data joint probability function for a Gaussian mixture model_ is the joint probability density function

    $$
    p\big(x^{(1)},\ldots,x^{(m)}, z^{(1)},\ldots,z^{(m)} ; \ \mu_0,\mu_1, \sigma_0^2, \sigma_1^2, \phi \big) = \prod_{i=1}^m p\big(x^{(i)}, z^{(i)} ; \ \mu_0,\mu_1, \sigma_0^2, \sigma_1^2, \phi \big).
    $$


```


```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

# import gaussian mixture model from scikit-learn
from sklearn.mixture import GaussianMixture

# import data and convert to numpy array
url = 'https://raw.githubusercontent.com/jmyers7/stats-book-materials/main/data/ch10-book-data-02.csv'
df = pd.read_csv(url)
x = df['x'].to_numpy().reshape(-1, 1)

# instantiate the model, fit it to the data, predict components
gmm = GaussianMixture(n_components=2, random_state=42)
z_hat = gmm.fit_predict(X=x)

# cluster the data based on predicted components
class_0 = x[z_hat == 0]
class_1 = x[z_hat == 1]

# pull out the learned parameters
means = gmm.means_
std = np.sqrt(gmm.covariances_)

# define gaussian random variables based on learned parameters
comp_0 = sp.stats.norm(loc=means[0][0], scale=std[0][0][0])
comp_1 = sp.stats.norm(loc=means[1][0], scale=std[1][0][0])

# plot the gaussian density curves
grid = np.linspace(-3, 13, num=300)
plt.plot(grid, comp_0.pdf(grid))
plt.plot(grid, comp_1.pdf(grid))

# plot the data with component labels
plt.hist(x=class_0, alpha=0.7, ec='black', bins=5, density=True, color=blue, label='class 0')
plt.hist(x=class_1, alpha=0.7, ec='black', bins=12, density=True, color=magenta, label='class 1')

plt.legend()
plt.xlabel('$x$')
plt.ylabel('density')
plt.gcf().set_size_inches(w=5, h=4)
plt.tight_layout()
```




