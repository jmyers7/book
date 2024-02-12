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

## A brief look at causal inference

Suppose that we are given two random variables $X$ and $Y$. As we explained in {numref}`cond-entropy-mutual-info-sec`, the two-way flow of "information" and "influence" between the random variables is conceptualized via the Markov kernels

$$
x\mapsto p(y|x) \quad \text{and} \quad y \mapsto p(x|y)
$$

given by the conditional distributions, which are both directly obtainable from the joint distribution of $X$ and $Y$. Mathematically, one may reverse the direction of the flow and obtain one Markov kernel from the other via Bayes' theorem; thus, as long as we have access to the joint distribution, there is no _a priori_ reason to prefer one direction over the other.

But there are very often situations in the real world where one of the directions of flow is more "natural," or at least easier to conceptualize, due to the two variables occurring in a cause and effect relationship. For example, in the case that $X$ and $Y$ are binary variables indicating the presence of a disease ($X$) and whether a test for the disease is positive ($Y$), we more naturally assume that the flow of influence goes from $X$ to $Y$, and not the other way around. Graphically, we might represent the situation as:

&nbsp;
```{image} ../img/stochastic-link.svg
:width: 25%
:align: center
```
&nbsp;

So, the arrow represents more than just the flow of information along the Markov kernel $x\mapsto p(x|y)$; by drawing $\rightarrow$ and not the reverse $\leftarrow$, we indicate that $X$ is the _cause_ and $Y$ is the _effect_.

Continuing with this example, imagine that we've collected data

$$
(x_1,y_1),(x_2,y_2),\ldots,(x_{10{,}000}, y_{10{,}000})
$$

on $10{,}000$ patients, where $x_i$ and $y_i$ indicate if the $i$-th patient has the disease and has tested positive for it. Then there is _absolutely nothing_ inherent or intrinsic to the dataset sufficient to determine the directionality of the cause and effect relationship between $X$ and $Y$. All the sample-based statistics that we may compute are derived from the empirical mass function, which is just a collection of relative frequencies. These statistics include the empirical correlation coefficient and the empirical mutual information, both of which will certainly be positive---but correlation is not the same as causation, as we are always warned, and so the data by itself tells us nothing about the causal relationship.

For another example, suppose that $U$ and $W$ are the proportions of people in a city that use an umbrella on a given day ($U$) and run their windshield wipers ($W$) on their drive to work. Then certainly $U$ and $W$ are positively correlated, but we would be quite skeptical if someone tried to convince us that the cause and effect relationship went like this:

&nbsp;
```{image} ../img/confounding-01.svg
:width: 25%
:align: center
```
&nbsp;

Indeed, the statistical correlation is not a result of a direct cause and effect relationship between the variables, but rather it is a result of the presence of a _confounding variable_ $R$, indicating whether it rained on the given day, and which serves as a common cause of both $U$ and $W$. Then the proper cause and effect relationships would be indicated by the graph:

&nbsp;
```{image} ../img/confounding-02.svg
:width: 25%
:align: center
```
&nbsp;

```{margin}

It is important to remember that the directionality of the causal relationship does _not_ indicate that there is no flow of information backward from effect to cause. The communication channel is still reversible via Bayes' theorem! These transfers of information against the directionality indicated by the causal structure are sometimes called _backdoor paths_.
```

In this situation, the correlation between $U$ and $W$ will vanish if we condition on $R$; this just means that the only way information flows from $U$ to $W$ is through $R$, and if we know what value $R$ takes, then this flow of information is cut off.  So, using the language introduced in {numref}`cond-entropy-mutual-info-sec`, the casual relationships indicated by the graph show that $U$ and $W$ are conditionally independent given $R$. Importantly, this independence would be detectable by observation, via the factorization of the empirical (conditional) joint mass function into the product of the empirical (conditional) marginal mass functions.

But recall from our discussion in {numref}`cond-entropy-mutual-info-sec` that conditional independence also occurs when the variables are configured in a chain

&nbsp;
```{image} ../img/mediator.svg
:width: 45%
:align: center
```
&nbsp;

with $R$ serving as a _mediating variable_ rather than a confounding one. Since the only insight obtainable from observed data is independence, the data itself does not express a preference between the first causal structure with $R$ a confounding variable and the second one with $R$ a mediator. The causal structure would need to be determined some other way, beyond observation.

We may summarize the discussion in any one of the following ways:

> **Causal structures and probability**.
> * Relationships of cause and effect represent strictly more structure than a joint probability distribution.
> * A causal structure _refines_ a joint probability distribution; it encodes _more_ knowledge.
> * The mapping from causal structures to joint probability distributions is many-to-one.

The very simple types of graphs that we have drawn to represent causal structures are called _causal graphs_ in the literature; they are graphical representations of _structural causal models_. We will use identical graphs to represent _probabilistic graphical models_ (*PGM*s) throughout the rest of this book. Essentially, a PGM represents a factorization of a joint probability function into products of conditional and marginal probability functions based on the structure of the underlying graph---but different graphs may represent the same joint distribution and the same factorization. So, strictly speaking, a PGM is more than just a probabilistic object. When we draw its underlying causal graph with arrows pointing one way and not the other, we are indicating what we believe are the "natural" directions of flow of information or influence, or at least just the directed links of communication that we choose to model directly. The links pointing in the opposite directions are modeled indirectly, via Bayes' theorem.

This has been a very (_very_) short introduction to the ideas of the formal theory of causality, intended only to motivate the causal graphs that we will see over the next few sections. To learn more, see the introductions in Chapter 9 of {cite}`HardtRecht2022` and Chapter 36 of {cite}`Murphy2023`. For a more comprehensive treatment, see {cite}`Pearl2009`.










## Probabilistic graphical models

By way of introduction, let's begin with two deterministic vectors $\bx\in \bbr^n$ and $\by \in \bbr^m$. As we discussed at the beginning of {numref}`cond-entropy-mutual-info-sec`, by saying that there is a _deterministic flow of information_ from $\bx$ to $\by$, we shall mean simply that there is a function

$$
g: \bbr^n \to \bbr^m, \quad \by = g(\bx),
$$

called a _link function_. It will be convenient to depict this situation graphically by representing the variables $\bx$ and $\by$ as nodes in a [graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) and the link function $g$ as an arrow between them:

```{image} ../img/det-link.svg
:width: 25%
:align: center
```
&nbsp;

Very often, the label $g$ on the link function will be omitted. In the case that both $\bx$ and $\by$ are $1$-dimensional, we visualized a deterministic flow like this:

```{image} ../img/det-kernel.svg
:width: 100%
:align: center
```
&nbsp;

It could be the case that flow of influence is parametrized. For example, $g$ might be a linear transformation that is represented by a matrix $\bA \in \bbr^{m\times n}$, with the entries in the matrix serving as parameters for the flow. We would represent this situation as

```{image} ../img/det-link-2.svg
:width: 25%
:align: center
```
&nbsp;

where the parameter matrix is represented by an un-circled node.

For a more complex example, consider the following graph:

```{image} ../img/det-link-3.svg
:width: 25%
:align: center
```
&nbsp;

This might represent a link function of the form

$$
\bz = \bA \bx + \bB \by, \quad \bx \in \bbr^{n}, \ \by\in \bbr^{k}, \ \bz \in \bbr^{m},
$$

which is parametrized by matrices $\bA \in \bbr^{m\times n}$ and $\bB \in \bbr^{m\times k}$.

The vectors in our discussion might be random, rather than deterministic, say $\bX$ and $\bY$. In this case, a _stochastic flow of information_ from $\bX$ to $\bY$ would be visualized just as before:


```{image} ../img/random-link.svg
:width: 25%
:align: center
```
&nbsp;

This flow is represented mathematically via a _link function_ $\btheta = g(\bx)$ where $\bx$ is an observed value of $\bX$ and $\btheta$ is a parameter that uniquely determines the probability distribution of $\bY$. So, in this case, an observed value $\bx$ does _not_ determine a particular observed value $\by$ of $Y$, but rather an entire probability distribution over the $\by$'s. This probability distribution is conditioned on $\bX$, so the link function is often specified by giving the functional form of the conditional probability function $p(\by | \bx)$. In other words, a stochastic flow of information is exactly a Markov kernel (communication channel), as we discussed in {numref}`cond-entropy-mutual-info-sec`:


```{image} ../img/stochastic-flow.svg
:width: 100%
:align: center
```
&nbsp;

In the picture, both $\bX$ and $\bY$ are $1$-dimensional. Notice that only observed values $\bx$ of $\bX$ are used to determine the distribution of $\bY$ through the link---the distribution of $\bX$ itself plays no role.

These stochastic links might be parametrized. For example, suppose $\bY$ is $1$-dimensional, equal to a random variable $Y$, while $\bX\in \mathbb{R}^{n}$ is an $n$-dimensional random vector. Then, a particular example of a stochastic link is given by the graph


```{image} ../img/lin-reg-0.svg
:width: 45%
:align: center
```
&nbsp;

The parameters consist of a real number $\beta_0 \in \bbr$, a vector $\bbeta \in \bbr^{n}$, and a positive number $\sigma^2 >0$. A complete description of the link function at $Y$ is given by

$$
Y \mid \bX; \ \beta_0, \bbeta,\sigma^2 \sim \mathcal{N}(\mu, \sigma^2), \quad \text{where} \quad \mu \def \beta_0 + \bx^\intercal \bbeta.
$$

In fact, this is exactly a _linear regression model_, which we will see again in {numref}`lin-reg-sec` below, as well as in {numref}`Chapters %s <learning>` and {numref}`%s <lin-reg>`.


We shall take a flow of information of the form

```{image} ../img/mixed-1.svg
:width: 25%
:align: center
```
&nbsp;

from a deterministic vector $\bx$ to a stochastic one $\bY$ to mean that there is a link function $\btheta = g(\bx)$ where $\btheta$ is a parameter that uniquely determines the distribution of $\bY$. Such a link function is often specified by giving the functional form of the parametrized probability function $p(\by; \bx)$.

A flow of information of the form

```{image} ../img/mixed-2.svg
:width: 25%
:align: center
```
&nbsp;

from a random vector $\bX$ to a deterministic vector $\by$ means that there is a link function of the form $\by = g(\bx)$, so that observed values of $\bX$ uniquely determine values of $\by$.

The probabilistic graphical models that we will study in this chapter are meant to model real-world datasets. These datasets will often be conceptualized as observations of random or deterministic vectors, and these vectors are then integrated into a graphical model. These vectors are called _observed_ or _visible_, while all others are called _latent_ or _hidden_. To visually represent observed vectors in the graph structure, their nodes will be shaded; the nodes associated with _hidden_ vectors are left unshaded. For example, if we draw

```{image} ../img/shaded.svg
:width: 25%
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

with two observed random vectors and one hidden. Then, by saying that $\bY$ and $\bZ$ are observed, we mean that we have in possession a pair $(\by, \bz) $ consisting of observed values of $\bY$ and $\bZ$.

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

This is called _plate notation_, where the rectangle is called the _plate_. The visible nodes in the plate are assumed to be grouped as pairs $(\bY_i,\bZ_i)$, and altogether they form an IID random sample

$$
(\bY_1,\bZ_2),\ldots,(\bY_m,\bZ_m).
$$

We now have everything that we need to define our version of _probabilistic graphical models_. After the definition, the remaining sections in this chapter are devoted to the study of particular examples of such models.

```{prf:definition}
:label: pgm-def

A _probabilistic graphical model_ (_PGM_) consists of the following:

1. A set of vectors, some random and some deterministic, and some marked as observed and all others as hidden.

2. A graphical structure depicting the vectors as nodes and flows of influence (or information) as arrows between the nodes. If any of these flows are parametrized, then the graphical structure also has (un-circled) nodes for the parameters.

3. Mathematical descriptions of the flows as (possibly parametrized) link functions.
```













(lin-reg-sec)=
## Linear regression models

The type of PGM defined in this section is one of the simplest, but also one of the most important. Its goal is to model an observed dataset

$$
(\bx_1, y_1), (\bx_2,y_2),\ldots, (\bx_m,y_m) \in \bbr^{n} \times \bbr
$$

where we believe that

```{math}
:label: approx-linear-eqn

y_i \approx \beta_0 + \bx_i^\intercal \bbeta
```

for some parameters $\beta_0 \in \bbr$ and $\bbeta \in \bbr^{n}$. For example, let's consider the Ames housing dataset from the <a href="https://github.com/jmyers7/stats-book-materials/tree/main/programming-assignments">third programming assignment</a> and {numref}`Chapter %s <random-vectors>`; it consists of $m=2{,}930$ bivariate observations

$$
(x_1,y_1),(x_2,y_2),\ldots,(x_m,y_m) \in \bbr^2
$$

where $x_i$ and $y_i$ are the size (in square feet) and selling price (in thousands of US dollars) of the $i$-th house in the dataset. A scatter plot of the dataset looks like

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

where $\bX\in \bbr^{n}$. The model has the following parameters:

* A real parameter $\beta_0\in \mathbb{R}$.

* A parameter vector $\bbeta \in \mathbb{R}^{n}$.

* A positive real parameter $\sigma^2>0$.

The link function at $Y$ is given by

$$
Y \mid \bX; \ \beta_0,\bbeta,\sigma^2 \sim \mathcal{N}\big(\mu,\sigma^2\big), \quad \text{where} \quad \mu = \beta_0 + \bx^\intercal \bbeta.
$$
````

Before we introduce important terminology associated with linear regression models and look at an example, we need to discuss two probability functions that will play a crucial role in the [next chapter](learning). The first is just the conditional probability function of $Y$ given $\bX$, while the second is obtained from the plated version of a linear regression model: 

```{image} ../img/lin-reg-00-plated.svg
:width: 50%
:align: center
```
&nbsp;

Observations of the visible nodes correspond to an observed dataset.

```{prf:definition}
:label: linear-reg-pf-def

1. The _model probability function_ for a linear regression model is the conditional probability function

    $$
    p\big(y \mid \bx ; \ \beta_0, \bbeta, \sigma^2\big).
    $$

    On its support consisting of all $y\in \bbr$ and $\bx \in \bbr^{n}$, it is given by the formula

    $$
    p\big(y \mid \bx ; \ \beta_0, \bbeta, \sigma^2\big) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left(- \frac{1}{2\sigma^2} ( y - \mu)^2 \right),
    $$

    where $\mu = \beta_0 + \bx^\intercal \bbeta$.

2. Given an observed dataset

    $$
    (\bx_1,y_1),(\bx_2,y_2),\ldots,(\bx_m,y_m) \in \bbr^{n} \times \bbr,
    $$

    the _data probability function_ for a linear regression model is the conditional probability density function

    $$
    p\big(y_1,\ldots,y_m \mid \bx_1,\ldots,\bx_m; \ \beta_0, \bbeta,\sigma^2 \big) = \prod_{i=1}^m p\big(y_i \mid \bx_i ; \ \beta_0, \bbeta, \sigma^2\big).
    $$ (data-pf-eqn)
```

Note that the data probability function appears to be _defined_ as a product of model probability functions. However, using independence of the random sample

$$
(\bX_1,Y_1),(\bX_2,Y_2),\ldots,(\bX_m, Y_m),
$$

one may actually _prove_ that the left-hand side of {eq}`data-pf-eqn` is equal to the product on the right-hand side; see the homework for this section.

Returning to our discussion of the linear regression model, the components of the vector $\bX$ are referred to as _predictors_, _regressors_, _explanatory variables_, or _independent variables_, while the random variable $Y$ is called the _response variable_ or the _dependent variable_. In the case that $n=1$, the model is called a _simple linear regression model_; otherwise, it is called a _multiple linear regression model_.

Note that

$$
E\big(Y \mid \bX = \bx \big) = \mu = \beta_0 + \bx^\intercal \bbeta,
$$

and so a linear regression model assumes (among other things) that the conditional mean of the response variable is linearly related to the regressors through the link function

$$
\mu = \beta_0 + \bx^\intercal \bbeta.
$$ (lin-reg-line-eqn)

The parameter $\beta_0$ is often called the _intercept_ or _bias term_, while the other $\beta_j$'s (for $j>0$) are called _weights_ or _slope coefficients_ since they are exactly the (infinitesimal) slopes:

$$
\frac{\partial \mu}{\partial x_j} = \beta_j.
$$

The random variable

$$
\dev \stackrel{\text{def}}{=} Y - \beta_0 - \bX^\intercal\bbeta
$$

in a linear regression model is called the _error term_; note then that

$$
Y = \beta_0 + \bX^\intercal \bbeta + \dev \quad \text{and} \quad \dev \sim \mathcal{N}(0, \sigma^2).
$$ (random-lin-rel-eqn)

This is the manifestation in terms of random vectors and variables of the approximate linear relationship {eq}`approx-linear-eqn` described at the beginning of this section.

Suppose we are given an observed dataset

$$
(\bx_1,y_1),(\bx_2,y_2),\ldots,(\bx_m,y_m) \in \bbr^{n} \times \bbr.
$$

If for each $i=1,\ldots,m$, we define the _predicted values_

$$
\hat{y}_i = \beta_0 + \bx_i^\intercal\bbeta
$$

and the _residuals_

$$
\dev_i = y_i - \hat{y}_i,
$$

then from {eq}`random-lin-rel-eqn` we get

$$
y_i = \beta_0 + \bx^\intercal_i \bbeta + \dev_i.
$$

This shows that the residuals $\dev_i$ are observations of the error term $\dev \sim \mathcal{N}(0,\sigma^2)$. Thus, in a linear regression model, all residuals from a dataset are assumed to be modeled by a normal distribution with mean $0$ and a _fixed_ variance; the fixed-variance assumption is sometimes called _homoscedasticity_.

In {numref}`Chapter %s <learning>`, we will learn how to train a linear regression model on a dataset to obtain optimal values of the parameters $\beta_0$ and $\bbeta$. Using these training methods, we obtained values for the parameters $\beta_0$ and $\bbeta = \beta_1$ for the Ames housing dataset mentioned at the beginning of this section. The positively-sloped line in the scatter plot at the beginning of this section was the line traced out by the link function $\mu = \beta_0 + \beta_1 x $. The predicted values $\hat{y}_i$ lie along this line, and the magnitude of the residual $\dev_i$ may be visualized as the vertical distance from the true data point $y_i$ to this line. We may plot the residuals $\dev_i$ against the predictor variables $x_i$ to get:

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

As with the parameters $\beta_0$ and $\bbeta$, it is also possible to learn an optimal value of the variance $\sigma^2$. As another method of model checking, given all the learned parameters $\beta_0$, $\beta_1$, and $\sigma^2$ for the Ames dataset, we may generate a new dataset by sampling from the normal distributions  $\mathcal{N}\big(\hat{y}_i, \sigma^2\big)$ for each $i=1,2,\ldots,m$. A scatter plot of one simulated dataset is on the left in the following figure, while a KDE of the simulated dataset is compared against the "true" KDE on the right:

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

df['indicator'] = 'true data PDF'
df_gen['indicator'] = 'simulated data PDF'
df_combined = pd.concat(objs=[df, df_gen], axis=0)

# setup the figure
_, axes = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)

# plot the dataset
sns.scatterplot(data=df_gen, x='area', y='price', alpha=0.15, ax=axes[0])

# plot the original regression line
axes[0].plot(grid, beta_0 + beta * grid, color=magenta)

# plot the KDEs
g = sns.kdeplot(data=df_combined, x='area', y='price', hue='indicator', levels=6, ax=axes[1])
g.get_legend().set_title(None)

plt.tight_layout()
```

For smaller values of area, the distribution of the true prices is narrower compared to the simulated prices, while for larger values of area, the distribution of the true prices is wider.





















(log-reg-sec)=
## Logistic regression models

The types of models studied in this section are closely related to the linear regression models in the previous, but here the goal is to model a dataset of the form

$$
(\bx_1,y_1),(\bx_2,y_2),\ldots,(\bx_m,y_m) \in \bbr^{n} \times \{0,1\}.
$$

Such datasets arise naturally in _binary classification problems_, where we aim to determine which of two classes a given object lies in based on predictor features. The true class of the $i$-th object is indicated by the value of $y_i$, while the vector $\bx_i$ consists of the predictor features.

As a running example through this section, consider the data given in following scatter plot:

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
\bx_i^\intercal = \begin{bmatrix} x_{i1} & x_{i2} \end{bmatrix},
$$

while the color indicates the class $y_i \in \{0,1\}$. Our goal in this section is to capture the evident pattern in the data using a _logistic regression model_.

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

where $\bX\in \bbr^{n}$. The model has the following parameters:

* A real parameter $\beta_0\in \mathbb{R}$.

* A parameter vector $\bbeta \in \mathbb{R}^{n}$.

The link function at $Y$ is given by

$$
Y \mid \bX; \ \beta_0,\bbeta \sim \mathcal{B}er(\phi), \quad \text{where} \quad \phi = \sigma(\beta_0 + \bx^\intercal\bbeta),
$$

and where $\sigma$ is the sigmoid function.
````

Notice that the link function $\phi = \sigma(\beta_0 + \bx^\intercal\bbeta)$ in a logistic regression model is precisely the affine link function $\mu = \beta_0 + \bx^\intercal\bbeta$ of a linear regression model composed with the sigmoid function.

The two probability functions that we will use to train logistic regression models in the [next chapter](learning) are given as follows. The second is obtained from the plated version of a logistic regression model:

```{image} ../img/log-reg-00-plated.svg
:width: 50%
:align: center
```
&nbsp;

The two probability functions are:

```{prf:definition}
:label: log-reg-pf-def

1. The _model probability function_ for a logistic regression model is the conditional probability function

    $$
    p\big(y \mid \bx ; \ \beta_0, \bbeta\big).
    $$

    On its support consisting of all $y\in \{0,1\}$ and $\bx \in \bbr^{n}$, it is given by the formula

    $$
    p\big(y \mid \bx ; \ \beta_0, \bbeta\big) = \phi^y (1-\phi)^{1-y}
    $$

    where $\phi = \sigma(\beta_0 + \bx^\intercal \bbeta)$.

2. Given a dataset

    $$
    (\bx_1,y_1),(\bx_2,y_2),\ldots,(\bx_m,y_m) \in \bbr^{n} \times \{0,1\},
    $$

    the _data probability function for a logistic regression model_ is the conditional probability density function

    $$
    p\big(y_1,\ldots,y_m \mid \bx_1,\ldots,\bx_m; \ \beta_0, \bbeta\big) = \prod_{i=1}^m p\big(y_i \mid \bx_i ; \ \beta_0, \bbeta\big).
    $$ (log-reg-data-pf-eqn)
```

As in {prf:ref}`linear-reg-pf-def`, one may _prove_ that the data probability function of a logistic regression model is given by the product of model probability functions in {eq}`log-reg-data-pf-eqn`.

Let's return to our toy dataset introduced at the beginning of the section. To aid with training, it is often helpful to _standardize_ the predictor features

$$
\bx_1,\ldots,\bx_m \in \bbr^{n}.
$$

This means that we compute the (empirical) mean $\bar{x}_j$ and standard deviation $s_j$ of each sequence

$$
x_{1j},\ldots,x_{mj} \in \bbr
$$

of components, and then replace each $x_{ij}$ with

$$
\frac{x_{ij} - \bar{x}_j}{s_j}.
$$

It is convenient to visualize this process in terms of the so-called _design matrix_

$$
\mathbfcal{X} = \begin{bmatrix} \leftarrow & \bx_1^\intercal & \rightarrow \\ \vdots & \vdots & \vdots \\ \leftarrow & \bx_m^\intercal & \rightarrow \end{bmatrix} = \begin{bmatrix} x_{11} & \cdots & x_{1n} \\
\vdots & \ddots & \vdots \\
x_{m1} & \cdots & x_{mn}
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

Along with linear regression models, in the [next chapter](learning) we will see how to learn optimal values of the parameters $\beta_0$ and $\bbeta$ from data. With these parameters in hand, one way to check how well a logistic regression model captures the data is to draw a contour plot of the function $\phi = \sigma( \beta_0 + \bx^\intercal \bbeta)$. This contour plot appears on the left in the following:

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

To interpret this plot, remember that $\phi = \sigma( \beta_0 + \bx^\intercal \bbeta)$ is the probability parameter for the class indicator variable $Y \sim \Ber(\phi)$, so we should interpret $\phi$ as the probability that the point $\bx$ is in class $1$ (corresponding to $y=1$). In the right-hand plot, we have "thresholded" the probability $\phi$ at $0.5$, creating a _predictor function_

$$
f:\bbr^{2} \to \{0,1\}, \quad f(\bx) = \begin{cases}
0 & : \sigma(\beta_0 + \bx^\intercal\bbeta) < 0.5, \\
1 & : \sigma(\beta_0 + \bx^\intercal\bbeta) \geq 0.5. \\
\end{cases}
$$

The _decision boundary_ is exactly the curve in $\bbr^2$ consisting of those $\bx$ for which the predictor $f$ is "flipping a coin," i.e., it consists of those points $\bx$ such that

$$
\sigma(\beta_0 + \bx^\intercal \bbeta) = 0.5,
$$

which is equivalent to

$$
\beta_0 + \bx^\intercal \bbeta = 0.
$$

Notice that this defines a _linear_ decision boundary that separates $\bbr^2$ into two unbounded half planes based on whether

$$
\beta_0 + \bx^\intercal \bbeta > 0 \quad \text{or} \quad \beta_0 + \bx^\intercal \bbeta < 0.
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
\rho(\bx)^\intercal \def \begin{bmatrix} \rho(x_1) & \cdots & \rho(x_n) \end{bmatrix}.
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

where $\bX\in \bbr^{n}$ and $\bz \in \bbr^{k}$. The model has the following parameters:

* A parameter matrix $\mathbf{W} \in \mathbb{R}^{n\times k}$.

* A parameter vector $\bb \in \mathbb{R}^{k}$.

* A parameter vector $\bw \in \mathbb{R}^{k}$.

* A real parameter $b \in \mathbb{R}$.

The link function at $\mathbf{z}$ is given by

$$
\mathbf{z} = \rho(\mathbf{x}^\intercal\bW + \bb),
$$

while the link function at $Y$ is given by

$$
Y ;\  \mathbf{z}, \bw, b \sim \mathcal{B}er\big(\phi\big), \quad \text{where} \quad \phi = \sigma(\bz^\intercal\bw + b).
$$

Here, $\rho$ is the ReLU function and $\sigma$ is the sigmoid function.
````

The name "neural network" comes from a loose analogy with networks of biological neurons in the human brain. For this reason, sometimes neural networks just defined are called _artificial neural networks_ (*ANN*s).

Following the pattern begun with linear and logistic regression models, we first want to give the probability functions that we will use in the [next chapter](learning) to train neural network models. The second one is obatained from the plated version of a neural network:

```{image} ../img/nn-00-plated.svg
:width: 50%
:align: center
```
&nbsp;

The two probability functions are:

```{prf:definition}
:label: neural-net-pf-def

1. The _model probability function_ for a neural network model is the conditional probability function

    $$
    p\big(y \mid \bx ; \ \bW, \bb, \bw, b \big).
    $$

    On its support consisting of all $y\in \{0,1\}$ and $\bx \in \bbr^{n}$, it is given by the formula

    $$
    p\big(y \mid \bx ; \ \bW, \bb, \bw, b \big) = \phi^y (1-\phi)^{1-y}
    $$

    where $\phi = \sigma(\bz^\intercal \bw + b)$ and $\bz = \sigma(\bx^\intercal \bW + \bb)$.


2. Given a dataset

    $$
    (\bx_1,y_1),(\bx_2,y_2),\ldots,(\bx_m,y_m) \in \bbr^{n} \times \{0,1\},
    $$

    the _data probability function_ for a neural network model is the conditional probability function

    $$
    p\big(y_1,\ldots,y_m \mid \bx_1,\ldots,\bx_m; \ \bW, \bb, \bw, b\big) = \prod_{i=1}^m p\big(y_i \mid \bx_i ; \ \bW, \bb, \bw, b\big).
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
z_j = \rho(\bx^intercal \bw_j + b_j),
$$

where

$$
\bW = \begin{bmatrix} \uparrow & \uparrow & \uparrow & \uparrow \\ \bw_1 & \bw_2 & \bw_3 & \bw_4 \\
\downarrow & \downarrow & \downarrow & \downarrow \end{bmatrix} \quad \text{and} \quad \bb = \begin{bmatrix} b_{1} & b_2 & b_3 & b_{4} \end{bmatrix}.
$$

The link function at $Y$ is given by the same formula as before using the sigmoid function. In the literature, the ReLU function $\rho$ and the sigmoid function $\sigma$ are often called _activation functions_ of the network. The parameters $\bW$ and $\bw$ are called _weights_, while the parameters $\bb$ and $b$ are called _biases_.

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

There are now _two_ weight matrices $\bW_1$ and $\bW_2$, along with _two_ bias vectors $\bb_1$ and $\bb_2$. The link functions at $\bz_1$ and $\bz_2$ are given by

$$
\bz_\ell = \rho\big(\bz_{\ell-1}^\intercal \bW_\ell + \bb_\ell \big) \quad \text{for $\ell=1,2$,}
$$

where we set $\bz_\ell = \bx$. The link function at $Y$ is the same as it was before:

$$
Y; \ \bz_2, \bw, b \sim \Ber(\phi), \quad \text{where} \quad \phi = \sigma \big( \bz_2^\intercal \bw + b\big).
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
\phi = \sigma\big(\bz_3^\intercal\bw + b\big)
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

To be written: Blah, blah blah, talk about activations:


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