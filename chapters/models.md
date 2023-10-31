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

## First examples

### Head-to-tail nodes

### Tail-to-tail nodes

### Head-to-head nodes



## DAGs and PGMs

At the most basic level, a _directed graph_ $G$ consists of two sets $V$ and $E$ of _vertices_ and _edges_. The vertices are visualized as nodes, and the edges are arrows that run between the nodes. For example, consider the following graph:

```{image} ../img/graph-01.svg
:width: 90%
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
:width: 90%
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


## Independence and $d$-separation

## More examples

### Linear regression

### Logistic regression

### Markov models

### Gaussian mixture models

## Gradient-based optimization

### Vanilla gradient descent

### Stochastic gradient descent (SGD)

## Maximum likelihood estimation (MLE)

### The basics

### MLE for linear regression

### MLE for logistic regression

### MLE for Markov models

## Expectation maximization (EM)

### Vanilla EM

### Monte Carlo EM

### EM for Gaussian mixture models