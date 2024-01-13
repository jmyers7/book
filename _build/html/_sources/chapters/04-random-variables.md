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

(random-variables)=
# Random variables

## Random variables

I do not think it is too much of a stretch to say that _random variables_ are the objects that most concern practitioners who use probability and statistics in their work. In this section, I want to give a concrete introduction to these gadgets, saving their precise, mathematical definition for a little later.

As a first example, let's think about the population that you and I are all members of: The current population $S$ of the planet Earth. As members of this population, we all have different characteristics, qualities, or features that distinguish us from one another. For example, we all have:

* An age.

* A country of legal residence.

* An annual income (converted to some standard monetary system, like USD or euros).

* A primary profession.

* *etc*.

If you're interested in business, advertising, and sales applications, then you might have a _different_ population $S$ in mind. For example, your population might be all past customers who have purchased your company's product, and you might query the various features of this population in order to construct better future advertising campaigns. Can you think of some features of past customers that might be of interest?

Or, your population $S$ might not even be a population of _people_; for example, you might work for a manufacturing company, and $S$ could be the collection of all devices manufactured in the past year. Then, some features of this population might be:

* Date of manufacture of the device.

* A binary 'yes' or 'no' depending on whether the device turned out to be defective within a certain time period.

* Country of sale.

* *etc*.

Hopefully you can see that this idea of _features_ is very broad, and can be applied in many situations and cases.

All the features mentioned above are also called _variables_, because they _vary_ across members of the population. For example, we have the 'age' variable defined on the population of planet Earth, as well as the 'primary profession' variable, and so on. Some of these variables take on numerical values, while others do not; for example, 'age' is a numerical variable, while 'primary profession' is not. The types of variables that will most interest us (for now) are the numerical ones.

Now, so far nothing about probability has been mentioned. But when the population under consideration is a probability space, it is traditional to refer to variables (or features) of the population as _random variables_, the word 'random' simply reflecting that there is now a probability measure in the mix. The word 'random' is perhaps not the best, because for many of us 'random' already has such a strong meaning from everyday language. For this reason, sometimes _random variables_ are called _stochastic variables_, so that we're not misled in to thinking that there is something "random" about them (there isn't).

Though the term _random variable_ is common in statistics, the term _feature_ is much more common in the machine learning community. Often, the plain term _variable_ is also used. I will bounce back and forth between these pieces of terminology without explicit mention. But no matter what you call them, they will be _very_ important throughout the rest of this course. In fact, as you will see, the probability spaces that we've worked so hard to understand---and which serve as the domains of our random variables---will essentially recede into the background, and the random variables themselves will become the things that we talk about on a daily basis.

Alright, good. We feel like we have an intuitive, concrete understanding of what _random variables_ (or _features_, or _variables_) are. They are simply numerical characteristics or qualities of samples points in a probability space. But how might we encode this notion in mathematics? How might we take our intuitive definition of _random variable_ and turn it into a precise mathematical definition?

```{margin}
Apparently, there are only 32 of us---I counted the stick people!---and we all live on a rectangular planet Earth.
```
Let's return to the sample space $S$ representing the population of the planet Earth:

```{image} ../img/population.svg
:width: 45%
:align: center
```
&nbsp;

Let's talk about the 'annual income' variable (or feature). We might conceptualize it as a "machine" into which we feed individuals from the population, and then the machine spits out their annual income:

```{image} ../img/income.svg
:width: 90%
:align: center
```
&nbsp;

I've given this machine the name $I$, to stand for **I**ncome. We might also be interested in the 'age' variable; this leads us to consider another "machine," this time named $A$:

```{image} ../img/age.svg
:width: 90%
:align: center
```
&nbsp;

The obsession that mathematicians have with "machines" is because they are concrete and easily understandable metaphors for the more formal notion of a _mathematical function_. Above, the "machines" $I$ and $A$ are simply objects that have inputs and outputs---indeed, both have inputs consisting of people, and the outputs are numbers. But this is _exactly_ what a formal _mathematical function_ is---it's an object that has inputs and outputs! So, our "machines" above are really none other than _functions_, in the precise, mathematical sense.

To drive this point home, we might even use functional notation:

```{image} ../img/funcnotation.svg
:width: 90%
:align: center
```
&nbsp;

This way of writing things is modelled after the familiar functional notation used in other parts of mathematics:

```{image} ../img/funcnotation2.svg
:width: 35%
:align: center
```
&nbsp;

Our functions $I$ and $A$ may both be written as

$$I:S \to \mathbb{R} \quad \text{and} \quad A: S \to \mathbb{R}.$$

This means that the domain of both $I$ and $A$ is the population $S$, while the outputs lie in the set $\mathbb{R}$ of real numbers.

So, does this mean that a _random variable_ is secretly just a function? Yup! Here's the definition:

```{prf:definition}
:label: random-var-def

Let $S$ be a probability space. A _random variable_ on $S$ is a function $X:S \to \mathbb{R}$.
```

```{margin} Looking beyond...
Technically, not *every* function on a sample space qualifies as a random variable. As you would learn if you took a (*really*) advanced course in theoretical probability, random variables must have the additional property of being <a href="https://en.wikipedia.org/wiki/Measurable_function">*measurable*</a>. But we won't worry about this. At all. So you can forget I ever said anything.
```

Therefore, a random variable on a sample space is nothing but a function that takes sample points as inputs, and spits out real numbers. That's it!

Before we do some practice problems with random variables, let me add a few quick remarks:

* It is traditional in probability theory to use capital letters toward the end of the alphabet to name generic random variables, like $X$, $Y$, and $Z$, instead of the more familiar lower case letters like $f$, $g$, and $h$. $X$.

* If a capital letter is used to represent a random variable itself, then the lowercase version of the same letter is often used to represent the generic _output_ of a random variable. So, the functional notation template $f(x)=y$ that you are familiar with from calculus becomes

    $$X(s) = x \quad \text{or} \quad Y(s) = y \quad \text{or} \quad Z(s) = z$$

    in probability theory.

* With all this being said, this 'capital letter $=$ random variable' convention is *not* always strictly followed, especially if you read other textbooks, research papers, *etc*. Frankly, struggling through an author's choice for statistical and probabilistic notation can sometimes be quite a nightmare.

Now:

```{admonition} Problem Prompt
Let's get some practice with random variables! Do problems 1-4 on the worksheet.
```






(prob-measure-rv)=
## Probability measures of random variables

We know that a random variable on a probability space $S$ is simply a real-valued function

$$X:S \to \mathbb{R}.$$

Since $S$ is a probability space, it comes equipped with a probability measure $P$, an object which measures the probabilities of events $A \subset S$. However, the random variable $X$ "transports" the probability measure $P$ to a second probability measure that I will denote $P_X$ (since it depends on $Y$) which measures probabilities of events $A\subset \mathbb{R}$. Here's a figure to help you remember *where* each of these probability measures live:

```{image} ../img/pushforward.svg
:width: 70%
:align: center
```
&nbsp;

It is best to introduce these concepts by way of example. So, let's return to our example of the 'annual income' variable $I$ from the previous section. But because I want you to get used to seeing $X$ for the name of a random variable, let's rename $I$ to $X$:

$$I:S \to \mathbb{R} \quad \xrightarrow{\text{rename}} \quad X:S \to \mathbb{R}.$$

And remember, the sample space $S$ is the current population of the planet Earth.

Now, for (extreme!) simplicity, let's suppose that there are only four possible annual incomes in the entire population:

$$\text{range of $X$} = \{35, 40, 45, 50\}.$$

The units are thousands of US dollars, so, for example,

$$X(\text{John}) = 35$$

means that my annual income (Hi, I'm John) is 35,000 USD. We may group together all individuals in the sample space $S$ based on their annual incomes:

```{margin}
By the way, if you remember the definition of a *partition* from an earlier chapter, what we have done is partition the sample space $S$ based on the values of the random variable $Y$.
```

```{image} ../img/discretedist0.svg
:width: 40%
:align: center
```
&nbsp;

I tend to view functions like $X$ as "active" transformations, so I would picture the action of $X$ as follows:

```{image} ../img/discretedist1.svg
:width: 90%
:align: center
```
&nbsp;

Notice that the variable $X$, as a function, is carrying certain portions of the sample space to certain values along $\mathbb{R}$, which I am picturing as a number line.

Let's now bring in the probability measures. Let's suppose for the sake of argument that the size of our sample space $S$ (i.e., the population) is only 32 individuals, and that the probability measure $P$ on $S$ is uniform. Thus, the probability of choosing any *one* person from $S$ is:

$$P(\{\text{a single person}\}) = \frac{1}{32}.$$

If you count all the little stick people, you will find:

* There are 7 people with annual income 35.

* There are 8 people with annual income 40.

* There are 12 people with annual income 45.

* There are 5 people with annual income 50.

Therefore, we compute:

\begin{align*}
P(\{\text{people with annual income 35}\}) &= 7/32 \approx 0.22, \\
P(\{\text{people with annual income 40}\}) &= 8/32 = 0.25, \\
P(\{\text{people with annual income 45}\}) &= 12/32 \approx 0.38, \\
P(\{\text{people with annual income 50}\}) &= 5/32 \approx 0.16.
\end{align*}

Now, remember that in addition to the probability measure $P$ on $S$, there is supposed to be a *second* probability measure $P_X$ on $\mathbb{R}$. It turns out, however, that these previous four probabilities _are_ the probabilities that come from $P_X$, by definition! Indeed, we _define_ the probability measure $P_X$ on $\mathbb{R}$ by setting

$$P_X(\{x\}) = P(\{\text{people with annual income $x$}\})$$ (inv-01-eqn)

for each $x\in \mathbb{R}$, so that we may rewrite the above equations using $P_X$ as:

\begin{align*}
P_X(\{35\}) &= 7/32 \approx 0.22, \\
P_X(\{40\}) &= 8/32 = 0.25, \\
P_X(\{45\}) &= 12/32 \approx 0.38, \\
P_X(\{50\}) &= 5/32 \approx 0.16.
\end{align*}

The equation {eq}`inv-01-eqn` is the fundamental bridge that relates the two probability measures $P$ (on $S$) and $P_X$ (on $\mathbb{R})$, so make sure that you understand it! (It will reappear below in the abstract definitions.)

The probability measure $P$ lives on the original probability space $S$, so we can't really *visualize* it since $S$ doesn't have a nice linear structure like $\mathbb{R}$. However, the probability measure $P_X$ lives on $\mathbb{R}$, so we _can_ visualize _it_ using a probability histogram:

```{image} ../img/discretedist.svg
:width: 90%
:align: center
```
&nbsp;

The heights of the bars above each numerical value in $\mathbb{R}$ represent the probabilities measured by $P_X$.

Ok. Having worked our way through a concrete example, we are now ready for the abstract definitions.

```{prf:definition}
:label: prob-measure-X-defn

Let $X:S\to \mathbb{R}$ be a random variable on a probability space $S$ with probability measure $P$. We define the *probability measure of $X$*, denoted $P_X$, via the formula

$$P_X(A) = P \big( \{ s\in S : X(s) \in A\} \big),$$ (inv2-eqn)

for all events $A\subset \mathbb{R}$.
```

For a given event $A\subset \mathbb{R}$, notice that the set

$$
\{s \in S : X(s) \in A)\} \subset S
$$

inside the probability measure on the right-hand side of {eq}`inv2-eqn` consists exactly of those sample points $s\in S$ that land in $A$ under the action of the random variable $X$; I would visualize this as:

```{image} ../img/pushforward-2.svg
:width: 70%
:align: center
```
&nbsp;


```{admonition} Alternate notation

In the *mathematical* theory of probability, the notation $P_X$ is very common for the probability measure induced by $X$. However, in elementary statistics it is much more common to write

$$P(X\in A) \quad \text{in place of} \quad P_X(A),$$

and

$$P(X=x) \quad \text{in place of} \quad P_X\big(\{x\} \big),$$

and

$$P(a\leq X \leq b) \quad \text{in place of} \quad P_X\big([a,b] \big).$$

In deference to tradition, I will use these alternate notations almost always.

However, I want to point out the following:

* Do notice that these alternate notations are actually *misuses* of notation, since the expressions "$X\in A$", "$X=x$", and "$a\leq X \leq b$" are complete nonsense. Indeed, remember that $X$ is a function, while $x$ is a number, so there is **no way** that $X$ could be equal to $x$, since they are *different* types of objects!

* In the alternate notation, notice that the single letter $P$ is used to denote *both* the original probability measure on $S$ *and* the second probability measure of $X$ on $\mathbb{R}$. This might make you think that there is only *one* probability measure in play, but remember that there are actually *two*, and they live on different probability spaces!
```

Now:

```{admonition} Problem Prompt
Let's practice! Have a go at problems 5-7 on the worksheet.
```







## Discrete and continuous random variables

Two types of random variables will be the ones that are most frequently encountered in this class. Their definitions follow below. Before reading them, however, it might be worth reviewing our discussions of discrete and continuous probability measures in {numref}`Sections %s <discrete-prob>` and {numref}`%s <cont-prob>`.

```{prf:definition}
:label: discrete-continuous-rv-def

Let $X:S\to \mathbb{R}$ be a random variable.

1. We shall say $X$ is *discrete* if there exists a function $p:\mathbb{R} \to \mathbb{R}$ such that
    
    \begin{equation*}
    P(X\in A) = \sum_{x\in A} p(x)
    \end{equation*}
    
    for all events $A\subset \mathbb{R}$. In this case, $p(x)$ is called the *probability mass function of $X$*.

2. We shall say $X$ is *continuous* if there exists a function $f:\mathbb{R} \to \mathbb{R}$ such that

    \begin{equation*}
    P(X\in A) = \int_A f(x) \ \text{d} x
    \end{equation*}

    for all events $A\subset \mathbb{R}$. In this case, $f(x)$ is called the *probability density function of $X$*.
```

```{margin} Looking beyond...

Here's where things get really deep!

Though it seems that discrete and continuous random variables are really rather different because the former type involve discrete summations while the latter involve continuous integrals, there is a very general mathematical theory of integration in which they are *united*. This is the powerful theory of <a href="https://en.wikipedia.org/wiki/Lebesgue_integration">Lebesgue integration</a>.

Indeed, in this much more general theory, discrete summations actually *are* integrals of a particular type: They are  integrals "against a <a href="https://en.wikipedia.org/wiki/Counting_measure">counting measure</a>." Also, in this theory the *probability mass functions* of discrete random variables and the *probability density functions* of continuous ones are united, for they are both examples of <a href="https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem#">Radon-Nikodym derivatives</a> of measures that are <a href="https://en.wikipedia.org/wiki/Absolute_continuity#Absolute_continuity_of_measures">absolutely continuous</a> with respect to an ambient measure, either a <a href="https://en.wikipedia.org/wiki/Counting_measure">counting measure</a> in the discrete case, or the <a href="https://en.wikipedia.org/wiki/Lebesgue_measure">Lebesgue measure</a> in the continuous one.
```

Notice that a random variable $X$ is discrete if and only if its probability measure $P_X$ is discrete in the sense defined in {numref}`Section %s <discrete-prob>`, while it is continuous if and only if $P_X$ is continuous in the sense defined in {numref}`Section %s <cont-prob>`.


```{admonition} Problem Prompt
Let's get some practice recognizing discrete and continuous random variables, and computing some of their probability measures. Do problems 8 and 9 on the worksheet.
```

As I mentioned, discrete and continuous random variables will be the only types of random variables that we work with in this class---at least in problems where we need to *compute* things. As we continue, you will notice the contrast in the definition above, where discrete random variables involve summations $\sum$ and continuous random variables involve integrals $\int$, will be replicated over and over again. This is another good intuition to have in mind for these two types of random variables: The discrete random variables are those in which you "add things" to compute various quantities, whereas continuous random variables are those in which you must "integrate things."














(dist-func-rv)=
## Distribution and quantile functions

If $X$ is any type of random variable (discrete, continuous, or neither), then its probability measure $P_X$ lives on $\mathbb{R}$. As such, it has both distribution and quantile functions. We studied these latter types of functions in {numref}`Section %s <dist-quant>`. But in case you forgot their definitions, we review them in this section in the context of random variables.

```{prf:definition}
:label: cdf-rv-def

Let $X$ be a random variable. The *distribution function of $X$* is the function $F:\mathbb{R} \to \mathbb{R}$ defined by

\begin{equation*}
F(x) = P(X \leq x).
\end{equation*}

In particular:

1. If $X$ is discrete with probability mass function $p(x)$, then

    \begin{equation*}
    F(x) = \sum_{y \leq x} p(y),
    \end{equation*}

    where the sum ranges over all $y\in \mathbb{R}$ with $y\leq x$.

2. If $X$ is continuous with density function $f(x)$, then

    \begin{equation*}
    F(x) = \int_{-\infty}^x f(y) \ \text{d} y.
    \end{equation*}
```

And here's the definition of quantile functions:

```{prf:definition}
:label: quantile-rv-def

Let $X$ be a random variable with distribution function $F:\mathbb{R} \to [0,1]$. The *quantile function of $X$* is the function $Q: [0,1] \to \mathbb{R}$ defined so that

\begin{equation*}
Q(p) = \min\{x\in \mathbb{R} : p \leq F(x)\}.
\end{equation*}

In other words, the value $x=Q(p)$ is the smallest $x\in \mathbb{R}$ such that $p\leq F(x)$.

1. The value $Q(p)$ is called the *$p$-th quantile of $X$*.

2. The quantile $Q(0.5)$ is called the *median of $X$*.
```

Even though we had considerable practice with distribution and quantile functions in {numref}`Section %s <dist-quant>`, it won't hurt to do another practice problem:

```{admonition} Problem Prompt
Do problem 10 on the worksheet.
```










## Expected values

You should review problem 6 from the worksheet if it isn't already fresh in your mind. In that problem, we saw the discrete random variable

$$
X: S \to \mathbb{R}, \quad X = \text{ largest of two numbers}.
$$

We saw that the probability measure of $X$ is described by the following probability histogram:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 70%

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from itertools import product
import matplotlib as mpl 
plt.style.use('../aux-files/custom_style_light.mplstyle')
mpl.rcParams['figure.dpi'] = 600

support = [2, 3, 4, 5]
probs = [0.1, 0.2, 0.3, 0.4]
plt.bar(support, probs, width=0.25)
plt.xlabel('x')
plt.ylabel('probability')
plt.tight_layout()
```

Given this information, I want to ask: What is the *mean* (i.e., *average*) value of $X$? You might say: "Well, the range of $X$ consists of the numbers $2,3,4,5$, so the mean value of $X$ is just the mean of these four numbers:"

$$
\frac{2+3+4+5}{4} = 3.5.
$$

But this is wrong!

Don't believe me? Let's have the computer randomly draw two balls and record the value of $X$. Then, let's have the computer place the balls back into the urn, and then repeat the procedure, recording another (possibly different) value of $X$. Each time the computer draws a pair of balls we will call a *trial*. Let's have the computer complete, say, $n=10$ trials, and then have it compute the mean of the $n=10$ values of $X$. This produces a mean value of $X$ over $n=10$ trials.

But why stop with a mean value of $X$ over only $n=10$ trials? Why not let $n$ get bigger? What's the mean value of $X$ over $n=50$ trials, $n=100$ trials, or even $n=200$ trials? Answer:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 70%

np.random.seed(42)
sizes = range(1, 201)
means = [np.mean(np.random.choice(support, size=n, p=probs)) for n in sizes]
plt.plot(sizes, means)
plt.xlabel('$n=$number of trials')
plt.ylabel('mean value of $X$')
plt.axhline(y=4, color='red', linestyle='--')
plt.tight_layout()
```

As you can see, the mean values of $X$ appear to stabilize near $4$ (not $3.5$!) as the number of trials increases. And it is *this* theoretical limiting value of $4$ that we will call the *true* mean value of the random variable $X$.

Then, here's the central question:

> **Q**: How could we have computed this (theoretical, limiting) mean value without resorting to a computer simulation?

The answer is surprisingly easy. First, let's take a look at the *wrong* mean value that we computed above:

$$
\frac{2+3+4+5}{4} = 2\left(\frac{1}{4}\right) +3\left(\frac{1}{4}\right)+4\left(\frac{1}{4}\right)+5\left(\frac{1}{4}\right).
$$

The expression on the right-hand side is called a *weighted sum* of the numbers $2,3,4,5$, because it is the sum of these four numbers, but there are (multiplicative) weights placed on each of them, namely the number $1/4$. Notice that there are four $1/4$'s appearing in the weighted sum, and that

$$
\frac{1}{4} + \frac{1}{4} + \frac{1}{4} + \frac{1}{4} = 1.
$$

We interpret these four $1/4$'s as the *uniform* probability measure on the range $2,3,4,5$, where each of these four numbers gets a probability of $1/4$. Therefore, the *wrong* mean value computed above is wrong simply because it uses the wrong probability measure! The probability measure on the range of $X$ is *not* uniform, rather, it is given by the values in the probability histogram from above:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 70%

plt.bar(support, probs, width=0.25)
plt.xlabel('x')
plt.ylabel('probability')
plt.tight_layout()
```

So, what happens if we substitute the *correct* probability measure into the weighted sum? Here it is:

$$
2\left(0.1\right) +3\left(0.2\right)+4\left(0.3\right)+5\left(0.4\right) =4.
$$

And just like that, we get the correct answer!

Just by looking at the probability histogram of $X$, you can *see* that the mean value must be bigger than $3.5$ (which lies smack in the middle of the histogram). This is because the probability measure of $X$ has more "probability mass" concentrated on the right-hand side of the histogram, which will "pull" the mean value in that direction. In other words, the values of $X$ on the right-hand side of the histogram have more "weight" and thus contribute more to the mean value. Here's a plot of the probability histogram again, but with a vertical line representing the mean value:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 70%

expected_val = sum([x * p for x, p in zip(support, probs)])
plt.bar(support, probs, width=0.25)
plt.axvline(x=expected_val, color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('probability')
plt.tight_layout()
```

Let me give you four more random variables, $X_1$, $X_2$, $X_3$, and $X_4$, along with their probability histograms and vertical lines representing their mean values:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 100%

params = [[(2, 6), (6, 2)], [(0.1, 0.1), (2, 2)]]
support = np.linspace(0.1, 0.9, 9)
_, axes = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(10, 6))

for i, j in product(range(2), repeat=2):
    a = params[i][j][0]
    b = params[i][j][1]
    X = sp.stats.beta(a, b)
    probs = np.array([X.pdf(x) for x in support])
    probs = probs / np.sum(probs)
    expected_val = sum([x * p for x, p in zip(support, probs)])

    axes[i, j].bar(support, probs, width=0.05)
    axes[i, j].set_xticks(support)
    axes[i, j].axvline(x=expected_val, color='red', linestyle='--')
    axes[i, j].set_ylabel('probability')

axes[0, 0].set_xlabel('$X_1$')
axes[0, 1].set_xlabel('$X_2$')
axes[1, 0].set_xlabel('$X_3$')
axes[1, 1].set_xlabel('$X_4$')   
plt.tight_layout()

```

Notice that the mean values of the random variables $X_1$ and $X_2$ in the first row are "pulled" in the direction of higher "probability mass," while the mean values of $X_3$ and $X_4$ in the second row lie on the "axes of symmetry" of the probability distributions. Think of the mean value of a random variable as the "center of mass" of its probability distribution.

Before giving you the precise definitions, you need to know that mean values are also called *expected values*. (Quantum physicists also call them *expectation values*.) So, in the official, technical lingo, what we were computing above was the *expected value* of the random variable $X$.

```{prf:definition}
:label: expected-val-rv-def

Let $X$ be a random variable.

* If $X$ is discrete with probability mass function $p(x)$, then its *expected value*, denoted $E(X)$, is the sum
  
  $$
  E(X) = \sum_{x\in \mathbb{R}} x\cdot p(x).
  $$ (sum-01-eqn)

* If $X$ is continuous with probability density function $f(x)$, then its *expected value*, denoted $E(X)$, is the integral
  
  $$
  E(X) = \int_{\mathbb{R}} x\cdot f(x) \ \text{d} x.
  $$

In both cases, the expected value $E(X)$ is also often called the *mean value* of $X$ (or just *mean*) and denoted $\mu_X$ or just $\mu$.
```

Here we see the "sum vs. integral" dichotomy between discrete and continuous random variables that we saw in previous sections.

```{margin} Looking beyond...

Though I have only defined expected values of discrete and continuous random variables, technically *all* random variables have expected values (though there may be convergence issues). However, in order to even <a href="https://en.wikipedia.org/wiki/Expected_value#Arbitrary_real-valued_random_variables">*define*</a> these expected values, we would need the theory of Lebesgue integration that I mentioned above.
```

Now, the sum $\sum_{x\in \mathbb{R}}$ in {eq}`sum-01-eqn` simply means that we are adding up all terms of the form $x\cdot p(x)$, as $x$ ranges over all real numbers in $\mathbb{R}$. But remember, the random variable $X$ is assumed to be *discrete*, so there are actually either only finitely many $p(x)$'s that are nonzero---in which case the sum is actually finite---or there are a countable infinity that are nonzero---in which case the sum is of the form $\sum_{n=1}^\infty$. In the latter case, there is the possibility that the expected value $E(X)$ doesn't exist, because the infinite series doesn't converge. However, at this level, except for a few special cases, we won't worry about such things. We shall *always* assume in this class that the infinite series converge and all our expected values exist. Even more, we shall assume that the series converge *absolutely*, which you might(?) remember means that the series of absolute values

$$
\sum_{x\in \mathbb{R}} |x\cdot p(x)| = \sum_{x\in \mathbb{R}} |x|\cdot p(x)
$$

converges. One byproduct of absolute convergence is that the order of the summands in the infinite series doesn't matter; if you've taken real analysis and have studied <a href="https://en.wikipedia.org/wiki/Riemann_series_theorem#">rearrangements</a>, this might sound familiar to you. If not, don't worry about it.

Likewise, in the continuous case there is the possibility that $E(X)$ doesn't exist, in the sense that the improper integral

$$
\int_{-\infty}^\infty |x| \cdot f(x) \ \text{d} x
$$

doesn't converge. But as in the discrete case, we won't worry about these situations.

```{admonition} Problem Prompt
Do problems 11-15 on the worksheet.
```












## The algebra of random variables

Remember that, technically, random variables are *functions*. But as you will see as you go through your training in probability and statistics, we often want to treat them *as if* they were numbers and add and subtract them, multiply and divide them, and even plug them into functions like we would a numerical variable. In short: We want to develop an *algebra* of random variables.

Let's see how this works in the case of *addition* of random variables. Given two random variables

$$
X,Y: S \to \mathbb{R}
$$

on a probability space $S$, we ask: What should the sum $X+Y$ be? Well, it should *itself* be a random variable on $S$, so, in particular, it needs to be a function

$$
X+Y : S \to \mathbb{R}.
$$

To evaluate $X+Y$ at a sample point $s\in S$, the most natural thing to do is to add the corresponding outputs of $X$ and $Y$, i.e., to define

$$
(X+Y)(s) = X(s) + Y(s).
$$ (add-eqn)

This addition operation on random variables is called the *pointwise sum*.

Yikes. This seems overly complicated. But it's pretty easy to bring down to earth if we imagine for simplicity that $X$ and $Y$ are discrete random variables defined on a *finite* sample space. For example, suppose that

$$
S = \{1,2,3,4\}.
$$

Then, suppose that the outputs of $X$ and $Y$ are given by the values in the table:

$$
\begin{array}{c|cc}
s & X(s) & Y(s) \\ \hline
1 & -1 & 0 \\
2 & 1 & 2 \\
3 & 3 & -1 \\
4 & 0 & 3 
\end{array}
$$

Thus, for example, we have $X(1) = -1$ and $Y(4) = 3$. We obtain the pointwise sum $X+Y$ defined by {eq}`add-eqn` simply by adding the corresponding outputs of $X$ and $Y$, which is essentially just the rowwise sum:

$$
\begin{array}{c|ccc}
s & X(s) & Y(s) & (X+Y)(s) \\ \hline
1 & -1 & 0 & -1 + 0 =-1 \\
2 & 1 & 2 & 1+2 = 3 \\
3 & 3 & -1 & 3-1 = 2 \\
4 & 0 & 3 & 0+3=3
\end{array}
$$

Ok, cool. So we've defined $X+Y$ as a function on $S$. But remember, $X+Y$ is supposed to be a *random variable*, so it's supposed to have a probability measure. We will compute it in:

```{admonition} Problem Prompt

Do problem 16 on the worksheet.
```

In general, given the probability measures of arbitrary random variables $X$ and $Y$, it can be difficult to compute the probability measure of $X+Y$ except in very special cases.

```{margin} Looking beyond...

I've previewed some of the more advanced *analytic* theory in the previous few "Looking beyond..." margin notes. Now, how about the *algebraic* theory?

* The pointwise addition and product operations defined above give the set of all random variables on a fixed probability space the structure of a [commutative ring](https://en.wikipedia.org/wiki/Commutative_ring).

* The pointwise addition and scaling operations defined above give the set of all random variables on a fixed probability space the structure of an abstract [vector space](https://en.wikipedia.org/wiki/Vector_space).

* These ring and vector space structures interact "coherently" in the sense that together they form a [commutative algebra](https://en.wikipedia.org/wiki/Algebra_over_a_field).
```

You saw the pointwise product $XY$ in problem 16 on the worksheet. I bet you can guess the definition of the pointwise difference $X-Y$ and pointwise quotient $X/Y$ (watch out for when $Y=0$ in the latter!). Moreover, given a constant $c\in \mathbb{R}$, there is also a very natural definition of $cY$, where $Y$ is a random variable; indeed, it is given pointwise by

$$
(cY)(s) = cY(s).
$$ (scale-eqn)

Taking $c=4$ in the example above, we have:

$$
\begin{array}{c|cc}
s & Y(s) & (4Y)(s) \\ \hline
1 & 0 & 4\cdot 0 = 0 \\
2 & 2 & 4\cdot 2 = 8 \\
3 & -1 & 4 \cdot(-1) = -4 \\
4 & 3 & 4\cdot 3 = 12
\end{array}
$$

It thus appears that the pointwise scaling operation {eq}`scale-eqn` is just columnwise scaling.

So there we have it: We now have an *algebra* of random variables. Easy enough. But be aware that if your random variables are *continuous*, then we obviously cannot represent them via a finite table of values like we did above, and perform rowwise and columnwise operations. In this case, your only option is to resort to the defining equations for the algebraic operations:

* $(X\pm Y)(s) = X(s) \pm Y(s)$,

* $(XY)(s) = X(s) Y(s)$,

* $(X/Y)(s) = X(s)/Y(s)$, when $Y(s)\neq 0$,

for $s\in S$.












## Functions of random variables

In addition to an algebra of random variables, I mentioned at the beginning of the previous section that we will want to plug random variables into functions, just like we would numerical variables. It's difficult to convey at this early stage exactly *why* we would want to do this, but I encourage patience. You'll see soon enough how often this procedure is used!

To see how this works, let's suppose a random variable $X$ is defined on a finite sample space $S=\{1,2,3,4\}$ with:

$$
\begin{array}{c|c}
s & X(s)  \\ \hline
1 & 0  \\
2 & 2  \\
3 & -1  \\
4 & 3 
\end{array}
$$

Now, consider the good ol' quadratic function $g(x) = x^2$. How should I make sense of $g(X) = X^2$?

First, notice that the expression $g(X)$ technically doesn't make any sense. This is because $X$ is a *function*, while the domain of $g$ is a set of *numbers*, so mathematically speaking you cannot plug $X$ into $g$ since $X$ is not in the domain of $g$. Instead, what $g(X)$ *really* represents is the composite function $g\circ X$ which is a legitimate function on $S$ like any random variable is supposed to be.

Given this interpretation, how would we actually compute $g(X) = X^2$ given the table of values above? Easy! Simply apply the squaring function $g(x) = x^2$ rowwise to each output of $X$:

$$
\begin{array}{c|cc}
s & X(s) & X^2(s) \\ \hline
1 & 0 & 0^2 = 0  \\
2 & 2 & 2^2 = 4  \\
3 & -1 & (-1)^2 = 1  \\
4 & 3 & 3^2 = 9 
\end{array}
$$

While this is pretty straightforward in the case that $X$ is defined on a finite probability space, in the case that $X$ is *continuous*, we cannot compute $g(X)$ using a finite table of values. In this case, you must resort to the *definition*:

$$
g(X) = g \circ X.
$$

Now: 

```{admonition} Problem Prompt

Have a go at problem 17 on the worksheet.
```











## Expectations of functions of random variables and the LotUS

In the next two sections, we list some of the most useful properties of expectations of functions of random variables. Two such properties will be of particular importance: The first is encoded in something called the "Law of the Unconscious Statistician," which we will talk about in this section, while the second property is called *linearity* and will be discussed in the next section.

Here's the situation: Suppose we have a discrete random variable $X:S\to \mathbb{R}$ on a probability space, and a real-valued function $y=g(x)$ on $\mathbb{R}$. As I explained above, we can "plug $X$ into $g$," obtaining the "transformed" random variable $Y = g(X)$ which is really the composite $g\circ X$ in disguise. The goal in this section is to derive a formula for the expected value $E(Y) = E(g(X))$.

Now, **by definition**, this expected value is given by the formula

$$
E(Y) = \sum_{y\in \mathbb{R}} y \cdot P\left(Y=y\right).
$$ (right-eqn)

However, if you weren't paying close attention and were mindlessly and unconsciously computing things, you might be tempted by the (valid!) equations $Y = g(X)$ and $y=g(x)$ to compute this expected value by

$$
E(g(X)) = \sum_{x\in \mathbb{R}} g(x)\cdot P(X=x).
$$ (wrong1-eqn)

It could happen, right?

But you *need* to notice that it is *not* obvious that the expression on the right-hand side of {eq}`wrong1-eqn` correctly computes the expected value $E(Y) = E(g(X))$. For one, notice that the sum in {eq}`wrong1-eqn` iterates over all $x\in \mathbb{R}$, and that these values are inserted into $g$ to obtain $y=g(x)$. But there is no reason to believe that you obtain *all* values of $y\in \mathbb{R}$ in this way, and yet this is what the *correct* formula {eq}`right-eqn` demands.

So, you see there is no *obvious* reason why the two formulas {eq}`right-eqn` and {eq}`wrong1-eqn` should be the same. And yet, if you mistakenly compute $E(Y) =E(g(X))$ using {eq}`wrong1-eqn` and then return later in a panic to re-do your computations using {eq}`right-eqn`, you'll find that they were *right* the entire time! This is because these two formulas secretly *are* the same, but exactly *why* they are equal is not obvious.

```{margin}
And by the way: Though I am a mathematician by training and get annoyed rather easily with sloppy mathematical rigor, I did *not* <a href="https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician">make up</a> this name. :)
```

The fact that the formula {eq}`wrong1-eqn` correctly computes the expected value $E(Y) = E(g(X))$ is called the "Law of the Unconscious Statistician," named in honor of all those unconscious statisticians who believe the formulas {eq}`right-eqn` and {eq}`wrong1-eqn`are **obviously** the same thing!


Now, let me begin to explain *why* {eq}`wrong1-eqn` correctly computes the expected value by drawing a few pictures. First, remember that $Y=g(X)$ is a real-valued function on $S$ like any other, so I would picture it like this:

```{image} ../img/trans0.svg
:width: 75%
:align: center
```
&nbsp;

In the *definition* {eq}`right-eqn` of $E(g(X))$, notice that we must compute the probabilities

$$
P(Y=y) = P(g(X)=y)
$$ (yeah-eqn)

for each $y\in \mathbb{R}$. This probability is precisely the probability of the set in $S$ consisting of those sample points $s$ that "hit" $y$ when mapped via $g(X)$. I have shaded this set in:

```{image} ../img/trans3.svg
:width: 75%
:align: center
```
&nbsp;


It could be that there are actually *no* sample points in $S$ that "hit" $y$; in this case, the shaded set in $S$ is empty, and the probability {eq}`yeah-eqn` is $0$.

Now, remember also that $g(X)$ technically stands for the composite function $g\circ X$, so on its way from $S$ to $\mathbb{R}$ (left to right), the random variable $g(X)$ first takes a detour through $X$ and another copy of $\mathbb{R}$:

```{image} ../img/trans1.svg
:width: 75%
:align: center
```
&nbsp;


If we bring back our point $y\in \mathbb{R}$ from above, this means that the points in $S$ that "hit" $y$ must *also* first take a detour through $X$:

```{image} ../img/trans2.svg
:width: 75%
:align: center
```
&nbsp;

The collection of shaded "intermediate points" in the diagram is denoted $g^{-1}(y)$ and is called the *preimage* of $y$ under $g$; in set notation:

$$
g^{-1}(y) \stackrel{\text{def}}{=} \{x\in \mathbb{R} : g(x) = y \}.
$$

Now, any sample point $s\in S$ that "hits" $y$ under the function $g(X)$ must have the property that $X(s) \in g^{-1}(y)$. That makes sense, right? If $s$ "hits" $y$, then it must first "hit" one of the shaded "intermediate points" in the preimage $g^{-1}(y)$ in the last picture.

If we suppose that there are only three points $x_1$, $x_2$, and $x_3$ in the preimage $g^{-1}(y)$ (so that the picture is accurate), then each of the three shaded sets in $S$ can be labelled as follows:

```{image} ../img/trans4.svg
:width: 100%
:align: center
```
&nbsp;

Therefore, we have the equation of sets

$$
\{s\in S: g(X(s)) = y \} =  \bigcup_{k=1}^3 \{s\in S : X(s) = x_k\}.
$$ (easy-eqn)

There's a lot going on in this equality, so make sure that you pause and ponder it for a while if you need to.

But it might be the case that the preimage $g^{-1}(y)$ doesn't consist only of three $x$-values; indeed, it could even contain infinitely many values! Since I don't know ahead of time how many points it contains, it is better to rewrite {eq}`easy-eqn` as

$$
\{s\in S: g(X(s)) = y \} =  \bigcup_{x\in g^{-1}(y)} \{s\in S : X(s) = x\}.
$$ (harder-eqn)

This looks like a more complicated equality, but the intuition behind it is *exactly* the same as {eq}`easy-eqn`.

If I now apply the probability measure $P$ to both sides of this last equality {eq}`harder-eqn`, and use the fact that the union on the right-hand side is disjoint, I get:

$$
P\left(g(X)=y\right) = \sum_{x\in g^{-1}(y)} P(X=x).
$$ (fundamental-eqn)

Now, it turns out that *this* is the key equality for unlocking the proof of the Law of the Unconscious Statistician. There are two versions of this law, one for discrete random variables and the other for continuous ones. Let's formally state the law before I continue with the proof:

```{prf:theorem} Law of the Unconscious Statistician (LotUS)
:label: lotus-thm

Let $X$ be a random variable and $g:\mathbb{R}^2 \to \mathbb{R}$ a function.

* If $X$ is discrete with mass function $p(x)$, then
  
  $$
  E(g(X)) = \sum_{x\in \mathbb{R}} g(x)\cdot p(x).
  $$

* If $X$ is continuous with density function $f(x)$, then
  
  $$
  E(g(X)) = \int_{\mathbb{R}} g(x) \cdot f(x) \ \text{d} x.
  $$
```

```{margin} Looking beyond... 

From a much more high-brow point of view, the Law of the Unconscious Statistician is a corollary of a simple <a href="https://en.wikipedia.org/wiki/Pushforward_measure#Main_property:_change-of-variables_formula">change-of-variables formula</a> for Lebesgue integrals.
```

We will only prove the result in the case that $X$ is discrete. And in view of the work we did above, the proof is easy: Beginning with the definition {eq}`right-eqn` of $E(g(X))$ and using {eq}`fundamental-eqn`, we compute:

\begin{align*}
E(g(X)) &= \sum_{y\in \mathbb{R}} y\cdot P\left(g(X)=y\right) \\
&= \sum_{y\in \mathbb{R}} \sum_{x\in g^{-1}(y)} y\cdot P(X=x) \\
&= \sum_{x\in \mathbb{R}} g(x)\cdot P(X=x),
\end{align*}

where the last equality follows from the observation that $g(x)=y$ if $x\in g^{-1}(y)$.

Et voilà! Just like that, we've proved the LotUS by simply drawing a bunch of pictures (at least in the discrete case)! Now try the following:

```{admonition} Problem Prompt

Do problems 18 and 19 on the worksheet.
```









## Linearity of expectation

We've learned that if $X$ and $Y$ are two random variables, then their pointwise sum $Z=X+Y$ is also a random variable. In the discrete case, we may compute the expectations of $X$ and $Y$, respectively, via the definition as

$$
E(X) = \sum_{x\in \mathbb{R}}x\cdot P(X=x) \quad \text{and} \quad E(Y) = \sum_{y\in \mathbb{R}}y\cdot P(Y=y).
$$ (huh1-eqn)

How would we compute the expectation of the sum $Z = X+Y$? By definition, it would be

$$
E(X+Y) = \sum_{z\in \mathbb{R}} z \cdot P(X+Y=z).
$$ (huh2-eqn)

But it is not clear at first glance that there is any sort of simple relationship between the three expectations in {eq}`huh1-eqn` and {eq}`huh2-eqn`. Nevertheless, as I will explain in a later chapter---after we've learned about *joint distributions*---these expectations are related through the first equation in:

```{prf:theorem} Linearity of Expectations
:label: linearity-init-thm

Let $X$ and $Y$ be two random variables and let $c\in \mathbb{R}$ be a constant. Then:

$$
E(X+Y) = E(X) + E(Y),
$$ (linear-eqn)

and

$$
E(cX) = c E(X).
$$ (homog-eqn)
```

You couldn't possibly hope for any simpler relationship than {eq}`linear-eqn`! But as I just mentioned, I can't yet explain *why* this equation holds. The explanation will come later, in the form of {prf:ref}`linear-exp-thm`.

However, we *can* prove {eq}`homog-eqn` quite easily: Just use the LotUS with the function $g(x) = cx$. Can you supply the details to this argument?

Now, even though we can't prove {eq}`linear-eqn` at this point, we won't need it quite yet, so we're not in any danger. Instead, the following "weak" form of linearity is all we need. (For convenience, I've taken the identity {eq}`homog-eqn`---which you *just* proved---and included it in this special case.)

```{prf:theorem} "Weak" Linearity of Expectations
:label: weak-linear-thm

Let $X$ be a discrete or continuous random variable, let $y=g_1(x)$ and $y=g_2(x)$ be two real-valued functions, and let $c\in \mathbb{R}$ be a constant. Then:

$$
E(g_1(X) + g_2(X)) = E(g_1(X)) + E(g_2(X)),
$$

and

$$
E(cX) = c E(X).
$$
```

This "weak" form of linearity is a simple application of the LotUS. To see this, first define the real-valued function $y=g(x)$ by setting

$$
g(x) = g_1(x) + g_2(x)
$$

for each $x$. Then $g(X) = g_1(X) + g_2(X)$, and an application of the LotUS (in the continuous case) gives:

\begin{align*}
E(g_1(X) + g_2(X)) &= E(g(X)) \notag \\
&= \int_{-\infty}^\infty g(x)\cdot f(x) \ \text{d} x \notag \\
&= \int_{-\infty}^\infty (g_1(x) + g_2(x)) f(x) \ \text{d} x \notag \\
& = \int_{-\infty}^\infty g_1(x)\cdot f(x) \ \text{d} x + \int_{-\infty}^\infty g_2(x)\cdot f(x) \ \text{d} x \\
&= E(g_1(X)) + E(g_2(X)).
\end{align*}

Can you see exactly where I used the LotUS? And can you adapt this argument on your own to cover the discrete case?

We need just one more fact before proceeding to practice problems. To state it, observe that any constant $c\in \mathbb{R}$ may be considered a random variable. Indeed, it may be identified with the unique random variable that sends the *entire* sample space to the constant $c$:

```{image} ../img/constant.svg
:width: 50%
:align: center
```
&nbsp;

Thus, every constant $c\in \mathbb{R}$ may be considered a *constant* random variable. Your intuition would then suggest that the mean value of a *constant* random variable should be pretty easy to compute. And you'd be right!

```{prf:theorem} Expectations of Constants
:label: expectation-constant-thm

Let $c\in \mathbb{R}$ be a constant, viewed as a constant random variable. Then $E(c) = c$.
```

To see why $E(c) = c$ holds, simply note that

$$
E(c) = \sum_{x\in \mathbb{R}} x\cdot P(c=x) = c,
$$

since

$$
P(c=x) = \begin{cases} 1 & : x = c, \\ 0 & : x\neq c. \end{cases}
$$

Now:

```{admonition} Problem Prompt

Do problem 20 on the worksheet.
```













## Variances and standard deviations

The last few sections were quite technical, packed with all sorts of definitions, formulas, and equations. But we now return to the same circle of ideas that we used to *motivate* the definition of the expected value of a random variable $X$: It is precisely the "mean value" of $X$, and if we think physically, it is the "center of mass" of the probability distribution of $X$. As such, it provides a rough description of the "shape" of the distribution, telling us where the majority of the "probability mass" is centered. In fact, for this reason, the expected value of $X$ is often called a *measure of centrality*.

In this section, we study two so-called *measures of dispersion*, or *measures of spread*. To motivate them, first consider the following two possible probability distributions for a discrete random variable $X$:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 100%

params = [(10, 10), (2, 2)]
support = np.linspace(0.1, 0.9, 9)
_, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 3))

for i in range(2):
    a = params[i][0]
    b = params[i][1]
    X = sp.stats.beta(a, b)
    probs = np.array([X.pdf(x) for x in support])
    probs = probs / np.sum(probs)
    expected_val = sum([x * p for x, p in zip(support, probs)])
    
    axes[i].bar(support, probs, width=0.05)
    axes[i].set_xticks(support)
    axes[i].axvline(x=expected_val, color='red', linestyle='--')
    axes[i].set_ylabel('probability')

axes[0].set_xlabel('$y$')
axes[1].set_xlabel('$y$')
plt.tight_layout()
```

Notice that both probability distributions have the same expected value $\mu = 0.5$, as indicated by the two vertical lines. But the second distribution is clearly more dispersed---or "spread out"---as compared to the first.

> **Q**: How might we measure the spread of these distributions?

Here's one way. Imagine choosing random numbers in the range of $X$, which according to the histograms above, is equal to the set

$$
\{0.1,0.2,\ldots,0.9\}.
$$

If you choose these numbers according to the left-hand distribution, then *on average*, your numbers should be relatively close to the mean value $\mu=0.5$. On the other hand, if you choose these numbers according to the right-hand distribution, then *on average*, your numbers would likely spend more time further away from the mean value $\mu=0.5$. So, what we're really talking about is the *average distance from the mean*.

Now, if $x$ is one of your chosen numbers in the range of $X$, then its distance to the mean is $|x - \mu|$. (The absolute value bars ensure that this distance is always non-negative, as any distance should be.) Thus, a measure of dispersion of the distributions would be the mean values of the distances $|x-\mu|$, as $x$ varies over the range of $X$. But expected values *are* mean values, and so this suggests that a good measure of dispersion would be the expected value

$$
E(|X-\mu|).
$$ (disp-eqn)

In fact, if I compute these expected values for the two probability measures above, I get:

$$
E(|X-\mu|) \approx \begin{cases} 0.08 & : \text{left distribution}, \\ 0.18 & : \text{right distribution}. \end{cases}
$$

So, according to the dispersion measure {eq}`disp-eqn`, the right-hand distribution is, in fact, more spread out than the left-hand distribution. This is precisely what we expected!

But while {eq}`disp-eqn` is a perfectly good measure of dispersion, for certain technical reasons it is not the one most often used. (For example, the function $g(x) = |x-\mu|$ is not differentiable at $x=\mu$, which can complicate some things.) Instead, we don't look at the average *actual* distance to the mean, rather we look at the average *squared distance* to the mean:

```{prf:definition}
:label: variance-rv-def

Let $X$ be a random variable with expected value $\mu = E(X)$. The *variance* of $X$, denoted $V(X)$, is given by

$$
V(X) = E\left( (X-\mu)^2 \right).
$$ (var-eqn)

The variance of $X$ is also denoted $\sigma^2_X$ or just $\sigma^2$.
```

As with expected values, there is the question of whether the variance actually *exists*. Moreover, technically we have only defined expected values for discrete and continuous random variables, so I *should* insert those qualifiers in the definition. But we don't worry about these technicalities at this level.

One of the advantages that the variance {eq}`var-eqn` has over {eq}`disp-eqn` is that the quadratic function $g(x) = (x-\mu)^2$ is nice and smooth, whereas the absolute value function $g(x) = |x-\mu|$ is not. Further differences can be uncovered by considering the graphs of these two functions: The first gives more weight to $x$-values far away from $\mu$ as compared to the second, while it gives less weight to $x$-values close to $\mu$ as compared to the second. (Draw the graphs!) These types of considerations are important when one studies <a href="https://en.wikipedia.org/wiki/Loss_function">loss, cost and error functions</a>.

Due to the squaring operation, the units of the variance $V(X)$ are the units of $X$ *squared*. This is sometimes undesirable---for example, if $X$ is measured in degrees Fahrenheit, then $V(X)$ is measured in degrees Fahrenheit *squared*. So, in order to compare apples to apples, sometimes it is more useful to consider the following related measure of dispersion:

```{prf:definition}
:label: std-rv-def

Let $X$ be a random variable. The *standard deviation* of $X$, denoted $\sigma_X$ or just $\sigma$, is the positive square root of the variance:

$$
\sigma_X = \sqrt{V(X)}.
$$
```

Let's compute a few variances and standard deviations straight from the definitions:

```{admonition} Problem Prompt

Do problems 21-23 on the worksheet.
```

Sometimes the following formula can shorten computations:

```{prf:theorem} Shortcut Formula for Variance
:label: shortcut-var-thm

Let $X$ be a random variable. Then

$$
V(X) = E(X^2) - E(X)^2 =  E(X^2) - \mu_X^2.
$$
```

Why is this formula true? Here's why:

\begin{align*}
V(X) &= E\left( (X-\mu)^2 \right) \\
&= E \left( X^2 - 2\mu X + \mu^2 \right) \\
&= E(X^2) + E(-2\mu X) +E(\mu^2) \\
&= E(X^2) -2\mu E(X) + \mu^2 \\
&= E(X^2) - 2\mu^2 + \mu^2 \\
&= E(X^2) - \mu^2.
\end{align*}

Go through these computations, line by line, and see if you can identify which properties of expectations I used! Be very careful in identifying them!

Now, given a function $g(X)$ of a random variable $X$, the LotUS gives a very simple way to compute the expectation $E(g(X))$. We might wish for something similar for variances, but unfortunately things aren't so simple in general. However, if the function $g(x)$ is a so-called *affine transformation* of the form

$$
g(x) = ax+b,
$$

where $a$ and $b$ are constants, then we *can* say something in general:

```{prf:theorem} Variance of an Affine Transformation
:label: var-affine-thm

Let $X$ be a random variable and $a$ and $b$ constants. Then

\begin{equation}\notag
V(aX +b) = a^2 V(X).
\end{equation}
```

Why is this formula true? Here's why:

\begin{align*}
V(aX+b) &= E\left( (aX+b)^2 \right) - E(aX+b)^2 \\
&= E\left( a^2 X^2 + 2abX+b^2 \right) - \left(aE(X) + b \right)^2 \\
&= a^2 E(X) + 2ab E(X) + b^2 - a^2 E(X)^2 -2abE(X) - b^2 \\
&= a^2\left( E(X^2) - E(X)^2 \right) \\
&= a^2 V(X).
\end{align*}

Again, make sure you can pick out exactly what properties of expectations and variances I used in moving from line to line.

```{admonition} Problem Prompt

Holy smokes. After a very, *very* long discussion of random variables, we've finally reached the end! Now finish off the worksheet and do problems 24 and 25.
```