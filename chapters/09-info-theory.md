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

(information-theory)=
# Information theory

This chapter marks a pivotal shift in the book, moving from our focused exploration of abstract probability theory to practicalities of building and training probabilistic models. Subsequent chapters construct estimators and statistics and develop their theories, all of this with the overarching goal of leveraging these newfound tools to discover answers to specific questions or inquiries of particular interest. Most texts on mathematical statistics make a similar transition to similar material that they call _inferential statistics_---but whatever it might be called, the end goal is the same: _Learn from data_.

The current chapter describes tools and techniques drawn from the _theory of information_, which is a charming amalgamation of practical engineering and theoretical mathematics. Among other things, this theory provides us with a method for measuring a particular form of _information_, but it also gives us techniques for quantifying related degrees of _surprise_, _uncertainty_, and _entropy_. In the upcoming chapters, our primary use for these measures will be to train and choose probabilistic models, but the theory reaches way beyond into physics, coding theory, computer science, neuroscience, biology, economics, the theory of complex systems, and even philosophy.

We will begin the chapter with a motivational section to help introduce the very specific type of _information_ that this theory purports to study. While the hurried reader may skip this initial section if they so choose, they should also be warned that the abstract definitions in later sections will be quite difficult to "grok" at first sight without the context and setting provided by the first section. Throughout the remaining sections in the chapter, we will try to hit all the highlights of the theory, but we will fall well short of comprehensive coverage. But fortunately, there are several textbook-length treatments of information theory that are very approachable---the standard reference seems to be {cite}`CoverThomas2006`, though my preferences lean much more toward {cite}`MacKay2003`. I also quite enjoy {cite}`Ash2012`, though it is much older than the other two references and is written in the definition-theorem-proof style of pure mathematics (which some appreciate, some don't).










## Preview: How do we measure information?

The word _information_ is often used in different contexts to mean different things. We all have some intuitive sense for what the term means, but it is a notoriously difficult thing to nail down _precisely_---no single definition seems to exist that covers _all_ the ways in which it is used. Both professional philosophers (which I am not) and amateur armchair philosophers (which I am) like to argue about it. The fastest way for someone to tell you that they have no clue what information is, is to tell you that they know what information is.

This being said, one of the central quantities that we will define (precisely!) and study in this chapter is something called _Shannon information_. The name comes from Claude Shannon, who is credited with laying down most of the foundations of the mathematical theory of information in {cite}`Shannon1948`, though he referred to it as the mathematical theory of _communication_. Indeed, on this point I cannot resist quoting another one of the pioneers of the field, Robert Fano:

```{margin}

This is the same quote that opens the fantastic survey article {cite}`Rioul2021` on information theory, which I enthusiastically recommend. I learned of this quote from that article.
```

```{epigraph}
"I didn’t like the term 'information theory.' Claude [Shannon] didn’t like it either. You see, the term 'information theory' suggests that it is a theory about information—--but it’s not. It’s the transmission of information, not information. Lots of people just didn’t understand this."
```

Despite these misgivings from the founders, the terms _information theory_ and _Shannon information_ have stuck, and that's what we will call them. Once you learn the precise definition of _Shannon information_, it will be up to you to decide if it comports with and captures your prior intuitive understanding of "information."

At first blush, you might imagine that this _information_ resides in data, but that's not true. Rather, this particular form of _information_ is initially attached to our _beliefs_ about the data---or, more precisely, this _information_ is associated with a probabilistic model of the data. But if we have successfully cooked up a model that we think truly captures the data, then this form of _information_ might (with caution!) be transferred from the model and attributed to the data. In any case, it's a point that you would do well to remember: _Information-theoretic measures are associated with models, not datasets!_

To give you an initial sense of how this special notion of _information_ arises, let's go through a simple and concrete example. Let's suppose that we have two simple data sources that produce bit strings, or strings of $0$'s and $1$'s. (Bit $=$ binary digit.) We will assign them the names Source 1 and Source 2, and we then collect data:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import huffman
from itertools import product
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.style.use('../aux-files/custom_style_light.mplstyle')
blue = '#486AFB'
magenta = '#FD46FC'

def generate_data(theta, block_length, message_length, random_state=None):
    if random_state != None:
        np.random.seed(random_state)
    num_bits = block_length * message_length
    n1 = int(theta * num_bits)
    n0 = int((1 - theta) * num_bits)
    ones = np.ones(n1, dtype=int)
    zeros = np.zeros(n0, dtype=int)
    data = np.concatenate((ones, zeros))
    np.random.shuffle(data)
    data_string = ''.join([str(num) for num in data])
    data_blocks = [data_string[i:i+5] for i in range(0, num_bits, block_length)]
    return data_string, data_blocks

block_length = 5
theta1 = 0.5
theta2 = 0.2    # four times as many 0's as 1's
message_length = 100

data1_string, data1 = generate_data(theta=theta1, block_length=block_length, message_length=message_length, random_state=42)
data2_string, data2 = generate_data(theta=theta2, block_length=block_length, message_length=message_length, random_state=42)

print('Data 1: ', data1_string)
print('Data 2: ', data2_string)
```

As the names indicate, the first bit string is drawn from Source 1, while the second from Source 2. Each string is 500 bits long.

Models are suggested through the identification of patterns, regularities, and other types of special and particular structure. But when you scroll through the bit strings, it appears that the $0$'s and $1$'s are produced by the two sources in a random and haphazard manner; there is no detectable _deterministic_ regularities. But probabilistic models are not built to capture such regularities, so this should not worry us; rather, such models are designed to capture _probabilistic_ or _statistical_ properties.

We _do_ notice that the second string contains many more $0$'s than $1$'s. With this in mind, we count the number of $0$'s and $1$'s in each string and compute the ratio of these two numbers. We find:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

n1 = data1_string.count('1')
n0 = len(data1_string) - n1
theta1 = n1 / len(data1_string)
print(f"Ratio of 0's to 1's for Data 1:  {n0 / n1:0.2f}")

n1 = data2_string.count('1')
n0 = len(data2_string) - n1
theta2 = n1 / len(data1_string)
print(f"Ratio of 0's to 1's for Data 2:  {n0 / n1:0.2f}")
```

Ah ha! The first bit string contains exactly the same number of $0$'s and $1$'s, while the second bit string contains exactly four $0$'s for every $1$. This immediately suggests the models: Source 1 and Source 2 will be modeled, respectively, by the random variables

$$
X_1 \sim \Ber(0.5) \quad \text{and} \quad X_2 \sim \Ber(0.2).
$$

We conceptualize the bit strings as subsequent draws from these random variables, each bit in the string being produced independently of all those that came before it.

Now we ask the central question: Using these models, how might we measure or quantify the _information_ contained in each bit string? (Take care to notice that we are asking this question only _after_ we have chosen probabilistic models, not _before_. Indeed, in accord with what we mentioned above, the notion of _information_ that we are ultimately after is a feature of the model, _not_ the raw data.)

Of course, the question is hopelessly unanswerable, because _information_ is as yet undefined. But instead of trying to find some abstract and highfalutin description that _directly_ aims to characterize _information_ in all its diverse manifestations, we seek some sort of proxy that allows us to _indirectly_ "get at" this slippery notion.

One such proxy is inspired and motivated by practical engineering considerations: The information content in these strings should be related to our ability to losslessly _compress_ the strings. Indeed, a larger compression ratio (i.e., the number of original bits to the number of compressed bits) should reflect that the string contains little information, while a smaller compression ratio should mean the opposite. As an extreme example, think of the bit string consisting of exactly five hundred $1$'s; we might imagine that it is the output of a third data source modeled via the random(?) variable $X_3 \sim \Ber(1)$. Intuitively, there is little information content carried by such a string, which is reflected in the fact that it may be highly compressed: If we design an encoding scheme with _block length_ equal to $500$ (see below), then this string would be compressed to the length-$1$ string consisting of just the single bit $1$. This is a $500$ to $1$ compression factor---quite large indeed!

How do the probabilistic models fit into these considerations? Remember, the models were chosen to capture statistical properties of the bit strings. We can take advantage of these properties by splitting the strings into substrings of a specified _block length_; for example, if we use a block length of $5$, we get:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

print('Data 1: ', data1)
print('Data 2: ', data2)
```

The first model $X_1 \sim \Bin(0.5)$ tells us that any given block is just as likely to appear in the data as any other; however, since the second model $X_2 \sim \Bin(0.2)$ assigns different probabilities to observing one or the other of the two bits $0$ and $1$, certain blocks in the second data string are more likely to appear than others. So then the idea is simple: To compress the second data string, we assign short code words to blocks that are more likely to appear according to the model.

One routine to find suitably short code words is called _Huffman coding_ which, conveniently, may be easily implemented in Python. The following code cell contains a dictionary representing the codebook obtained by running this routine on the second data string above. The keys to the dictionary consist of all $2^5 = 32$ possible blocks, while the values are the code words.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

strings = [''.join([str(tup[i]) for i in range(block_length)]) for tup in product(range(2), repeat=block_length)]

def generate_codebook(theta):
    prob_dist = {}
    for string in strings:
        n1 = string.count('1')
        n0 = block_length - n1
        prob = (theta ** n1) * ((1 - theta) ** n0)
        prob_dist = prob_dist | {string: prob}
    codebook = huffman.codebook(prob_dist.items())
    return codebook, prob_dist

codebook2, prob_dist2 = generate_codebook(theta=theta2)
print('codebook: ', codebook2)
```

Notice that the blocks `00000` and `11111` are assigned, respectively, the code words `11` and `0111100100`. The difference in length of the code words reflects the difference in probability of observing the two blocks, $0.8^5 \approx 0.3277$ for the first versus $0.2^5 \approx 0.0003$ for the second. We print out the original blocks and their code words in the next cell, along with bit counts and the (reciprocal) compression ratio:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

compressed_data2 = [codebook2[word] for word in data2]
data2_spaced = []
compressed_data2_spaced = []

for block, compressed_block in zip(data2, compressed_data2):
    diff = len(block) - len(compressed_block)
    if diff >= 0:
        blanks = (' ' * diff)
        data2_spaced.append(block)
        compressed_data2_spaced.append(blanks + compressed_block)
    else:
        blanks = (' ' * -diff)
        data2_spaced.append(blanks + block)
        compressed_data2_spaced.append(compressed_block)
    
print('Data 2:                              ', data2_spaced)
print('Compressed data 2:                   ', compressed_data2_spaced)
print('Number of bits in data 2:            ', len(data2_string))
print('Number of bits in compressed data 2: ', len(''.join(compressed_data2)))
print('(reciprocal) Compression ratio 2:   ', len(''.join(compressed_data2)) / len(data2_string))
```

If we run the same routine on the first data string drawn from the model $X_1 \sim \Ber(0.5)$, we get:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

codebook1, prob_dist1 = generate_codebook(theta=theta1)
compressed_data1 = [codebook1[word] for word in data1]
data1_spaced = []
compressed_data1_spaced = []

for block, compressed_block in zip(data1, compressed_data1):
    diff = len(block) - len(compressed_block)
    if diff >= 0:
        blanks = (' ' * diff)
        data1_spaced.append(block)
        compressed_data1_spaced.append(blanks + compressed_block)
    else:
        blanks = (' ' * -diff)
        data1_spaced.append(blanks + block)
        compressed_data1_spaced.append(compressed_block)
    
print('Data 1:                              ', data1_spaced)
print('Compressed data 1:                   ', compressed_data1_spaced)
print('Number of bits in data 1:            ', len(data1_string))
print('Number of bits in compressed data 1: ', len(''.join(compressed_data1)))
print('(reciprocal) Compression ratio 1:   ', len(''.join(compressed_data1)) / len(data1_string))
```

Notice that the code words all have length $5$, resulting in a $1$ to $1$ compression ratio.

Continue later...





## KL divergence, entropy, and cross entropy

The types of measures $P$ and $Q$ that we shall work with initially are ones defined on a finite probability space $S$, so that they have mass functions $p(s)$ and $q(s)$ with finite support. The basic measure that we use in this chapter to compare them is the mean logarithmic relative magnitude. 

```{margin}

Of course, the two notions of _absolute relative magnitude_ and _logarithmic relative magnitude_ make sense for any pair of numbers, not necessarily probabilities.
```

Precisely, the _absolute relative magnitude_ of the probability $p(s)$ to the probability $q(s)$ ordinarily refers to the ratio $p(s)/q(s)$, while the _logarithmic relative magnitude_ refers to the base-$10$ logarithm of the absolute relative magnitude:

$$
\log_{10}\left( \frac{p(s)}{q(s)} \right).
$$

The intuition for this number is that it is the _order_ of the absolute relative magnitude; indeed, if we have $p(s) \approx 10^k$ and $q(s) \approx 10^l$, then the logarithmic relative magnitude is roughly the difference $k-l$.

Perhaps the most obvious immediate benefit of introducing the logarithm is that it yields a workable number when $p(s)$ and $q(s)$ are each on different scales. For example, let's suppose that the mass functions $p(s)$ and $q(s)$ are given by

$$
p(s) = \binom{10}{s} (0.4)^s(0.6)^{10-s} \quad \text{and} \quad q(s) = \binom{10}{s} (0.9)^s(0.1)^{10-s}
$$

for $s\in \{0,1,\ldots,10\}$; these are the mass functions of a $\Bin(10,0.4)$ and $\Bin(10,0.9)$ random variable, respectively. We then plot histograms for these mass functions, along with histograms of the absolute and logarithmic relative magnitudes:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center

grid = np.arange(0, 11)
p = sp.stats.binom(n=10, p=0.4).pmf(grid)
q = sp.stats.binom(n=10, p=0.9).pmf(grid)
titles = ['$p(s)$',
          '$q(s)$',
          '$\\frac{p(s)}{q(s)}$',
          '$\\log_{10}\\left(\\frac{p(s)}{q(s)}\\right)$']
probs = [p,
         q,
         p / q,
         np.log10(p / q)]
ylims = [(0, 0.4),
         (0, 0.4),
         (-50, 0.75e8),
         (-5, 10)]

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5), sharex=True)

for title, prob, ylim, axis in zip(titles, probs, ylims, axes.flatten()):
    axis.bar(grid, prob, width=0.4)
    axis.set_xticks(ticks=grid)
    axis.set_ylim(ylim)
    axis.set_title(title)
    axis.set_xlabel('$s$')

plt.tight_layout()
```

The second row in the figure drives home the point: The absolute relative magnitudes are on such widely different scales that the plot is nearly useless and numerical computations in a machine will likely be unstable.

We obtain a single-number summary of the logarithmic relative magnitudes by averaging with weights drawn from the mass function $p(s)$; this yields the number

$$
\sum_{s\in S} p(s) \log_{10}\left( \frac{p(s)}{q(s)} \right).
$$ (first-kl-eq)

Observe that we could have drawn the averaging weights from the mass function $q(s)$ to instead obtain the single-number summary

$$
\sum_{s\in S} q(s) \log_{10}\left( \frac{p(s)}{q(s)} \right).
$$ (second-kl-eq)

But observe that

$$
\sum_{s\in S} q(s) \log_{10}\left( \frac{p(s)}{q(s)} \right) = - \sum_{s\in S} q(s) \log_{10}\left( \frac{q(s)}{p(s)} \right),
$$

where the right-hand side is the negative of a number of the form {eq}`first-kl-eq`. So, at least up to sign, it doesn't really matter which of the two numbers {eq}`first-kl-eq` or {eq}`second-kl-eq` we use to develop our theory. As we will see, our choice of {eq}`first-kl-eq` has the benefit of making the KL divergence nonnegative. Moreover, we can also alter the base of the logarithm in {eq}`first-kl-eq` without altering the core of the theory, since the change-of-base formula for logarithms tells us that the only difference is a multiplicative constant. In the following official definition, we will select the base-$2$ logarithm to make the later connections with bit strings in coding theory more transparent.

```{prf:definition}
:label: KL-def

Let $P$ and $Q$ be two probability measures on a finite probability space $S$ with mass functions $p(s)$ and $q(s)$. Then the _Kullback-Leibler divergence_ (or just _KL divergence_) from $P$ to $Q$, denoted $D(P \parallel Q)$, is the mean order of relative magnitude of $P$ to $Q$. Precisely, it is given by

$$
D(P \parallel Q) \def \sum_{s\in S} p(s) \log_2\left( \frac{p(s)}{q(s)} \right).
$$ (kl-eq)
```

Since averages of the form {eq}`kl-eq` will reoccur so often in the next few chapters, it will be convenient to introduce new notations for them. So, if $P$ is a discrete probability measure with mass function $p(s)$ on a sample space $S$ and $g:S\to \bbr$ is a real-valued function, we will define

$$
E_P(g(s)) \def \sum_{s\in S} g(s) p(s).
$$

Alternatively, if we want to explicitly call attention to the mass function $p(s)$ rather than the probability measure $P$ itself, we will write

$$
E_{s\sim p(s)}(g(s)) \def \sum_{s\in S} g(s) p(s).
$$

We refer to these sums as the _mean value_ or _expected value_ of $g(s)$. Note that these are legitimately new usages of the expectation symbol $E$, since there is no random variable given _a priori_. To see the connection with the previous usage of $E$ for a discrete random variable $X$ with mass function $p_X(x)$, suppose that $g:\bbr \to \bbr$ and note

$$
E_{P_X}(g(x)) = \sum_{x\in \bbr}g(x) p_X(x) = E(g(X)).
$$

Indeed, the first equality follows from the definition of $E_{P_X}(g(x))$ given above, while the second equality follows from the LotUS. Therefore, using this new notation, we may rewrite {eq}`kl-eq` as

$$
D(P \parallel Q) = E_P\left[ \log_2\left(\frac{p(s)}{q(s)}\right)\right] = E_{s\sim p(s)} \left[ \log_2\left(\frac{p(s)}{q(s)}\right)\right].
$$

Notice again that the mean is with respect to $P$. This breaks the symmetry between $P$ and $Q$ so that

$$
D( P \parallel Q) \neq D(Q \parallel P),
$$

except in special cases. Problems are encountered in a strict interpretation of the formula {eq}`kl-eq` when one or the other (or both) of the mass functions are $0$. In these cases, it is conventional to define:

$$
p \log_2\left( \frac{p}{q} \right) = \begin{cases}
0 & : p =0, \ q=0, \\
0 & : p = 0, \ q\neq 0, \\
\infty & : p \neq 0, \ q=0.
\end{cases}
$$

Let's take a look at an example problem:


```{admonition} Problem Prompt

To problem 1 on the worksheet.
```

The KL divergence turns out to decompose into a sum of two of the most important quantities in information theory:

```{prf:definition}
:label: entropy-def

Let $P$ and $Q$ be two probability measures on a finite probability space $S$ with mass functions $p(s)$ and $q(s)$.

1. The _cross entropy_ from $P$ to $Q$, denoted $H(P \parallel Q)$, is the number defined by

  $$
  H(P \parallel Q) \def - \sum_{s\in S} p(s)\log_2(q(s)).
  $$

2. The _entropy_ of $P$, denoted $H(P)$, is the number given by

    $$
    H(P) \def - \sum_{s\in S} p(s) \log_2(p(s)).
    $$

When $\bX$ is a random vector with finite range, we shall often write $H(\bX)$ in place of $H(P_\bX)$.
```

```{admonition} Problem Prompt

Do problem 2 on the worksheet.
```


The connection between these entropies and the KL divergence is given in the next theorem. Its proof is a triviality.

```{prf:theorem} KL divergence and entropy
:label: KL-and-entropy-thm

Let $P$ and $Q$ be two probability measures on a finite probability space $S$. Then

$$
D(P\parallel Q) = H(P \parallel Q) - H(P).
$$
```

The inequality in the first part of the following result is perhaps the most important in the foundations of the theory and ultimately justifies our conception of the KL divergence as a "directed distance" between two probability distributions. The second part shows that the maximum-entropy distributions are exactly the uniform ones.

```{prf:theorem} Optimization of KL divergences and entropies
:label: kl-entropy-optim-thm

Let $P$ and $Q$ be two probability measures on a finite probability space $S$.

1. _The Information Inequality_. We have

    $$
    D(P \parallel Q) = H(P \parallel Q) - H(P) \geq 0
    $$

    for all $P$ and $Q$, with equality if and only if $P=Q$.

2. We have 

    $$
    H(P) \leq \log_2{|S|}
    $$

    for all $P$, with equality if and only if $P$ is uniform.
```

```{prf:proof}
For the first part, suppose that $p_1,\ldots,p_n$ and $q_1,\ldots,q_n$ are numbers in $(0,1]$ such that

$$
\sum_{i=1}^n p_i = \sum_{i=1}^n q_i = 1.
$$ (constraint-lagrance-eq)

It will suffice, then, to show that the objective function

$$
J(q_1,\ldots,q_n) \def -\sum_{i=1}^n p_i \log_2{q_i},
$$

is globally minimized when $p_i = q_i$. But it is an easy exercise (using Lagrange multipliers) to show that a minimum can only occur when $p_i = q_i$ for each $i=1,\ldots,n$; one may confirm that this indeed yields a global maximum by showing that the objective function $J$ is convex (its Hessian matrix is positive definite) and noticing that the second constraint in {eq}`constraint-lagrance-eq` is affine. (See [here](https://math.stackexchange.com/a/1739181) for an explanation of the latter fact.) The proof of the second part follows the same pattern, with only the obvious changes. Q.E.D.
```

So, when $P$ is uniform, we have

$$
H(P) = \log_2|S|.
$$ (max-ent-eq)

It is pleasing to compare this maximum-entropy equation to the [Boltzmann equation](https://en.wikipedia.org/wiki/Boltzmann%27s_entropy_formula) for entropy in statistical mechanics. The definitional equation

$$
H(P) = - \sum_{s\in S} p(s) \log_2(p(s))
$$

is the analog of the [Gibbs equation](https://en.wikipedia.org/wiki/Entropy_(statistical_thermodynamics)#Gibbs_entropy_formula) for Boltzmann entropy.

In his initial paper, Shannon described entropy $H(P)$ as a measure of _uncertainty_. From this perspective, the rationale behind the maximum-entropy equation {eq}`max-ent-eq` becomes clear: If one were to randomly draw a number from a probability distribution, the uniform distribution is the one that would result in the highest level of uncertainty regarding the outcome.

















## Conditional entropy, mutual information
























## Source coding

We now describe a coding-theoretic interpretation that sheds additional light on entropy and KL divergence. Rather than quantifying the degree of "uncertainty" present in a probability distribution, in this framework entropy gives a lower bound on the (average) minimum description length of the data modeled by a random variable or vector.

```{margin}

According to our definitions, technically the codomain of a random variable must be a subset of $\bbr$; but we can get around this minor annoyance by assuming that $a=1$, $b=2$, $c=3$, and $d=4$.
```

By way of introduction, suppose that $X$ is a discrete random variable with range $R = \{a,b,c,d\}$. Our goal is to construct an _encoding_ of $X$, by which we mean an assignment of a bit string to each symbol in $R$. For example, we might encode $X$ as

$$
a \leftrightarrow 00, \quad b \leftrightarrow 011, \quad c \leftrightarrow 10, \quad d \leftrightarrow 1100.
$$ (encoding-eqn)

We can _visualize_ this encoding by drawing a binary tree with five levels (including the root):

```{image} ../img/tree-01.svg
:width: 90%
:align: center
```
&nbsp;

To read this tree, begin at the root node at the top; then, follow the edges downward to find the nodes labeled by the symbols in $R$. A positively sloped edge represents a $0$, while a negatively edge represents a $1$. Thus, for example, to reach $d$ beginning from the root node, we follow edges labelled $1$, $1$, $0$, and $0$. This sequence of binary digits is exactly the code word for $d$, and thus paths through the tree represent code words. The numbered levels $\ell$ of the tree appear along the left-hand side of the figure; notice that these numbers are also the lengths of the code words.

Notice also that every path through the tree beginning at the root node and ending at a leaf in the lowest level contains at most one symbol in $R$. This is in contrast to the encoding of $X$ represented by the following tree:

```{image} ../img/tree-bad.svg
:width: 90%
:align: center
```
&nbsp;

with corresponding code words

$$
a \leftrightarrow 00, \quad b \leftrightarrow 001, \quad c \leftrightarrow 10, \quad d \leftrightarrow 1000.
$$

Indeed, in this latter encoding, there are _multiple_ paths from the top to the bottom level that contain more than one symbol in $R$. These paths manifest themselves as code words that are prefixes of other code words: The code word for $a$ appears as a prefix in the code word for $b$, and the code word for $c$ appears as a prefix in the code word for $d$. For this reason, encodings like the first {eq}`encoding-eqn` are called _prefix-free codes_.

Now, returning to our prefix-free code, consider the set of all descendants of symbols in $R$ that are in the lowest level, including any symbols in $R$ that happen to lie in the lowest level; these are all highlighted in:

```{image} ../img/tree-02.svg
:width: 90%
:align: center
```
&nbsp;

If a symbol in $R$ is on level $\ell_i$, then it has $4 - \ell_i$ descendents in the lowest level. Then obviously $\sum_{i=1}^4 2^{4-\ell_i} \leq 2^{4}$ (count the highlighted nodes!), and so

$$
\sum_{i=1}^4 2^{-\ell_i} \leq 1.
$$

In fact, this latter inequality should _always_ be true for _any_ encoding of $X$, provided that the code is prefix free. Can you see why?







## Conditional entropy and mutual information



```{prf:definition}
:label: mutual-info-def

Let $\bX$ and $\bY$ be two random vectors with finite ranges. The *mutual information between $\bX$ and $\bY$* is the KL divergence

$$
I(\bX, \bY) \def D( P_{(\bX,\bY)} \parallel P_{\bX} P_{\bY}).
$$
```

```{prf:theorem} Mutual information and entropy
:label: other-info-thm

Let $\bX$ and $\bY$ be two random vectors with finite ranges. Then:

$$
I(\bX,\bY) = H(\bX) + H(\bY) - H(\bX,\bY).
$$
```

```{prf:proof}

The proof is a computation:

\begin{align*}
I(\bX,\bY) &= \sum_{\bx\in \bbr^n}\sum_{\by \in \bbr^m} p(\bx,\by) \log_2\left( \frac{p(\bx,\by)}{p(\bx)p(\by)} \right) \\
&= \sum_{\bx\in \bbr^n}\sum_{\by \in \bbr^m} p(\bx,\by) \log_2\left(p(\bx,\by)\right) - \sum_{\bx\in \bbr^n}\sum_{\by \in \bbr^m} p(\bx,\by) \log_2 \left(p(\bx)\right) \\
&\quad - \sum_{\bx\in \bbr^n}\sum_{\by \in \bbr^m} p(\bx,\by) \log_2\left(p(\by)\right) \\
&= - H(\bX,\bY) - \sum_{\bx \in \bbr^n} p(\bx) \log_2\left( p(\bx) \right) - \sum_{\by \in \bbr^m} p(\by) \log_2\left( p(\by)\right) \\
&= H(\bX) + H(\bY) - H(\bX, \bY),
\end{align*}

as desired. Q.E.D.
```