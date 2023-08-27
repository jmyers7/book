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

# Rules of probability

Having learned the basic theory of probability in the previous chapter, in this chapter we begin developing various tools and techniques that will allow us to actually _compute_ probabilities.

The first two sections in this chapter deal with techniques for _counting_, and are more widely applicable than just computing probabilities. Indeed, you might have seen some of this material in a course in discrete mathematics or combinatorics. We will treat this material with a much lighter touch than most elementary probability textbooks, believing that these techniques are not _as important_ as many those texts might lead you to believe.

The rest of the chapter is devoted to probability theory in which, among other things, we introduce the fundamental topics of _conditioning_ and _independence_, as well as the _Law of Total Probability_ and _Bayes' Theorem_. All four of these topics are absolutely crucial to the rest of the book, and we will return to them again in the context of random variables in a [later](random-vectors) chapter.











## The Product and Sum Rules for Counting

The main rules that we will use for counting are:

```{prf:theorem} The Product Rule for Counting

Suppose that a procedure can be broken down into a sequence of two tasks. If there are $m$ ways to do the first task, and for each of these ways of doing the first task there are $n$ ways to do the second task, then there are $mn$ total ways to do the procedure.
```

```{prf:theorem} The Sum Rule for Counting

If a procedure can be done either in one of $m$ ways or in one of $n$ ways, where none of the set of $m$ ways is the same as any of the set of $n$ ways, then there are $m+n$ ways to do the procedure.
```

Let's get some practice:

```{admonition} Problem Prompt
Do problem 1 on the worksheet.
```















## Permutations and combinations

The Product Rule for Counting tells us how to count the number of ways to accomplish a task that can be broken down into a sequence of smaller, sub-tasks. One very simple (yet very important) example of such a task is choosing, in order, a collection of objects from a larger collection of objects. Such collections have names:

```{prf:definition}

An ordered collection of $k$ distinct objects is called a _permutation_ of those objects. The number of permutations of $k$ objects selected from a collection of $n$ objects will be designated by the symbol $P^n_k$.
```

Mathematically, a permutation consisting of $k$ objects is often represented as $k$-tuple. This is because, for a permutation, the ordering of the objects **matters**. But order matters in $k$-tuples as well; for example, the $2$-tuple $(1,2)$ is _not_ the same as the $2$-tuple $(2,1)$.


```{image} ../img/permutation.svg
:width: 75%
:align: center
```
&nbsp;

We will often need to count permutations:

```{prf:theorem} Formula for Counting Permutations

For $0 \leq k \leq n$, we have

$$P^n_k = \frac{n!}{(n-k)!}.$$

_Note_: Remember the convention that $0! = 1$.
```

Why is this true? Remember that a permutation of $k$ objects may be represented as a $k$-tuple:

$$(\text{object $1$},\text{object $2$},\ldots, \text{object $k$}).$$

But then, written like this, we see that counting the number of permutations is a simple application of the Product Rule for Counting. Indeed, the number of permutations is the product

````{div} full-width
$$(\text{# ways to choose object $1$}) \times (\text{# ways to choose object $2$}) \times \cdots \times (\text{# ways to choose object $k$}),$$
````

which is equal to

$$n \times (n-1) \times (n-2) \times \cdots \times (n-k+1) = \frac{n!}{(n-k)!}.$$

```{admonition} Problem Prompt
Do problem 2 on the worksheet.
```

The partners to permutations are defined in:

```{prf:definition}

An unordered collection of $k$ distinct objects is called a _combination_ of those objects. The number of combinations of $k$ objects selected from a collection of $n$ objects will be designated by the symbol $C^n_k$ or $\binom{n}{k}$.
```

Permutations are represented as $k$-tuples because order matters. But for combinations, order does **not** matter, and therefore they are often represented as sets:

```{image} ../img/combination.svg
:width: 75%
:align: center
```
&nbsp;


```{prf:theorem} Formula for Counting Combinations

For $0 \leq k \leq n$, we have

$$\binom{n}{k} = C^n_k = \frac{n!}{(n-k)!k!}.$$

_Note_: Remember the convention that $0! = 1$.
```

To see why this formula works, think of the process of forming a *permutation* as a two-step process:

1. Choose a combination.
2. Rearrange the objects in your chosen combination to fit the order of your desired permutation.

Then, it follows from the Product Rule for Counting that the total number of permutations, $P^n_k$, is the product of the number of ways to accomplish the first task with the number of ways to accomplish the second task. In an equation, this means:

$$P^n_k = C^n_k \cdot P^k_k.$$

Indeed, on the right-hand side of this equation you see $C^n_k$, which is the number of ways to accomplish the first task, while $P^k_k$ represents the number of permutations of $k$ objects selected from a total number of $k$ objects (which are just rearrangements of those objects). But we have

$$P^n_k = \frac{n!}{(n-k)!} \quad \text{and} \quad P^k_k = k!,$$

which means that we must have

$$C^n_k = \frac{ n!}{(n-k)!k!}.$$

But this is exactly the desired formula for $C^n_k$!

```{admonition} Problem Prompt
Do problem 3 on the worksheet.
```


















## The Sum Rule for Probability

We now transition back to talking about probability theory.


```{prf:theorem} The Sum Rule for Probability

The probability of the union of two events $A$ and $B$ is

$$P(A \cup B) = P(A) + P(B) - P(A\cap B).$$

In particular, if $A$ and $B$ are disjoint, then

$$P(A\cup B) = P(A) + P(B).$$
```

So, the Sum Rule for Probability tells you how to compute the probability of a *union* of two events. We will see [later](prod-rule-prob) a different rule that tells you how to compute the probability of an *intersection*.

In the very special case that the sample space $S$ is finite and the probability measure $P$ is uniform, I claim that the Sum Rule for Probability is "obvious."

Let me explain.

Notice that we have

$$|A\cup B| = |A| + |B| - |A\cap B|,$$(inc-ex-eqn)

where the vertical bars denote cardinalities. Can you see why this is true? If you want to count the number of points in the union $A\cup B$, then you count all the points in $A$, then all the points in $B$, but you have to subtract out the points in $A\cap B$ because they would be double counted (this is called [inclusion-exclusion](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle)). 

Now, if we divide both sides of {eq}`inc-ex-eqn` by the cardinality $|S|$, we get

$$\frac{|A\cup B|}{|S|} = \frac{|A|}{|S|} + \frac{|B|}{|S|} - \frac{|A\cap B|}{|S|}.$$

But because we are working with a uniform probability measure $P$, each of these four ratios _are_ the probabilities:

$$P(A \cup B) = P(A) + P(B) - P(A\cap B).$$

Simple!

But what if $P$ is _not_ uniform? Why is the Sum Rule still true?

To explain, imagine the usual Venn diagram with the events $A$ and $B$ inside the sample space S:

```{image} ../img/eventssum.svg
:width: 25%
:align: center
```
&nbsp;

The argument that I am about to give depends crucially on the two equations

$$A\cup B = A \cup ( A^c \cap B) \quad \text{and} \quad B = (A^c \cap B) \cup ( A\cap B),$$(fund-eqn)

where for brevity I've written $A^c$ for the complement $S\smallsetminus A$. If you're not quite sure why these equations are true, here's a visual explanation using Venn diagrams:

```{image} ../img/sumrule.svg
:width: 70%
:align: center
```
&nbsp;

The first row of this figure simply shows you how to visualize the intersection $A^c \cap B$, and the second two rows explain the fundamental equations {eq}`fund-eqn` above. The crucial point to notice is that the two pairs of sets on the right-hand sides of the equations {eq}`fund-eqn` are _disjoint_, which you can see easily from the Venn diagrams. Thus, if we apply $P$ to both sides of these equations, we get:

$$P(A\cup B) = P(A) + P(A^c\cap B) \quad \text{and} \quad P(B) = P(A^c \cap B) +P(A\cap B).$$

Now subtract these equations to get

$$P(A\cup B) - P(B) = P(A) + P(A^c\cap B) - (P(A^c \cap B) + P(A\cap B)),$$

which we may rearrange to get the Sum Rule:

$$P(A\cup B) = P(A) + P(B) - P(A\cap B).$$

And there we have it! This explains why the Sum Rule works.

```{admonition} Problem Prompt
Have a go at problems 4 and 5 in the worksheet.
```



















(cond-prob)=
## Conditional probability

So far, we have studied just plain probabilities $P(A)$. Intuitively, this value represents the probability that you expect the event $A$ to occur, _in the absence of any additional given information_. But what if you actually _had_ additional information that might affect the probability of the event occurring? How would you update the probability $P(A)$?

Let's do an example. Suppose that you roll two fair, six-sided dice. The sample space $S$ in this scenario is the collection of all ordered pairs $(\text{roll 1}, \text{roll 2})$. Here's a picture:

```{image} ../img/conditionalSS.svg
:width: 45%
:align: center
```
&nbsp;

Since there are six possibilities for each roll, according to the Product Rule for Counting, the cardinality of the sample space is $6^2 = 36$. And, since each die is fair, any number is equally likely to be rolled as any other, and hence the probability measure is uniform: Each possible pair of numbers has probability $1/36$ of being rolled.

Now, suppose I ask you to compute the probability of the event $A$ that your two rolls together sum to 6. This computation is pretty easy, since you can just count the possibilities by hand:

$$(5,1), (4,2), (3,3), (2,4), (1,5). $$(who-eqn)

We may visualize the situation like this:

```{image} ../img/conditionalrolls.svg
:width: 85%
:align: center
```
&nbsp;

The figure on the left shows all sums for all possible pair of rolls in the sample space $S$, while the figure on the right shows the event $A$. Since the probability is uniform, we have

$$P(A) = \frac{5}{36} \approx 0.139.$$

However, what if I _now_ tell you that your first roll is an odd number---does this affect your expected probability for $A$?

_It does_!

To see why, let's first give the event that the first roll is odd a name, say $B$. Then we're not interested in the _full_ event $A$ anymore, since it includes the rolls $(4,2)$ and $(2,4)$. Rather, we are _now_ interested in the intersection $A\cap B$:

```{image} ../img/conditionalrollsInt.svg
:width: 85%
:align: center
```
&nbsp;

The intersection $A\cap B$ is where the blue horizontal highlights (representing $B$) intersect the magenta diagonal highlight (representating $A$). But there are three pairs of rolls in this intersection, and the cardinality of $B$ is $18$, so our probability of $A$ occuring changes from $P(A) = 5/36$, to $3/18$. But notice that $3/18$ is the _same_ as the ratio

$$\frac{P(A\cap B)}{P(B)}.$$

These considerations suggest the following fundamental definition:

```{prf:definition}

Let $A$ and $B$ be two events in a probability space $S$. Then the _conditional probability of $A$ given $B$_ is the ratio

$$P(A|B) \stackrel{\text{def}}{=} \frac{P(A\cap B)}{P(B)},$$

provided that $P(B) >0$. The event $B$ is called the _conditioning event_, and the conditional probability $P(A|B)$ is often called a probability _conditioned on $B$_.
```

In plain language, the conditional probability $P(A|B)$ represents the probability that $A$ will occur, _provided that you already know that event $B$ has occurred_. So, in our die-rolling scenario above where $A$ is the event that the sum of the two rolls is $6$, and $B$ is the event that the first roll is odd, we would write

$$ P(A|B) = \frac{3}{18}.$$

Conceptually, a conditional probability $P(A|B)$ may be thought of as collapsing the full probability space $S$ down to the event $B$. In our dice-roll scenario above, this collapse looks like:

```{image} ../img/collapse.svg
:width: 95%
:align: center
```
&nbsp;

```{admonition} Problem Prompt

Do problems 6 and 7 on the worksheet.
```





















(independence-first)=
## Independence

Sometimes the outcomes of two experiments or processes do not affect each other. For example, if I flip a single coin twice, then we would expect that the outcome of the first flip should have no effect on the outcome of the second flip. Such events are called _independent_, and they are defined formally in:

```{prf:definition}

Two events $A$ and $B$ in a sample space are _independent_ if

$$P(A\cap B) = P(A)P(B).$$ (ind2-eqn)
```

The definition of independence technically only applies to two events, but there's an obvious generalization of the definition to an arbitrary number of events. Indeed, we say that events $A_1,A_2,\ldots,A_n$ are _independent_ if

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1)P(A_2) \cdots P(A_n).$$

Here's a test for independence that involves conditional probabilities:

```{prf:theorem} Conditional Criterion for Independence

Two events $A$ and $B$ with nonzero probability are independent if and only if

$$P(A|B) = P(A) \quad \text{or} \quad P(B|A) = P(B).$$(ind-eqn)
```

Do you see why this criterion follows immediately from the definition {eq}`ind2-eqn` of independence and the definition of conditional probability? See if you can work it out for yourself.

Think about what the first equation in {eq}`ind-eqn` is attempting to tell you: It says that the probability that $A$ occurs, _given that $B$ has occurred_, is just the original probability $P(A)$. This means that whether $B$ has occurred or not has _no_ effect on whether $A$ occurs. This captures precisely the intuitive idea that the formal definition of independence {eq}`ind2-eqn` is attempting to get at.

```{admonition} Problem Prompt
Do problems 8 and 9 on the worksheet.
```
















(prod-rule-prob)=
## The Product Rule for Probability

Just as the Sum Rule for Probability expresses the probability of a union $A\cup B$ of two events as a sum of probabilities, the next rule expresses the probability of an intersection $A\cap B$ as a product of probabilities:

```{prf:theorem} The Product Rule for Probability

The probability of the intersection of two events $A$ and $B$ is

$$P(A \cap B) = P(A|B) P(B) = P(B|A)P(A).$$ (product-eqn)
```

Notice that the equations in {eq}`product-eqn` are immediate consequences of the definition of conditional probability, so there's not really any mystery for why they hold.




























(total-prob-bayes)=
## The Law of Total Probability and Bayes' Theorem

A *partition* of a sample space $S$ is exactly what it sounds like: It's a division of the sample space into a collection of disjoint subsets. For example, the six subsets $B_1,B_2,\ldots,B_6$ of the sample space $S$ in the following Venn diagram form a partition:

```{image} ../img/partition.svg
:width: 35%
:align: center
```
&nbsp;

There are two things required of a partition: All the sets in the partition must be pairwise disjoint, and their union must be the entire sample space $S$.

Using the concept of a partition, we may state the following law of probability:

```{prf:theorem} The Law of Total Probability

Suppose that $\{B_1,B_2,\ldots,B_n\}$ is a partition of a sample space $S$, where each $B_k$ is an event. Then for any event $A$, we have

$$P(A) = \sum_{k=1}^n P(A|B_k) P(B_k).$$
```

Why is this law true? Notice first that

$$A = \bigcup_{k=1}^n ( A\cap B_k),$$

and that the sets in the union on the right-hand side are pairwise disjoint. (Why?) Therefore, if we apply the probability measure $P$ to both sides of this last equation, we get

$$P(A) = \sum_{k=1}^n P(A\cap B_k).$$

But $P(A\cap B_k) = P(A|B_k) P(B_k)$ by the Product Rule for Probability, and hence we get

$$P(A) = \sum_{k=1}^n P(A|B_k) P(B_k),$$

which is what we wanted to prove.

Often times, the Law of Total Probability is applied when only *two* events are involved. In this case, the formula simplifies considerably:

```{prf:theorem} The Law of Total Probability (Two-Event Version)

Let $A$ and $B$ be two events in a sample space $S$. Then

$$P(A) = P(A|B)P(B) + P(A|B^c)P(B^c),$$

where I've written $B^c$ for $S \smallsetminus B $.
```

Indeed, this version of the Law of Total Probability is the special case of the first where the partition is $\{B,B^c\}$. Check this!

```{admonition} Problem Prompt
Do problem 10 on the worksheet.
```

Finally, our last result is following very important theorem:

```{prf:theorem} Bayes' Theorem

Let $A$ and $B$ be two events in a sample space $S$ with $P(B)>0$. Then

$$ P(A| B) = \frac{P(B|A)P(A)}{P(B)}.$$
```

Bayes' Theorem allows you to switch the order of conditioning: If you want to compute $P(A|B)$ but you only know $P(B|A)$, then Bayes' Theorem allows you to find the former, provided that you know $P(A)$ and $P(B)$.

```{admonition} Problem Prompt
Do problem 11 on the worksheet.
```