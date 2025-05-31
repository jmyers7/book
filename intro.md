---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(intro)=
# Probability Theory with a View Toward Machine Learning

This textbook is the result of my search for a book that combines a mathematically rigorous treatment of probability theory with applications in machine learning and coding in Python: I could not find such a textbook, so I wrote one.

The intended audience for the book is upper-level undergraduate students and first-year graduate students specializing in mathematics, statistics, computer science, and other STEM disciplines that make heavy use of probabilistic concepts and machine learning.

* On the mathematical side, I assume that the reader has completed the calculus sequence typical at most US universities, up to and including multi-variable calculus. I also make good use of linear algebra in the middle chapters. The reader should know the basics of vectors, matrices, determinants, and eigenvalues and eigenvectors. Other concepts from linear algebra will be introduced when they are needed.

* On the programming side, I assume that the reader knows only the basics of Python and writing code in Jupyter notebooks. Nothing advanced is needed---just a familiarity with the fundamental data structures in the language, as well as basic logic, control flow, and functions. The big libraries (e.g., NumPy, pandas, matplotlib, seaborn, SciPy, PyTorch, scikit-learn, _etc_.) will be introduced throughout the course.

## How to use this book

The supporting material for this book is contained in a public GitHub repository linked [here](https://github.com/jmyers7/stats-book-materials). This repository is also accessible from any page in the book by clicking the GitHub link near the top.

The book is written to suit how I _actually teach_ in the classroom:

* Each chapter is divided into sections. Depending on length, a few sections might be covered each class period.

* The sections contain boxes for all the important definitions and theorems connected with exposition and explanation, just as one would expect. However, the text itself contains almost no examples or worked problems; in their place, the reader will discover many boxes like this:

    ```{admonition} Problem Prompt

    Do problems 8-9 on the worksheet.
    ```

    The worksheets that these Problem Prompts point to are contained in the GitHub repository I linked above. There is one worksheet per chapter, they are printed out and distributed to the students, and we complete them together as a class. By handing out worksheets with example problems printed out, I don't have to waste time in writing problem statements on the board.
    
* The public repository also contains solutions to the worksheets, but these tend to be quite concise. The brevity of these written solutions is mitigated by the fact that the students and I will have discussed the worksheet problems in depth during class.

* Every line of Python code used to produce the content in the book is included for the reader in drop-down code cells; for example:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:   figure:
:       align: center
:   image:
:       width: 70%

print('Hello, world!')
```

* However, I believe the task of learning the mathematical theory _and_ its computer implementation is made more difficult if both are attempted to be learned _simultaneously_. Therefore, the textbook itself does not explain the Python implementations; rather, these are left to the programming assignments in the Jupyter notebooks contained in the GitHub repository. These notebooks contain ample guidance in the markdown cells, which is also supplemented by in-class demonstrations during which I introduce the programming assignments.

## A description of the content matter

The first four chapters of the book cover the basics of abstract probability theory. Through real-world datasets and their associated empirical distributions, the fifth chapter provides a respite from the deluge of theory in the early chapters and marks the first time in the book that probabilistic models are introduced. The sixth and seventh chapters tie up some probabilistic loose ends.

Except for the Python code and (perhaps) my special emphasis on mathematical rigor, there is not much that differentiates the first seven chapters of this book from other texts on probability theory. But, beginning with the eighth chapter, probability theory moves to a supporting role while probabilistic models become the main objects of study. The novelty in my treatment is that these models are studied from the perspective of a (mathematically inclined) machine learning engineer, rather than from a statistician's perspective. This difference in perspective accounts for the ninth chapter on information theory, where the reader will find treatments of Shannon entropy and information, along with the fundamental concept of Kullback-Leibler divergence.

The overarching goal of the tenth chapter is a description of the foundational gradient descent algorithms, including the stochastic version. The treatment does not shy away from mathematical details and thus it might be the most difficult chapter in the book. The reader can expect to thoroughly exercise their understanding of multivariable calculus and linear algebra. (This being said, when I taught this chapter to a real class I skipped many technical details.)

The eleventh chapter describes a trio of the most basic families of probabilistic models encountered in machine learning: Linear and logistic regression models, as well as (fully-connected, feedforward) neural networks. Mixture models and Naive Bayes models appear in the homework and programming assignment for this chapter. All these models are presented in the context of probabilistic **graphical** models (PGMs)---the expert reader who knows something of PGMs from other textbooks will recognize that there are some differences in my definitions and terminology. Beware!

Finally, in the culminating twelfth chapter of the book, we get down to the business of training PGMs. The main theoretical result of the chapter is a proof of the equivalence of four training objectives: (1) Minimizing Kullback-Leibler divergence, (2) minimizing cross entropy, (3) minimizing surprisal, (4) maximizing likelihood. Following this result, the reader will see concrete examples of how these training objectives can be attacked via the gradient descent algorithms introduced earlier in the book. In the programming assignment for the chapter, the reader will try their hand at training a Naive Bayes model to classify spam emails.

---

Even if nobody but myself uses the _entire_ book in their classroom or for self study, I hope at least some of its content proves to be useful to somebody somewhere! If you _do_ use something, I would appreciate a link back to this webpage or a citation of some sort. I would also appreciate any feedback, comments and suggestions that you might have!

--- john myers, ph.d. \ [johnmyersmath.com](https://www.johnmyersmath.com/) \ <a href = "mailto: jmmyers25@gmail.com">email</a>