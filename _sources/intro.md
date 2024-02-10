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
# Mathematical Statistics with a View Toward Machine Learning

```{image} ./img/wallpaper4.png
:width: 100%
:align: center
```
&nbsp;

This textbook is the result of my search for a book that combines a mathematically rigorous treatment of probability and statistics with applications in machine learning and coding in Python: I could not find such a textbook, so I wrote one. (Or, depending on when you are reading this, I am in the _process_ of writing one.)

The intended audience is upper-level undergraduate students majoring in mathematics, statistics, computer science, and other STEM disciplines that make heavy use of statistical concepts and techniques.

* On the mathematical side, I assume that the reader has completed the calculus sequence typical at most US universities, up to and including multi-variable calculus. We also make good use of linear algebra in the middle chapters. The reader should know the basics of vectors, matrices, determinants, and eigenvalues and eigenvectors. Other concepts from linear algebra will be introduced when they are needed.

* On the programming side, I assume that the reader knows only the basics of Python and writing code in Jupyter notebooks. Nothing advanced is needed---just a familiarity with the fundamental data structures in the language, as well as basic logic, control flow, and functions. The big libraries (e.g., NumPy, pandas, matplotlib, seaborn, SciPy, PyTorch, scikit-learn, statsmodels, _etc_.) will be introduced throughout the course.

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

## References and further reading

Besides the text itself, much of the intellectual effort in this endeavor has been directed toward the programming assignments which are all developed and written from scratch. Therefore, in order to expedite writing, I have had to borrow significantly from other textbooks for the worksheet problems. These other texts include those written by:

* {cite:ts}`DeGrootSchervish2014`,
* {cite:ts}`Dekking2005`,
* {cite:ts}`Devore2021`,
* {cite:ts}`Rice2006`,
* {cite:ts}`Wackerly2014`, and
* {cite:ts}`Wasserman2004`.

Depending on my future time commitments and motivation, I might return and compose more original worksheet problems. But that won't happen any time soon. In any case, I highly recommend each of these texts and their problem banks for further study.

---

Even if nobody but myself uses the _entire_ book in their classroom or for self study, I hope at least some of its content proves to be useful to somebody somewhere! If you _do_ use something, I would appreciate a link back to this webpage or a citation of some sort. I would also appreciate any feedback, comments and suggestions that you might have!

--- john myers, ph.d. \ [johnmyersmath.com](https://www.johnmyersmath.com/) \ [mml.johnmyersmath.com](https://mml.johnmyersmath.com/) \ <a href = "mailto: jmmyers25@gmail.com">email</a>