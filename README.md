# PARALAAOMPI
A simple parallel implementation of the all-at-once method.

The all-at-once method is outlined in the paper 

    "Preconditioning and iterative solution of all-at-once systems for evolutionary partial differential equations"
        by McDonald, Pestana, Wathen.

In this paper it is claimed that the all-at-once method is a parallel method. This code seeks to be a proof-of-concept
code. This code is my first attempt at parallel computing and as a result there are many things that need to be improved. 

I would be interested in writing a more general (and improved) implementation if the interest is there. 

The results from this code have been submitted for review:

https://arxiv.org/abs/1810.00615

There has also been some work on a GPU implementation and an application to non-linear systems which I include as techreport.pdf in this repo.
