# PARALAAOMPI
A simple parallel implementation of the all-at-once method.

The all-at-once method is outlined in the paper 

    "Preconditioning and iterative solution of all-at-once systems for evolutionary partial differential equations"
        by McDonald, Pestana, Wathen.

In this paper it is claimed that the all-at-once method is a parallel method. This code seeks to be a proof-of-concept
code. There are a few improvements suggested:

    (1) Merge wave equation and heat equation implementations to one file and modularise the formation
        of "W" and "A". These files literally differ by a few lines. 

    (2) Move GMRES functions to a separate file.

    (3) With some effort, this code can be generalised.
        I hope create some software with MassStiff (another project of mine)
        that will enable the user to give only the domain and the order of
        the time dervaitve approximation of the heat or wave equation.
        The monolithic systems can then be automatically generated and
        solved using GMRES.

There have been some recent improvements made that reduce the execution time by ~30% compared to the timed
results provided in the paper "Wathen and Goddard".
