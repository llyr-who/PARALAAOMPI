# PARALAAOMPI
A simple parallel implementation of the all-at-once method.

The all-at-once method is outlined in the paper 

    "Preconditioning and iterative solution of all-at-once systems for evolutionary partial differential equations"

In this paper it is claimed that the all-at-once method is a parallel method. This code seeks to be a proof-of-concept
code. There are a few improvements suggested (I need to make):

    (1) Merge wave equation and heat equation implementations to one file and modularise the formation
        of "W" and "A". These files literally differ by a few lines. 

    (2) Move GMRES functions to a separate file.


    (3) Add more comments to ParallelRoutines.cpp

