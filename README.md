# PARALAAOMPI
A simple parallel implementation of the all-at-once method.

The all-at-once method is outlined in the paper 

    "Preconditioning and iterative solution of all-at-once systems for evolutionary partial differential equations"
        by McDonald, Pestana, Wathen.

In this paper it is claimed that the all-at-once method is a parallel method. This code seeks to be a proof-of-concept
code. This code is my first attempt at parallel computing and as a result there are many things that need to be improved. 

I would be interested in writing a more general (and improved) implementation if the interest is there. 

The results from this have been published in ETNA : 

http://etna.mcs.kent.edu/volumes/2011-2020/vol51/abstract.php?vol=51&pages=135-150

There has also been some work on a GPU implementation and an application to non-linear systems which I include as techreport.pdf in this repo. 

As this code was a proof-of-concept code designed and written under very tight time contraints the solver is not modular
in any way. In order to run this code you will have to choose which equation you would like to solve, the heat equation
or the wave equation. Open the makefile and select which one you would like to solve. Then type

    make

at the command line.

Then to run this code

    mpirun -np 4 paralaaompi -N 128 -L 32

where you are requesting 4 processes, a spatial discretisaion of 128 nodes and a temporal discretisation of 32 nodes. If
there are any problems then either create an issue on GitHub or contact me on antg3254 - AT - gmail.com


