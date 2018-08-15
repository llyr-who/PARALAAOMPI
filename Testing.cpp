/*
 *  Filename: Testing.cpp
 *
 *  This file is used for testing purposes. 
 *
 *  Mainly to test improvements and changes to the parallel rotuines.
 *
 */


#include <iostream>
#include <iomanip>
#include <mpi.h>
#include "MatrixHelper.h"
#include "ParallelRoutines.h"


int main(int argc, char * argv[])
{


//  DECLARATION OF VARIABLES
    double start,end;           /* Used in timing the GMRES routine */
    int i,j,k, N = 4,L=4;   /* N is the number of spatial steps, L is the number of time steps */
    double h = 1.0/(N-1);       /* The size of the spatial discretisaion step */ 
    double timestep = 1.0/L;    /* Timestep length */
    double delta = 0.0;         /* Perturbation of temproal domain */


    std::complex<double> *A,*x,*q,*y,*pointertolargeblocked,*b;
    std::complex<double> *massContig,*stiffContig;
    std::complex<double> *F,*D,*Ft;

    std::vector<double> timesteps,times,perts;

    int totalnodes,mynode;
    std::vector<std::complex<double>*> Wblocks,Ablocks;
    std::complex<double>** UMonolithic,**UtMonolithic;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    x = new std::complex<double>[N*L];
    y =  new std::complex<double>[N*L];
    q = new std::complex<double>[N];


        for(i=0;i<N*L;i++)
            x[i] = i*1.0;
        

    if(mynode==0)
    {
        for(i=0;i<N;i++)
            q[i] = i*1.0;

        FFT(q,N,0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MultiplicationByIdentityKronU_usingFFT(mynode,totalnodes,N,L,x,y,1);
    MultiplicationByIdentityKronU_usingFFT(mynode,totalnodes,N,L,y,x,0);
    MPI_Barrier(MPI_COMM_WORLD);
    if(mynode == 0)
    {

        for(i=0;i<N*L;i++)
            std::cout << x[i] << " " <<  y[i] << std::endl;

        for(i=0;i<N;i++)
            std::cout << q[i] << std::endl;
    }

    MPI_Finalize();
}
