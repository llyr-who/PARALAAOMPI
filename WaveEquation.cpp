/* 
 *  Filename: WaveEquation.h
 *
 *  Author: Anthony Goddard. 
 *
 *  GitHub: anthonyjamesgoddard/PARALAAOMPI
 *
 *  Discription: 
 *
 *  This code is an implementation of the all-at-once formulation of the 
 *  wave equation. 
 * 
 */


#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <math.h>
#include "MatrixHelper.h"
#include "ParallelRoutines.h"

using namespace std;


// Declarations of routines used in GMRES
// (This will idealy be moved to a separate file in future)
void Update(std::complex<double>*x,int lengthofx, int k,int m, std::complex<double>*h,std::complex<double>*s,std::vector<std::complex<double>*>v);
void GeneratePlaneRotation(std::complex<double> &dx, std::complex<double> &dy, std::complex<double> &cs, std::complex<double> &sn);
void ApplyPlaneRotation(std::complex<double> &dx, std::complex<double> &dy, std::complex<double> &cs, std::complex<double> &sn);



int main(int argc, char * argv[])
{

//  DECLARATION OF VARIABLES
    double start,end;           /* Used in timing the GMRES routine */
    int i,j,k, N = 320,L=512;   /* N is the number of spatial steps, L is the number of time steps */
    double h = 1.0/(N-1);       /* The size of the spatial discretisaion step */ 
    double timestep = 1.0/L;    /* Timestep length */

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

// RESERVATION OF MEMORY

    // Intermediate calculation vectors
    y = new std::complex<double>[N*L];
    x = new std::complex<double>[N*L];
    q = new std::complex<double>[N*L];


    // Right hand side vector
    b = new std::complex<double>[N*L];


    // All vectors are given to all nodes. While this increases the 
    // memory requirments of the program it significantly reduces
    // the communication cost.


// FORMATION OF THE MATRICES ON NODE 0

    // We take a different ideology with matrices. Due to thier size
    // only small matrices will be distributed across all nodes.

    if(mynode==0)
    {
        // Reservation of memory on the master node.
        tridiag mass; CreateTridiag(N,mass);
        tridiag stiff; CreateTridiag(N,stiff);
        F = CreateMatrixContiguous(L,L);
        D = CreateMatrixContiguous(L,L);
        Ft = CreateMatrixContiguous(L,L);


        // Formation of the mass and stiffness matrix.
        // The Fourier basis matrix F, its transpose Ft,
        // and the diagonal matrix of eigenvalues D are
        // also formed.
        FormMassStiff(h,N,mass,stiff);
        FormFourier_Diag_FourierTranspose(L,F,D,Ft);

//
// FORMATION OF W: BEGIN
//

        // W is the large tridiagonal matrix that needs to be inverted.
        // Reference: See Goddard Wathen

        // Temporary structures used in formation of matrices
        tridiag stiffA0;CreateTridiag(N,stiffA0);
        SetTriDiagEqualTo(N,stiff,stiffA0);
 
        MultiplyTriDiagByConst(N,timestep*timestep,stiffA0);

        // At this point we have
        //  stiffA0 = timestep*timestep*K

        std::vector<tridiag> tridaigVec;

        // Reserve memory 
        for(int i=0;i<L;i++)
        {
            tridiag temp; CreateTridiag(N,temp);
            tridaigVec.push_back(temp);
        }
        // Multiply A1 by eigenvalue and ADD to A0
        for(int i=0;i<L;i++)
        {
            tridiag temp;CreateTridiag(N,temp);
            SetTriDiagEqualTo(N,mass,temp);
            MultiplyTriDiagByConst(N,(1.0 - 2.0*D[i*L+i] + D[i*L+i]*D[i*L+i]),temp);
            AddTriDiag(N,temp,stiffA0,tridaigVec[i]);
        }
        // Reserve memory for contiguous counterparts
        for(int i=0;i<L;i++)
        {
            std::complex<double> *pointertolargeblocked = new std::complex<double>[3*N-2];
            Wblocks.push_back(pointertolargeblocked);
        }
        // Fill the contiguous counterparts
        for(int i=0;i<L;i++)
        {
            for(j=0;j<N-1;j++) Wblocks[i][j] = std::get<0>(tridaigVec[i])[j];
            for(j=0;j<N  ;j++) Wblocks[i][N-1+j] = std::get<1>(tridaigVec[i])[j];
            for(j=0;j<N-1;j++) Wblocks[i][2*N-1+j] = std::get<2>(tridaigVec[i])[j];
            Wblocks[i][N-1] = 1;
            Wblocks[i][2*N-2] = 1;
        }

//
// FORMATION OF W: END
//

// We form the blocks of A and pass the -M as a seperate argument

//
// FORMATION OF Ablocks: BEGIN
//

        // We form the monolithic A by considering
        // block diagonal entries tobe A0 = M + timestep[i]*K
        // and the subdiagonal entries to have 
        // A1 = -M.
        tridaigVec.clear();
        // Reserve memory
        for(int i=0;i<L;i++)
        {
            tridiag temp; CreateTridiag(N,temp);
            tridaigVec.push_back(temp);
        }
        // Multiply A1 by timestep and package
        for(int i=0;i<L;i++)
        {
            tridiag temp;CreateTridiag(N,temp);
            SetTriDiagEqualTo(N,stiff,temp);
            MultiplyTriDiagByConst(N,timestep*timestep,temp);
            AddTriDiag(N,temp,mass,tridaigVec[i]);
        }
        // Reserve memory for contiguous counterparts
        for(int i=0;i<L;i++)
        {
            std::complex<double> *pointertolargeblocked = new std::complex<double>[3*N-2];
            Ablocks.push_back(pointertolargeblocked);
        }
        // Fill the contiguous counterparts
        for(int i=0;i<L;i++)
        {
            for(j=0;j<N-1;j++) Ablocks[i][j] = std::get<0>(tridaigVec[i])[j];
            for(j=0;j<N  ;j++) Ablocks[i][N-1+j] = std::get<1>(tridaigVec[i])[j];
            for(j=0;j<N-1;j++) Ablocks[i][2*N-1+j] = std::get<2>(tridaigVec[i])[j];
            Ablocks[i][N-1] = 1;
            Ablocks[i][2*N-2] = 1;
        }
//
// FORMATION OF Ablocks: END
//
        // U and U^* of the paper 
        UMonolithic = CreateMatrix(L,L);
        UtMonolithic = CreateMatrix(L,L);

        for(i=0;i<L;i++)
        {
            for(j=0;j<L;j++)
            {
                UMonolithic[i][j] = F[j*L+i];
                UtMonolithic[i][j] = Ft[j*L+i];
            }
        }


        // Reservation of memory for massContig
        massContig = new std::complex<double>[3*N-2];
        // Formation of contiguous mass
        for(j=0;j<N-1;j++) massContig[j] = std::get<0>(mass)[j];
        for(j=0;j<N  ;j++) massContig[N-1+j] = std::get<1>(mass)[j];
        for(j=0;j<N-1;j++) massContig[2*N-1+j] = std::get<2>(mass)[j];

//
// FORMATION OF b: BEGIN
//
        // b = [Mu_0,-Mu_0, ... 0] 

        // Initial condition
        std::complex<double>* u0 = new std::complex<double>[N];
        std::complex<double>* U0 = new std::complex<double>[N];
        
        // The smooth initial condition.
        for(i=0;i<N;i++) u0[i] = sin(2*M_PI*i*h);
    
/*
        // The non-smooth initial condition.
        for(i=0;i<N;i++)
        {
           if(i*h<0.5-1.0/8 || i*h > 0.5+1.0/8)
               u0[i] = 0.0;
           else
           {
               u0[i] = cos(4*M_PI*(i*h-0.5))*cos(4*M_PI*(i*h-0.5));
           }
        }
*/
        std::complex<double> prod =0;
        U0[0] = massContig[N-1]*u0[0] +  massContig[2*N-1]*u0[1];
        for(j=1;j<N-1;j++)
        {
            U0[j]= massContig[j-1]*u0[j-1] + massContig[N-1+j]*u0[j] + massContig[2*N-1+j]*u0[j+1];
        }
        U0[N-1] = massContig[2*N-2]*u0[N-1] + massContig[N-2]*u0[N-2];

        for(i=0;i<N;i++) b[i] = U0[i];
        for(i=N;i<N+1;i++) b[i] = -1.0*U0[i-N];
//
// FORMATION OF b: END
//
    }

    // The right hand size is broadcast. This is consistent 
    // with our ethos for this code.
    MPI_Bcast(b,N*L,MPI_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);

//
//  THE CALCULATION PHASE
//  


    // Temp vectors used in GMRES rotuine
    std::complex<double>* temp = new std::complex<double>[N*L]; 
    std::complex<double>* temp2 = new std::complex<double>[N*L];

    // Residual vector 
    std::complex<double>* r0 = new std::complex<double>[N*L];
    

    std::complex<double> normb,beta,resid; /* norm of preconditioned RHS*/

    // If we set the tolerance to 10^{-4} we get a fixed iteration count of 2 for 
    // sufficiently large values of n and l. For  10^{-5} we get a fixed iteration 
    // count of 3.

    std::complex<double> tol = 0.0001;    /* tolerance */ 

    int complete = 0;
    int max_iter = 10;
    int m = max_iter;

    std::complex<double>* s = new std::complex<double>[m+1];
    std::complex<double>* sn = new std::complex<double>[m+1];
    std::complex<double>* cs = new std::complex<double>[m+1];

    std::complex<double>* H = CreateMatrixContiguous(m+1,m+1);

    // Standard procedure is to measure the time taken to complete
    // the calculation on the master node
    if(mynode == 0)
    {
        start = MPI_Wtime();
    }

    
// ----------------------Calculaution Phase Begin ----------------------------------------------------

    // As we are measuring on node 0, we want all processes to
    // "meet up" before entering the calculation stage.

    MPI_Barrier(MPI_COMM_WORLD);


// ---------------------------- GMRES BEGIN ----------------------------------------------------------

    // Our GMRES implementation is essentially a copy of the "wiki" implementation of GMRES

    j=1;
    
    // Calculation of ||P^{-1}b||
    ApplyPreconditioner(mynode,totalnodes,N,L,UMonolithic,UtMonolithic,Wblocks,b,temp);
    CalculateNorm(mynode,totalnodes,N,L,temp,normb);
    // Calculation of r = P^{-1}(b-A*x)
    MultiplyByWaveSystem(mynode,totalnodes,N,L,Ablocks,massContig,x,temp); 
    VectorSubtraction(mynode,totalnodes,N,L,b,temp,temp2);
    ApplyPreconditioner(mynode,totalnodes,N,L,UMonolithic,UtMonolithic,Wblocks,temp2,r0);

    // Calculate beta = ||r||
    CalculateNorm(mynode,totalnodes,N,L,r0,beta);

    if(normb.real() ==0)
        normb = 1;
    resid = beta/normb;
    if(resid.real() < tol.real())
    {
        tol = resid;
        max_iter = 0;
        complete = 1;
    }

    std::vector<std::complex<double>*> v; 
    for(int i=0;i<m+1;i++)
    {
        std::complex<double> *pointer = new std::complex<double>[N*L];
        v.push_back(pointer);
    }
    while(j<=max_iter)
    {
        // Calculate v[0] = r0 * (1/beta)
        SetEqualTo(mynode,totalnodes,N,L,r0,v[0],(1.0/beta));
        s[0] = beta;
        
        for(i=0;i<m && j <= max_iter;i++,j++)
        {
            
            if(complete) continue;

//==========================================  ARNODLI BEGIN
            MultiplyByWaveSystem(mynode,totalnodes,N,L,Ablocks,massContig,v[i],temp);
            ApplyPreconditioner(mynode,totalnodes,N,L,UMonolithic,UtMonolithic,Wblocks,temp,temp2);
            for(k=0;k<=i;k++)
            {
                // w = temp2
                std::complex<double> dotprodoutput;
                DotProduct(mynode,totalnodes,N,L,temp2,v[k],dotprodoutput);
                H[k+i*m] = dotprodoutput;
                PlusEqualTo(mynode,totalnodes,N,L,v[k],temp2,-1.0*dotprodoutput);
            }
            CalculateNorm(mynode,totalnodes,N,L,temp2,H[(i+1)+i*m]); 
            SetEqualTo(mynode,totalnodes,N,L,temp2,v[i+1],(1.0/H[(i+1)+i*m]));
//=========================================  ARNOLDI END

            for(k=0;k<i;k++)
                ApplyPlaneRotation(H[k+i*m],H[(k+1)+i*m],cs[k],sn[k]);

            GeneratePlaneRotation(H[i+i*m], H[(i+1)+i*m], cs[i], sn[i]);
            ApplyPlaneRotation(H[i+i*m], H[(i+1)+i*m], cs[i], sn[i]);
            ApplyPlaneRotation(s[i], s[i+1], cs[i], sn[i]);
        
            resid = fabs(s[i+1]);
            if(resid.real()/normb.real() < tol.real())
            {
                Update(x,N*L,m,m,H,s,v);
                tol = resid;
                max_iter = j;
                complete = 1;
            }
            if(complete) continue;
             
        }
        
        Update(x,N*L,m-1,m,H,s,v);
        
        // Calculation of r = P^{-1}(b-A*x)
        MultiplyByWaveSystem(mynode,totalnodes,N,L,Ablocks,massContig,x,temp);
        VectorSubtraction(mynode,totalnodes,N,L,b,temp,temp2);
        ApplyPreconditioner(mynode,totalnodes,N,L,UMonolithic,UtMonolithic,Wblocks,temp2,r0);


        // Calculate beta = ||r||
        CalculateNorm(mynode,totalnodes,N,L,r0,beta);
        resid = beta/normb;
        if(resid.real() < tol.real())
        {
            tol = resid;
            max_iter = j;
            complete = 1;
        }
        if (complete)continue;
        
    }
   
//
// --------------------------- GMRES END ----------------------------------------------------------
//




// ----------------------Calculaution Phase End ---------------------------------------------------


    MPI_Barrier(MPI_COMM_WORLD);
    if(mynode == 0)
    {
        end = MPI_Wtime();
        // Print the time taken for the calculation to complete
        std::cout << end- start << std::endl;
        std::cout << std::endl;
        // How many iterations did it take for GMRES to terminate?
        std::cout << j << std::endl;
    }
    MPI_Finalize();
}


// Definitions of the routines used in GMRES


void Update(std::complex<double>*x,int lengthofx, int k,int m, std::complex<double>*h,std::complex<double>*s,std::vector<std::complex<double>*>v)
{
    std::complex<double>*y = new std::complex<double>[k+1];
    for(int i=0;i<k+1;i++) y[i] = s[i];

    for(int i=k;i>=0;i--)
    {
        if(h[i+i*m] == 0.0) continue;
        y[i] = y[i]/h[i+i*m];
        for(int j=i-1; j>= 0;j--)
        {
            y[j] -= h[j+i*m]*y[i];
        }
    }
    
    for(int j=0;j<=k;j++)
    {
        for(int mm =0;mm<lengthofx;mm++)
            x[mm] += v[j][mm]*y[j];
    }
    
}

void GeneratePlaneRotation(std::complex<double> &dx, std::complex<double> &dy, std::complex<double> &cs, std::complex<double> &sn)
{
    if (dy == 0.0)
    {
        cs = 1.0;
        sn = 0.0;
    }
    else if (fabs(dy.real()) > fabs(dx.real()))
    {
        std::complex<double> temp = dx / dy;
        sn = 1.0 / sqrt( 1.0 + temp*temp );
        cs = temp * sn;
    }
    else
    {
        std::complex<double> temp = dy / dx;
        cs = 1.0 / sqrt( 1.0 + temp*temp );
        sn = temp * cs;
    }
}


void ApplyPlaneRotation(std::complex<double> &dx, std::complex<double> &dy, std::complex<double> &cs, std::complex<double> &sn)
{
    std::complex<double> temp  =  cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
}










