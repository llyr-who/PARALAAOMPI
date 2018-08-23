/*****************************************************
 *      Parallel routines used to solve
 *      all-at-once formualtions.
 *
 *      (Currently there are also serial routines as well.)
 *
 *      Code Author: Anthony Goddard
 *
 *      Github: anthonyjamesgoddard
 *
 *      Contact: Please feel free to contact me to implement
 *               suggested improvements. This is my first 
 *               parallel software and so there are some
 *               glaring software development practice issues.
 *
 *         
 *
 *
 *      VERY IMPORTANT :
 *          This will only work if the number of processes it is passed
 *          divides L exactly. This is by design. "odd" problem sizes
 *          would significantly slow down the calculation even if the 
 *          functionality was present.
 *
 *          There are further restrictions when dealing using the FFT.
 *          To be safe the input parameters N,L must be powers of 2.
 *
 *      TODO: 
 *		    Separate serial and parallel rotuines.
 *		    Remove mixed usage of AllGather and AllGatherv to just use AllGather.
 *
 * ***************************************************/




#include<mpi.h>
#include<complex>
#include<tuple>
#include<math.h>
#include<cmath>
#include "MatrixHelper.h"
#include "ParallelRoutines.h"

using namespace std;

//////////////////////////////////////////////////////////////
//
// PREAMBLE FUNCTIONS THAT !!DO NOT!! NEED TO BE DISTRIBUTED 
//
//////////////////////////////////////////////////////////////

// Multiplication of a tridiagonal matrix by a constant
//
// coeff : input constant
// matrix: the matrix that will bte multiplied.

void MultiplyTriDiagByConst(int N, std::complex<double> coeff, tridiag& matrix)
{
    int i;
    for(i=0;i<N;i++)
    {
        std::get<1>(matrix)[i] *= coeff;
    }
    for(i=0;i<N-1;i++)
    {
        std::get<0>(matrix)[i] *= coeff;
        std::get<2>(matrix)[i] *= coeff;
    }
}

// Addition of two tridiagonal matrices
//
// matrix1,matrix2  : are the input matrices
// matrix 3         : the output matrix.

void AddTriDiag(int N, tridiag& matrix1,tridiag& matrix2,tridiag& matrix3)
{
    int i;
    for(i=0;i<N;i++)
    {
        std::get<1>(matrix3)[i] = std::get<1>(matrix1)[i] + std::get<1>(matrix2)[i];
    }
    for(i=0;i<N-1;i++)
    {
        std::get<0>(matrix3)[i] = std::get<0>(matrix1)[i] + std::get<0>(matrix2)[i];
        std::get<2>(matrix3)[i] = std::get<2>(matrix1)[i] + std::get<2>(matrix2)[i];
    }
}

// The following code generates the Mass and Stiffness matrices
// 
// mass, stiff: output matrices
// h,N : input parameters. 
// 
// Having both h and N is redundant.

void FormMassStiff(double h,int N,tridiag& mass, tridiag& stiff)
{
    int i;
    double eStiff[4];
    double eMass[4];

    eStiff[0] = (1.0/h);   eStiff[2] = -1.0*eStiff[0];
    eStiff[1] = eStiff[2]; eStiff[3] = eStiff[0];

    eMass[0] = (h/3.0);    eMass[2] = 0.5*eMass[0];
    eMass[1] = eMass[2];   eMass[3] = eMass[0];
    for(i=0;i<N-1;i++)
    {
        std::get<1>(stiff)[i] =  std::get<1>(stiff)[i] + eStiff[0];
        std::get<0>(stiff)[i] =  std::get<0>(stiff)[i] + eStiff[2];
        std::get<2>(stiff)[i] =  std::get<2>(stiff)[i] + eStiff[1];
        std::get<1>(stiff)[i+1] =  std::get<1>(stiff)[i+1] + eStiff[3];

        std::get<1>(mass)[i] =  std::get<1>(mass)[i] + eMass[0];
        std::get<0>(mass)[i] =  std::get<0>(mass)[i] + eMass[2];
        std::get<2>(mass)[i] =  std::get<2>(mass)[i] + eMass[1];
        std::get<1>(mass)[i+1] =  std::get<1>(mass)[i+1] + eMass[3];
    }
    std::get<1>(stiff)[N-1] = std::get<1>(stiff)[N-1]  + (1.0/h);
    std::get<1>(mass)[N-1] = std::get<1>(mass)[N-1]  + (h/3.0);
    std::get<0>(mass)[0] = 0;
    std::get<2>(mass)[0] = 0;
    std::get<0>(stiff)[0] = 0;
    std::get<2>(stiff)[0] = 0;
    std::get<0>(mass)[N-2] = 0;
    std::get<2>(mass)[N-2] = 0;
    std::get<0>(stiff)[N-2] = 0;
    std::get<2>(stiff)[N-2] = 0;
    std::get<1>(mass)[0] = 0;
    std::get<1>(mass)[N-1] = 0;
    std::get<1>(stiff)[0] = 0;
    std::get<1>(stiff)[N-1] = 0;
}

// Forms the fourier matrix, its transpose and the 
// eigenvalue matrix of the Circulant $\Sigma$
//
// F : Fourier matrix
// D : Diagonal matrix of eigenvalues
// F^* : conjugate transpose of F

void FormFourier_Diag_FourierTranspose(int N,std::complex<double>*F,std::complex<double>*D,std::complex<double>*Ft)
{
    std::complex<double> imag(0,1);
    std::complex<double> arg;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            arg = (2.0/N)*M_PI*i*j*imag;
            F[i*N+j] =std::exp(arg)/sqrt(N);
            arg = (-2.0/N)*M_PI*i*j*imag;
            Ft[i*N+j]  = std::exp(arg)/sqrt(N);
        }
    }
    int mm=0;
    for(int i=mm;i<N+mm;i++)
    {
        arg = -1*(2.0/N)*M_PI*i*imag;
        D[(i-mm)*N+(i-mm)] = std::exp(arg);
    }
}

// This is the routine that carries out the FFT.
// It only returns a correct result if it is given
// a input whos length is a power of 2.


// It is essentially a copy and paste from Wiki as it is a 
// solid implementation. 
// There are better method than this that involve bit switching.
// These methods are much easier on the memory requirements

//-------------------------------------------FROM WIKI

// separate even/odd elements to lower/upper halves of array respectively.
// Due to Butterfly combinations, this turns out to be the simplest way 
// to get the job done without clobbering the wrong elements.

void separate (std::complex<double>* a, int n) {
    std::complex<double>* b = new std::complex<double>[n/2];  // get temp heap storage
    for(int i=0; i<n/2; i++)    // copy all odd elements to heap storage
        b[i] = a[i*2+1];
    for(int i=0; i<n/2; i++)    // copy all even elements to lower-half of a[]
        a[i] = a[i*2];
    for(int i=0; i<n/2; i++)    // copy all odd (from heap) to upper-half of a[]
        a[i+n/2] = b[i];
    delete[] b;                 // delete heap storage
}

// N must be a power-of-2, or bad things will happen.
// Currently no check for this condition.
//
// N input samples in X[] are FFT'd and results left in X[].
// Because of Nyquist theorem, N samples means 
// only first N/2 FFT results in X[] are the answer.
// (upper half of X[] is a reflection with no new information).

void FFT(std::complex<double>* X, int N,int inverse) {
    int factor = 1;
    if(N < 2) {
        // bottom of recursion.
        // Do nothing here, because already X[0] = x[0]
    } else {
        separate(X,N);      // all evens to lower half, all odds to upper half
        FFT(X,     N/2,inverse);   // recurse even items
        FFT(X+N/2, N/2,inverse);   // recurse odd  items
        // combine results of two half recursions
        for(int k=0; k<N/2; k++) {
            std::complex<double> e = X[k    ];   // even
            std::complex<double> o = X[k+N/2];   // odd
            // w is the "twiddle-factor"
            if(inverse) factor=-1;
            std::complex<double> w = exp( std::complex<double>(0,-2.*M_PI*factor*k/N) );
            X[k    ] = e + w * o;
            X[k+N/2] = e - w * o;
        }
    }
}

//----------------------------------------END FROM WIKI

///////////////////////////////////////////////////////
//
// PARALLEL FUNCTIONS THAT !!DO!! NEED TO BE DISTRIBUTED
//
////////////////////////////////////////////////////////


// This routine carries out the vector transpose in parallel.
//
// x : input vector
// z : output transposed vector
//
// Comments:
//          The input is assummed to be in collumn major form. Since we know that the
//          vector is of length N*L we can split it up into chunks and treat each of 
//          these N-chunks (or L-chunks) as a collumn vector of a matrx.
//          By using some simple index arithmatic we can then map the entries of the old
//          vector to a new vector.
//
void VecTranspose(int mynode,int numnodes,int N,int L, std::complex<double>*x,std::complex<double>*z)
{
    int i,j,k;
    std::complex<double>*y;
    int local_rows = (int)((double)L/numnodes);
    int local_offset = local_rows*mynode;
    y = new std::complex<double>[N*local_rows];
    for(j =0;j< local_rows;j++)
    {
        for(i=0;i<N;i++)
        {
            k = j + local_offset;
            y[j*N+i] = x[i*L+k];
        }
    }
    MPI_Allgather(y,local_rows*N,MPI_DOUBLE_COMPLEX,z,N*local_rows,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD); 
}


// This routine solves "q=Ax" for x where A is block tridiagonal. The matrices blocks are split up
// into groups of size blocks_local and then solved in the usuual serial way.
//
// blocks: the diagonal blocks
// q : as above.
// x as above, the solution.

void BlockTriDiagSolve_Thomas(int mynode, int numnodes,int N,int L, std::vector<std::complex<double> *>&blocks,std::complex<double> *x,std::complex<double>* q)
{
    int i,j,k;
    int local_offset,blocks_local,last_blocks_local;
    int *count;
    int *displacements;

    // The number of blocks each processor is dealt
    blocks_local = L/numnodes;

    // The part of the output vector per process.
    std::complex<double> *z = new std::complex<double>[N*blocks_local];

    // Container of local blocks
    std::vector<std::complex<double>*> localblocks(blocks_local);

    // the offset
    local_offset = mynode*blocks_local;

    MPI_Status status;

    /* Distribute the blocks across the processes */

    // At this point node 0 has the matrix. So we only need
    // to distribute among the remaining nodes, using the 
    // last node as a cleanup.
    if(mynode ==0)
    {
        for(i=0;i<blocks_local;i++) localblocks[i] = CreateTDMatrixContiguous(N);
        for(i=0;i<blocks_local;i++) localblocks[i] = blocks[i];
        // This deals the matrix between processes 1 to numnodes -2
        for(i=1;i<numnodes;i++)
        {
            for(j=0;j<blocks_local;j++)
            {
                MPI_Send(blocks[i*blocks_local+j],3*N-2,MPI_DOUBLE_COMPLEX,i,j,MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        /*This code allows other processes to obtain the chunks of data*/
        /* sent by process 0 */
        for(i=0;i<blocks_local;i++) localblocks[i] = CreateTDMatrixContiguous(N);
        /* rows_local has a different value on the last processor, remember */
        for(i=0;i<blocks_local;i++)
        {
            MPI_Recv(localblocks[i],3*N-2,MPI_DOUBLE_COMPLEX,0,i,MPI_COMM_WORLD,&status);
        }
    }


    //ENTERING THE CALCULATION STAGE

    
    for(k = 0;k<blocks_local;k++)
    {
        // Unpacking of contig. data structure.
        std::complex<double> * a = new std::complex<double>[N];
        std::complex<double> * am1 = new std::complex<double>[N-1];
        std::complex<double> * ap1 = new std::complex<double>[N-1];
        
        for(j=0;j<N-1;j++) am1[j] = localblocks[k][j];
        for(j=0;j<N;j++) a[j] = localblocks[k][N-1 + j];
        for(j=0;j<N-1;j++) ap1[j] = localblocks[k][2*N-1 + j];
        
        // Entering Thomas
        std::complex<double> *l,*u,*d,*y;
        l = new std::complex<double>[N];
        u = new std::complex<double>[N];
        d = new std::complex<double>[N];
        y = new std::complex<double>[N];

        // LU 
        d[0] = a[0];
        u[0] = ap1[0];
        for(i=0;i<N-2;i++)
        {
            l[i]    = am1[i]/d[i];
            d[i+1]  = a[i+1] - l[i]*u[i];
            u[i+1]  = ap1[i+1];
        }
        l[N-2] = am1[N-2]/d[N-2];
        d[N-1] = a[N-1] - l[N-2]*u[N-2];

        // Forward substitution 
        y[0] = q[(k+local_offset)*N];
        for(i=1;i<N;i++)
        {
            y[i] = q[(k+local_offset)*N+i] - l[i-1]*y[i-1];
        }

        // Backward substitution 
        z[k*N+N-1] = y[N-1]/d[N-1];
        for(i=N-2;i>=0;i--)
        {
            z[k*N+i] = (y[i]-u[i]*z[k*N+i+1])/d[i];
        }
        delete [] l;
        delete [] u;
        delete [] d;
        delete [] y;
        
    }

    count = new int[numnodes];
    displacements = new int[numnodes];

    for(i=0;i<numnodes;i++)
    {
        count[i] = N*(L/numnodes);
        displacements[i] = i*count[i];
    }

    MPI_Allgatherv(z,N*blocks_local,MPI_DOUBLE_COMPLEX,x,count,displacements,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);

    delete [] z;

}



// Block matrix vector multiplication.
void BlockMatVecMultiplication(int mynode, int numnodes,int N,int L, std::vector<std::complex<double> *>&blocks,std::complex<double> *x,std::complex<double>* y)
{
    int i,j;
    int local_offset,blocks_local,last_blocks_local;
    int *count;
    int *displacements;
    std::complex<double> prod;

    // The number of blocks each processor is dealt
    blocks_local = L/numnodes;

    // The part of the output vector per process.
    std::complex<double> *temp = new std::complex<double>[N*blocks_local];

    // Container of local blocks
    std::vector<std::complex<double>*> localblocks(blocks_local);

    // the offset
    local_offset = mynode*blocks_local;

    MPI_Status status;

    /* Distribute the blocks across the processes */

    // At this point node 0 has the matrix. So we only need
    // to distribute among the remaining nodes, using the 
    // last node as a cleanup.
    if(mynode ==0)
    {
        for(i=0;i<blocks_local;i++) localblocks[i] = CreateMatrixContiguous(N,N);
        for(i=0;i<blocks_local;i++) localblocks[i] = blocks[i];
        // This deals the matrix between processes 1 to numnodes -2
        for(i=1;i<numnodes;i++)
        {
            for(j=0;j<blocks_local;j++)
            {
                MPI_Send(blocks[i*blocks_local+j],N*N,MPI_DOUBLE_COMPLEX,i,j,MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        /*This code allows other processes to obtain the chunks of data
         sent by process 0 */
        for(i=0;i<blocks_local;i++) localblocks[i] = CreateMatrixContiguous(N,N);
        /* rows_local has a different value on the last processor, remember */
        for(i=0;i<blocks_local;i++)
        {
            MPI_Recv(localblocks[i],N*N,MPI_DOUBLE_COMPLEX,0,i,MPI_COMM_WORLD,&status);
        }
    }


    // The calculations can now be carried out.
    // We need to evaluate Ax on each process.

    
    for(i = 0;i<blocks_local;i++)
    {
        for(j=0;j<N;j++)
        {
            prod = 0;
            for(int k=0;k<N;k++)
            {
                prod += localblocks[i][k*N+j]*x[local_offset*N + i*N + k];
            }
            temp[i*N+j] = prod;
        }
        
    }

    count = new int[numnodes];
    displacements = new int[numnodes];

    for(i=0;i<numnodes;i++)
    {
        count[i] = N*(L/numnodes);
        displacements[i] = i*count[i];
    }

    MPI_Allgatherv(temp,N*blocks_local,MPI_DOUBLE_COMPLEX,y,count,displacements,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);

    delete [] temp;

}


//    This routine evaluates (Kron(U,eye(N))) x = y 

void MultiplicationByUKronIdentity(int mynode, int numnodes,int N,int L,std::complex<double>**U,std::complex<double> *x,std::complex<double>* y)
{
    int i,j;
    int local_offset,blocks_local,last_blocks_local;
    int *count;
    int *displacements;
    std::complex<double> prod;

    // The number of blocks each processor is dealt
    blocks_local = L/numnodes;

    // The part of the output vector per process.
    std::complex<double> *temp = new std::complex<double>[N*blocks_local];

    // Contains the rows of U that will be needed 
    // by each process.
    std::complex<double>** Ulocal;

    // the offset
    local_offset = mynode*blocks_local;

    MPI_Status status;

    // Distribution of U across the processes.
    //
    // NOTE:
    // At this point node 0 has the matrix. So we only need
    // to distribute among the remaining nodes, using the 
    // last node as a cleanup.

    if(mynode ==0)
    {
        // This deals the matrix between processes 1 to numnodes -2
        Ulocal = CreateMatrix(blocks_local,L);
        for(i=0;i<blocks_local;i++)
        {
            Ulocal[i] = U[i];
        }
        for(i=1;i<numnodes;i++)
        {
            for(j=0;j<blocks_local;j++)
            {
                MPI_Send(U[i*blocks_local+j],L,MPI_DOUBLE_COMPLEX,i,j,MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        //This code allows other processes to obtain the chunks of data
        // sent by process 0 
        Ulocal = CreateMatrix(blocks_local,L);
        //rows_local has a different value on the last processor, remember 
        for(i=0;i<blocks_local;i++)
        {
            MPI_Recv(Ulocal[i],L,MPI_DOUBLE_COMPLEX,0,i,MPI_COMM_WORLD,&status);
        }
    }


    // Carries out the "strip" multiplication.
    
    for(i = 0;i<blocks_local;i++)
    {
        for(j=0;j<N;j++)
        {
            prod = 0;
            for(int k=0;k<L;k++)
            {
                prod += Ulocal[i][k]*x[k*N + j];
            }
            temp[i*N+j] = prod;
        }
        
    }

    count = new int[numnodes];
    displacements = new int[numnodes];

    for(i=0;i<numnodes;i++)
    {
        count[i] = N*(L/numnodes);
        displacements[i] = i*count[i];
    }

    MPI_Allgatherv(temp,N*blocks_local,MPI_DOUBLE_COMPLEX,y,count,displacements,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);

    delete [] temp;
    delete [] count;
    delete [] displacements;
    
}


// This evaluates (Kron(eye(N),U))*x = y (OR (Kron(eye(N),U^*))*x = y ) using the fast fourier tranform.
//
// x: is the input vector
// y: is the output vector
// inverse: indicates whether we are are using U or U^*.

void MultiplicationByIdentityKronU_usingFFT(int mynode, int numnodes,int N,int L,std::complex<double> *x,std::complex<double>* y,int inverse)
{
    int i;
    int local_offset,blocks_local;
    // The number of blocks each processor is dealt
    blocks_local = N/numnodes;

    // The part of the output vector per process.
    std::complex<double> *temp = new std::complex<double>[L*blocks_local];

    // the offset
    local_offset = mynode*blocks_local;

    for(i = 0;i<blocks_local;i++)
    {
		std::copy(&x[local_offset*L + i*L], &x[local_offset*L + (i+1)*L], &temp[i*L]);
		FFT(&temp[i*L],L,inverse);
    }
    for(i=0;i<L*blocks_local;i++) temp[i] *= (1.0/std::sqrt(L));
    MPI_Allgather(temp,L*blocks_local,MPI_DOUBLE_COMPLEX,y,L*blocks_local,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD); 
}


// x is input and y output
void MultiplyByHeatSystem(int mynode, int numnodes,int N,int L, std::vector<std::complex<double> *>&blocks,std::complex<double>*mass,std::complex<double> *x,std::complex<double>* y)
{
    int i,j;
    int local_offset,blocks_local,last_blocks_local;
    int *count;
    int *displacements;
    std::complex<double> prod;

    // The number of blocks each processor is dealt
    blocks_local = L/numnodes;

    // The part of the output vector per process.
    std::complex<double> *temp = new std::complex<double>[N*blocks_local];

    // Container of local blocks
    std::vector<std::complex<double>*> localblocks(blocks_local);

    // the offset
    local_offset = mynode*blocks_local;

    MPI_Status status;

    /* Distribute the blocks across the processes */

    // At this point node 0 has the matrix. So we only need
    // to distribute among the remaining nodes, using the 
    // last node as a cleanup.
    //
    if(mynode ==0)
    {
        for(i=0;i<blocks_local;i++) localblocks[i] = CreateMatrixContiguous(N,N);
        for(i=0;i<blocks_local;i++) localblocks[i] = blocks[i];
        // This deals the matrix between processes 1 to numnodes -2
        for(i=1;i<numnodes;i++)
        {
            for(j=0;j<blocks_local;j++)
            {
                MPI_Send(blocks[i*blocks_local+j],3*N-2,MPI_DOUBLE_COMPLEX,i,j,MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        //This code allows other processes to obtain the chunks of data
        //  sent by process 0 
        mass = new std::complex<double>[3*N-2];
        for(i=0;i<blocks_local;i++) localblocks[i] = CreateMatrixContiguous(N,N);
        /* rows_local has a different value on the last processor, remember */
        for(i=0;i<blocks_local;i++)
        {
            MPI_Recv(localblocks[i],3*N-2,MPI_DOUBLE_COMPLEX,0,i,MPI_COMM_WORLD,&status);
        }
    }
    // Reserve space for the mass matrix on each 
    // process
    MPI_Bcast(mass,3*N-2,MPI_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);

    // At this point the diagonal blocks of
    // A (heat) have been distributed across
    // all processes. Also the mass matrix
    // is available on all processes.


    // Here we have the block
    for(i = 0;i<blocks_local;i++)
    {
        // We have to distinguish between the
        // first block row and the rest.
        if(mynode == 0 && i ==0)
        {
            prod = 0;
            prod+=localblocks[i][N-1]*x[0];
            prod+=localblocks[i][2*N-1]*x[1];
            temp[0] = prod;
            for(j=1;j<N-1;j++)
            {
                prod = 0;
                prod += localblocks[i][j-1]*x[j-1];
                prod += localblocks[i][N-1+j]*x[j];
                prod += localblocks[i][2*N-1+j]*x[j+1];
                temp[j] = prod;
            }
            prod = 0;
            prod+=localblocks[i][2*N-2]*x[N-1];
            prod+=localblocks[i][N-2]*x[N-2];
            temp[N-1] = prod;
        }
        else
        {
            // this deals with the remaining block rows
            prod = 0;
            prod+=localblocks[i][N-1]*x[local_offset*N + i*N];
            prod+=localblocks[i][2*N-1]*x[local_offset*N + i*N + 1];
            prod+=-1.0*mass[N-1]*x[local_offset*N + (i-1)*N];
            prod+=-1.0*mass[2*N-1]*x[local_offset*N + (i-1)*N + 1];
            temp[i*N] = prod;
            for(j=1;j<N-1;j++)
            {
                prod = 0;
                prod += localblocks[i][j-1]*x[local_offset*N + i*N + j-1];
                prod += localblocks[i][N-1+j]*x[local_offset*N + i*N + j];
                prod += localblocks[i][2*N-1+j]*x[local_offset*N + i*N + j+1];
                prod += -1.0*mass[j-1]*x[local_offset*N + (i-1)*N + j-1];
                prod += -1.0*mass[N-1+j]*x[local_offset*N + (i-1)*N + j];
                prod += -1.0*mass[2*N-1+j]*x[local_offset*N + (i-1)*N + j+1];
                temp[i*N+j] = prod;
            }
            prod = 0;
            prod+=localblocks[i][2*N-2]*x[local_offset*N + i*N + N-1];
            prod+=localblocks[i][N-2]*x[local_offset*N + i*N + N-2];
            prod+=-1.0*mass[N-2]*x[local_offset*N + (i-1)*N + N-1];
            prod+=-1.0*mass[2*N-2]*x[local_offset*N + (i-1)*N + N-2];
            temp[i*N+N-1] = prod;
        }
    }
        
    

    count = new int[numnodes];
    displacements = new int[numnodes];

    for(i=0;i<numnodes;i++)
    {
        count[i] = N*(L/numnodes);
        displacements[i] = i*count[i];
    }

    MPI_Allgatherv(temp,N*blocks_local,MPI_DOUBLE_COMPLEX,y,count,displacements,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);
    delete [] count;
    delete [] displacements;
    delete [] temp;

    //clean up

}

// x is input and y output
void MultiplyByWaveSystem(int mynode, int numnodes,int N,int L, std::vector<std::complex<double> *>&blocks,std::complex<double>*mass,std::complex<double> *x,std::complex<double>* y)
{
    int i,j;
    int local_offset,blocks_local,last_blocks_local;
    int *count;
    int *displacements;
    std::complex<double> prod;

    // The number of blocks each processor is dealt
    blocks_local = L/numnodes;

    // The part of the output vector per process.
    std::complex<double> *temp = new std::complex<double>[N*blocks_local];

    // Container of local blocks
    std::vector<std::complex<double>*> localblocks(blocks_local);

    // the offset
    local_offset = mynode*blocks_local;

    MPI_Status status;

    /* Distribute the blocks across the processes */

    // At this point node 0 has the matrix. So we only need
    // to distribute among the remaining nodes, using the 
    // last node as a cleanup.
    //
    if(mynode ==0)
    {
        for(i=0;i<blocks_local;i++) localblocks[i] = CreateMatrixContiguous(N,N);
        for(i=0;i<blocks_local;i++) localblocks[i] = blocks[i];
        // This deals the matrix between processes 1 to numnodes -2
        for(i=1;i<numnodes;i++)
        {
            for(j=0;j<blocks_local;j++)
            {
                MPI_Send(blocks[i*blocks_local+j],3*N-2,MPI_DOUBLE_COMPLEX,i,j,MPI_COMM_WORLD);
            }
        }
    }

    else
    {
        //This code allows other processes to obtain the chunks of data
        //  sent by process 0 
        mass = new std::complex<double>[3*N-2];
        for(i=0;i<blocks_local;i++) localblocks[i] = CreateMatrixContiguous(N,N);
        /* rows_local has a different value on the last processor, remember */
        for(i=0;i<blocks_local;i++)
        {
            MPI_Recv(localblocks[i],3*N-2,MPI_DOUBLE_COMPLEX,0,i,MPI_COMM_WORLD,&status);
        }
    }
    // Reserve space for the mass matrix on each 
    // process
    MPI_Bcast(mass,3*N-2,MPI_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);

    // At this point the diagonal blocks of
    // A (heat) have been distributed across
    // all processes. Also the mass matrix
    // is available on all processes.


    // Here we have the block
    for(i = 0;i<blocks_local;i++)
    {
        // Along the first row block is ennesntially equal to Iu_0 = u_0
        if(mynode == 0 && i <2)
        {
            if(i==0)
            {
                for(j=0;j<N;j++)
                {
                    temp[j] = x[j];
                }
            }
            if(i==1)
            {
                // this deals with the remaining block rows
                prod = 0;
                prod+=localblocks[i][N-1]*x[local_offset*N + i*N];
                prod+=localblocks[i][2*N-1]*x[local_offset*N + i*N + 1];
                prod+=-2.0*mass[N-1]*x[local_offset*N + (i-1)*N];
                prod+=-2.0*mass[2*N-1]*x[local_offset*N + (i-1)*N + 1];
                temp[i*N] = prod;
                for(j=1;j<N-1;j++)
                {
                    prod = 0;
                    prod += localblocks[i][j-1]*x[local_offset*N + i*N + j-1];
                    prod += localblocks[i][N-1+j]*x[local_offset*N + i*N + j];
                    prod += localblocks[i][2*N-1+j]*x[local_offset*N + i*N + j+1];
                    prod += -2.0*mass[j-1]*x[local_offset*N + (i-1)*N + j-1];
                    prod += -2.0*mass[N-1+j]*x[local_offset*N + (i-1)*N + j];
                    prod += -2.0*mass[2*N-1+j]*x[local_offset*N + (i-1)*N + j+1];
                    temp[i*N+j] = prod;
                }
                prod = 0;
                prod+=localblocks[i][2*N-2]*x[local_offset*N + i*N + N-1];
                prod+=localblocks[i][N-2]*x[local_offset*N + i*N + N-2];
                prod+=-2.0*mass[N-2]*x[local_offset*N + (i-1)*N + N-1];
                prod+=-2.0*mass[2*N-2]*x[local_offset*N + (i-1)*N + N-2];
                temp[i*N+N-1] = prod;
            }
        }
        else
        {
            // this deals with the remaining block rows
            prod = 0;
            prod+=localblocks[i][N-1]*x[local_offset*N + i*N];
            prod+=localblocks[i][2*N-1]*x[local_offset*N + i*N + 1];
            prod+=-2.0*mass[N-1]*x[local_offset*N + (i-1)*N];
            prod+=-2.0*mass[2*N-1]*x[local_offset*N + (i-1)*N + 1];
            prod+=    mass[N-1]*x[local_offset*N + (i-2)*N];
            prod+=    mass[2*N-1]*x[local_offset*N + (i-2)*N + 1];
            temp[i*N] = prod;
            for(j=1;j<N-1;j++)
            {
                prod = 0;
                prod += localblocks[i][j-1]*x[local_offset*N + i*N + j-1];
                prod += localblocks[i][N-1+j]*x[local_offset*N + i*N + j];
                prod += localblocks[i][2*N-1+j]*x[local_offset*N + i*N + j+1];
                prod += -2.0*mass[j-1]*x[local_offset*N + (i-1)*N + j-1];
                prod += -2.0*mass[N-1+j]*x[local_offset*N + (i-1)*N + j];
                prod += -2.0*mass[2*N-1+j]*x[local_offset*N + (i-1)*N + j+1];
                prod +=      mass[j-1]*x[local_offset*N + (i-2)*N + j-1];
                prod +=      mass[N-1+j]*x[local_offset*N + (i-2)*N + j];
                prod +=      mass[2*N-1+j]*x[local_offset*N + (i-2)*N + j+1];
                temp[i*N+j] = prod;
            }
            prod = 0;
            prod+=localblocks[i][2*N-2]*x[local_offset*N + i*N + N-1];
            prod+=localblocks[i][N-2]*x[local_offset*N + i*N + N-2];
            prod+=-2.0*mass[N-2]*x[local_offset*N + (i-1)*N + N-1];
            prod+=-2.0*mass[2*N-2]*x[local_offset*N + (i-1)*N + N-2];
            prod+=     mass[N-2]*x[local_offset*N + (i-2)*N + N-1];
            prod+=     mass[2*N-2]*x[local_offset*N + (i-2)*N + N-2];
            temp[i*N+N-1] = prod;
        }
    }
        
    

    count = new int[numnodes];
    displacements = new int[numnodes];

    for(i=0;i<numnodes;i++)
    {
        count[i] = N*(L/numnodes);
        displacements[i] = i*count[i];
    }

    MPI_Allgatherv(temp,N*blocks_local,MPI_DOUBLE_COMPLEX,y,count,displacements,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);
    delete [] count;
    delete [] displacements;
    delete [] temp;

    //clean up

}

//
// The following routines are obvious parallel implementations of simple
// vector operations.
//


void DotProduct(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>* y,std::complex<double>&result)
{
    int i;
    int local_chunks = L/numnodes;
    int local_offset = mynode*local_chunks;
    std::complex<double> temp=0;

    for(i=0;i<local_chunks*N;i++)
    {
        temp += x[local_offset*N + i]*std::conj(y[local_offset*N + i]);
    }
    MPI_Allreduce(&temp,&result,1,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD);

}

void VectorAddition(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>* y,std::complex<double>*result)
{
    int i;
    int local_chunks = L/numnodes;
    int local_offset = mynode*local_chunks;
    std::complex<double>* temp = new std::complex<double>[N*local_chunks];

    for(i=0;i<local_chunks*N;i++)
    {
        temp[i] = x[local_offset*N + i]+y[local_offset*N + i];
    }
    MPI_Allgather(temp,N*local_chunks,MPI_DOUBLE_COMPLEX,result,N*local_chunks,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);
    delete [] temp;
}

void VectorSubtraction(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>* y,std::complex<double>*result)
{
    int i;
    int local_chunks = L/numnodes;
    int local_offset = mynode*local_chunks;
    std::complex<double>* temp = new std::complex<double>[N*local_chunks];

    for(i=0;i<local_chunks*N;i++)
    {
        temp[i] = x[local_offset*N + i]-y[local_offset*N + i];
    }
    MPI_Allgather(temp,N*local_chunks,MPI_DOUBLE_COMPLEX,result,N*local_chunks,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);
    delete [] temp;
}


void SetEqualTo(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>*q,std::complex<double>scalar)
{ 
    int i;
    int local_chunks = L/numnodes;
    int local_offset = mynode*local_chunks;
    std::complex<double>* temp = new std::complex<double>[N*local_chunks];  
    for(i=0;i<local_chunks*N;i++)
    {
        temp[i] = scalar*x[local_offset*N + i];
    }
    MPI_Allgather(temp,N*local_chunks,MPI_DOUBLE_COMPLEX,q,N*local_chunks,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);
    delete [] temp;
}
void PlusEqualTo(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>*q,std::complex<double>scalar)
{ 
    int i;
    int local_chunks = L/numnodes;
    int local_offset = mynode*local_chunks;
    std::complex<double>* temp = new std::complex<double>[N*local_chunks];  
    for(i=0;i<local_chunks*N;i++)
    {
        temp[i] = q[local_offset*N + i] +  scalar*x[local_offset*N + i];
    }
    MPI_Allgather(temp,N*local_chunks,MPI_DOUBLE_COMPLEX,q,N*local_chunks,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);
    delete [] temp;
}



// Application of the preconditioners

void ApplyPreconditioner(int mynode,int totalnodes,int N, int L,std::complex<double>**U,std::complex<double>**Ut,std::vector<std::complex<double>*>& Wblks,std::complex<double>* q, std::complex<double>* x)
{
    // We need to make a copy of q and conduct calculations on that. 
    //
    std::complex<double>*b = new std::complex<double>[N*L];
    SetEqualTo(mynode,totalnodes,N,L,q,b,1.0);
    
    // This follows the paper exactly. The non-FFT implementation.
    MultiplicationByUKronIdentity(mynode,totalnodes,N,L,Ut,b,x);
    BlockTriDiagSolve_Thomas(mynode,totalnodes,N,L,Wblks,b,x);
    MultiplicationByUKronIdentity(mynode,totalnodes,N,L,U,b,x);

    delete [] b;
}

void ApplyPreconditionerFFT(int mynode,int totalnodes,int N, int L,std::vector<std::complex<double>*>& Wblks,std::complex<double>* q, std::complex<double>* x)
{
    // As we can see there is a significant memory footprint by using this
    // method.
    std::complex<double>*b = new std::complex<double>[N*L];
    SetEqualTo(mynode,totalnodes,N,L,q,b,1.0);

    VecTranspose(mynode,totalnodes,L,N,b,x);
    MultiplicationByIdentityKronU_usingFFT(mynode, totalnodes,N,L,x,b,1); 
    VecTranspose(mynode,totalnodes,N,L,b,x);

    BlockTriDiagSolve_Thomas(mynode,totalnodes,N,L,Wblks,b,x);

    VecTranspose(mynode,totalnodes,L,N,b,x);
    MultiplicationByIdentityKronU_usingFFT(mynode, totalnodes,N,L,x,b,0); 
    VecTranspose(mynode,totalnodes,N,L,b,x);

    delete [] b;
}

void ApplyNonUniformPreconditionerWithOneTerm(int mynode,int totalnodes,int N, int L,std::complex<double>**U,std::complex<double>**Ut,std::vector<std::complex<double>*>& Wblks,std::vector<std::complex<double>*> sigmaKronK,std::complex<double>* q, std::complex<double>* x)
{
	// Use buffers to store intermediate results
    std::complex<double> *buffer1 = new std::complex<double>[N*L];
    std::complex<double> *buffer2 = new std::complex<double>[N*L];
    std::complex<double> *buffer3 = new std::complex<double>[N*L];

	// Apply preconditioner to buffer and store for later use
    ApplyPreconditioner(mynode,totalnodes,N,L,U,Ut,Wblks,q,buffer1);

	// Apply block matrix multiplication and store for later use
    BlockMatVecMultiplication(mynode,totalnodes,N,L,sigmaKronK,buffer1,buffer2);

	// Apply preconditioner to result of above calculations
    ApplyPreconditioner(mynode,totalnodes,N,L,U,Ut,Wblks,buffer2,buffer3);

	// carry out the subtraction
    VectorSubtraction(mynode,totalnodes,N,L,buffer1,buffer3,x);
}

void CalculateNorm(int mynode, int totalnodes,int N, int L,std::complex<double>* vectocalnorm, std::complex<double>& result)
{
    DotProduct(mynode,totalnodes,N,L,vectocalnorm,vectocalnorm,result);
    result = sqrt(result);
}
