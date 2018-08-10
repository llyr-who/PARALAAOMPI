/*****************************************************
 *      Parallel routines used to solve
 *      monolithic problems outlined in
 *      
 *      A Note on ...
 *
 *      by Goddard and Wathen.
 *
 *
 *      Code Author: Anthony Goddard
 *
 *      Github: anthonyjamesgoddard
 *
 *      Contact: Please feel free to 
 *      contact me on updates surrounding
 *      the code, etc.
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



// PREAMBLE FUNCTIONS THAT !!DO NOT!! NEED TO BE DISTRIBUTED 

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

// h: stepssize
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


// PARALLEL FUNCTIONS THAT !!DO!! NEED TO BE DISTRIBUTED

// input is x, output is z
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



// The diagonal blocks are serialised. This means that the first N-1 entries are the 
// first subdiagonal, the next N entries are the main diagonal and the remaining entries
// contain the remaining superdiagonal.


// The right hand side is q
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
			l[i]	= am1[i]/d[i];
			d[i+1]	= a[i+1] - l[i]*u[i];
			u[i+1]	= ap1[i+1];
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



//    VERY IMPORTANT :
//        This will only work if the number of processes it is passed
//        divides L exactly.
//
//    For our routines in particular, this routine deals with the
//    component of the preconditioner G( [I \otimes U] <------ THIS  )G^T



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


//    This routine seeks to remove the communication induced
//    by the vector transpose operator matrix.
//
//    This routine currently abuses the fact that the U's
//    that we are using are given are symmetric.


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

// q is input and x is output.


void PostProcessing(int mynode, int numnodes,int N,int L, std::complex<double> *x)
{ 
    int i;
    int local_chunks = L/numnodes;
    int local_offset = mynode*local_chunks;
    std::complex<double>* temp = new std::complex<double>[N*local_chunks];  
    for(i=0;i<local_chunks*N;i++)
    {
        temp[i] = x[local_offset*N + i];
    }
    for(i=0;i<local_chunks*N;i++)
    {
        temp[i] = x[N*L-(i+1 + local_offset*N)].real();
    }
    MPI_Allgather(temp,N*local_chunks,MPI_DOUBLE_COMPLEX,x,N*local_chunks,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);
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
void ApplyPreconditioner(int mynode,int totalnodes,int N, int L,std::complex<double>**U,std::complex<double>**Ut,std::vector<std::complex<double>*>& Wblks,std::complex<double>* q, std::complex<double>* x)
{
    // We need to make a copy of q and conduct calculations on that. 
    //
    std::complex<double>*b = new std::complex<double>[N*L];
    SetEqualTo(mynode,totalnodes,N,L,q,b,1.0);
    
    //for(int i=0;i<N*L;i++) b[i] = q[i];
    /*     
    VecTranspose(mynode,totalnodes,L,N,b,x);
    BlockMatVecMultiplication(mynode,totalnodes,L,N,Ut,x,b); 
    VecTranspose(mynode,totalnodes,N,L,b,x);
    BlockTriDiagSolve_Thomas(mynode,totalnodes,N,L,Wblks,b,x);
    VecTranspose(mynode,totalnodes,L,N,b,x);
    BlockMatVecMultiplication(mynode,totalnodes,L,N,U,x,b);
    VecTranspose(mynode,totalnodes,N,L,b,x);
    */

    MultiplicationByUKronIdentity(mynode,totalnodes,N,L,Ut,b,x);
    BlockTriDiagSolve_Thomas(mynode,totalnodes,N,L,Wblks,b,x);
    MultiplicationByUKronIdentity(mynode,totalnodes,N,L,U,b,x);

    //for(int i=0;i<N*L;i++) b[i]=x[i]; 
    //for(int i=0;i<N*L;i++) x[N*L-i-1]=b[i].real(); 
    //for(int i=N*L-N;i<N*L;i++) x[i] = b[i].real();
    //PostProcessing(mynode,totalnodes,N,L,x);
    delete [] b;
}

void ApplyNonUniformPreconditionerWithOneTerm(int mynode,int totalnodes,int N, int L,std::complex<double>**U,std::complex<double>**Ut,std::vector<std::complex<double>*>& Wblks,std::vector<std::complex<double>*> sigmaKronK,std::complex<double>* q, std::complex<double>* x)
{
    std::complex<double> *b = new std::complex<double>[N*L];
    std::complex<double> *buffer = new std::complex<double>[N*L];

    ApplyPreconditioner(mynode,totalnodes,N,L,U,Ut,Wblks,q,b);
    BlockMatVecMultiplication(mynode,totalnodes,N,L,sigmaKronK,b,buffer);
    VectorSubtraction(mynode,totalnodes,N,L,b,buffer,x);
}

void CalculateNorm(int mynode, int totalnodes,int N, int L,std::complex<double>* vectocalnorm, std::complex<double>& result)
{
    DotProduct(mynode,totalnodes,N,L,vectocalnorm,vectocalnorm,result);
    result = sqrt(result);
}
