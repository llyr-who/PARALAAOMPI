/**************************************************************************
*
*
*************************************************************************/


#include "MatrixHelper.h"
#include<tuple>

std::complex<double> ** CreateMatrix(int m, int n)
{
    std::complex<double> ** mat;
    mat = new std::complex<double>*[m];
    for(int i=0;i<m;i++)
    {
        mat[i] = new std::complex<double>[n];
        for(int j=0;j<n;j++)
        {
            mat[i][j] = 0.0;
        }
  }
  return mat;
}

std::complex<double> * CreateMatrixContiguous(int m, int n)
{
	std::complex<double>* mat = new std::complex<double>[m*n];
	for(int i=0;i<m*n;i++)
	{
		mat[i] = 0;
	}
	return mat;
}

std::complex<double> * CreateTDMatrixContiguous(int n)
{
	std::complex<double>* mat = new std::complex<double>[3*n-2];
	for(int i=0;i<3*n-2;i++)
	{
		mat[i] = 0;
	}
	return mat;
}

void CreateTridiag(int N,tridiag & TD)
{
	std::complex<double> *two = new std::complex<double>[N];
	std::complex<double> *one = new std::complex<double>[N-1];
	std::complex<double> *three = new std::complex<double>[N-1];

	for(int i=0;i<N;i++){two[i] = 0;}
	for(int i=0;i<N-1;i++){one[i] = 0;}
	for(int i=0;i<N-1;i++){three[i] = 0;}

	std::get<0>(TD) = one;
	std::get<1>(TD) = two;
	std::get<2>(TD) = three;
}



void DestroyMatrix(double ** mat, int m, int n)
{
    for(int i=0;i<m;i++)
        delete[] mat[i];
    delete[] mat;
}


void SetTriDiagEqualTo(int N, tridiag& oldTD, tridiag& newTD )
{
    for(int i=0;i<N-1;i++)  std::get<0>(newTD)[i] = std::get<0>(oldTD)[i];
    for(int i=0;i<N;i++)  std::get<1>(newTD)[i]  = std::get<1>(oldTD)[i];
    for(int i=0;i<N-1;i++) std::get<2>(newTD)[i] = std::get<2>(oldTD)[i];
}
           
