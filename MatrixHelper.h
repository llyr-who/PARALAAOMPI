/*
*  Filename: MatrixHelper.h
*
*
*
*/

#ifndef _MATRIXHELP
#define _MATRIXHELP

#include <iostream>
#include <math.h>
#include<complex>

// A simple implemtation of a tridiagonal matrix that is convenient to work with
typedef std::tuple<std::complex<double>*,std::complex<double>*,std::complex<double>*> tridiag;


// Create a "vector of vectors" matrix"
std::complex<double> ** CreateMatrix(int m,int n);

// Create a contiguous matrix.
// Returns an array length m*n
std::complex<double> * CreateMatrixContiguous(int m, int n);

// Create a contiguous tridiagonal square matrix.
// Returns an array of length 3*n-2
std::complex<double> * CreateTDMatrixContiguous(int n);

// Create a tridiagonal matrix.
// This reserves memory for our convenient tridiag structure
void CreateTridiag(int N,tridiag & TD);

// Destroys "vector of vector" matrices.
void DestroyMatrix(std::complex<double> ** mat, int m, int n); 

// Memory clean-up routines need to be implemented for
// the other types of matrices used


// This function sets a tridiagonal matrix equal to another one
void SetTriDiagEqualTo(int N, tridiag& oldTB,tridiag& newTD);



#endif

