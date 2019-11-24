/**************************************************************************
*
* Author: llyr-who
*
* Discription:
*
*   This file provided all the parallel routines needed to implment
*   a simple version of the all at once method.
*
*
*************************************************************************/


#ifndef _PARAROUTINES
#define _PARAROUTINES

#include <iostream>
#include <math.h>
#include <vector>
#include <complex>
#include <tuple>
#include "MatrixHelper.h"

// Serial Procedures

void MultiplyTriDiagByConst(int N, std::complex<double> coeff, tridiag& matrix);
void AddTriDiag(int N, tridiag& matrix1,tridiag& matrix2,tridiag& matrix3);
void FormMassStiff(double h,int N,tridiag& mass, tridiag& stiff);
void FormFourier_Diag_FourierTranspose(int N,std::complex<double>*F,std::complex<double>*D,std::complex<double>*Ft);
void FFT(std::complex<double>*x,int N,int inverse);

// Parallel Procedures

void MatVecMultiplication(int mynode, int numnodes,int N, std::complex<double> **A,std::complex<double> *x,std::complex<double>* y);
void VecTranspose(int mynode,int numnodes,int N,int L, std::complex<double>*x,std::complex<double>*z);
void BlockTriDiagSolve_Thomas(int mynode, int numnodes,int N,int L, std::vector<std::complex<double> *>&blocks,std::complex<double> *x,std::complex<double>* q);
void BlockMatVecMultiplication(int mynode, int numnodes,int N,int L, std::vector<std::complex<double>*>&blocks,std::complex<double> *x,std::complex<double>* y);


void MultiplicationByUKronIdentity(int mynode, int numnodes,int N,int L,std::complex<double>**U,std::complex<double> *x,std::complex<double>* y);
void MultiplicationByIdentityKronU_usingFFT(int mynode, int numnodes,int N,int L,std::complex<double> *x,std::complex<double>* y,int inverse);

void MultiplyByHeatSystem(int mynode, int numnodes,int N,int L, std::vector<std::complex<double> *>&blocks,std::complex<double>*mass,std::complex<double> *x,std::complex<double>* y);
void MultiplyByWaveSystem(int mynode, int numnodes,int N,int L, std::vector<std::complex<double> *>&blocks,std::complex<double>*mass,std::complex<double> *x,std::complex<double>* y);



void DotProduct(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>* y,std::complex<double>&result);
void VectorAddition(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>* y,std::complex<double>*result);
void VectorSubtraction(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>* y,std::complex<double>*result);

void ApplyPreconditioner(int mynode,int numnodes,int N, int L,std::complex<double>**U,std::complex<double>**Ut,std::vector<std::complex<double>*>& Wblks,std::complex<double>* q, std::complex<double>* x);

void ApplyPreconditionerFFT(int mynode,int totalnodes,int N, int L,std::vector<std::complex<double>*>& Wblks,std::complex<double>* q, std::complex<double>* x);

void ApplyNonUniformPreconditionerWithOneTerm(int mynode,int totalnodes,int N, int L,std::complex<double>**U,std::complex<double>**Ut,std::vector<std::complex<double>*>& Wblks,std::vector<std::complex<double>*> sigmaKronK,std::complex<double>* q, std::complex<double>* x);



void CalculateNorm(int mynode, int totalnodes,int N, int L,std::complex<double>* vectocalnorm, std::complex<double>& result);

void SetEqualTo(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>*q,std::complex<double> scalar);
void PlusEqualTo(int mynode, int numnodes,int N,int L, std::complex<double> *x,std::complex<double>*q,std::complex<double>scalar);

#endif
