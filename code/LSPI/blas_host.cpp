/**
 * Provides a wrapper around cblas and some custom blas-like extension functions for computation on the CPU.
 */

#include "stdafx.h"
#include "blas.h"
#include "cblas.h"

using namespace thrust;

static int inc = 1;

/**
* Computes x = alpha*x. 
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::scal(host_vector<float>& x, float alpha)
{
	int m = (int)x.size();
	sscal_(&m, &alpha, raw_pointer_cast(x.data()), &inc);
	return 0;
}

/**
* Computes result = x dot y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::dot(const host_vector<float>& x, const host_vector<float>& y, float& result)
{
	int m = (int)x.size();
	result = sdot_(&m, raw_pointer_cast(x.data()), &inc, raw_pointer_cast(y.data()), &inc);
	return 0;
}
	
/**
* Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, float beta, bool transpose)
{
	char trans;
	if(transpose)
	{
		trans = TRANSPOSE;
	}
	else
	{
		trans = NORMAL;
	}

	sgemv_(&trans, &A.rows, &A.cols, &alpha, raw_pointer_cast(A.vector.data()), &A.rows, raw_pointer_cast(x.data()), &inc, &beta, 
		   raw_pointer_cast(y.data()), &inc);
	return 0;
}

/**
* Computes y = alpha*A*x. For alpha*x*A set transpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, bool transpose)
{
	return blas::gemv(A, x, y, alpha, 0.0, transpose);
}

/**
* Computes y = A*x. For x*A set tranpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, bool transpose)
{
	return blas::gemv(A, x, y, 1.0, 0.0, transpose);
}

/**
* Computes y = alpha*x + y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::axpy(const host_vector<float>& x, host_vector<float>& y, float alpha)
{
	int m = (int)x.size();
	saxpy_(&m, &alpha, raw_pointer_cast(x.data()), &inc, raw_pointer_cast(y.data()), &inc);
	return 0;
}

/**
* Computes y = x + y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::axpy(const host_vector<float>& x, host_vector<float>& y)
{
	return blas::axpy(x, y, 1.0);
}

/**
* Computes A = alpha*x*y.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::ger(const host_vector<float>& x, const host_vector<float>& y, Matrix<host_vector<float>>& A, float alpha)
{
	int m = (int)x.size();
	int n = (int)y.size();
	sger_(&m, &n, &alpha, raw_pointer_cast(x.data()), &inc, raw_pointer_cast(y.data()), &inc, raw_pointer_cast(A.vector.data()), &A.rows);
	return 0;
}

/**
* Computes A = x*y.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::ger(const host_vector<float>& x, const host_vector<float>& y, Matrix<host_vector<float>>& A)
{
	blas::ger(x, y, A, 1.0);
	return 0;
}