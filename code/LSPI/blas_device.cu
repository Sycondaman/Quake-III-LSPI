/**
 * Provides a wrapper around cublas for computation on the GPU.
 */

#include "stdafx.h"
#include "blas.h"

cublasHandle_t blas::handle;

/**
* Computes x = alpha*x. 
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::scal(device_vector<float>& x, float alpha)
{
	cublasStatus_t status = cublasSscal(blas::handle, (int)x.size(), &alpha, raw_pointer_cast(x.data()), 1);

	if(status == CUBLAS_STATUS_SUCCESS)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}

/**
* Computes result = x dot y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::dot(const device_vector<float>& x, const device_vector<float>& y, float& result)
{
	cublasStatus_t status = cublasSdot(blas::handle, (int)x.size(), raw_pointer_cast(x.data()), 1, raw_pointer_cast(y.data()), 1, &result);

	if(status == CUBLAS_STATUS_SUCCESS)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}
	
/**
* Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, float alpha, float beta, bool transpose)
{
	cublasStatus_t status;
	
	if(transpose)
	{
		status = cublasSgemv(blas::handle, CUBLAS_OP_T, A.rows, A.cols, &alpha, raw_pointer_cast(A.vector.data()), A.rows, raw_pointer_cast(x.data()),
						     1, &beta, raw_pointer_cast(y.data()), 1);
	}
	else
	{
		status = cublasSgemv(blas::handle, CUBLAS_OP_N, A.rows, A.cols, &alpha, raw_pointer_cast(A.vector.data()), A.rows, raw_pointer_cast(x.data()),
							 1, &beta, raw_pointer_cast(y.data()), 1);
	}

	if(status == CUBLAS_STATUS_SUCCESS)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}

/**
* Computes y = alpha*A*x. For alpha*x*A set transpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, float alpha, bool transpose)
{
	return blas::gemv(A, x, y, alpha, 0.0, transpose);
}

/**
* Computes y = alpha*A*x. For alpha*x*A set tranpose to true.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, bool transpose)
{
	return blas::gemv(A, x, y, 1.0, 0.0, transpose);
}

/**
* Computes y = alpha*x + y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::axpy(const device_vector<float>& x, device_vector<float>& y, float alpha)
{
	cublasStatus_t status = cublasSaxpy(blas::handle, (int)x.size(), &alpha, raw_pointer_cast(x.data()), 1, raw_pointer_cast(y.data()), 1);
	
	if(status == CUBLAS_STATUS_SUCCESS)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}

/**
* Computes y = x + y
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::axpy(const device_vector<float>& x, device_vector<float>& y)
{
	return blas::axpy(x, y, 1.0);
}

/**
* Computes A = alpha*x*y.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::ger(const device_vector<float>& x, const device_vector<float>& y, Matrix<device_vector<float>>& A, float alpha)
{
	cublasStatus_t status = cublasSger(blas::handle, (int)x.size(), (int)y.size(), &alpha, raw_pointer_cast(x.data()), 1, raw_pointer_cast(y.data()), 1,
									   raw_pointer_cast(A.vector.data()), A.rows);

	if(status == CUBLAS_STATUS_SUCCESS)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}

/**
* Computes A = x*y.
* Returns 0 if the operation was successful, an error code otherwise
*/
int blas::ger(const device_vector<float>& x, const device_vector<float>& y, Matrix<device_vector<float>>& A)
{
	return blas::ger(x, y, A, 1.0);
}