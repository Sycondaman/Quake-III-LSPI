#pragma once

#include "Matrix.h"
#include <cublas_v2.h>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>

using namespace thrust;

namespace blas
{
	extern cublasHandle_t handle;

	//********** HOST CALLS **********//

	/**
	 * Computes x = alpha*x. 
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int scal(host_vector<float>& x, float alpha); 

	/**
	 * Computes result = x dot y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int dot(const host_vector<float>& x, const host_vector<float>& y, float& result);
	
	/**
	 * Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, float beta, bool transpose);

	/**
	 * Computes y = alpha*A*x. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, float alpha, bool transpose);

	/**
	 * Computes y = A*x. For x*A set tranpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<host_vector<float>>& A, const host_vector<float>& x, host_vector<float>& y, bool transpose);

	/**
	 * Computes y = alpha*x + y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int axpy(const host_vector<float>& x, host_vector<float>& y, float alpha);

	/**
	 * Computes y = x + y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int axpy(const host_vector<float>& x, host_vector<float>& y);

	// TODO: Provide an optimized implementation for just computing alpha*x*y^T
	/**
	 * Computes A = alpha*x*y + A.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int ger(const host_vector<float>& x, const host_vector<float>& y, Matrix<host_vector<float>>& A, float alpha);

	/**
	 * Computes A = x*y + A.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int ger(const host_vector<float>& x, const host_vector<float>& y, Matrix<host_vector<float>>& A);

	//********** DEVICE CALLS **********//

	/**
	 * Computes x = alpha*x. 
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int scal(device_vector<float>& x, float alpha); 

	/**
	 * Computes result = x dot y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int dot(const device_vector<float>& x, const device_vector<float>& y, float& result);
	
	/**
	 * Computes y = alpha*A*x + beta*y. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, float alpha, float beta, bool transpose);

	/**
	 * Computes y = alpha*A*x. For alpha*x*A set transpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, float alpha, bool transpose);

	/**
	 * Computes y = A*x. For x*A set tranpose to true.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int gemv(const Matrix<device_vector<float>>& A, const device_vector<float>& x, device_vector<float>& y, bool transpose);

	/**
	 * Computes y = alpha*x + y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int axpy(const device_vector<float>& x, device_vector<float>& y, float alpha);

	/**
	 * Computes y = x + y
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int axpy(const device_vector<float>& x, device_vector<float>& y);

	/**
	 * Computes A = alpha*x*y.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int ger(const device_vector<float>& x, const device_vector<float>& y, Matrix<device_vector<float>>& A, float alpha);

	/**
	 * Computes A = x*y.
 	 * Returns 0 if the operation was successful, an error code otherwise
	 */
	int ger(const device_vector<float>& x, const device_vector<float>& y, Matrix<device_vector<float>>& A);
};