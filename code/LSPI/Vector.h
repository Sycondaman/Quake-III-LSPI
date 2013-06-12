/**
 * Represents a vector object which performs internal calculations using CUDA.
 * Requires a cublas handle and expects that CUDA has already been initialized.
 */

#include <cublas_v2.h>

class Vector
{
	public:
		/**
		 * Creates a vector. The cublas handle specified will be used
		 * to execute all GPU accelerated operations. The vector has no values explicitly set.
		 */
		Vector(int size, cublasHandle_t handle);

		/**
		 * Frees the memory allocated to contain the vector.
		 */
		~Vector();

		/**
		 * Sets the vector to all zeros.
		 */
		void makeZeros();

		/**
		 * Sets the value of the xth element to val.
		 */
		void set(int x, float val);

		/**
		 * Returns the value of coordinate (row, col).
		 */
		float get(int row, int col);

		/**
		 * Returns a string representation of the matrix (each row is on a new line).
		 */
		char* toString();
};