#pragma once

/**
 * Provides a thin wrapper around thrust vectors to support matrix style operations.
 */
template <class vec_type> 
class Matrix
{
	public:
		vec_type vector;
		int rows, cols;

		/**
		 * Creates a matrix of size NxN. The matrix has no values explicitly set.
		 */
		Matrix(int n) : rows(n), cols(n), vector(n*n) {	}

		/**
		 * Creates a matrix of size MxN. The matrix has no values explicitly set.
		 */
		Matrix(int m, int n) : rows(m), cols(n), vector(m*n) { }

		// TODO: I don't think we actually need to do this
		///**
		// * Frees the memory allocated to contain the matrix.
		// */
		//~Matrix()
		//{
		//	vector.clear();

		//}

		/**
		 * Sets the value of coordinate (row, col) to val.
		 */
		void set(int row, int col, float val)
		{
			vector[col*rows + row] = val;
		}

		/**
		 * Returns the value of coordinate (row, col).
		 */
		float get(int row, int col)
		{
			return vector[col*rows + row];
		}

		/**
		 * Prints a string representation of the matrix.
		 */
		void print()
		{
			printf("\n");
			for(int row = 0; row < rows; row++)
			{
				for(int col = 0; col < cols; col++)
				{
					printf("%.3f ", get(row, col));
				}
				printf("\n");
			}
		}
};