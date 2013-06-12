/*
 * Enumerated and derived types
 */
#define NORMAL 'N'
#define TRANSPOSE 'T'

#ifdef __cplusplus
extern "C" {
#endif

void sgemv_(const char *trans, const int *m, const int *n, const float *alpha, const float *a, const int *lda, 
		   const float *x, const int *incx, const float *beta, float *y, const int *incy);

void sscal_(const int *n, const float *sa, float *sx, const int *incx);

double sdot_(const int *n, const float *sx, const int *incx, const float *sy, const int *incy);

void saxpy_(const int *n, const float *sa, const float *sx, const int *incx, float *sy, const int *incy);

void sger_(const int *m, const int *n, const float *alpha, const float *x, const int *incx, const float *y,
		  const int *incy, float *a, const int *lda);

#ifdef __cplusplus
}
#endif