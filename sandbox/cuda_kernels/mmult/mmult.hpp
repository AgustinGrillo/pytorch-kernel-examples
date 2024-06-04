#include <cublas_v2.h>

namespace custom {
void mmult_naive(float *out, float *a, float *b, int m, int k, int n);
void mmult_cublas(cublasHandle_t cublas_handle, float *out, float *a, float *b,
                  int m, int k, int n);
void mmult_mkl_baseline(float *out, float *a, float *b, int m, int k, int n);
} // namespace custom
