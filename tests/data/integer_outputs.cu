// Small kernels for tests/test_integer_outputs.py; regenerate the PTX with
//   nvcc -ptx -arch=compute_89 -o integer_outputs.ptx integer_outputs.cu
extern "C" __global__ void add_i32(const int *a, const int *b, int *c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

extern "C" __global__ void widen_i8(const signed char *in, int *out) {
  int i = threadIdx.x;
  out[i] = (int) in[i];
}

extern "C" __global__ void sum_i32(const int *a, int *out) {
  if (threadIdx.x == 0) {
    int s = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      s += a[k];
    }
    out[0] = s;
  }
}
