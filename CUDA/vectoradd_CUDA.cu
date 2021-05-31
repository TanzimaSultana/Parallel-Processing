/*
Vector addition code for CS 4380 / CS 5351

Copyright (c) 2019 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <sys/time.h>

static const int ThreadsPerBlock = 512;

static __global__ void vadd(int c[], const int a[], const int b[], const int size)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  // perform vector addition
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  printf("Vector addition v1.0\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s vector_size\n", argv[0]); exit(-1);}
  const int size = atoi(argv[1]);
  if (size < 8) {fprintf(stderr, "ERROR: vector_size must be at least 8\n"); exit(-1);}
  printf("vector size: %ld\n", size);

  // allocate vectors
  int* const a = new int [size];
  int* const b = new int [size];
  int* const c = new int [size];

  // initialize vectors
  for (int i = 0; i < size; i++) a[i] = i;
  for (int i = 0; i < size; i++) b[i] = size - i;
  for (int i = 0; i < size; i++) c[i] = -1;

  // allocate vectors on GPU
  int* d_a;
  int* d_b;
  int* d_c;
  if (cudaSuccess != cudaMalloc((void **)&d_a, sizeof(int) * size)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_b, sizeof(int) * size)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_c, sizeof(int) * size)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  // initialize vectors on GPU
  if (cudaSuccess != cudaMemcpy(d_a, a, sizeof(int) * size, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
  if (cudaSuccess != cudaMemcpy(d_b, b, sizeof(int) * size, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
  if (cudaSuccess != cudaMemcpy(d_c, c, sizeof(int) * size, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  vadd<<<(size + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_c, d_a, d_b, size);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // get result from GPU
  CheckCuda();
  if (cudaSuccess != cudaMemcpy(c, d_c, sizeof(int) * size, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  // verify result
  for (int i = 0; i < size; i++) {
    if (c[i] != size) {fprintf(stderr, "ERROR: incorrect result\n"); exit(-1);}
  }
  printf("verification passed\n");

  // clean up
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete [] a;
  delete [] b;
  delete [] c;
  return 0;
}
