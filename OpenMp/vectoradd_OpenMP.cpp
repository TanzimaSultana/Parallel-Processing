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
#include <sys/time.h>

static void vadd(int c[], const int a[], const int b[], const int size, const int threads)
{
  // perform vector addition
  #pragma omp parallel for default(none) shared(a, b, c) num_threads(threads)
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char *argv[])
{
  printf("Vector addition v1.0\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s vector_size thread_count\n", argv[0]); exit(-1);}
  const int size = atoi(argv[1]);
  if (size < 8) {fprintf(stderr, "ERROR: vector_size must be at least 8\n"); exit(-1);}
  printf("vector size: %d\n", size);
  const int threads = atoi(argv[2]);
  if (threads < 1) {fprintf(stderr, "ERROR: thread_count must be at least 1\n"); exit(-1);}
  printf("thread count: %d\n", threads);

  // allocate vectors
  int* const a = new int [size];
  int* const b = new int [size];
  int* const c = new int [size];

  // initialize vectors
  for (int i = 0; i < size; i++) a[i] = i;
  for (int i = 0; i < size; i++) b[i] = size - i;
  for (int i = 0; i < size; i++) c[i] = -1;

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  vadd(c, a, b, size, threads);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // verify result
  for (int i = 0; i < size; i++) {
    if (c[i] != size) {fprintf(stderr, "ERROR: incorrect result\n"); exit(-1);}
  }
  printf("verification passed\n");

  // clean up
  delete [] a;
  delete [] b;
  delete [] c;
  return 0;
}
