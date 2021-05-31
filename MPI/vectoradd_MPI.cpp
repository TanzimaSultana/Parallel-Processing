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
#include <mpi.h>

static void vadd(int c[], const int a[], const int b[], const int my_start, const int my_end)
{
  // perform vector addition
  for (int i = my_start; i < my_end; i++) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char *argv[])
{
  // set up MPI
  int comm_sz, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) printf("Vector addition v1.0\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s vector_size\n", argv[0]); exit(-1);}
  const int size = atoi(argv[1]);
  if (size < 8) {fprintf(stderr, "ERROR: vector_size must be at least 8\n"); exit(-1);}
  if ((size % comm_sz) != 0) {fprintf(stderr, "ERROR: vector_size must be a multiple of the number of processes\n"); exit(-1);}
  if (my_rank == 0) {
    printf("vector size: %ld\n", size);
    printf("processes: %d\n", comm_sz);
  }

  // compute range
  const int my_start = my_rank * (long)size / comm_sz;
  const int my_end = (my_rank + 1) * (long)size / comm_sz;
  const int range = my_end - my_start;

  // allocate vectors
  int* const a = new int [size];
  int* const b = new int [size];
  int* const c = new int [size];
  int* res = NULL;
  if (my_rank == 0) res = new int [size];

  // initialize vectors
  for (int i = my_start; i < my_end; i++) a[i] = i;
  for (int i = my_start; i < my_end; i++) b[i] = size - i;
  for (int i = my_start; i < my_end; i++) c[i] = -1;

  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);  // for better timing
  gettimeofday(&start, NULL);

  // execute timed code
  vadd(c, a, b, my_start, my_end);

  // gather the resulting frames
  MPI_Gather(&c[my_start], range, MPI_INT, res, range, MPI_INT, 0, MPI_COMM_WORLD);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  if (my_rank == 0) {
    printf("compute time: %.4f s\n", runtime);

    // verify result
    for (int i = 0; i < size; i++) {
      if (res[i] != size) {fprintf(stderr, "ERROR: incorrect result\n"); exit(-1);}
    }
    printf("sum: %d\n", *res);
    printf("verification passed\n");

    delete [] res;
  }

  // clean up
  MPI_Finalize();
  delete [] a;
  delete [] b;
  delete [] c;
  return 0;
}
