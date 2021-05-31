/*
Maximal independent set code for CS 4380 / CS 5351

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
#include <cuda.h>
#include "ECLgraph.h"

static const unsigned char in = 2;
static const unsigned char out = 1;
static const unsigned char undecided = 0;

static const int ThreadsPerBlock = 512;

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
__device__ unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static __global__ void init(const ECLgraph d_g, unsigned char* const d_status, unsigned int* const d_random)
{
  // initialize arrays
  const int v = threadIdx.x + blockIdx.x * blockDim.x;

  if(v < d_g.nodes)
  {
    d_status[v] = undecided;
    d_random[v] = hash(v + 1);
  }

  //for (int v = 0; v < g.nodes; v++) status[v] = undecided;
  //for (int v = 0; v < g.nodes; v++) random[v] = hash(v + 1);
}

static __global__ void mis(const ECLgraph d_g, bool* d_missing)
{

  const int v = threadIdx.x + blockIdx.x * blockDim.x;
  if(v < d_g.nodes)
  {

    if (d_status[v] == undecided) 
    {

      int i = d_g.nindex[v];
      // try to find a neighbor whose random number is lower
      while ((i < d_g.nindex[v + 1]) && ((d_status[d_g.nlist[i]] == out) || (d_random[v] < d_random[d_g.nlist[i]]) || ((d_random[v] == d_random[d_g.nlist[i]]) && (v < d_g.nlist[i])))) {
        i++;
      }
      if (i < d_g.nindex[v + 1]) {
        // found such a neighbor -> status still unknown
        *d_missing = true;
      } else {
        // no such neighbor -> status is "in" and all neighbors are "out"
        d_status[v] = in;
        for (int i = d_g.nindex[v]; i < d_g.nindex[v + 1]; i++) {
          d_status[d_g.nlist[i]] = out;
        }
      }

    }

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

int main(int argc, char* argv[])
{
  printf("Maximal Independent Set CUDA\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s input_file\n", argv[0]); exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("configuration: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);

  // allocate arrays
  unsigned char* const status = new unsigned char [g.nodes];
  unsigned int* const random = new unsigned int [g.nodes];

  // ---- GPU ----- //
  ECLgraph d_g = g;
  cudaMalloc((void **)&d_g.nindex, sizeof(int) * (g.nodes + 1));
  cudaMalloc((void **)&d_g.nlist, sizeof(int) * g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice);

  unsigned char* d_status;
  unsigned int* d_random;
  if (cudaSuccess != cudaMalloc((void **)&d_status, sizeof(unsigned char) * g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n");
  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_random, sizeof(unsigned int) * g.nodes)) {fprintf(stderr, "ERROR: could not allocate memory\n"); 
  exit(-1);}

  // initialize vectors on GPU
  if (cudaSuccess != cudaMemcpy(d_status, status, sizeof(unsigned char) * g.nodes, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); 
  exit(-1);}
  if (cudaSuccess != cudaMemcpy(d_random, random, sizeof(unsigned int) * g.nodes, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); 
  exit(-1);}

  // ----- GPU ----- //

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_status, d_random);

  bool* missing;
  bool* d_missing;
  if (cudaSuccess != cudaMalloc((void **)&d_missing, sizeof(bool) * 1)) {fprintf(stderr, "ERROR: could not allocate memory\n"); 
  exit(-1);}

  // repeat until all nodes' status has been decided
  do {

    *missing = false;

    if (cudaSuccess != cudaMemcpy(d_missing, missing, sizeof(bool) * 1, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n");
    exit(-1);}

    //mis<<<(d_g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_status, d_random, d_missing);
    mis<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_missing);

    // get result from GPU
    CheckCuda();
    if (cudaSuccess != cudaMemcpy(missing, d_missing, sizeof(bool) * 1, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); 
    exit(-1);}

  }while (*missing);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // Copy from GPU
  CheckCuda();
  if (cudaSuccess != cudaMemcpy(status, d_status, sizeof(unsigned char) * g.nodes, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); 
  exit(-1);}

  // determine and print set size
  int count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == in) {
      count++;
    }
  }
  printf("elements in set: %d (%.1f%%)\n", count, 100.0 * count / g.nodes);

  // verify result
  for (int v = 0; v < g.nodes; v++) {
    if ((status[v] != in) && (status[v] != out)) {fprintf(stderr, "ERROR: found unprocessed node\n"); exit(-1);}
    if (status[v] == in) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n"); exit(-1);}
      }
    } else {
      bool flag = true;
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {
          flag = false;
          break;
        }
      }
      if (flag) {fprintf(stderr, "ERROR: set is not maximal\n"); exit(-1);}
    }
  }

  // clean up

  
  cudaFree(d_status);
  cudaFree(d_random);
  cudaFree(d_missing);

  cudaFree(d_g.nindex);
  cudaFree(d_g.nlist);

  freeECLgraph(g);
  delete [] status;
  delete [] random;
  return 0;
}

