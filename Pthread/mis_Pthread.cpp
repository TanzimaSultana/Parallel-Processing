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
#include <pthread.h>
#include <sys/time.h>
#include "ECLgraph.h"

static const unsigned char in = 2;
static const unsigned char out = 1;
static const unsigned char undecided = 0;

// shared variables
static long thread_count;
static unsigned char* status;
static unsigned int* rndval;
static ECLgraph g;

static pthread_barrier_t   barrier;

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static void* mis(void* arg)
{
  const long my_rank = (long)arg;
  
  const int beg = my_rank * (long)g.nodes / thread_count;
  const int end = (my_rank + 1) * (long)g.nodes / thread_count;

  // initialize arrays
  //static unsigned int* const random = new unsigned int [g.nodes];
  for (int v = beg; v < end; v++) status[v] = undecided;
  for (int v = beg; v < end; v++) rndval[v] = hash(v + 1);

  pthread_barrier_wait(&barrier);

  bool missing;
  // repeat until all nodes' status has been decided
  do {
    missing = false;
    // go over all the nodes
    for (int v = beg; v < end; v++) {

      if (status[v] == undecided) {
        int i = g.nindex[v];
        // try to find a neighbor whose random number is lower
        while ((i < g.nindex[v + 1]) && ((status[g.nlist[i]] == out) || (rndval[v] < rndval[g.nlist[i]]) || ((rndval[v] == rndval[g.nlist[i]]) && (v < g.nlist[i])))) {
          i++;
        }
        if (i < g.nindex[v + 1]) {
          // found such a neighbor -> status still unknown
          missing = true;
        } else {
          // no such neighbor -> status is "in" and all neighbors are "out"
          status[v] = in;
          for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
            status[g.nlist[i]] = out;
          }
        }
      }

    }

  } while (missing);

  return NULL;
}

int main(int argc, char* argv[])
{
  printf("Maximal Independent Set Pthread\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s input_file\n", argv[0]); exit(-1);}
  thread_count = atol(argv[2]);
  if (thread_count < 1) {fprintf(stderr, "ERROR: threads must be at least 1\n"); exit(-1);}

  // read input
  g = readECLgraph(argv[1]);
  printf("configuration: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);
  printf("threads: %ld\n", thread_count);

  // allocate arrays
  status = new unsigned char [g.nodes];
  rndval = new unsigned int [g.nodes];

  // initialize pthread variables
  pthread_t* const handle = new pthread_t [thread_count - 1];
  
  // barrier
  pthread_barrier_init (&barrier, NULL, thread_count);

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  //mis(g, status, random);

  // launch threads
  for (long thread = 0; thread < thread_count - 1; thread++) {
    pthread_create(&handle[thread], NULL, mis, (void *)thread);
  }

  // work for master
  mis((void*)(thread_count - 1));

  // join threads
  for (long thread = 0; thread < thread_count - 1; thread++) {
    pthread_join(handle[thread], NULL);
  }

  // end time
  gettimeofday(&end, NULL);

  pthread_barrier_destroy(&barrier);

  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

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
  freeECLgraph(g);
  delete [] status;
  delete [] rndval;
  return 0;
}

