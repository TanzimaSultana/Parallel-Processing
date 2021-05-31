
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NOT_MARK 0
#define MARK 1

int main (int argc, char ** argv) {
  
  int my_rank;
  int comm_size;
  
  double runtime;

  int n;
  int block_size;
  int start_val, end_val, start_idx;

  int *mark_array;
  int prime_idx, prime;

  int count, global_count;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  
  MPI_Barrier(MPI_COMM_WORLD);
  runtime = -MPI_Wtime();
  
  // Check for the command line argument.
  if (argc != 2) {
    if (my_rank == 0) printf("Invalid argument.\n");
    MPI_Finalize();
    exit(1);
  }
  
  n = atoi(argv[1]);

  if(my_rank == 0){
    printf("N : %d, P : %d\n", n, comm_size);
  }

  // Check no of processor between 1 - 32
  if(comm_size < 0 || comm_size > 32){
    if (my_rank == 0) printf("Invalid no of processor.\n");
    MPI_Finalize();
    exit(1);
  }
  
  // Check if all the primes used for sieving are not all held by processor zero.
  if ((2 + (n - 1 / comm_size)) < (int) sqrt((double) n)) {
    if (my_rank == 0) printf("Too many processor.\n");
    MPI_Finalize();
    exit(1);
  }
  
  // Start & End Index
  block_size = n / comm_size;
  start_val = my_rank * block_size;

  if(my_rank == comm_size - 1) end_val = n;
  else end_val = (my_rank + 1) * block_size;

  //printf(" P : %d, start : %d, end : %d\n", my_rank, start_val, end_val);
  
  mark_array = new int[block_size + 1];

  // Only marking/not-marking odd number
  if(start_val % 2 == 0) start_idx = 1;
  else start_idx = 0;

  for(int i = start_idx; i < block_size + 1; i = i + 2) mark_array[i] = NOT_MARK; 
  
  if (my_rank == 0) prime_idx = 5;
  prime = 3;
  
  while (prime * prime <= n) {

    // ----- Marking the multiples of prime ----- //
        
    int m;
    if(prime * prime > start_val) m = (prime * prime) - start_val;
    else{
        if(start_val == 0) m = prime * prime;
        else if(start_val % prime == 0) m = 0;
        else m = prime - (start_val % prime);
    }

    //printf("Prime : %d, M : %d\n", prime, m);
    
    for(int i = m; i < block_size + 1; i = i + prime){
        //printf("P : %d, idx : %d, Mark : %d\n", my_rank, i, i + start_val);
        mark_array[i] = MARK;
    }

    // ----- Select the next prime ----- //
    
    if (my_rank == 0) {
    
        while (1){
            
            if(mark_array[prime_idx] == 0){
                prime = prime_idx;
                prime_idx = prime_idx + 2;
                break;
            }
            else{
                prime_idx = prime_idx + 2;
            }
        }

        //printf("Next Prime : %d\n", prime);
    }

    if(comm_size > 1) MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);

  }

  MPI_Barrier(MPI_COMM_WORLD);  

  // ----- Count & Print the primes -----//

  //count = 0;
  if(my_rank == 0) printf("2 ");

  for(int i = start_idx; i < block_size + 1; i = i + 2) {
      if(mark_array[i] == NOT_MARK) {
          printf("%d ", i + start_val);
          //count++;
      }
  }
  

  runtime += MPI_Wtime();

  if (my_rank == 0) {
    printf("\n\nP : %d, time : %f\n\n", comm_size, runtime);
  }

  free(mark_array);
  
  MPI_Finalize();
  
  return 0;
}


