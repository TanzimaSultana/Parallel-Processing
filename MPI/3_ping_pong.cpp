#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LOOP 1000
#define B 1
#define KB 2
#define BYTE 1024

int main(int argc, char *argv[]){

    int my_rank;
    int comm_size;
    double runtime;

    long int b1, b2, incr;
    int op;
	
  	MPI_Init(&argc, &argv);
  	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  
  	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Check for the command line argument.
    if (argc != 5) {
        if (my_rank == 0) printf("Invalid argument.\n");
        MPI_Finalize();
        exit(1);
    }

    op = atol(argv[1]);
    b1 = atol(argv[2]);
    b2 = atol(argv[3]);
    incr = atol(argv[4]);

    // Check no of processor is 2
    if(comm_size != 2){
        if(my_rank == 0) printf("Invalid no of processor.\n");
        MPI_Finalize();
        exit(0);
    }

    // bytes
    if(op == KB){
        b1 = b1 * BYTE;
        b2 = b2 * BYTE;
        incr = incr * BYTE;
    }
    else{
        b1 = 32;
    }

    for(long int bytes = b1; bytes <= b2; bytes += incr){

        long int size = bytes / 8;
        //printf("size : %d\n", size);

        double *array = (double*)malloc(size*sizeof(double));
        for(int i = 0; i < size; i++){
            array[i] = 0.0;
        }

        int tag1 = 10;
        int tag2 = 20;

        double start_time, end_time, elapsed_time;
        start_time = MPI_Wtime();

        double comm_start, comm_end, comm_time;
        comm_time = 0.0;

        // Time ping-pong for loop_count

        for(int i = 0; i <= LOOP; i++){
            if(my_rank == 0){

                comm_start = MPI_Wtime();

                MPI_Send(array, size, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(array, size, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                comm_end = MPI_Wtime();
                comm_time += (comm_end - comm_start);
            }
            else if(my_rank == 1){
                MPI_Recv(array, size, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(array, size, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }

        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;

        double latency = elapsed_time / (2.0*(double)LOOP);
        double communication_time = comm_time/(double)LOOP;

        if(my_rank == 0) {
            if(op == KB)
                printf("%d, %15.9f, %15.9f\n", bytes / BYTE, latency, communication_time);
            else
                printf("%d, %15.9f, %15.9f\n", bytes, latency, communication_time);
            
        }

        free(array);
    }

  	MPI_Finalize();	

    return 0;	
}