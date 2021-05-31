#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double function(double x){
	double result = 4 / (1.0 + (x * x));
	return result;
}

int main(int argc, char *argv[]){

    int my_rank;
    int comm_size;
    double runtime;

    int n, a, b;
    double x, h, start, end;
    double result, global_result;
	
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
	
	n = atol(argv[1]);
    a = 0;
    b = 1;
	
    // ----- Start & End ----- //
	start = (double) ( b - a ) / comm_size * my_rank;
	end = (double) ( b - a ) / comm_size * ( my_rank + 1.0);
    
    // ----- Each processsor calculates partial sum ----- //
    h = (end - start) / n; 
    x = start;
    result = function(x); 
    x = x + h;

    int i = 1;
    while(x < end){

        if(i % 2 == 0) result += 2 * function(x);
        else result += 4 * function(x);

        x = x + h;
        i++;
    }

    result += function(x);
    result = result / ( 3.0 ) * h;
    
    // ----- Partial sum is reduced in processor 0 -----//
    MPI_Reduce(&result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    runtime += MPI_Wtime();

    if (my_rank == 0) {
        printf("Pi : %f, time : %f\n", global_result, runtime);
    }

  	MPI_Finalize();	

    return 0;	
}