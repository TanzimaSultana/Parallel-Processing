
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include<math.h>


void SHIFT(int my_rank, int comm_size, int array[], int result_array[], int shift_int){


    int n, size;
    int recv_proc;

    size = abs(shift_int);

    int *send_array = new int[size];
    int *recv_array = new int[size];

    n = sizeof(array) / sizeof(int);

    // ----- Positive Shift ----- //

    if(shift_int > 0){
        // ----- Each processor sends to the next processor ----- //
        // ----- Last processor sends to first processor
        if(my_rank == comm_size - 1) recv_proc = 0;
        else recv_proc = my_rank + 1;

        // ----- Send the last 'shift_int' no of elements to the next processor
        int idx = 0;
        for(int i = n - size; i < n; i++){
            send_array[idx++] = array[i];
        }

        MPI_Send(send_array, size, MPI_INT, recv_proc, 0, MPI_COMM_WORLD);

        if(my_rank != 0){
            MPI_Recv(recv_array, size, MPI_INT, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("\nProcessor : %d\n", my_rank + 1);
            printf("Original Array : ");
            for(int i = 0; i < n; i++){
                printf("%d ", array[i]);
            }
            printf("\n");

            for(int i = 0; i < size; i++){
                array[i + 1] = array[i];
                array[i] = recv_array[i];
            }

            printf("Result array : ");
            for(int i = 0; i < n; i++){
                printf("%d ", array[i]);
            }
            printf("\n\n");

        }
        else{
             MPI_Recv(recv_array, size, MPI_INT, comm_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("\nProcessor : %d\n", my_rank + 1);
            printf("Original Array : ");
            for(int i = 0; i < n; i++){
                printf("%d ", array[i]);
            }
            printf("\n");

            for(int i = 0; i < size; i++){
                array[i + 1] = array[i];
                array[i] = recv_array[i];
            }

            printf("Result array : ");
            for(int i = 0; i < n; i++){
                printf("%d ", array[i]);
            }
            printf("\n\n");
        }
    }

    // ----- Negative Shift ----- //

    else{
        // ----- Each processor sends to the previous processor ----- //
        // ----- First processor sends to last processor
        if(my_rank == 0) recv_proc = comm_size - 1;
        else recv_proc = my_rank - 1;

        //printf("P : %d, Recv P : %d\n", my_rank, recv_proc);

        // ----- Send the first 'shift_int' no of elements to the previous processor
        int idx = 0;
        for(int i = 0; i < size; i++){
            send_array[idx++] = array[i];
        }

        MPI_Send(send_array, size, MPI_INT, recv_proc, 0, MPI_COMM_WORLD);

        if(my_rank != comm_size - 1){
            MPI_Recv(recv_array, size, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("\nProcessor : %d\n", my_rank + 1);
            printf("Original Array : ");
            for(int i = 0; i < n; i++){
                printf("%d ", array[i]);
            }
            printf("\n");


            int j = n - 1;
            for(int i = size - 1; i >= 0; i--){

                if(j < 0) j = 1;

                array[j - 1] = array[j];
                array[j] = recv_array[i];
                j--;
            }

            printf("Result array : ");
            for(int i = 0; i < n; i++){
                printf("%d ", array[i]);
            }
            printf("\n\n");

        }
        // ----- Last Processor ----- //
        else{
             MPI_Recv(recv_array, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("\nProcessor : %d\n", my_rank + 1);
            printf("Original Array : ");
            for(int i = 0; i < n; i++){
                printf("%d ", array[i]);
            }
            printf("\n");

            int j = n - 1;
            for(int i = size - 1; i >= 0; i--){

                if(j < 0) j = 1;

                array[j - 1] = array[j];
                array[j] = recv_array[i];
                j--;
            }

            printf("Result array : ");
            for(int i = 0; i < n; i++){
                printf("%d ", array[i]);
            }
            printf("\n\n");
        }
    }
}

int main (int argc, char ** argv) {

    int my_rank;
    int comm_size;
    double runtime;

    int n;
    int *array, *result_array;
    int shift_int;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    runtime = -MPI_Wtime();
    
    n = atoi(argv[1]);

    if(my_rank == 0){
        printf("N : %d, P : %d\n", n, comm_size);
    }

    array = new int[n];
    result_array = new int[n];

    // ----- Input ----- //
    int start_idx = (my_rank * n) + 2;
    int idx = 0;
    for(int i = start_idx; i < start_idx + n;i++){
        array[idx++] = atoi(argv[i]);
    }

    // ----- Shift Integer ----- //

    shift_int = atoi(argv[(n * comm_size) + 2]);
    //printf("Shift Int : %d\n", shift_int);

    if (shift_int > n) {
        if (my_rank == 0) printf("Invalid shift integer.\n");
        MPI_Finalize();
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    SHIFT(my_rank, comm_size, array, result_array, shift_int);

    delete [] array;
    delete [] result_array;
    
    MPI_Finalize();
  
  return 0;
}


