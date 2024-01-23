#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <numaif.h> 
#define MAX_MPI_SIZE 5
#define MAX_OMP_THREADS_PER_MPI 5

// A simple MPI+OpenMP+CUDA program, which contains memory reuse strategy
// 一段简单的MPI+OpenMP+CUDA程序，其中包含了内存复用策略

void cuda_test(int is, float *d_data1, float *d_data2, float *d_result, int size);


int main(int argc, char* argv[]) {
    int myid, numprocs;
    int ns = 50; // 震源数量 Shots number

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Barrier(MPI_COMM_WORLD);
    // 若numprocs大于MAX_MPI_SIZE,则定义MPI_SIZE为MAX_MPI_SIZE,否则MPI_SIZE为用户设置的值(numprocs)
    // If numprocs is greater than MAX_MPI_SIZE, then define MPI_SIZE as MAX_MPI_SIZE, otherwise MPI_SIZE is the value set by the user (numprocs)
    int MPI_SIZE = numprocs > MAX_MPI_SIZE ? MAX_MPI_SIZE : numprocs;
    
    // 计算子集大小, 确保所有ns都被计算
    // Calculate the subset size to ensure that all ns are calculated
    int subset_size = (ns + MPI_SIZE - 1) / MPI_SIZE;
    
    int size = 5;

    if (myid == 0)
        printf("subset_size=%d, numprocs=%d\n", subset_size, MPI_SIZE);
    for(int subset=myid; subset<MPI_SIZE; subset=subset+numprocs)
    {
        // 直接设置OpenMP线程数量
        // Set the number of OpenMP threads directly
        #pragma omp parallel num_threads(MAX_OMP_THREADS_PER_MPI)
        {
            float *d_data1, *d_data2, *d_result;
            cudaMalloc((void**)&d_data1, sizeof(float) * size);
            cudaMalloc((void**)&d_data2, sizeof(float) * size);
            cudaMalloc((void**)&d_result, sizeof(float) * size);
#pragma omp  for  
            for(int is=subset*subset_size; is<(subset+1)*subset_size; is++) 
            {
                if (is < ns)
                {    
                    cudaMemset(d_data1, 0, sizeof(float) * size); // 重置内存 Reset memory
                    cudaMemset(d_data2, 0, sizeof(float) * size); // 重置内存
                    cudaMemset(d_result, 0, sizeof(float) * size); // 重置内存
                    printf("MPI %d, OpenMP %d, is=%d\n", myid, omp_get_thread_num(), is);
                    cuda_test(is, d_data1, d_data2, d_result, size); 
                }
            }
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}